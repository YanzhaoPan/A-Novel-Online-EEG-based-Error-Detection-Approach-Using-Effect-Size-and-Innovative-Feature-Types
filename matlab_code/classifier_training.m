% Script Description:
%
% This script processes EEG data for multiple subjects to train and evaluate 
% a Support Vector Machine (SVM) classifier. The process involves reading EEG 
% files, preprocessing and epoching the data, calculating noise thresholds, 
% and extracting features. The features include temporal averages, differences, 
% and dynamics. The script then selects the most discriminative features 
% using various measures and combines these features for classifier training.
%
% Author: Yanzhao Pan
% Date Created: 20.06.2023
% Last Modified: 27.11.2023
% Affiliation: Intuitive XR Labs, BTU Cottbus-Senftenberg, Germany
%
% Key Steps:
% 1. Reading EEG Data: Loads EEG files for each subject.
% 2. Preprocessing and Epoching: Applies preprocessing steps and extracts epochs 
%    for target and distractor stimuli.
% 3. Noise Threshold Calculation: Determines a threshold to identify noise in EEG epochs (used in online stage).
% 4. Feature Extraction: Extracts various features like temporal average, difference, 
%    and dynamics from the epochs.
% 5. Feature Selection: Selects the top features based on their discriminative power.
% 6. Classifier Training: Trains an SVM classifier using the selected features and 
%    evaluates its performance with cross-validation.
%
% Input Data:
% - EEG files for each subject (IJCAI'23 CC6 Competition) [Data set]
%
% Output:
% - Trained SVM Model
% - Validation accuracy
% - Average confusion matrices for training and validation
% - Struct of confusion matrices for each fold in cross-validation
%
% External Dependencies:
% - EEGLAB toolbox for EEG preprocessing
% - SMOTE for class balancing in training data
%
% Note:
% - The script is part of a larger project on EEG and EMG dataset analysis 
%   for error detection in the context of an active orthosis device.
%

eeglab;
subjects = {'AA56D', 'AC17D', 'AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D'};
n_subjects = length(subjects);

% Initialize struct to store evaluation metrics for all participants
evaluation_metrics = struct('Subject', [], 'TPR', [], 'TNR', [], 'BalancedAccuracy', []);

for sub = 1:n_subjects
    this_subject = subjects{sub};
    
    %% classifier training
    % Get all the EEG files, refer to {Kueper, N., Chari, K., Bütefür, J., Habenicht, J., Rossol, T., Kim, S. K., Tabie, M., Kirchner, F., & Kirchner, E. A. (2023). 
    % EEG and EMG dataset for the detection of errors introduced by an active orthosis device (IJCAI'23 CC6 Competition) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8345429}
    % modify the path
    path = fullfile('...');
    disp(path)
    train_files = dir(fullfile(path,'*.vhdr'));
    
    all_target_epochs = [];
    all_distractor_epochs = [];
    
    for i_train_file = 1:length(train_files)
        this_file_name = train_files(i_train_file).name;
        this_file_name = fullfile(path, this_file_name);
    
        % Read eeg data
        EEG = pop_biosig(this_file_name, 'channels', 1:64);
        fs = EEG.srate;
    
        % Perform preprocessing and epoch extraction
        [target_epoch, distractor_epoch] = preprocess_and_epoch(EEG);
        all_target_epochs = cat(3, all_target_epochs, target_epoch);
        all_distractor_epochs = cat(3, all_distractor_epochs, distractor_epoch);
    end
    
    % Calculate noise threshold
    threshold_noise = calculate_noise_threshold(all_target_epochs);
    
    % Specify some parameters for feature extraction
    time_start = 0;
    time_end = 800;
    n_selected_initial = 100;
    n_features = 150;
    
    % Get dimensions of the data
    [n_chans, ~, n_tar] = size(all_target_epochs);
    [~, ~, n_dist] = size(all_distractor_epochs);
    n_epoch = n_tar + n_dist;
    
    % Specify the parameters for the moving window
    win_size = 80;  % in ms, specify size of moving window
    win_step = 80;  % in ms, specify step of moving window
    win_step_ind = round(win_step / 1000 * fs);
    n_win = round((time_end - time_start) / win_size);
    
    % Calculate the average features for target and distractor epochs
    features_tar_avg = NaN(n_chans, n_win, n_tar);
    features_dist_avg = NaN(n_chans, n_win, n_dist);
    for i_win = 1:n_win
        features_tar_avg(:, i_win, :) = mean(all_target_epochs(:, (i_win-1)*win_step_ind+1:i_win*win_step_ind, :), 2);
        features_dist_avg(:, i_win, :) = mean(all_distractor_epochs(:, (i_win-1)*win_step_ind+1:i_win*win_step_ind, :), 2);
    end
    
    % Calculate the difference features for target and distractor epochs
    features_tar_diff = features_tar_avg(:, 3:end, :) - features_tar_avg(:, 1:end-2, :);
    features_dist_diff = features_dist_avg(:, 3:end, :) - features_dist_avg(:, 1:end-2, :);
    
    % Calculate dynamic features by taking differences of differences
    diff_tar = features_tar_avg(:, 2:end, :) - features_tar_avg(:, 1:end-1, :);
    diff_dist = features_dist_avg(:, 2:end, :) - features_dist_avg(:, 1:end-1, :);
    features_tar_dyn = diff_tar(:, 4:end, :) - diff_tar(:, 1:end-3, :);
    features_dist_dyn = diff_dist(:, 4:end, :) - diff_dist(:, 1:end-3, :);
    
    while true  
        % Select top features based on various measures (average, difference, dynamic)
        [indices_pairs_avg, sorted_values_avg, cohens_d_avg, cohens_d_avg_vis] = select_top_features(features_tar_avg, features_dist_avg, n_selected_initial);
        [indices_pairs_diff, sorted_values_diff, cohens_d_diff, cohens_d_diff_vis] = select_top_features(features_tar_diff, features_dist_diff, n_selected_initial);
        [indices_pairs_dyn, sorted_values_dyn, cohens_d_dyn, cohens_d_dyn_vis] = select_top_features(features_tar_dyn, features_dist_dyn, n_selected_initial);
    
        % Extract indices pairs with common channels
        chan_common = intersect(intersect(indices_pairs_avg(:, 1), indices_pairs_diff(:, 1)), indices_pairs_dyn(:, 1));
        indices_pairs_avg = indices_pairs_avg(ismember(indices_pairs_avg(:, 1), chan_common), :);
        indices_pairs_diff = indices_pairs_diff(ismember(indices_pairs_diff(:, 1), chan_common), :);
        indices_pairs_dyn = indices_pairs_dyn(ismember(indices_pairs_dyn(:, 1), chan_common), :);
    
        % Refine the selected features based on ratios and desired total number of features
        sum_all_sorted_values = sum(sorted_values_avg) + sum(sorted_values_diff) + sum(sorted_values_dyn);
        ratio_avg = sum(sorted_values_avg) / sum_all_sorted_values;
        ratio_diff = sum(sorted_values_diff) / sum_all_sorted_values;
        ratio_dyn = sum(sorted_values_dyn) / sum_all_sorted_values;
        n_selected_avg = round(ratio_avg * n_features); 
        n_selected_diff = round(ratio_diff * n_features);
        n_selected_dyn = round(ratio_dyn * n_features);
    
        % Check if any n_selected_* exceeds the size of its corresponding indices_pairs matrix
        condition_avg = n_selected_avg > size(indices_pairs_avg, 1);
        condition_diff = n_selected_diff > size(indices_pairs_diff, 1);
        condition_dyn = n_selected_dyn > size(indices_pairs_dyn, 1);
    
        if condition_avg || condition_diff || condition_dyn  % If any condition is true
            % Update n_selected_initial
            n_selected_initial = n_selected_initial + 50;
            fprintf('increasing n_selected_initial by 50\n');
    
            % Check if n_selected_initial exceeds the total number of possible features
            if n_selected_initial > n_chans*size(features_tar_dyn,2)
                warning('n_selected_initial exceeds the total number of possible dynamics features. Please reduce n_features');
                break;
            end
        else
            break;  % If no condition is true, break the loop
        end
    end
    
    fprintf('Number of selected temporal average features: %d\n', n_selected_avg);
    fprintf('Number of selected temporal changes features: %d\n', n_selected_diff);
    fprintf('Number of selected temporal dynamics features: %d\n', n_selected_dyn);
    
    indices_pairs_avg = indices_pairs_avg(1:n_selected_avg, :);
    indices_pairs_diff = indices_pairs_diff(1:n_selected_diff, :);
    indices_pairs_dyn = indices_pairs_dyn(1:n_selected_dyn, :);
    
    % Extract and combine selected features
    features_avg = cat(3, features_tar_avg, features_dist_avg);
    features_selected_avg = zeros(n_epoch, n_selected_avg);
    for i_feat = 1:n_selected_avg
        rowIdx = indices_pairs_avg(i_feat, 1);
        colIdx = indices_pairs_avg(i_feat, 2);
        features_selected_avg(:, i_feat) = features_avg(rowIdx, colIdx, :);
    end
    
    features_diff = cat(3, features_tar_diff, features_dist_diff);
    features_selected_diff = zeros(n_epoch, n_selected_diff);
    for i_feat = 1:n_selected_diff
        rowIdx = indices_pairs_diff(i_feat, 1);
        colIdx = indices_pairs_diff(i_feat, 2);
        features_selected_diff(:, i_feat) = features_diff(rowIdx, colIdx, :);
    end
    
    features_dyn = cat(3, features_tar_dyn, features_dist_dyn);
    features_selected_dyn = zeros(n_epoch, n_selected_dyn);
    for i_feat = 1:n_selected_dyn
        rowIdx = indices_pairs_dyn(i_feat, 1);
        colIdx = indices_pairs_dyn(i_feat, 2);
        features_selected_dyn(:, i_feat) = features_dyn(rowIdx, colIdx, :);
    end
    
    % Concatenate selected features and generate labels
    X_features = horzcat(features_selected_avg, features_selected_diff, features_selected_dyn);
    labels = [ones(n_tar, 1); zeros(n_dist, 1)];
    
    %% train classifier
    % Perform SMOTE for class balancing  
    [X_features, labels] = smote(X_features, [], 'Class', labels);
    
    % training a Support Vector Machine (SVM) classifier and evaluating its performance
    [SVMModel, val_accuracy, avg_cm_train, avg_cm_val, confusion_matrices] = train_and_evaluate_svm(X_features, labels);

    % Extract values from avg_cm_val
    TP = avg_cm_val(1,1); % True Positives
    FN = avg_cm_val(1,2); % False Negatives
    FP = avg_cm_val(2,1); % False Positives
    TN = avg_cm_val(2,2); % True Negatives
    
    % Calculate TPR and TNR
    TPR = TP / (TP + FN);
    TNR = TN / (TN + FP);
    
    % Add calculated metrics to the evaluation metrics struct
    evaluation_metrics(sub).Subject = this_subject;
    evaluation_metrics(sub).TPR = TPR;
    evaluation_metrics(sub).TNR = TNR;
    evaluation_metrics(sub).BalancedAccuracy = 0.5 * (TPR + TNR);
end

% Extract metrics from the struct array
tpr_values = [evaluation_metrics.TPR];
tnr_values = [evaluation_metrics.TNR];
balanced_accuracy_values = [evaluation_metrics.BalancedAccuracy];

% Calculate mean and standard deviation for each metric
mean_tpr = mean(tpr_values);
std_tpr = std(tpr_values);
mean_tnr = mean(tnr_values);
std_tnr = std(tnr_values);
mean_balanced_accuracy = mean(balanced_accuracy_values);
std_balanced_accuracy = std(balanced_accuracy_values);

% Display the results
fprintf('TPR: Mean ± SD = %.3f ± %.3f\n', mean_tpr, std_tpr);
fprintf('TNR: Mean ± SD = %.3f ± %.3f\n', mean_tnr, std_tnr);
fprintf('Balanced Accuracy: Mean ± SD = %.3f ± %.3f\n', mean_balanced_accuracy, std_balanced_accuracy);

