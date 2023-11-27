% Function Name: extract_features.m
%
% Description: 
% This function extracts and selects features from EEG epoch data for use in 
% machine learning models, specifically targeting and distractor epochs. It 
% calculates average, difference, and dynamic features and selects the most 
% discriminative features using a tailored feature selection process based on 
% Cohen's d values. The function is flexible with several optional parameters 
% to customize the feature extraction process.
%
% Author: Yanzhao Pan
% Date Created: 20.06.2023
% Last Modified: 27.11.2023
% Affiliation: Intuitive XR Labs, BTU Cottbus-Senftenberg, Germany
%
% Inputs:
% 1. fs: Sampling frequency of the EEG data.
% 2. target_epochs: 3D matrix of target epochs (channels x time x epochs).
% 3. distractor_epochs: 3D matrix of distractor epochs (channels x time x epochs).
% 4. time_start (Optional): Start time of the feature extraction window in milliseconds. Default is 0 ms.
% 5. time_end (Optional): End time of the feature extraction window in milliseconds. Default is 800 ms.
% 6. n_selected_initial (Optional): Initial number of features for each feature type. Default is 100.
% 7. n_features (Optional): Total number of features to be selected. Default is 150.
%
% Process:
% 1. Feature Calculation: Calculates average, difference, and dynamic features for target and distractor epochs.
% 2. Feature Selection: Applies a feature selection algorithm to identify the most discriminative features.
% 3. Feature Extraction: Extracts the selected features and prepares them for input into a machine learning model.
%
% Outputs:
% 1. X_features: Matrix of concatenated selected features.
% 2. labels: Vector of labels corresponding to the features (1 for targets, 0 for distractors).
% 3. indices_pairs_avg: Indices of selected average features.
% 4. indices_pairs_diff: Indices of selected difference features.
% 5. indices_pairs_dyn: Indices of selected dynamic features.
%
% Usage Example:
% [features, labels, idx_avg, idx_diff, idx_dyn] = 
%    extract_features(fs, target_epochs, distractor_epochs, 'n_features', 200);
%


function [X_features, labels, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn] = extract_features(fs, target_epochs, distractor_epochs, varargin)
    % Create an input parser object
    p = inputParser;
    
    % Add required and optional parameters
    addRequired(p, 'fs');
    addRequired(p, 'target_epochs');
    addRequired(p, 'distractor_epochs');
    addOptional(p, 'time_start', 0);
    addOptional(p, 'time_end', 800);
    addOptional(p, 'n_selected_initial', 100);
    addOptional(p, 'n_features', 150);
    
    % Parse the inputs
    parse(p, fs, target_epochs, distractor_epochs, varargin{:});
    
    % Extract values from the input parser
    time_start = p.Results.time_start;
    time_end = p.Results.time_end;
    n_selected_initial = p.Results.n_selected_initial;
    n_features = p.Results.n_features;

    % Get dimensions of the data
    [n_chans, ~, n_tar] = size(target_epochs);
    [~, ~, n_dist] = size(distractor_epochs);
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
        features_tar_avg(:, i_win, :) = mean(target_epochs(:, (i_win-1)*win_step_ind+1:i_win*win_step_ind, :), 2);
        features_dist_avg(:, i_win, :) = mean(distractor_epochs(:, (i_win-1)*win_step_ind+1:i_win*win_step_ind, :), 2);
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
        [indices_pairs_avg, sorted_values_avg] = select_top_features(features_tar_avg, features_dist_avg, n_selected_initial);
        [indices_pairs_diff, sorted_values_diff] = select_top_features(features_tar_diff, features_dist_diff, n_selected_initial);
        [indices_pairs_dyn, sorted_values_dyn] = select_top_features(features_tar_dyn, features_dist_dyn, n_selected_initial);

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
end
