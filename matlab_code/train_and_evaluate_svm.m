% Function Name: train_and_evaluate_svm.m
%
% Description: 
% This function trains and evaluates a Support Vector Machine (SVM) classifier 
% on the provided feature matrix and labels. It performs k-fold cross-validation 
% to assess the model's performance and provides evaluation metrics such as 
% validation accuracy and confusion matrices for both training and testing data.
%
% Inputs:
% 1. X_features: Feature matrix (predictors).
% 2. labels: Vector of target labels.
% 3. num_folds (Optional): Number of folds for cross-validation. Default is 10.
%
% Process:
% 1. Training: Trains a standard SVM classifier on the provided data.
% 2. Cross-validation: Performs k-fold cross-validation to estimate the model's performance.
% 3. Evaluation: Computes confusion matrices for each fold and calculates average 
%    confusion matrices, miss rates, and false alarm rates for training and testing data.
%
% Outputs:
% 1. SVMModel: The trained SVM model.
% 2. val_accuracy: Validation accuracy of the model.
% 3. avg_cm_train: Average confusion matrix for the training data.
% 4. avg_cm_test: Average confusion matrix for the testing data.
% 5. confusion_matrices: Struct containing individual confusion matrices for each fold.
%
% Usage Example:
% [model, accuracy, cm_train, cm_test, conf_matrices] = 
%    train_and_evaluate_svm(features, labels, 'num_folds', 5);
%
% Notes:
% - The function requires MATLAB's Statistics and Machine Learning Toolbox for SVM training 
%   and evaluation.
% - It provides a comprehensive evaluation of the SVM's performance in classification tasks.
%


function [SVMModel, val_accuracy, avg_cm_train, avg_cm_test, confusion_matrices] = train_and_evaluate_svm(X_features, labels, varargin)
    % Create an input parser object
    p = inputParser;
    
    % Add required and optional parameters
    addRequired(p, 'X_features');
    addRequired(p, 'labels');
    addOptional(p, 'num_folds', 10);
    
    % Parse the inputs
    parse(p, X_features, labels, varargin{:});
    
    % Extract values from the input parser
    X_features = p.Results.X_features;
    labels = p.Results.labels;
    num_folds = p.Results.num_folds;

    % Train a Support Vector Machine (SVM) classifier
    SVMModel = fitcsvm(X_features, labels, 'Standardize', true);
    
    % Perform cross-validation
    cv_svm_model = crossval(SVMModel, 'KFold', num_folds);
    loss = kfoldLoss(cv_svm_model);
    val_accuracy = 1-loss;
    
    % Initialize struct to store confusion matrices for each fold
    confusion_matrices = struct('Train', [], 'Test', []);
    
    for fold = 1:num_folds
        % Get the training and validation data for the current fold
        train_idx = training(cv_svm_model.Partition, fold);
        test_idx = test(cv_svm_model.Partition, fold);
        X_train = X_features(train_idx, :);
        y_train = labels(train_idx);
        X_test = X_features(test_idx, :);
        y_test = labels(test_idx);
    
        % Get the SVM model trained within the current fold
        svm_model = cv_svm_model.Trained{fold};
    
        % Predict labels for the training and testing data using the SVM model
        y_train_pred = predict(svm_model, X_train);
        y_test_pred = predict(svm_model, X_test);
    
        % Calculate confusion matrix for training and testing data
        train_conf_mat = confusionmat(y_train, y_train_pred, 'Order', [1 0]);
        test_conf_mat = confusionmat(y_test, y_test_pred, 'Order', [1 0]);
    
        % Store the confusion matrices in the struct
        confusion_matrices(fold).Train = train_conf_mat;
        confusion_matrices(fold).Test = test_conf_mat;
    end
    
    % Calculate and print evaluation metrics
    avg_cm_train = mean(cat(3, confusion_matrices.Train), 3);
    miss_rate_train = avg_cm_train(1,2)/(avg_cm_train(1,1)+avg_cm_train(1,2));
    fa_rate_train = avg_cm_train(2,1) / (avg_cm_train(2,2) + avg_cm_train(2,1));
    disp("Average Confusion Matrix for Training Data:");
    disp(avg_cm_train);
    disp("Miss Rate for Training Data:");
    disp([sprintf('%.2f', miss_rate_train * 100), '%']);
    disp("False Alarm Rate for Training Data:");
    disp([sprintf('%.2f', fa_rate_train * 100), '%']);

    avg_cm_test = mean(cat(3, confusion_matrices.Test), 3);
    miss_rate_test = avg_cm_test(1,2)/(avg_cm_test(1,1)+avg_cm_test(1,2));
    fa_rate_test = avg_cm_test(2,1)/(avg_cm_test(2,2) + avg_cm_test(2,1));
    disp("Average Confusion Matrix for Testing Data:");
    disp(avg_cm_test);
    disp("Miss Rate for Testing Data:");
    disp([sprintf('%.2f', miss_rate_test * 100), '%']);
    disp("False Alarm Rate for Testing Data:");
    disp([sprintf('%.2f', fa_rate_test * 100), '%']);

    disp("Validation Accuracy:");
    disp([sprintf('%.2f', val_accuracy * 100), '%']);
end