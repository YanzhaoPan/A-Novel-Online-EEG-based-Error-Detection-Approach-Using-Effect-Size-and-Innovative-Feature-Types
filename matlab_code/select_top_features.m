% Function Name: select_top_features.m
%
% Description: 
% This function selects the top features for EEG data based on their effect size, 
% measured by Cohen's d. It computes Cohen's d for each feature, comparing target 
% and distractor epochs, and then selects the features with the largest absolute 
% values of Cohen's d. The function is intended to identify features with the 
% most significant differences between the two conditions.
%
% Author: Yanzhao Pan
% Date Created: 20.06.2023
% Last Modified: 27.11.2023
% Affiliation: Intuitive XR Labs, BTU Cottbus-Senftenberg, Germany
%
% Inputs:
% 1. features_tar: 3D matrix of features from target epochs (channels x time x epochs).
% 2. features_dist: 3D matrix of features from distractor epochs (channels x time x epochs).
% 3. n_selected: Number of top features to select based on Cohen's d.
%
% Process:
% 1. Calculate Mean and Standard Deviation: Computes the mean and standard deviation for 
%    target and distractor features separately.
% 2. Compute Pooled Standard Deviation: Determines the pooled standard deviation across both conditions.
% 3. Compute Cohen's d: Calculates the effect size for each feature.
% 4. Select Top Features: Identifies the features with the largest effect sizes.
%
% Outputs:
% 1. indices_pairs: Indices of the selected features (row and column indices).
% 2. sorted_values: The sorted values of Cohen's d for the selected features.
% 3. cohens_d_abs: Absolute values of Cohen's d for all features.
% 4. cohens_d: Actual values of Cohen's d for all features.
%
% Usage Example:
% [idx_pairs, values, cohens_d_abs, cohens_d] = 
%    select_top_features(features_target, features_distractor, 100);
%


function [indices_pairs, sorted_values, cohens_d_abs, cohens_d] = select_top_features(features_tar, features_dist, n_selected)
    % Select top features based on effect size (Cohen's d).

    % Get dimensions of the data
    [~, ~, n_tar] = size(features_tar);
    [~, ~, n_dist] = size(features_dist);

    % Compute mean and standard deviation for targets and distractors separately
    mean_tar = mean(features_tar, 3);
    mean_dist = mean(features_dist, 3);
    std_tar = std(features_tar, [], 3);
    std_dist = std(features_dist, [], 3);

    % Compute pooled standard deviation (sd)
    sd = sqrt(((n_tar - 1) .* std_tar.^2 + (n_dist - 1) .* std_dist.^2) / (n_tar + n_dist - 2));

    % Compute effect size (Cohen's d) for each feature
    cohens_d_abs = abs(mean_tar - mean_dist) ./ sd;
    cohens_d = (mean_tar - mean_dist) ./ sd;

    % Find the n_selected largest values and their indices
    [sorted_values, sorted_indices] = maxk(cohens_d_abs(:), n_selected);

    % Convert linear indices to row and column indices
    [row_indices, col_indices] = ind2sub(size(cohens_d_abs), sorted_indices);

    % Output the row and column indices in pairs
    indices_pairs = [row_indices, col_indices];
end

