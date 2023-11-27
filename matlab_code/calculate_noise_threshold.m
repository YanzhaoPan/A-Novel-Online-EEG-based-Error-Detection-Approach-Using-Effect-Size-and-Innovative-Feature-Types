% Function Name: calculate_noise_threshold.m
%
% Description: 
% This function calculates a noise threshold for EEG epochs. It computes 
% the maximum and minimum difference for each epoch and determines the threshold 
% based on the mean and standard deviation of the top max-min differences. 
% The function allows specifying the number of top values to consider through 
% an optional parameter.
%
% Inputs:
% 1. epochs: A 3D matrix of EEG epochs (channels x time x epochs).
% 2. num_top_values (Optional): Number of top max-min values to consider for 
%    threshold calculation. Default is 6.
%
% Process:
% 1. Calculate Max-Min Difference: For each epoch, finds the maximum and minimum 
%    difference across all channels.
% 2. Sorting: Sorts the max-min differences in ascending order.
% 3. Mean Calculation: Calculates the mean of the top specified number of max-min differences.
% 4. Threshold Calculation: Determines the noise threshold based on the mean and standard 
%    deviation of the mean max-min differences.
%
% Output:
% 1. threshold_noise: The calculated noise threshold value.
%
% Usage Example:
% noise_threshold = calculate_noise_threshold(EEG_epochs, 'num_top_values', 5);
%


function threshold_noise = calculate_noise_threshold(epochs, varargin)
    % Create an input parser object
    p = inputParser;
    
    % Add required and optional parameters
    addRequired(p, 'epochs');
    addOptional(p, 'num_top_values', 6); % Default value if not specified
    
    % Parse the inputs
    parse(p, epochs, varargin{:});
    
    % Extract values from the input parser
    num_top_values = p.Results.num_top_values;
    
    % Calculate max-min difference for each epoch
    max_min_difference = squeeze(max(epochs, [], 2) - min(epochs, [], 2));    
    
    % Sort the max-min differences
    sorted_max_min_diff = sort(max_min_difference, 1);
    
    % Calculate mean of the top values of max-min differences
    mean_max_min_diff = mean(sorted_max_min_diff(end-num_top_values+1:end, :), 1);
    
    % Calculate the noise threshold
    threshold_noise = mean(mean_max_min_diff) + std(mean_max_min_diff);
end

