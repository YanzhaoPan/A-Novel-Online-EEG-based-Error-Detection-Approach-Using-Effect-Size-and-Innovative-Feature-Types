% Function Name: preprocess_and_epoch.m
% 
% Description: 
% This function preprocesses and epochs EEG data. It accepts an EEG 
% structure and performs re-referencing, filtering, and epoch extraction 
% for target and distractor conditions. The function allows customization 
% of the time window and frequency range for filtering through optional 
% parameters.
%
% Inputs:
% 1. EEG: EEG structure containing raw EEG data.
% 2. time_start (Optional): Start time of the epoching window in milliseconds. Default is 0 ms.
% 3. time_end (Optional): End time of the epoching window in milliseconds. Default is 800 ms.
% 4. low_cutoff (Optional): Lower cutoff frequency for bandpass filter in Hz. Default is 0.1 Hz.
% 5. high_cutoff (Optional): Upper cutoff frequency for bandpass filter in Hz. Default is 15 Hz.
%
% Processing Steps:
% 1. Re-referencing: Standardizes the EEG data.
% 2. Filtering: Applies bandpass filter based on low_cutoff and high_cutoff.
% 3. Epoching: Segments EEG data into epochs for target ('condition 7', 'condition 8') and 
%    distractor ('S 48') conditions.
% 4. Baseline Removal: Removes baseline from each epoch.
%
% Outputs:
% 1. target_epoch: EEG data array for target epochs within specified time range.
% 2. distractor_epoch: EEG data array for distractor epochs within specified time range.
%
% Usage Example:
% [targets, distractors] = preprocess_and_epoch(EEG_data, 'time_start', 100, 'time_end', 700, 
%                                               'low_cutoff', 1, 'high_cutoff', 30);
%
% Notes:
% - Requires EEGLAB toolbox functions (pop_reref, pop_eegfiltnew, pop_epoch).
% - Adaptable for various EEG analyses requiring preprocessing and epoching.
%


function [target_epoch, distractor_epoch] = preprocess_and_epoch(EEG, varargin)
    % Create an input parser object
    p = inputParser;
    
    % Add required and optional parameters
    addRequired(p, 'EEG');
    addOptional(p, 'time_start', 0);
    addOptional(p, 'time_end', 800);
    addOptional(p, 'low_cutoff', 0.1);
    addOptional(p, 'high_cutoff', 15);
    
    % Parse the inputs
    parse(p, EEG, varargin{:});
    
    % Extract values from the input parser
    time_start = p.Results.time_start;
    time_end = p.Results.time_end;
    low_cutoff = p.Results.low_cutoff;
    high_cutoff = p.Results.high_cutoff;

    % Perform preprocessing
    EEG = pop_reref(EEG, []);
    EEG = pop_eegfiltnew(EEG, 'locutoff', low_cutoff, 'plotfreqz', 0);
    EEG = pop_eegfiltnew(EEG, 'hicutoff', high_cutoff, 'plotfreqz', 0);

    % Epoching for targets('condition 7' and 'condition 8' in eeglab)
    EEG_tar = pop_epoch( EEG, {  'condition 7', 'condition 8'  }, [-0.1  1], 'newname', 'EEG_epochs', 'epochinfo', 'yes');
    EEG_tar = pop_rmbase( EEG_tar, [-100 0] ,[]);
    
    % Epoching for distractors('S 48')
    EEG_dist = pop_epoch( EEG, {  'S 48'  }, [-0.1  1], 'newname', 'EEG_epochs', 'epochinfo', 'yes');
    EEG_dist = pop_rmbase( EEG_dist, [-100 0] ,[]);

    % Convert time range to indices
    time_start_ind = find(EEG_tar.times >= time_start, 1);
    time_end_ind = find(EEG_tar.times < time_end, 1, 'last');

    % Extract epochs for targets in specified time range
    target_epoch = EEG_tar.data(:, time_start_ind:time_end_ind, :);

    % Extract epochs for distractors in specified time range
    distractor_epoch = EEG_dist.data(:, time_start_ind:time_end_ind, :);
end
