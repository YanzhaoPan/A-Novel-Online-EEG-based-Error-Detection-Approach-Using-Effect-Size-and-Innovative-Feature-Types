import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import mne
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import pickle


def read_eeg(file_path):
    """
    Read EEG data from a BrainVision file.

    Parameters:
    - file_path (str): Path to the BrainVision file (.vhdr) containing EEG data.

    Returns:
    - raw_data (mne.io.Raw): MNE Raw object containing EEG data.
    - fs (float): Sampling frequency of the EEG data.

    This function reads EEG data from a BrainVision file and returns an MNE Raw object and the sampling frequency.

    Example usage:
    ```
    raw_data, fs = read_eeg(file_path)
    ```

    """
    raw_data = mne.io.read_raw_brainvision(file_path, scale=1e6, preload=True)
    raw_data.pick_types(eeg=True)
    fs = raw_data.info['sfreq']

    return raw_data, fs


def preprocessing_and_epoching(raw_data, l_freq=0.1, h_freq=15, time_start=0, time_end=0.8):
    """
    Preprocess EEG data and extract epochs for specific events.

    Parameters:
    - raw_data (mne.io.Raw): MNE Raw object containing EEG data.
    - l_freq (float, optional): Lower frequency of the bandpass filter. Default is 0.1 Hz.
    - h_freq (float, optional): Upper frequency of the bandpass filter. Default is 15 Hz.
    - time_start (float, optional): Start time of the desired epoch in seconds. Default is 0.
    - time_end (float, optional): End time of the desired epoch in seconds. Default is 0.8.

    Returns:
    - target_epochs (numpy.ndarray): Extracted epochs of target events (shape: [n_epochs, n_channels, n_times]).
    - distractor_epochs (numpy.ndarray): Extracted epochs of distractor events (shape: [n_epochs, n_channels, n_times]).

    This function preprocesses EEG data and extracts epochs for specific events of interest.

    Steps:
    1. Apply average reference to the EEG data.
    2. Apply a bandpass filter to the EEG data within the specified frequency range.
    3. Extract events and event IDs from annotations in the raw data.
    4. Construct epochs for target and distractor events using the event IDs.
    5. Convert the specified time range to indices for epoch extraction.
    6. Extract epochs for target and distractor events within the specified time range.

    Example usage:
    ```
    target_epochs, distractor_epochs = preprocessing_and_epoching(raw_data, l_freq=0.1, h_freq=15, time_start=0, time_end=0.8)
    ```

    """
    # Apply average reference
    raw_data.set_eeg_reference("average", projection=False, verbose=False)
    
    # Apply bandpass filter
    raw_data.filter(l_freq=l_freq, h_freq=h_freq)
    
    # Extract events and event IDs
    events, event_id = mne.events_from_annotations(raw_data, verbose=False)
    
    # Define epoch time boundaries
    tmin, tmax = -0.1, 1
    
    # Create epochs for target and distractor events
    eeg_tar = mne.Epochs(raw_data, events=events, event_id=event_id["Stimulus/S 96"], 
                         tmin=tmin, tmax=tmax, baseline=(-0.1, 0), verbose=False)
    eeg_dist = mne.Epochs(raw_data, events=events, event_id=event_id["Stimulus/S 48"], 
                          tmin=tmin, tmax=tmax, baseline=(-0.1, 0), verbose=False)
    
    # Convert time range to indices
    start_ind = eeg_tar.time_as_index(time_start)[0]
    end_ind = eeg_tar.time_as_index(time_end)[0]
    
    # Extract epochs for targets and distractors within the specified time range
    target_epochs = eeg_tar.get_data()[:, :, start_ind:end_ind]
    distractor_epochs = eeg_dist.get_data()[:, :, start_ind:end_ind]

    return target_epochs, distractor_epochs


def calculate_noise_threshold(epochs, num_top_values=6):
    """
    Calculate the noise threshold for online prediction.

    Parameters:
    - epochs (numpy.ndarray): EEG data of targets (shape: [n_epochs, n_channels, n_times]).
    - num_top_values (int, optional): Number of top max-min differences to use in threshold calculation. Default is 6.

    Returns:
    - threshold_noise (float): The calculated noise threshold.

    This function calculates the noise threshold based on the top `num_top_values` max-min differences
    of EEG epochs. It provides a measure of noise level in the EEG data.

    Steps:
    1. Calculate the max-min differences for each channel within each epoch and rank the max-min differences for each epoch separately
    2. Calculate the mean of the top `num_top_values` ranked differences for each epoch.
    3. Compute the threshold_noise as the mean of the above means plus the standard deviation.

    Note:
    - A higher `num_top_values` can capture more extreme noise values but may also include more EEG signal variations.

    Example usage:
    ```
    threshold = calculate_noise_threshold(eeg_epochs, num_top_values=6)
    ```

    """
    # Calculate the noise threshold based on the top num_top_values max-min differences
    max_min_difference = np.squeeze(np.max(epochs, axis=2) - np.min(epochs, axis=2))
    sorted_max_min_diff = np.sort(max_min_difference, axis=1)
    mean_max_min_diff = np.mean(sorted_max_min_diff[:, -num_top_values:], axis=1)
    threshold_noise = np.mean(mean_max_min_diff) + np.std(mean_max_min_diff)

    return threshold_noise


def select_top_features(features_tar, features_dist, n_selected=100):
    """
    Select top features based on effect size (Cohen's d).

    Parameters:
    - features_tar (numpy.ndarray): Features extracted from target epochs (shape: [n_epochs, n_channels, n_times]).
    - features_dist (numpy.ndarray): Features extracted from distractor epochs (shape: [n_epochs, n_channels, n_times]).
    - n_selected (int, optional): Number of top features to select. Default is 100.

    Returns:
    - indices_pairs (numpy.ndarray): Indices pairs for selected features (shape: [n_selected, 2]).
    - sorted_values (numpy.ndarray): Sorted effect size values corresponding to selected features.

    This function selects the top features based on the effect size (Cohen's d) between target and distractor epochs.

    Steps:
    1. Compute the mean and standard deviation for target and distractor features separately.
    2. Compute the pooled standard deviation (sd) based on mean and standard deviation of target and distractor features.
    3. Calculate Cohen's d (effect size) for each feature by comparing means and pooled standard deviation.
    4. Find the indices of the k largest effect size values.
    5. Convert linear indices to row and column indices for the selected features.
    6. Output the row and column indices in pairs along with the corresponding sorted effect size values.

    Example usage:
    ```
    indices_pairs, sorted_values = select_top_features(features_tar, features_dist, n_selected=100)
    ```

    """
    n_tar = features_tar.shape[0]
    n_dist = features_dist.shape[0]

    # Step 1: Compute mean and standard deviation for targets and distractors separately
    mean_tar = np.mean(features_tar, axis=0)
    mean_dist = np.mean(features_dist, axis=0)
    std_tar = np.std(features_tar, axis=0)
    std_dist = np.std(features_dist, axis=0)

    # Step 2: Compute pooled standard deviation (sd)
    sd = np.sqrt(((n_tar - 1) * std_tar ** 2 + (n_dist - 1) * std_dist ** 2) / (n_tar + n_dist - 2))

    # Step 3: Compute effect size (Cohen's d) for each feature
    cohens_d = np.abs(mean_tar - mean_dist) / sd

    # Step 4: Find the n_selected largest values and their indices
    sorted_indices = np.argsort(cohens_d.ravel())[::-1][:n_selected]
    sorted_values = cohens_d.ravel()[sorted_indices]

    # Step 5: Convert linear indices to row and column indices
    row_indices, col_indices = np.unravel_index(sorted_indices, cohens_d.shape)

    # Step 6: Output the row and column indices in pairs
    indices_pairs = np.column_stack((row_indices, col_indices))

    return indices_pairs, sorted_values


def extract_features(fs, target_epochs, distractor_epochs, time_start=0, time_end=0.8, n_features=150):
    """
    Extract and select features for classification.

    Parameters:
    - fs (float): Sampling frequency of the EEG data.
    - target_epochs (numpy.ndarray): Target epochs (shape: [n_epochs, n_channels, n_times]).
    - distractor_epochs (numpy.ndarray): Distractor epochs (shape: [n_epochs, n_channels, n_times]).
    - time_start (float, optional): Start time for feature extraction in seconds. Default is 0.
    - time_end (float, optional): End time for feature extraction in seconds. Default is 0.8.
    - n_features (int, optional): Total number of selected features. Default is 150.

    Returns:
    - X_features (numpy.ndarray): Selected features for classification (shape: [n_epochs, n_selected_features]).
    - labels (numpy.ndarray): Labels corresponding to the target (1) and distractor (0) epochs.
    - indices_pairs_avg (numpy.ndarray): Indices pairs for selected average features.
    - indices_pairs_diff (numpy.ndarray): Indices pairs for selected difference features.
    - indices_pairs_dyn (numpy.ndarray): Indices pairs for selected dynamics features.

    This function extracts and selects features from target and distractor epochs for classification.

    Steps:
    1. Perform temporal feature extraction using moving windows on target and distractor epochs.
    2. Select top features for temporal average, difference, and dynamics using effect size (Cohen's d).
    3. Find common channels among selected features.
    4. Refine the selected features based on the ratio of summed cohen's d and desired total number of features.
    5. Extract and concatenate selected temporal average, difference, and dynamics features.
    6. Generate labels for target and distractor epochs.
    7. Return the selected features, labels, and indices pairs for each feature type.

    Example usage:
    ```
    X_features, labels, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn = extract_features(fs, target_epochs, distractor_epochs)
    ```

    """
    n_tar, n_chans, _ = target_epochs.shape
    n_dist = distractor_epochs.shape[0]
    n_epoch = n_tar + n_dist

    # Step 1: Temporal feature extraction using moving windows
    win_size = 80  # in ms, specify size of moving window 
    win_step = 80  # in ms, specify step of moving window 
    win_step_ind = int(win_step / 1000 * fs)
    n_win = int((time_end - time_start)*1000 / win_size)
    features_tar_avg = np.empty((n_tar, n_chans, n_win))
    features_dist_avg = np.empty((n_dist, n_chans, n_win))
    for i_win in range(n_win):
        features_tar_avg[:, :, i_win] = np.mean(target_epochs[:, :, i_win*win_step_ind:(i_win+1)*win_step_ind], axis=2)
        features_dist_avg[:, :, i_win] = np.mean(distractor_epochs[:, :, i_win*win_step_ind:(i_win+1)*win_step_ind], axis=2)

    # Step 2: Extract temporal average features based on effect size (Cohen's d)
    indices_pairs_avg, sorted_values_avg = select_top_features(features_tar_avg, features_dist_avg)

    # Step 3: Extract difference and dynamics features
    features_tar_diff = features_tar_avg[:, :, 2:] - features_tar_avg[:, :, :-2]
    features_dist_diff = features_dist_avg[:, :, 2:] - features_dist_avg[:, :, :-2]
    indices_pairs_diff, sorted_values_diff = select_top_features(features_tar_diff, features_dist_diff)

    diff_tar = features_tar_avg[:, :, 1:] - features_tar_avg[:, :, :-1]
    diff_dist = features_dist_avg[:, :, 1:] - features_dist_avg[:, :, :-1]
    features_tar_dyn = diff_tar[:, :, 3:] - diff_tar[:, :, :-3]
    features_dist_dyn = diff_dist[:, :, 3:] - diff_dist[:, :, :-3]
    indices_pairs_dyn, sorted_values_dyn = select_top_features(features_tar_dyn, features_dist_dyn)

    # Step 4: Extract indices pairs with common channels
    chan_common = np.intersect1d(np.intersect1d(indices_pairs_avg[:, 0], indices_pairs_diff[:, 0]), indices_pairs_dyn[:, 0])
    indices_pairs_avg = indices_pairs_avg[np.isin(indices_pairs_avg[:, 0], chan_common)]
    indices_pairs_diff = indices_pairs_diff[np.isin(indices_pairs_diff[:, 0], chan_common)]
    indices_pairs_dyn = indices_pairs_dyn[np.isin(indices_pairs_dyn[:, 0], chan_common)]

    # Step 5: Refine the selected features based on ratios and desired total number of features
    sum_all_sorted_values = sum(sorted_values_avg) + sum(sorted_values_diff) + sum(sorted_values_dyn)
    ratio_avg = sum(sorted_values_avg) / sum_all_sorted_values
    ratio_diff = sum(sorted_values_diff) / sum_all_sorted_values
    ratio_dyn = sum(sorted_values_dyn) / sum_all_sorted_values

    n_selected_avg = int(ratio_avg * n_features)
    n_selected_diff = int(ratio_diff * n_features)
    n_selected_dyn = int(ratio_dyn * n_features)
    print("Number of selected temporal average features:", n_selected_avg)
    print("Number of selected temporal changes features:", n_selected_diff)
    print("Number of selected temporal dynamics features:", n_selected_dyn)

    indices_pairs_avg = indices_pairs_avg[:n_selected_avg, :]
    indices_pairs_diff = indices_pairs_diff[:n_selected_diff, :]
    indices_pairs_dyn = indices_pairs_dyn[:n_selected_dyn, :]

    # Step 6: Extract and combine selected features
    features_avg = np.concatenate((features_tar_avg, features_dist_avg), axis=0)
    features_selected_avg = np.empty((n_epoch, n_selected_avg))
    for i_feat in range(n_selected_avg):
        rowIdx = indices_pairs_avg[i_feat, 0]
        colIdx = indices_pairs_avg[i_feat, 1]
        features_selected_avg[:, i_feat] = features_avg[:, rowIdx, colIdx]

    features_diff = np.concatenate((features_tar_diff, features_dist_diff), axis=0)
    features_selected_diff = np.empty((n_epoch, n_selected_diff))
    for i_feat in range(n_selected_diff):
        rowIdx = indices_pairs_diff[i_feat, 0]
        colIdx = indices_pairs_diff[i_feat, 1]
        features_selected_diff[:, i_feat] = features_diff[:, rowIdx, colIdx]

    features_dyn = np.concatenate((features_tar_dyn, features_dist_dyn), axis=0)
    features_selected_dyn = np.empty((n_epoch, n_selected_dyn))    
    for i_feat in range(n_selected_dyn):
        rowIdx = indices_pairs_dyn[i_feat, 0]
        colIdx = indices_pairs_dyn[i_feat, 1]
        features_selected_dyn[:, i_feat] = features_dyn[:, rowIdx, colIdx]

    # Step 7: concatenate selected features and generate labels
    X_features = np.hstack((features_selected_avg, features_selected_diff, features_selected_dyn))
    labels = np.hstack((np.ones(n_tar), np.zeros(n_dist)))

    return X_features, labels, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn 


def train_and_evaluate_svm(X_features, labels, nFold=10):
    """
    Train and evaluate a Support Vector Machine (SVM) classifier.

    Parameters:
    - X_features (numpy.ndarray): Selected features for classification (shape: [n_epochs, n_selected_features]).
    - labels (numpy.ndarray): Labels corresponding to the target (1) and distractor (0) epochs.
    - nFold (int, optional): Number of folds for cross-validation. Default is 10.

    Returns:
    - mean_accuracy (float): Mean validation accuracy across folds.
    - avg_cm_train (numpy.ndarray): Average confusion matrix for training data.
    - avg_cm_test (numpy.ndarray): Average confusion matrix for testing data.

    Example usage:
    ```
    mean_accuracy, avg_cm_train, avg_cm_test = train_and_evaluate_svm(X_features, labels, nFold=10)    
    ```

    """
    SVMModel = SVC(kernel='linear', C=1.0)

    # Initialize lists to store results of testing accuracy, confusion matrix for each fold
    val_accuracies = []
    cm_train = []
    cm_test = []

    # Perform cross-validation using KFold
    for train_idx, test_idx in KFold(n_splits=nFold, shuffle=True, random_state=42).split(X_features):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Train the SVM model on the training data
        SVMModel.fit(X_train, y_train)

        # Make predictions on the training and testing data
        y_train_pred = SVMModel.predict(X_train)
        y_test_pred = SVMModel.predict(X_test)

        # Calculate the confusion matrix for training and testing data
        cm_train_fold = confusion_matrix(y_train, y_train_pred)
        cm_test_fold = confusion_matrix(y_test, y_test_pred)

        # Append accuracy and confusion matrices to respective lists
        val_accuracies.append(SVMModel.score(X_test, y_test))
        cm_train.append(cm_train_fold)
        cm_test.append(cm_test_fold)

    # Calculate and print evaluation metrics
    mean_accuracy = np.mean(val_accuracies)
    print(f"\nMean Validation Accuracy: {mean_accuracy*100:.2f}%")

    avg_cm_train = np.mean(cm_train, axis=0)
    print("\nAverage Confusion Matrix for Training Data:\n", avg_cm_train)

    miss_rate_train = avg_cm_train[1, 0] / (avg_cm_train[1, 0] + avg_cm_train[1, 1])
    fa_rate_train = avg_cm_train[0, 1] / (avg_cm_train[0, 0] + avg_cm_train[0, 1])
    print(f"\nMiss Rate for Training Data: {miss_rate_train*100:.2f}%")
    print(f"False Alarm Rate for Training Data: {fa_rate_train*100:.2f}%")

    avg_cm_test = np.mean(cm_test, axis=0)
    print("\nAverage Confusion Matrix for Testing Data:\n", avg_cm_test)

    miss_rate_test = avg_cm_test[1, 0] / (avg_cm_test[1, 0] + avg_cm_test[1, 1])
    fa_rate_test = avg_cm_test[0, 1] / (avg_cm_test[0, 0] + avg_cm_test[0, 1])
    print(f"\nMiss Rate for Testing Data: {miss_rate_test*100:.2f}%")
    print(f"False Alarm Rate for Testing Data: {fa_rate_test*100:.2f}%")

    return mean_accuracy, avg_cm_train, avg_cm_test


def simulate_online_prediction(error_container, this_subject, svm_model, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn, threshold_noise, i_test=0):
    """
    Simulate online prediction using test data in the folder.

    Parameters:
    - error_container: Container of actual error latencies for each test set, which are announced after the first stage of competition.
    - this_subject (str): Name of the subject for online stage of competition.
    - svm_model (sklearn.svm.SVC): Trained Support Vector Machine (SVM) model for classification.
    - indices_pairs_avg (numpy.ndarray): Indices pairs for averaging feature extraction (shape: [n_selected_avg, 2]).
    - indices_pairs_diff (numpy.ndarray): Indices pairs for difference feature extraction (shape: [n_selected_diff, 2]).
    - indices_pairs_dyn (numpy.ndarray): Indices pairs for dynamic feature extraction (shape: [n_selected_dyn, 2]).
    - threshold_noise (float): Threshold for detecting noise in EEG data.
    - i_test (int, optional): Index of the current test set. Default is 0.

    Returns:
    - None

    This function simulates online prediction using test set. It extracts relevant features from EEG segments and predicts target events using a trained SVM classifier.

    Steps:
    1. Load EEG data for the current subject and test set.
    2. Preprocess EEG data by applying averaging reference and bandpass filtering.
    3. Extract EEG segments for feature extraction and baseline correction.
    4. Detect noise in EEG segments using the provided noise threshold.
    5. Extract averaged, difference, and dynamic features from EEG segments and concatenate selected features for prediction.
    6. Use the SVM model to predict target events and obtain decision scores.
    7. Apply noise filtering to predictions and scores if noise is detected.
    8. Store predictions, scores, and timestamps in corresponding lists.
    9. Visualize the results including scatter plots of probability, prediction for targets(blue lines) and actual error (red lines).

    Example usage:
    ```
    simulate_online_prediction(error_container, this_subject, svm_model, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn, threshold_noise, i_test=0):
    ```

    """
    # Load EEG data for the current subject and test set.    
    folder_path = os.path.join(os.getcwd(), 'python_code', 'test_data', this_subject)
    eeg_files = [file for file in os.listdir(folder_path) if file.endswith('.vhdr')]
    test_file_names = [os.path.splitext(file)[0] for file in eeg_files]
    test_file_name = test_file_names[i_test]
    file_path = os.path.join(folder_path, test_file_name + '.vhdr')
    raw_data, fs = read_eeg(file_path)
    eeg_data = raw_data.get_data()  
    timestamps = raw_data.times * 1000
    target_latencies = error_container[i_test]   # get the actual error latencies for this test set

    # Initialize variables for storing results
    is_target_list = []   # prediction for the sample, target - 1, distractor - 0
    score_list = []   # probability of every prediction
    ts = []   # timestamp of the first timepoint for every sampling
    noise_exist = []   # check the artifact 

    # Define parameters
    buffer_size = 900   # in ms, specify size of buffer
    buffer_size_ind = int(buffer_size * fs / 1000)
    sampling_step = 40   # in ms, specify the step of data fetching, to simulate the online scenario
    sampling_step_ind = int(sampling_step * fs / 1000)
    num_sampling = int((len(timestamps) - buffer_size_ind) / sampling_step_ind)
    baseline_range = 100
    win_size = 80  # in ms, specify size of moving window for temporal feature extraction
    win_step = 80  # in ms, specify step of moving window for temporal feature extraction
    win_step_ind = int(win_step / 1000 * fs)
    num_chan = eeg_data.shape[0]
    num_win = int((buffer_size - baseline_range) / win_size)   # number of moving windows for temporal feature extraction

    # Simulate the online data fetching and prediction
    for i_data_win in range(num_sampling):
        buffer_start_ind = i_data_win * sampling_step_ind
        buffer_end_ind = buffer_start_ind + buffer_size_ind
        ts.append(buffer_start_ind)
        this_data = eeg_data[:, buffer_start_ind : buffer_end_ind]

        # do average reference
        mean_this_data = np.mean(this_data, axis=0)
        this_data = this_data - mean_this_data

        # bandpass filtering
        low_cutoff = 0.1
        high_cutoff = 15.0
        filter_order = 4
        nyquist_freq = 0.5 * fs
        low_cutoff_norm = low_cutoff / nyquist_freq
        high_cutoff_norm = high_cutoff / nyquist_freq
        b, a = butter(filter_order, [low_cutoff_norm, high_cutoff_norm], btype='band')
        this_data = filtfilt(b, a, this_data, axis=1)

        # baseline correction
        baseline_range_ind = int(baseline_range * fs / 1000)
        baseline = np.mean(this_data[:, :baseline_range_ind], axis=1, keepdims=True)
        this_data = this_data - baseline

        # segment
        segmented_data = this_data[:, baseline_range_ind:]

        # noise checking
        this_max_min_difference = np.max(segmented_data, axis=1) - np.min(segmented_data, axis=1)
        this_sorted_difference = np.sort(this_max_min_difference)[::-1]
        this_noise_value = np.mean(this_sorted_difference[:6])
        if this_noise_value > threshold_noise:
            this_noise = True
            noise_exist.append(1)
        else:
            this_noise = False
            noise_exist.append(0)

        # average in each moving window
        this_feature = np.empty((num_chan, num_win))
        for i_win in range(num_win):
            this_feature[:, i_win] = np.mean(segmented_data[:, (i_win * win_step_ind):(i_win + 1) * win_step_ind], axis=1)

        # extract average features
        this_feature_avg = []
        for i_feat in range(indices_pairs_avg.shape[0]):
            row_idx = indices_pairs_avg[i_feat, 0]
            col_idx = indices_pairs_avg[i_feat, 1]
            this_feature_avg.append(this_feature[row_idx, col_idx])

        # extract difference features
        feature_diff = this_feature[:, 2:] - this_feature[:, :-2]
        this_feature_diff = []
        for i_feat in range(indices_pairs_diff.shape[0]):
            row_idx = indices_pairs_diff[i_feat, 0]
            col_idx = indices_pairs_diff[i_feat, 1]
            this_feature_diff.append(feature_diff[row_idx, col_idx])

        # extract dynamics features
        feature_diff = this_feature[:, 1:] - this_feature[:, :-1]
        feature_dyn = feature_diff[:, 3:] - feature_diff[:, :-3]
        this_feature_dyn = []
        for i_feat in range(indices_pairs_dyn.shape[0]):
            row_idx = indices_pairs_dyn[i_feat, 0]
            col_idx = indices_pairs_dyn[i_feat, 1]
            this_feature_dyn.append(feature_dyn[row_idx, col_idx])

        # combine these features
        this_selected_feature = np.concatenate((this_feature_avg, this_feature_diff, this_feature_dyn))
        this_selected_feature = np.array(this_selected_feature).reshape(1, -1)

        # predict and store the result
        this_prediction = svm_model.predict(this_selected_feature)[0]
        this_score = svm_model.decision_function(this_selected_feature)[0]
        if this_noise is True:
            this_prediction = 0
            this_score = 0
        is_target_list.append(this_prediction)
        score_list.append(this_score)

    score_list_sorted = np.sort(score_list)[::-1]
    print(score_list_sorted[:30])
    score_threshold = 1.5   # specify the threshold
    ts = np.array(ts) + baseline_range_ind + 15  # leave some margin to avoid early prediction than the true error
    # find out the predictions above threshold
    one_indices = np.array(score_list) > score_threshold
    ts_selected = ts[one_indices]
    normalized_scores_selected = np.array(score_list)[one_indices] - score_threshold

    # plotting (dots: probability, blue: prediction, red: true errors)
    plt.scatter(ts_selected, np.zeros(len(ts_selected)), s=normalized_scores_selected*100, marker='o', alpha=normalized_scores_selected/10)
    for latency in target_latencies:
        plt.axvline(x=latency, color='red', linewidth=2)
    for ts in ts_selected:
        plt.axvline(x=ts, color='blue', linewidth=1)
    plt.show()


def main():
    # Define the subject
    this_subject = 'AQ59D'
    # Define the path to the EEG data for the current subject
    folder_path = os.path.join(os.getcwd(), 'python_code', 'training_data', this_subject)
    print(folder_path)
    # Get a list of all files with '.vhdr' extension in the folder
    eeg_files = [file for file in os.listdir(folder_path) if file.endswith('.vhdr')]
    # Extract file names without '.vhdr' suffix
    train_file_names = [os.path.splitext(file)[0] for file in eeg_files]

    # Initialize lists to store all the target and distractor epochs
    target_epochs_list = []
    distractor_epochs_list = []

    for name in train_file_names:
        file_path = os.path.join(folder_path, name + '.vhdr')
        raw_data, fs = read_eeg(file_path)
        target_epochs, distractor_epochs = preprocessing_and_epoching(raw_data) 

        # Append epochs to the overall lists
        target_epochs_list.append(target_epochs)
        distractor_epochs_list.append(distractor_epochs)

    # Concatenate the target and distractor epochs
    all_target_epochs = np.concatenate(target_epochs_list, axis=0)
    all_distractor_epochs = np.concatenate(distractor_epochs_list, axis=0)

    # get the threshold of noise
    threshold_noise = calculate_noise_threshold(all_target_epochs)
    print('threshold_noise: ', threshold_noise)

    # extract features
    X_features, labels, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn = extract_features(fs, all_target_epochs, all_distractor_epochs)

    # Perform SMOTE for class balancing
    smote = SMOTE(sampling_strategy='minority')
    X_features, labels = smote.fit_resample(X_features, labels)

    # training a Support Vector Machine (SVM) classifier and evaluating its performance
    avg_val_accuracy, avg_cm_train, avg_cm_test = train_and_evaluate_svm(X_features, labels)

    # scaler = StandardScaler()
    # scaler.fit(X_features)
    # X_features = scaler.transform(X_features)

    # train a SVM classifier, store the clf and other parameters for online use
    SVMModel = SVC(kernel='linear', C=1.0, probability=True)
    SVMModel.fit(X_features, labels)
    # np.save('indices_pairs_avg.npy', indices_pairs_avg)
    # np.save('indices_pairs_diff.npy', indices_pairs_diff)
    # np.save('indices_pairs_dyn.npy', indices_pairs_dyn)
    # pickle.dump(SVMModel, open('SVMModel', "wb"))
    # np.save('threshold_noise.npy', threshold_noise)

    error_container = ([30989, 45383, 80984, 94935, 105786, 116483], [20550, 32360, 51188, 62390, 73316, 119470])
    simulate_online_prediction(error_container, this_subject, SVMModel, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn, threshold_noise, i_test=0)


if __name__ == '__main__':
    main()