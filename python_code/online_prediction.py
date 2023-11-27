from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np 
from time import perf_counter 
import pickle
import time
from scipy.signal import butter, filtfilt
import threading


def print_meta(stream_info_obj):
    """
    This function prints some basic meta data of the stream
    """
    print("") 
    print("Meta data")
    print("Name:", stream_info_obj.name())
    print("Type:", stream_info_obj.type())
    print("Number of channels:", stream_info_obj.channel_count())
    print("Nominal sampling rate:", stream_info_obj.nominal_srate())
    print("Channel format:",stream_info_obj.channel_format())
    print("Source_id:",stream_info_obj.source_id())
    print("Version:",stream_info_obj.version())
    print("")


def get_ringbuffer_values(chunk, timestamps, current_local_time, timestamp_offset, data_buffer, timestamp_buffer): 
    """
    This function provides the most recent data samples and timestamps in a ringbuffer 
    (first val is oldest, last the newest) 

    Attributes:
        chunk               : current data chunk
        timestamps          : LSL local host timestamp for the data chunk 
        current_local_time  : LSL local client timestamp when the chunk is received
        timestamp offset    : correction factor that needs to be added to the timestamps to map it into the client's local LSL time
        data_buffer         : data buffer array of shape (buffer_size, n_channels)
        timestamp_buffer    : timestamps buffer of shape (buffer_size, 3). The 3 columns correspond to the host timestamp, the client local time and time correction offset resp.

    Returns:
        data_buffer         : data buffer array of shape (buffer_size, n_channels)
        timestamp_buffer    : timestamp buffer of shape (buffer_size, 3)
    """
    # data 
    current_chunk = np.array(chunk)
    n_samples = current_chunk.shape[0] # shape (samples, channels)

    temp_data = data_buffer[n_samples:, :]
    data_buffer[0:temp_data.shape[0], :] = temp_data
    data_buffer[temp_data.shape[0]:, :] = current_chunk

    # timestamps 
    current_timestamp_buffer = np.array(timestamps)

    temp_time = timestamp_buffer[n_samples:, 0]
    timestamp_buffer[0:temp_time.shape[0], 0] = temp_time
    timestamp_buffer[temp_time.shape[0]:, 0] = current_timestamp_buffer

    # current local time and offset correction 
    temp_local_time = timestamp_buffer[n_samples:, 1]
    timestamp_buffer[0:temp_local_time.shape[0], 1] = temp_local_time
    timestamp_buffer[temp_local_time.shape[0]:, 1] = current_local_time

    temp_offset_time = timestamp_buffer[n_samples:, 2]
    timestamp_buffer[0:temp_offset_time.shape[0], 2] = temp_offset_time
    timestamp_buffer[temp_offset_time.shape[0]:, 2] = timestamp_offset

    return data_buffer, timestamp_buffer


def send_detected_error(team_name, secret_id, timestamp_buffer_vals, local_clock_time):
    """
    This function gathers all the relevant results and sends it to the host.
    This function should be called everytime an error is detected.

    Attributes:
        team_name (str)         : each team will be assigned a team name which 
        secret_id (str)         : each team will be provided with a secret code
        timestamp_buffer_vals   : subset of the timestamp_buffer array at the instant when you have predicted an error and want to send the current result. Basically the i-th element of the timestamp_buffer array
        local_clock_time        : current LSL local clock time when you have run your classifier and predicted an error. This can be determined with the help of "local_clock()" call.
    """
    # calculate the final values for the timings 
    comm_delay = timestamp_buffer_vals[1] -timestamp_buffer_vals[0] -timestamp_buffer_vals[2]
    computation_time = local_clock_time - timestamp_buffer_vals[1]

    # connection to API for sending the results online 
    url = 'http://10.250.223.221:5000/results'
    myobj = {'team': team_name,
            'secret': secret_id,
            'host_timestamp': timestamp_buffer_vals[0], 
            'comp_time': computation_time, 
            'comm_delay': comm_delay}

    # x = requests.post(url, json = myobj)
    # print(x.text)


def process_and_predict(data_buffer, timestamp_buffer, team_name, secret_id, svm_model, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn, threshold_noise):
    """
    Perform data processing and prediction using the trained SVM model.

    Parameters:
    - data_buffer (numpy.ndarray): Buffer containing EEG data (shape: [buffer_size, n_channels]).
    - timestamp_buffer (numpy.ndarray): Buffer containing timestamps and time correction offsets (shape: [buffer_size, 3]).
    - team_name (str): Name of the team.
    - secret_id (str): Secret code for the team.
    - svm_model: Trained SVM model.
    - indices_pairs_avg (numpy.ndarray): Selected feature indices for average features (shape: [n_features_avg, 2]).
    - indices_pairs_diff (numpy.ndarray): Selected feature indices for difference features (shape: [n_features_diff, 2]).
    - indices_pairs_dyn (numpy.ndarray): Selected feature indices for dynamics features (shape: [n_features_dyn, 2]).
    - threshold_noise (float): Noise threshold for feature validation.

    """

    # specify some parameters
    sampling_freq = 500
    time_start = 0
    num_top_values = 6
    num_channels = 64
    num_time_windows = 10
    win_step = 80  # in ms, specify step of moving window for temporal feature extraction
    win_step_ind = int(win_step / 1000 * sampling_freq)
    n_features_avg = indices_pairs_avg.shape[0]
    n_features_diff = indices_pairs_diff.shape[0]
    n_features_dyn = indices_pairs_dyn.shape[0]

    # extract EEG data
    this_data = data_buffer.T  # (channels, samples)
    this_data = this_data[:64, :]

    # do average reference
    mean_this_data = np.mean(this_data, axis=0)
    this_data = this_data - mean_this_data

    # bandpass filtering
    low_cutoff = 0.1
    high_cutoff = 15.0
    filter_order = 4
    nyquist_freq = 0.5 * sampling_freq
    low_cutoff_norm = low_cutoff / nyquist_freq
    high_cutoff_norm = high_cutoff / nyquist_freq
    b, a = butter(filter_order, [low_cutoff_norm, high_cutoff_norm], btype='band')
    this_data = filtfilt(b, a, this_data, axis=1)

    # baseline correction
    baseline_ind_range = int(100 * sampling_freq / 1000)
    baseline = np.mean(this_data[:, :baseline_ind_range], axis=1, keepdims=True)
    this_data = this_data - baseline

    # segment
    segmented_data = this_data[:, int(time_start * sampling_freq + baseline_ind_range):]

    # check if there is noise
    max_min_difference = np.max(segmented_data, axis=1) - np.min(segmented_data, axis=1)
    sorted_difference = np.sort(max_min_difference)[::-1]
    noise_value = np.mean(sorted_difference[:num_top_values])
    if noise_value > threshold_noise:
        noise_exist = True
    else:
        noise_exist = False


    # temporal average feature extraction
    feature_matrix = np.empty((num_channels, num_time_windows))
    for i_win in range(num_time_windows):
        feature_matrix[:, i_win] = np.mean(segmented_data[:, (i_win * win_step_ind):(i_win + 1) * win_step_ind], axis=1)

    selected_feat_avg = []
    for i_feat in range(n_features_avg):
        row_idx = indices_pairs_avg[i_feat, 0]
        col_idx = indices_pairs_avg[i_feat, 1]
        selected_feat_avg.append(feature_matrix[row_idx, col_idx])

    # temporal difference feature extraction
    feature_diff = feature_matrix[:, 2:] - feature_matrix[:, :-2]
    selected_feat_diff = []
    for i_feat in range(n_features_diff):
        row_idx = indices_pairs_diff[i_feat, 0]
        col_idx = indices_pairs_diff[i_feat, 1]
        selected_feat_diff.append(feature_diff[row_idx, col_idx])

    # temporal dynamics feature extraction
    diff_matrix = feature_matrix[:, 1:] - feature_matrix[:, :-1]
    feature_dyn = diff_matrix[:, 3:] - diff_matrix[:, :-3]
    selected_feat_dyn = []
    for i_feat in range(n_features_dyn):
        row_idx = indices_pairs_dyn[i_feat, 0]
        col_idx = indices_pairs_dyn[i_feat, 1]
        selected_feat_dyn.append(feature_dyn[row_idx, col_idx])

    # concatenate these features
    selected_feat_combined = np.concatenate((selected_feat_avg, selected_feat_diff, selected_feat_dyn))

    # prediction for this buffer
    selected_feat_combined = np.array(selected_feat_combined).reshape(1, -1)
    score = svm_model.decision_function(selected_feat_combined)[0]
    error_index_in_buffer = baseline_ind_range

    if (not noise_exist) and (score > 2):
        print('ERROR DETECTED!!!!!', timestamp_buffer[error_index_in_buffer, :])
        # # if an error was detected, use the following lines to send the timepoint (timestamp) of detection
        # local_clock_time = local_clock()
        # send_detected_error(team_name, secret_id, timestamp_buffer[error_index_in_buffer, :], local_clock_time)
    

def main():
    # Parameters
    buffer_size = 450  # size of ringbuffer in samples (0.9 sec data times 500 Hz sampling rate)
    dt_read_buffer = 0.04  # time in seconds how often the buffer is read (updated with new incoming chunks)
    threads = []

    # team info
    team_name = 'neuro_xr_explorers'
    secret_id = 'xr_ij99'

    # Load model and other parameters
    svm_model = pickle.load(open('python_code/Model_stored/SVMModel', "rb"))
    indices_pairs_avg = np.load('python_code/Model_stored/indices_pairs_avg.npy')
    indices_pairs_diff = np.load('python_code/Model_stored/indices_pairs_diff.npy')
    indices_pairs_dyn = np.load('python_code/Model_stored/indices_pairs_dyn.npy')
    threshold_noise = np.load('python_code/Model_stored/threshold_noise.npy')
    print('threshold_noise: ', threshold_noise)
    print(len(indices_pairs_avg), len(indices_pairs_diff), len(indices_pairs_dyn))

    # First resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')  # create data stream

    # Create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    stream_info = inlet.info()
    print_meta(stream_info)  # print stream info

    # Inits
    data_buffer = np.zeros((buffer_size, stream_info.channel_count()))  # shape (buffer_size, n_channels)
    timestamp_buffer = np.zeros((buffer_size, 3))  # buffer for different kinds of timestamps or time values

    # Get timestamp offset
    timestamp_offset = inlet.time_correction()

    while True:
        chunk, timestamps = inlet.pull_chunk() # get a new data chunk

        if(chunk): # if list not empty (new data)
            # get timing info 
            current_local_time = local_clock()
            timestamp_offset = inlet.time_correction()

            # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
            data_buffer, timestamp_buffer = get_ringbuffer_values(chunk, timestamps, current_local_time, timestamp_offset, data_buffer, timestamp_buffer) 

            process_and_predict(data_buffer, timestamp_buffer, team_name, secret_id, svm_model, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn, threshold_noise)

            # # predicting in parallel
            # thread = threading.Thread(target=process_and_predict,
            #                       args=(data_buffer, timestamp_buffer, team_name, secret_id, svm_model, indices_pairs_avg, indices_pairs_diff, indices_pairs_dyn, threshold_noise))
            # threads.append(thread)
            # thread.start()
            # threads = [thread for thread in threads if thread.is_alive()]

            # wait for some time to ensure a "fixed" frequency to read new data from buffer 
            while((perf_counter()-old_time) < dt_read_buffer): 
                pass

            # # just for checking the loop frequency 
            # print("time: ", (perf_counter()-old_time)*1000)

        old_time = perf_counter()
    
    
if __name__ == '__main__':
    main()