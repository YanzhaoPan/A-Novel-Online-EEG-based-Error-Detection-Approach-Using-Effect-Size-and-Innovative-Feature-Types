import scipy.io
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock


# Load test data and timestamp
mat_data = scipy.io.loadmat('python_code/data4online_simulation/data_AQ59D_testset6.mat')
data = mat_data['data']
mat_times = scipy.io.loadmat('python_code/data4online_simulation/times_AQ59D_testset6.mat')
times = mat_times['times']
# mat_data = scipy.io.loadmat('data_AQ59D_testset7.mat')
# data = mat_data['data']
# mat_times = scipy.io.loadmat('times_AQ59D_testset7.mat')
# times = mat_times['times']

times = np.array(times)
sampling_rate = 500
num_channels = data.shape[0]
num_samples = data.shape[1]

# Create LSL outlet
info = StreamInfo('SimulatedEEG', 'EEG', num_channels, sampling_rate, 'float32', 'sourceid123456')
outlet = StreamOutlet(info)

print("\nWait for start, please press 1:")

try:
    input_var = int(input())
    if input_var == 1:
        print('Start sending streaming...')
        streaming_start = local_clock()
        # Simulate LSL input streaming
        for i in range(num_samples):
            sample = data[:, i]
            sample_timestamp = times[0, i]

            # Push the sample and timestamp through LSL outlet
            outlet.push_sample(sample, sample_timestamp)
            print(sample_timestamp)
            
            # Calculate elapsed time for precise timing
            elapsed_time = local_clock() - streaming_start

            # Wait until the desired time has passed for the next sample
            while elapsed_time < (i + 1) / sampling_rate:
                elapsed_time = local_clock() - streaming_start + 0.0002
            elapsed_time = local_clock() - streaming_start
            print('elapsed_time: ', elapsed_time)

        streaming_end = local_clock()
        print('Total streaming duration:', streaming_end - streaming_start)
        print('Expected duration:', num_samples / sampling_rate)

except ValueError as e:
    print(e)


