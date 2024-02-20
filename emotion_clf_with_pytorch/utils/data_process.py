import mne, warnings
import numpy as np # done in mac m1
from itertools import chain
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# Feature extraction
def get_feature(data):
    channel_no = [0, 2, 16, 19] # only taking these four channels
    feature_matrix = []
    for ith_video in range(data.shape[0]):
        features = []
        for ith_channel in channel_no:
            # power spectral density
            psd, freqs = plt.psd(data[ith_video][ith_channel], Fs = 128)
            # get frequency bands mean power
            theta_mean = np.mean(psd[np.logical_and(freqs >= 4, freqs <= 7)])
            alpha_mean = np.mean(psd[np.logical_and(freqs >= 8, freqs <= 13)])
            beta_mean  = np.mean(psd[np.logical_and(freqs >= 13, freqs <= 30)])
            gamma_mean = np.mean(psd[np.logical_and(freqs >= 30, freqs <= 40)])
            features.append([theta_mean, alpha_mean, beta_mean, gamma_mean])
        # flatten the features i.e. transform it from 2D to 1D
        feature_matrix.append(np.array(list(chain.from_iterable(features))))
    return np.array(feature_matrix)

# Label extraction
def get_labels(labels, class_type):
    # encoding
    num_labels = labels.shape[0]
    emotion = np.ones(num_labels)
    if(class_type=='valence'):
        for i in range(0, num_labels):
            if labels[i][0]>=5:
                emotion[i] = 0
            else:
                emotion[i] = 1
    elif(class_type=='arousal'):
        for i in range(num_labels):
            if labels[i][1]>=5:
                emotion[i] = 0
            else:
                emotion[i] = 1
    else:
        for i in range(num_labels):
            if(labels[i][0]>=5 and labels[i][1] >=5): # HVHA
                emotion[i] = 0
            elif(labels[i][0]>=5 and labels[i][1]<5): #HVLA
                emotion[i] = 1
            elif(labels[i][0]<5 and labels[i][1]>=5): #LVHA
                emotion[i] = 2
            else: #LVLA
                emotion[i] = 3
    return emotion

# Function to apply sliding window
def apply_sliding_window(data, window_size, overlap_size, sampling_rate):
    trials, channels, time_steps = data.shape
    window_size_in_samples = int(window_size * sampling_rate)
    overlap_size_in_samples = int(overlap_size * sampling_rate)

    new_time_steps = int((time_steps - window_size_in_samples) / overlap_size_in_samples) + 1

    # Initialize an empty array for the new data
    new_data = np.zeros((trials, channels, new_time_steps, window_size_in_samples))

    # Apply sliding window
    for trial in range(trials):
        for channel in range(channels):
            for i in range(new_time_steps):
                start_idx = i * overlap_size_in_samples
                end_idx = start_idx + window_size_in_samples
                new_data[trial, channel, i, :] = data[trial, channel, start_idx:end_idx]

    return new_data

# Features and labels extraction
def get_features_labels(dataset, args):
    # separate data and labels
    data = np.array(dataset['data']) # for current data
    labels = np.array(dataset['labels']) # for current labels
    # remove 3sec pre baseline
    data  = data[0:40,0:32,384:8064]

    # Apply sliding window to EEG data
    data = apply_sliding_window(data, args.window_size, args.overlap_size, args.sampling_rate)

    windowed_data = data.reshape(-1, data.shape[1], data.shape[3])
    windowed_labels = np.repeat(labels, data.shape[2], axis=0)

    features = get_feature(windowed_data)
    labels = get_labels(windowed_labels, args.class_type)

    return features, labels
