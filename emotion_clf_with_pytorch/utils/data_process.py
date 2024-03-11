import mne, warnings
import numpy as np # done in mac m1
from itertools import chain
import matplotlib.pyplot as plt
import scipy.io
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# Feature extraction
def get_psd(data):
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

def get_raw(data):
    
    raw_matrix = data.reshape(data.shape[0], -1)
    
    return raw_matrix

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
def get_features_labels(dataset, ch_no, args):
    # separate data and labels
    data = np.array(dataset['data']) # for current data
    labels = np.array(dataset['labels']) # for current labels
    # remove 3sec pre baseline
    data  = data[0:40,:,384:8064]
    data = np.take(np.array(data), ch_no, axis=1)

    # Apply sliding window to EEG data
    data = apply_sliding_window(data, args.window_size, args.overlap_size, args.sampling_rate)

    windowed_data = data.transpose(0, 2, 1, 3)
    windowed_data = windowed_data.reshape(windowed_data.shape[0]* windowed_data.shape[1], windowed_data.shape[2], windowed_data.shape[3])
    windowed_labels = np.repeat(labels, data.shape[2], axis=0)

    features = get_raw(windowed_data)
    labels = get_labels(windowed_labels, args.class_type)

    return features, labels

def load_data(data_path, sub_no, ch_no, args):

    data_path = data_path + sub_no + '.mat'
    deap_dataset = scipy.io.loadmat(data_path)
    features, labels = get_features_labels(deap_dataset, ch_no, args)

    return features, labels

def load_n_data(data_path, sub_no, ch_no, args):

    n_features = []
    n_labels = []
    for n in range(0, sub_no): 
        s = ''
        if (n+1) < 10:
            s += '0'
        s += str(n+1)

        filename = data_path +"s" + s + ".mat"
        deap_dataset = scipy.io.loadmat(filename)
        # separate data and labels   
        features, labels = get_features_labels(deap_dataset, ch_no, args)
        n_features.append(features)
        n_labels.append(labels)

    n_features = np.array(n_features)
    n_labels = np.array(n_labels)
    features = n_features.reshape(n_features.shape[0]*n_features.shape[1], n_features.shape[2])
    labels = n_labels.reshape(n_labels.shape[0]*n_labels.shape[1])

    return features, labels