import pickle, mne, os, random, warnings, copy, sys, shutil# done in mac m1
import numpy as np # done in mac m1
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay,\
 f1_score, confusion_matrix, precision_score, recall_score
import sklearn.model_selection as model_selection
import scipy.io
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


def get_four_class(val, ar):
    # decoding
    emotion = np.ones(val.shape[0])
    assert val.shape[0]==ar.shape[0]
    for i in range(0, val.shape[0]):
        if(val[i]==1 and ar[i]==1): # HVHA
            emotion[i] = 1
        elif(val[i]==1 and ar[i]==0): #HVLA
            emotion[i] = 4
        elif(val[i]==0 and ar[i]==1): #LVHA
            emotion[i] = 2
        else: #LVLA
            emotion[i] = 3
    return emotion

def get_class_labels(labels, class_type):
    # encoding
    emotion = np.ones(40)
    if(class_type=='valence'):
        for i in range(0, 40):
            if labels[i][0]>=5:
                emotion[i] = 0
            else:
                emotion[i] = 1
    elif(class_type=='arousal'):
        for i in range(40):
            if labels[i][1]>=5:
                emotion[i] = 0
            else:
                emotion[i] = 1
    else:
        for i in range(40):
            if(labels[i][0]>=5 and labels[i][1] >=5): # HVHA
                emotion[i] = 1
            elif(labels[i][0]>=5 and labels[i][1]<5): #HVLA
                emotion[i] = 4
            elif(labels[i][0]<5 and labels[i][1]>=5): #LVHA
                emotion[i] = 2
            else: #LVLA
                emotion[i] = 3
    return emotion

def get_feature(data):
    channel_no = [0, 2, 16, 19] # only taking these four channels
    feature_matrix = []
    for ith_video in range(40):
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

def get_data(subject_no, dataset_path):
    # read the data
    deap_dataset = scipy.io.loadmat(dataset_path + subject_no + '.mat')
    # separate data and labels
    data = np.array(deap_dataset['data']) # for current data
    labels = np.array(deap_dataset['labels']) # for current labels
    # remove 3sec pre baseline
    data  = data[0:40,0:32,384:8064]
    # signal processing
    # data = signal_pro(data)
    # feature extraction
    feature = get_feature(data)
    # class label
    four_class_labels = get_class_labels(labels, 'four_class')
    valence_labels = get_class_labels(labels, 'valence')
    arousal_labels = get_class_labels(labels, 'arousal')
    return feature, valence_labels, arousal_labels, four_class_labels

def emotion_classification_actual(data, label, type=None):

    model = load(f'./model/model_{type}.joblib')

    pred_label = model.predict(data)
    true_label = label

    return true_label, pred_label

if __name__ == '__main__':
    subject_no = 's01'
    dataset_path = './data/'
    features, valence_labels, arousal_labels, four_class_labels = get_data(subject_no, dataset_path)

    true_val, pred_val = emotion_classification_actual(features, valence_labels, 'valence')
    test_ar, pred_ar = emotion_classification_actual(features, arousal_labels, 'arousal')

    four_pred = get_four_class(pred_val, pred_ar)
    four_test = get_four_class(true_val, test_ar)

    for i in range(four_test.shape[0]):
        emotion_tag = four_pred[i]
        print(f"predicted: {int(emotion_tag)}, true: {int(four_test[i])}")

