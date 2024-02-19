import scipy.io
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from train import train
from test import test
from utils.data_process import *
from model.MLP import MLPClassifier

################################################################################
# config
################################################################################

parser = ArgumentParser()
# data related
parser.add_argument("--data_path", default='../data/deap/', type=str)
parser.add_argument("--subject_no", default='s01', type=str)
parser.add_argument("--window_size",default=10,type=int)
parser.add_argument("--overlap_size",default=5,type=float)
parser.add_argument("--sampling_rate",default=128,type=int)
parser.add_argument("--class_type",default='4-cls',type=str, choices=['4-cls', 'Arousal', 'Valence'])

# model related
parser.add_argument("--task_type", default='test', type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--hidden_size",default=50,type=int)
parser.add_argument("--weight_decay",default=.0,type=float)
parser.add_argument("--model_path",default='./save/models/',type=str)
parser.add_argument("--model_name",default='MLP',type=str)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
################################################################################
# main
################################################################################

if __name__=='__main__':
    deap_dataset = scipy.io.loadmat(args.data_path + args.subject_no + '.mat')

    # separate data and labels
    data = np.array(deap_dataset['data']) # for current data
    labels = np.array(deap_dataset['labels']) # for current labels
    # remove 3sec pre baseline
    data  = data[0:40,0:32,384:8064]

    # Apply sliding window to EEG data
    data = apply_sliding_window(data, args.window_size, args.overlap_size, args.sampling_rate)

    windowed_data = data.reshape(-1, data.shape[1], data.shape[3])
    windowed_labels = np.repeat(labels, data.shape[2], axis=0)

    features = get_feature(windowed_data)
    labels = get_class_labels(windowed_labels, args.class_type)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42) # Split the dataset into training and testing sets
    

    if args.task_type == 'train':

        model = MLPClassifier(input_size=features.shape[1], hidden_size= args.hidden_size, num_classes=len(np.unique(labels)))
        model = model.to(device)

        train(model, X_train, y_train, args)

        model_path = args.model_path + args.model_name + '.pt'
        torch.save(model, model_path)

    elif args.task_type == 'test':

        model_path = args.model_path + args.model_name + '.pt'
        model = torch.load(model_path)
        test(model, X_test, y_test)

