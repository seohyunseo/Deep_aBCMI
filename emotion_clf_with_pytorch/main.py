import scipy.io
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from configparser import ConfigParser
from train import train, train_kfold
from test import test, test_intermediate_features
from utils.data_process import load_data, load_n_data
from model.MLP import MLPClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)


    hidden_size = args.hidden_size
    hidden_size = [int(element) for element in hidden_size.split(', ')]
    channel_no = args.channel_no
    channel_no = [int(element)-1 for element in channel_no.split(', ')]

    features, labels = load_data(args.data_path, args.subject_no, channel_no, args)
    # print(features.shape)
    # print(labels.shape)
    # sys.exit()    

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False) # Split the dataset into training and testing sets

    if args.task_type == 'train':

        model = MLPClassifier(input_size=features.shape[1], hidden_sizes= hidden_size, num_classes=len(np.unique(labels)), dropout_rate=args.dropout_rate)
        model = model.to(device)

        if args.k_folds > 0:
            print(f"[*] Start {args.k_folds}-fold training...")
            train_kfold(model, X_train, y_train, args)
            model_path = args.model_path + args.model_name + '_' + args.class_type + '_' + str(args.k_folds)+'-folds.pt'
        else:
            print("[*] Start training...")
            train(model, X_train, y_train, args)
            model_path = args.model_path + args.model_name + '_' + args.class_type + '.pt'

        torch.save(model, model_path)

    elif args.task_type == 'test':

        print("[*] Start testing...")
        model_path = args.model_path + args.model_name + '_' + args.class_type + '.pt'
        model = torch.load(model_path)
        test(model, X_test, y_test)
    
    elif args.task_type == 'intermediate':

        print("[*] Start extracting intermediate feature...")
        model_path = args.model_path + args.model_name + '_' + args.class_type + '.pt'
        feature_path = args.feature_path  + args.model_name + '_' + args.feature_layer + '_' + hidden_size[-1] + '.csv'
        model = torch.load(model_path)
        intermediate_features = test_intermediate_features(model, X_test, y_test, feature_path, args.feature_layer)


if __name__=='__main__':

    configparser = ConfigParser()
    configparser.read('./config/config.ini')

    parser = ArgumentParser()
    # data related
    parser.add_argument("--data_path", default=configparser.get('data', 'data_path'), type=str)
    parser.add_argument("--subject_no", default='s01', type=str)
    parser.add_argument("--window_size",default=configparser.get('data', 'window_size'),type=int)
    parser.add_argument("--overlap_size",default=configparser.get('data', 'overlap_size'),type=float)
    parser.add_argument("--sampling_rate",default=configparser.get('data', 'sampling_rate'),type=int)
    parser.add_argument("--class_type",default='4-cls',type=str, choices=['4-cls', 'Arousal', 'Valence'])
    parser.add_argument("--channel_no",default=configparser.get('data', 'channel_no'),type=str)

    # model related
    parser.add_argument("--task_type", default='test', type=str, choices=['train', 'test', 'intermediate'])
    parser.add_argument("--k_folds", default=configparser.get('model', 'k_folds'), type=int)
    parser.add_argument("--batch_size", default=configparser.get('model', 'batch_size'), type=int)
    parser.add_argument("--learning_rate", default=configparser.get('model', 'learning_rate'), type=float)
    parser.add_argument("--num_epochs", default=configparser.get('model', 'num_epochs'), type=int)
    parser.add_argument("--hidden_size",default=configparser.get('model', 'hidden_size'),type=str)
    parser.add_argument("--weight_decay",default=configparser.get('model', 'weight_decay'),type=float)
    parser.add_argument("--model_path",default=configparser.get('model', 'model_path'),type=str)
    parser.add_argument("--model_name",default=configparser.get('model', 'model_name'),type=str)
    parser.add_argument("--dropout_rate",default=configparser.get('model', 'dropout_rate'),type=float)
    parser.add_argument("--random_seed",default=configparser.get('model', 'random_seed'),type=int)

    parser.add_argument("--feature_path",default=configparser.get('feature', 'feature_path'),type=str)
    parser.add_argument("--feature_layer",default=configparser.get('feature', 'layer'),type=str)
    
    args = parser.parse_args()
    main(args)

