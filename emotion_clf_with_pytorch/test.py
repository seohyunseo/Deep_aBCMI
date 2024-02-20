import torch
import csv
from torch.utils.data import DataLoader
from dataset.dataset import DEAPDataset
import numpy as np

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def test(model, X, y):

    test_torch = DEAPDataset(X, y)
    test_dataloader = DataLoader(test_torch, batch_size=1, shuffle=True)
    # Evaluate on the test set
    model.eval()

    with torch.no_grad():
        correct_predictions = 0

        for test_inputs, test_labels in test_dataloader:
            test_outputs = model(test_inputs)
            # print(test_outputs)
            _, predicted_labels = torch.max(test_outputs, 1)

            correct_predictions += (predicted_labels == test_labels).sum().item()

        test_accuracy = correct_predictions / len(test_dataloader.dataset)
        print(f'\nTest Accuracy: {test_accuracy:.4f}\n')


def test_intermediate_features(model, X, y, file_path, layer):

    test_torch = DEAPDataset(X, y)
    test_dataloader = DataLoader(test_torch, batch_size=1, shuffle=True)
    # Evaluate on the test set
    model.eval()
    intermediate_features = []

    getattr(model, layer).register_forward_hook(get_activation(layer))

    with torch.no_grad():
        correct_predictions = 0

        # Open the CSV file in write mode
        with open(file_path, 'w', newline='') as csvfile:
            # Create a CSV writer
            csvwriter = csv.writer(csvfile)
            # Write header row
            csvwriter.writerow(['Trial Number', 'Flattened Array', 'Prediected Label'])
            for idx, (test_inputs, test_labels) in enumerate(test_dataloader):
                test_outputs = model(test_inputs)
                # print(test_outputs)
                _, predicted_labels = torch.max(test_outputs, 1)
                
                if int(predicted_labels) == int(test_labels):

                    feature = [float('%.4f' % value) for value in activation[layer].cpu().flatten()]

                    print(f"Trial[{idx+1}] prediction: {int(predicted_labels)}, truth: {int(test_labels)}")
                    print(f"Intermediate features: {feature}\n")

                    intermediate_features.append(feature)
                    csvwriter.writerow([idx+1, feature, int(predicted_labels)])

                correct_predictions += (predicted_labels == test_labels).sum().item()

        test_accuracy = correct_predictions / len(test_dataloader.dataset)
        print('===========================Info==============================')
        print(f'Test Accuracy: {test_accuracy:.4f}')
        print(f'Returned Intermediate Features: {len(intermediate_features)}/{len(test_dataloader.dataset)}')
        print(f'Extracted Features from \'{layer}\' layer')
        print('=============================================================')

    return intermediate_features