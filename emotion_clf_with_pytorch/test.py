import torch
from torch.utils.data import DataLoader
from dataset.dataset import DEAPDataset

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
    idx = 0
    intermediate_features = []

    model.softmax.register_forward_hook(get_activation('softmax'))

    with torch.no_grad():
        correct_predictions = 0

        for test_inputs, test_labels in test_dataloader:
            test_outputs = model(test_inputs)
            # print(test_outputs)
            _, predicted_labels = torch.max(test_outputs, 1)

            intermediate_features.append(activation['softmax'].cpu().numpy())

            # print(f"Trial[{idx+1}] prediction: {int(predicted_labels)}, truth: {int(test_labels)}")
            idx += 1

            correct_predictions += (predicted_labels == test_labels).sum().item()

        test_accuracy = correct_predictions / len(test_dataloader.dataset)
        print(f'Test Accuracy: {test_accuracy:.4f}')

        # for i in range(len(intermediate_features)):
        #     print(f"Trial[{i+1}] {intermediate_features[i]}")