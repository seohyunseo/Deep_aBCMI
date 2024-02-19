import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DEAPDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.device = device

        # Standardize the data
        scaler = StandardScaler()
        self.features = scaler.fit_transform(features)

        assert self.features.shape[0] == self.labels.shape[0]

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):

        sample_features = self.features[idx]
        sample_labels = self.labels[idx]

        features = torch.tensor(sample_features, dtype=torch.float32).to(self.device)
        labels = torch.tensor(sample_labels, dtype=torch.float32).to(self.device)

        return features, labels