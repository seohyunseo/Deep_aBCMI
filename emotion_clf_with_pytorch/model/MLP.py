import torch.nn as nn

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.dropout2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x