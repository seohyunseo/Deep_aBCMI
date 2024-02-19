import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.dataset import DEAPDataset
from torch.utils.data import DataLoader

def train(model, X, y, args):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # Split the dataset into training and validation sets

    train_torch= DEAPDataset(X_train, y_train)
    train_dataloader = DataLoader(train_torch, batch_size=args.batch_size, shuffle=True)

    val_torch = DEAPDataset(X_val, y_val)
    val_dataloader = DataLoader(val_torch, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_losses = []
    val_losses = []

    for epoch in range(args.num_epochs):
        model.train()
        train_epoch_loss = 0.0
        train_correct_predictions = 0

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels.long())
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct_predictions += (predicted == labels).sum().item()


            with torch.no_grad():
                val_epoch_loss = 0.0
                for val_inputs, val_labels in val_dataloader:
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels.long())
                    val_epoch_loss += val_loss.item()


        average_train_loss = train_epoch_loss / len(train_dataloader.dataset)
        accuracy = train_correct_predictions / len(train_dataloader.dataset)
        train_losses.append(average_train_loss)

        average_val_loss = val_epoch_loss / len(val_dataloader.dataset)
        val_losses.append(average_val_loss)

        print(f'Epoch [{epoch + 1}/{args.num_epochs}] => Train Loss: {average_train_loss:.4f}, Accuracy: {accuracy:.4f} | Validation Loss {average_val_loss:.4f}')
