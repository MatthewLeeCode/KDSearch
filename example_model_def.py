import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn


def train_model(X, y, X_test, y_test, hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 3
    batch_size = 64
    learning_rate = hyperparameters["learning_rate"]
    dropout = hyperparameters["dropout"]
    
    # Define the model
    class FashionMNISTClassifier(nn.Module):
        # Standard MLP
        def __init__(self):
            super(FashionMNISTClassifier, self).__init__()
            self.layer1 = nn.Linear(28 * 28, 256)
            self.layer2 = nn.Linear(256, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            x = x.reshape(x.size(0), -1)
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            return x
           
    # Create the model and move it to the device
    model = FashionMNISTClassifier().to(device)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    train_loader = data.DataLoader(
        data.TensorDataset(X, y), batch_size=batch_size, shuffle=True
    )
    
    test_loader = data.DataLoader(
        data.TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move data to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # Move data to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
    return accuracy