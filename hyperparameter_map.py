""" Compares the results of KDSearch with:

- GridSearch
- RandomSearch
- BayesianOptimization
"""
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.utils.data as data
import torch.nn as nn
import kdsearch


# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 3
batch_size = 64


def train_model(X, y, X_test, y_test, hyperparameters):
    learning_rate = hyperparameters["learning_rate"]
    dropout = hyperparameters["dropout"]
    
    # Define the model
    class FashionMNISTClassifier(nn.Module):
        def __init__(self):
            super(FashionMNISTClassifier, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = x.view(-1, 1, 28, 28)
            x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
            x = nn.functional.dropout(x, training=self.training, p=dropout) 
            x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.dropout(x, training=self.training, p=dropout)
            x = self.fc2(x)
            return nn.functional.log_softmax(x, dim=1)
           
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


# Load data into X and y
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

X = train_data.train_data.float()
y = train_data.train_labels
X_test = test_data.test_data.float()
y_test = test_data.test_labels

hyper_parameter_ranges = {
    "learning_rate": [0.00001, 0.0001],
    "dropout": [0.1, 0.9]
}

#output = train_model(X, y, X_test, y_test, {"learning_rate": 0.001, "dropout": 0.5})
output = kdsearch.search(X, y, train_model, n_splits=1, depth=5, hyperparameter_ranges=hyper_parameter_ranges)
print(output)