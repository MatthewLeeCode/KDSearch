""" Compares the results of KDSearch with:

- GridSearch
- RandomSearch
- BayesianOptimization
"""
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np
from example_model_def import train_model
import kdsearch

torch.manual_seed(0)
np.random.seed(0)

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

# Generate a grid of hyperparameters
learning_step = 0.00001
dropout_step = 0.1

# Generate grid with 10 steps for learning rate and dropout
learning_rates = [0.00001 + learning_step * i for i in range(10)]
dropouts = [0 + dropout_step * i for i in range(10)]

matrix = np.zeros((len(learning_rates), len(dropouts)))
for i, lr in tqdm(enumerate(learning_rates)):
    for j, d in enumerate(dropouts):
        output = kdsearch.train_cross_validator(X, y, train_model, {"learning_rate": lr, "dropout": d}, 1, 0)
        matrix[i, j] = output
        
# Save numpy array
np.save("data/results/grid.npy", matrix)
