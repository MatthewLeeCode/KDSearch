import kdsearch
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np
from example_model_def import train_model
import json

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

hyperparameter_ranges = {
    "learning_rate": [0.00001, 0.0001],
    "weight_decay": [0, 0.9]
}

best_result, results, kdtree = kdsearch.search(
    X=X,
    y=y,
    model_func=train_model,
    hyperparameter_ranges=hyperparameter_ranges,
    num_best_branches=1,
    n_splits=1,
    depth=5,
    seed=0
)

#coords = kdtree.get_tree_hyperparameters()
#bboxs = kdtree.get_tree_bbox()

# Save the best result
with open("data/results/best_result.json", "w") as f:
    json.dump(best_result, f)

# Save results
with open("data/results/coords.json", "w") as f:
    json.dump(results, f)

#with open("data/results/bboxs.json", "w") as f:
    #json.dump(bboxs, f)
