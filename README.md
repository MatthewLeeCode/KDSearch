# KDTree-based Hyperparameter Optimization
<While functional, this is still an experimental work in progress>
This project implements a k-dimensional binary search tree (KDTree) algorithm for hyperparameter optimization. The algorithm is designed to search for the optimal hyperparameters for a model by dividing the hyperparameter space into smaller and smaller regions until the best hyperparameters are found.

The main components of the project are the KDTree class, the train_cross_validator function, the remove_duplicates function, and the search function.

## KDTree
The KDTree class creates a KDTree for a set of hyperparameters and their ranges. Each node in the tree represents a region of the hyperparameter space, and the hyperparameters of the node are the midpoint of the ranges of the hyperparameters in the region. The class provides methods to divide the tree into branches, get the bounding boxes and hyperparameters of the tree and its branches, and more.

## train_cross_validator
The train_cross_validator function trains a model using the given hyperparameters and evaluates its performance using cross validation. The function returns the mean score from cross validation.

## remove_duplicates
The remove_duplicates function removes duplicate branches from a list of KDTree objects. Unlike Quadtrees, KDTree branches can be identical, and the function ensures that we don't train the same model twice.

## search
The search function searches for the optimal hyperparameters for a model using the KDTree algorithm. The function takes a set of features, labels, a model creation function, and hyperparameter ranges as inputs, and returns the best hyperparameters found, a list of all hyperparameters found, and the root of the KDTree. The function uses the train_cross_validator and remove_duplicates functions to train and evaluate models and prune the tree, respectively.

Example usage
```python
import numpy as np
from kdsearch.kdtree import KDTree, train_cross_validator, remove_duplicates, search

# Define the hyperparameter ranges
hyperparameter_ranges = {
    "learning_rate": (0, 10),
    "alpha": (0, 1),
    "dropout": (0.5, 1)
}

# Define the model creation function
def create_model(X_train, y_train, X_test, y_test, hyperparameters):
    # Train and evaluate the model
    score = ...
    return score

# Search for the optimal hyperparameters
best_result, results, root = search(X, y, create_model, hyperparameter_ranges)
print("Best hyperparameters:", best_result["hyperparameters"])
```
Note: The create_model function should accept the training and test data, as well as the hyperparameters, and return the score of the model. The example usage is a simplified version, and the actual implementation will depend on the specifics of the model being trained.
