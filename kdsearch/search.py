""" Uses the KDTree to search for the optimal hyperparameters for a model. """
import numpy as np
from sklearn.model_selection import ShuffleSplit
from kdsearch.kdtree import KDTree


def train_cross_validator(
        X: np.ndarray,
        y: np.ndarray,
        model_func: callable,
        hyperparameters: dict,
        n_splits: int,
        seed: int = 0
    ) -> float:
    """ Train a model and return the score from cross validation
    
    Args:
        X: The features
        y: The labels
        model_func: The function to run the model. Should accept (X, y, X_test, y_test, hyper_parameters)
        hyperparameters: The hyperparameters to use as a dictionary {"learning_rate": 0.1, "alpha": 0.5, etc}
        n_splits: The number of splits for cross validation
        seed: The seed for the random number generator
    
    Returns:
        The mean score from cross validation
    """
    cv = ShuffleSplit(n_splits=n_splits, random_state=seed)
    
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        score = model_func(X_train, y_train, X_test, y_test, hyperparameters)
        scores.append(score)

    return np.mean(scores)


def search(
        X: np.ndarray,
        y: np.ndarray,
        model_func: callable,
        hyperparameter_ranges: dict[tuple],
        pruning_cutoff: float = 0.5,
        larger_is_better: bool = True,
        n_splits: int = 5,
        depth: int = 10,
        seed: int = 42
    ) -> list[dict]:
    """ Search for the optimal hyperparameters for a model
    using the KDTree algorithm.
    
    Splits the hyperparameter space into smaller and smaller spaces
    until the best hyperparameters are found.

    Will only subdivide the space if the result is in the top 'pruning_cutoff'
    
    Args:
        X: The features
        y: The labels
        model_func: The function to create the model
        hyperparameter_ranges: The ranges for the hyperparameters
        pruning_cutoff: The percentage of the best results to keep
        larger_is_better: Whether larger output values of model_func are better
        n_splits: The number of splits for cross validation
        depth: The depth of the KDTree to search
        seed: The seed for the random number generator
    Returns:
        A list of the best results found
    
    Example:
        {
            "hyperparameters": {
                "learning_rate": 0.1,
                "alpha": 0.5,
                "dropout": 0.8
            },
            "score": 0.8
        }
    """
    root = KDTree(hyperparameter_ranges)
    score = train_cross_validator(X, y, model_func, root.hyperparameters, n_splits, seed)
    results = [{"hyperparameters": root.hyperparameters, "score": score}]
    
    queue = root.divide()
    for _ in range(depth + 1):  # +1 because we start with the root
        # We want to store the depth results so we can prune the queue
        depth_results = []
        for branch in queue:
            score = train_cross_validator(X, y, model_func, branch.hyperparameters, n_splits, seed)
            depth_results.append({"hyperparameters": branch.hyperparameters, "score": score})
            
        # Sort the results and the queue
        combined = zip(depth_results, queue)
        combined = sorted(combined, key=lambda x: x[0]["score"], reverse=larger_is_better)
        
        # Prune the results
        cutoff = int(len(depth_results) * pruning_cutoff)
        
        # Add the best results to the list and divide the branches
        new_queue = []
        for result, branch in combined[:cutoff]:
            results.append(result)
            new_queue.extend(branch.divide())
            
        # Update the queue
        queue = new_queue
        
    return results
