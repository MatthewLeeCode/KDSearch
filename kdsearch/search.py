""" Uses the KDTree to search for the optimal hyperparameters for a model. """
import numpy as np
from sklearn.model_selection import ShuffleSplit
from kdsearch.kdtree import KDTree
from tqdm import tqdm


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


def remove_duplicates(kdtrees: list[KDTree]) -> list[KDTree]:
    """ Remove duplicate branches from a list of KDTree objects
    
    Unlike Quadtrees, KDTree branches can be identical. We don't want to
    train the same model twice, so we remove duplicates.
    
    Args:
        kdtrees: The list of KDTree objects
    Returns:
        The list of KDTree objects with duplicates removed
    """
    hyperparameters = [kdtree.hyperparameters for kdtree in kdtrees]
    unique_hyperparameters = []
    unique_kdtrees = []
    for i, hyperparameter in enumerate(hyperparameters):
        if hyperparameter not in unique_hyperparameters:
            unique_hyperparameters.append(hyperparameter)
            unique_kdtrees.append(kdtrees[i])

    return unique_kdtrees


def search(
        X: np.ndarray,
        y: np.ndarray,
        model_func: callable,
        hyperparameter_ranges: dict[tuple],
        num_best_branches: int = 4,
        larger_is_better: bool = True,
        n_splits: int = 5,
        depth: int = 10,
        seed: int = None
    ) -> list[dict, list, KDTree]:
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
        num_best_branches: The number of branches to keep (if positive) or remove \
            from the end (if negative). We pass the value into a list slice \
            [:num_best_branches]. If None, we keep all branches.
        larger_is_better: Whether larger output values of model_func are better
        n_splits: The number of splits for cross validation
        depth: The depth of the KDTree to search
        seed: The seed for the random number generator
    Returns:
        (the best result, A list of the best results found, the root of the KDTree)
    
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
    
    results = []
    queue = [root]
    for d in tqdm(range(1, depth + 1)):  # Start count at 1
        # Remove duplicate branches
        queue = remove_duplicates(queue)
        
        # We want to store the depth results so we can prune the queue
        depth_results = []
        for branch in queue:
            score = train_cross_validator(X, y, model_func, branch.hyperparameters, n_splits, seed)
            depth_results.append({"hyperparameters": branch.hyperparameters, "score": score, "depth": d})
            
        # Add all the results to the list
        results.extend(depth_results)
        
        # Sort the results and the queue
        combined = zip(depth_results, queue)
        combined = sorted(combined, key=lambda x: x[0]["score"], reverse=larger_is_better)
        
        if num_best_branches is not None:
            combined = combined[:num_best_branches]
        
        # Add the best results to the list and divide the branches
        new_queue = []
        for _, branch in combined:
            new_queue.extend(branch.divide())
            
        # Update the queue
        queue = new_queue
    
    results = sorted(results, key=lambda x: x["score"], reverse=larger_is_better)
    return results[0], results, root
