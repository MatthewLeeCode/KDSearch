from kdsearch.search import search, train_cross_validator
from unittest.mock import patch
import numpy as np


def test_train_cross_validator():
    """ Test the train_cross_validator function """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([1, 1, 1, 0, 0, 0])
    hyperparameters = {"alpha": 0.1, "learning_rate": 0.1}
    
    def model_func(X_train, y_train, X_test, y_test, hyperparameters):
        """ A model function that returns the accuracy score """
        return 0.5
    
    score = train_cross_validator(X, y, model_func, hyperparameters, 3)
    assert score == 0.5


def test_train_cross_validator_seed():
    """ Ensure that the seed is set correctly in train_cross_validator """
    X = np.array([[1, 2], [2, 1], [5, 6], [6, 5], [3, 2], [2, 3]])
    y = np.array([1, 0, 1, 0, 1, 0])
    hyperparameters = {"alpha": 0.1, "learning_rate": 0.1}
    
    def model_func(X_train, y_train, X_test, y_test, hyperparameters):
        """ A model function that returns the accuracy score """
        sum = np.sum(X_train) + np.sum(y_train) + np.sum(X_test) + np.sum(y_test)
        return sum
    
    score = train_cross_validator(X, y, model_func, hyperparameters, 3, seed=0)
    assert score == 41.0

@patch("kdsearch.search.train_cross_validator", return_value=0.5)
def test_search(mock_train_cross_validator):
    """ Test the search function """
    
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([1, 1, 1, 0, 0, 0])
    hyperparameter_ranges = {
        "alpha": (0.0, 1.0),
        "learning_rate": (0.0, 1.0)
    }
    
    def model_func(X_train, y_train, X_test, y_test, hyperparameters):
        """ A model function that returns the accuracy score """
        # Model function is never used in this test as its functionality is used
        # in train_cross_validator which is mocked
        return
    
    results = search(
        X=X, 
        y=y,
        model_func=model_func,
        hyperparameter_ranges=hyperparameter_ranges,
        num_best_branches=4,
        larger_is_better=True,
        depth=3)[1]
    assert len(results) == 9
    assert mock_train_cross_validator.call_count == 17  # 4 duplicates


@patch("kdsearch.search.train_cross_validator", return_value=0.5)
def test_search_seed(mock_train_cross_validator):
    """ Test the search function seed provides the same result """
    
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([1, 1, 1, 0, 0, 0])
    hyperparameter_ranges = {
        "alpha": (0.0, 1.0),
        "learning_rate": (0.0, 1.0)
    }
    
    def model_func(X_train, y_train, X_test, y_test, hyperparameters):
        """ A model function that returns the accuracy score """
        # Model function is never used in this test as its functionality is used
        # in train_cross_validator which is mocked
        return 
    
    results = search(
        X=X, 
        y=y,
        model_func=model_func,
        hyperparameter_ranges=hyperparameter_ranges,
        num_best_branches=4,
        larger_is_better=True,
        depth=3)[1]
    assert len(results) == 9
    assert mock_train_cross_validator.call_count == 17  # 4 duplicates