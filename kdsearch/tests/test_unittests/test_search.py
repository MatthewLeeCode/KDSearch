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
    
    results = search(X, y, model_func, hyperparameter_ranges, 0.5, True, 3, 2)
    assert len(results) == 15
    assert mock_train_cross_validator.call_count == 29
