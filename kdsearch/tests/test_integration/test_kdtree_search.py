""" Integration test that runs the search function using the KDTree algorithm. """
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn import svm
from kdsearch.search import search


def dummy_data():
    """ Returns dummy data for testing """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([1, 1, 1, 0, 0, 0])
    return X, y


def test_dummy_model():
    """ Tests the search function using a dummy model
    
    Dummy model returns the delta between the expected params and the actual params.
    It acts as an oracle that knows the optimal hyperparameters. The KDTree search
    should converge on the optimal hyperparameters.
    """
    hyperparameter_ranges = {"alpha": (0.0, 1.0), "learning_rate": (0.0, 1.0)}
    optimal_hyperparameters = {"alpha": 0.125, "learning_rate": 0.885}
    
    def dummy_model(X_train, y_train, X_test, y_test, hyperparameters):
        """ A model function that returns the delta between the expected params and the actual params """
        delta = 0
        for key, value in hyperparameters.items():
            delta += abs(value - optimal_hyperparameters[key])
        return delta
    
    results = search(
        X=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]),
        y=np.array([1, 1, 1, 0, 0, 0]),
        model_func=dummy_model,
        hyperparameter_ranges=hyperparameter_ranges,
        num_best_branches=2,
        larger_is_better=False,
        n_splits=1,
        depth=10
    )
    

def test_svr_model():
    """ Tests the search function using a svr model """
    data = load_digits()
    X = data.data
    y = data.target
    
    hyperparameter_ranges = {"tol": [1e-3, 1e-6], "C": [1, 3], "epsilon": [0.1, 0.5]}
    
    def model_func(X_train, y_train, X_test, y_test, hyperparameters):
        """ A model function that returns the delta between the expected params and the actual params """
        model = svm.SVR()
        model.set_params(**hyperparameters)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    
    results = search(
        X=X,
        y=y,
        model_func=model_func,
        hyperparameter_ranges=hyperparameter_ranges,
        num_best_branches=4,
        larger_is_better=True,
        n_splits=3,
        depth=10
    )
    
    print(results)
