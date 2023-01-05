from kdsearch.kdtree import KDTree


def test_kdtree_init():
    """ Test the KDTree init method """
    hyper_parameter_ranges = {
        "learning_rate": (0, 10),
        "alpha": (0, 1),
        "dropout": (0.5, 1)
    }
    
    expected_hyper_parameters = {
        "learning_rate": 5,
        "alpha": 0.5,
        "dropout": 0.75
    }
    
    kdt = KDTree(hyper_parameter_ranges)
    
    assert kdt.hyper_parameter_ranges == hyper_parameter_ranges, "hyper_parameter_ranges not set correctly"
    assert kdt.hyper_parameters == expected_hyper_parameters, "hyper_parameters not set correctly"