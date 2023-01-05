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
    

def test_kdtree_divide():
    """ Test the KDTree divide method 
    
    Does not abstract the __init__ method. If __init__ is changed, this test will fail.
    """
    hyper_parameter_ranges = {
        "learning_rate": (0, 10),
        "alpha": (0, 1),
        "dropout": (0.5, 1)
    }
    
    expected_subdivide_parameter_ranges = [
        {"learning_rate": (0, 5), "alpha": (0, 1), "dropout": (0.5, 1)},
        {"learning_rate": (5, 10), "alpha": (0, 1), "dropout": (0.5, 1)},
        {"learning_rate": (0, 10), "alpha": (0, 0.5), "dropout": (0.5, 1)},
        {"learning_rate": (0, 10), "alpha": (0.5, 1), "dropout": (0.5, 1)},
        {"learning_rate": (0, 10), "alpha": (0, 1), "dropout": (0.5, 0.75)},
        {"learning_rate": (0, 10), "alpha": (0, 1), "dropout": (0.75, 1)}
    ]
    
    kdt = KDTree(hyper_parameter_ranges)
    kdt.divide()
    
    # Get the ranges for each branch
    subdivide_parameter_ranges = [branch.hyper_parameter_ranges for branch in kdt.branches]
    
    assert subdivide_parameter_ranges == expected_subdivide_parameter_ranges, \
        "subdivide_parameter_ranges not set correctly"
