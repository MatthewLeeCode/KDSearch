

class KDTree:
    divided: bool = False
    branches: list = []

    def __init__(self, hyperparameter_ranges: dict[tuple]) -> None:
        """ Initialize the KDTree class with the parameters

        Args:
            hyperparameter_ranges: Parameters for the KDTree with the value
                being a tuple of (min, max) values for the parameter

        Example:
            hyperparameter_ranges = {
                "learning_rate": (0, 10),
                "alpha": (0, 1),
                "dropout": (0.5, 1)
            }
        """
        assert isinstance(hyperparameter_ranges, dict), "hyperparameter_ranges must be a dict"
        assert len(hyperparameter_ranges) > 0, "hyperparameter_ranges must have at least one parameter"

        self.hyperparameter_ranges = hyperparameter_ranges

        # Create the hyper parameters to use which is the mid point of the range
        self.hyperparameters = {}
        for key, value in self.hyperparameter_ranges.items():
            self.hyperparameters[key] = (value[0] + value[1]) / 2

    def divide(self) -> list:
        """ Divide the KDTree into branches
        
        Each branch will have a different set of the hyperparameter ranges.
        The number of branches will be len(hyperparameter_ranges) * 2
        """
        assert not self.divided, "KDTree has already been divided"

        self.divided = True
        
        # Create the branches
        for key, value in self.hyperparameter_ranges.items():
            # Create the new hyper parameter ranges
            new_hyperparameter_ranges = self.hyperparameter_ranges.copy()
            # Left branch
            new_hyperparameter_ranges[key] = (value[0], self.hyperparameters[key])
            self.branches.append(KDTree(new_hyperparameter_ranges.copy()))

            # Right branch
            new_hyperparameter_ranges = self.hyperparameter_ranges.copy()
            new_hyperparameter_ranges[key] = (self.hyperparameters[key], value[1])
            self.branches.append(KDTree(new_hyperparameter_ranges.copy()))
        
        return self.branches
