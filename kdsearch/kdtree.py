

class KDTree:
    divided: bool = False
    branches: list = []

    def __init__(self, hyper_parameter_ranges: dict[tuple]):
        """ Initialize the KDTree class with the parameters

        Args:
            hyper_parameter_ranges (dict): Parameters for the KDTree with the value
                being a tuple of (min, max) values for the parameter

        Example:
            hyper_parameter_ranges = {
                "learning_rate": (0, 10),
                "alpha": (0, 1),
                "dropout": (0.5, 1)
            }
        """
        assert isinstance(hyper_parameter_ranges, dict), "hyper_parameter_ranges must be a dict"
        assert len(hyper_parameter_ranges) > 0, "hyper_parameter_ranges must have at least one parameter"

        self.hyper_parameter_ranges = hyper_parameter_ranges

        # Create the hyper parameters to use which is the mid point of the range
        self.hyper_parameters = {}
        for key, value in self.hyper_parameter_ranges.items():
            self.hyper_parameters[key] = (value[0] + value[1]) / 2

    def divide(self):
        """ Divide the KDTree into branches 
        
        Each branch will have a different set of the hyperparameter ranges.
        The number of branches will be len(hyper_parameter_ranges) * 2
        """
        assert not self.divided, "KDTree has already been divided"

        self.divided = True
        
        # Create the branches
        for key, value in self.hyper_parameter_ranges.items():
            # Create the new hyper parameter ranges
            new_hyper_parameter_ranges = self.hyper_parameter_ranges.copy()
            # Left branch
            new_hyper_parameter_ranges[key] = (value[0], self.hyper_parameters[key])
            self.branches.append(KDTree(new_hyper_parameter_ranges.copy()))

            # Right branch
            new_hyper_parameter_ranges = self.hyper_parameter_ranges.copy()
            new_hyper_parameter_ranges[key] = (self.hyper_parameters[key], value[1])
            self.branches.append(KDTree(new_hyper_parameter_ranges.copy()))
