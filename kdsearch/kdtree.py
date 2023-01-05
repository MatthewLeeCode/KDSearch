

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
