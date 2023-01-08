

class KDTree:
    divided: bool = False

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
            
        self.branches = []

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
            # Left branch
            new_hyperparameter_ranges = self.hyperparameter_ranges.copy()
            new_hyperparameter_ranges[key] = (value[0], self.hyperparameters[key])
            left_branch = KDTree(new_hyperparameter_ranges.copy())
            self.branches.append(left_branch)

            # Right branch
            new_hyperparameter_ranges = self.hyperparameter_ranges.copy()
            new_hyperparameter_ranges[key] = (self.hyperparameters[key], value[1])
            right_branch = KDTree(new_hyperparameter_ranges.copy())
            self.branches.append(right_branch)
        
        return self.branches
    
    def get_bbox(self) -> list:
        """
        Returns the top-left and bottom-right coordinates of the bounding box of the region
        this branch is searching. The bounding box can be in any number of dimensions.

        Returns:
            [x1, y1, x2, y2] -> 2D
            [x1, y1, z1, x2, y2, z2] -> 3D
            [x1, y1, z1, w1, x2, y2, z2, w2] -> 4D
            (you get the idea)
        """
        top_left_values = []
        bottom_right_values = []
        for key, value in self.hyperparameter_ranges.items():
            # The value is the min and max of the hyperparameter
            top_left_values.append(value[0])
            bottom_right_values.append(value[1])
        
        bbox = top_left_values + bottom_right_values
        return bbox
    
    def get_tree_hyperparameters(self) -> list:
        """ Get the hyperparameters for the KDTree. Including the hyperparameters
        for the branches
        
        Returns:
            List of hyperparameters from the KDTree
        """
        hyperparameters = []
        hyperparameters.append(self.hyperparameters)
        
        for branch in self.branches:
            hyperparameters.extend(branch.get_tree_hyperparameters())
                
        return hyperparameters
    
    def get_tree_bbox(self) -> list:
        """
        Get the top-left and bottom-right coordinates of the bounding box of the
        KDTree. Including the bounding boxes of the branches. 
        
        Returns:
            List of bounding boxes from the KDTree
        """
        bbox = []
        bbox.append(self.get_bbox())
        
        for branch in self.branches:
            bbox.extend(branch.get_tree_bbox())
                
        return bbox
