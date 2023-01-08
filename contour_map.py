import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as patches

# Load the matrix
matrix = np.load("data/results/grid.npy")
coords = json.load(open("data/results/coords.json", "r"))
bboxs = json.load(open("data/results/bboxs.json", "r"))
best_coord = json.load(open("data/results/best_result.json", "r"))["hyperparameters"]

# Generate a grid of hyperparameters
learning_step = 0.00001
dropout_step = 0.1

# Generate grid with 10 steps for learning rate and dropout
learning_rates = [0.00001 + learning_step * i for i in range(10)]
dropouts = [0 + dropout_step * i for i in range(10)]

plt.gcf().subplots_adjust(bottom=0.25)
# Create contour map
contour_map = plt.contourf(learning_rates, dropouts, matrix, levels=10)

# Display the contour map
plt.colorbar(contour_map)
plt.xlabel("Learning rate")
plt.ylabel("Dropout")


# Rotate ticks
plt.xticks(rotation=45)
plt.title("Accuracy for Hyperparamter Values")
plt.savefig("data/figures/hyperparameter_contour_map.png")

# Plots the coords as dots to show where KDSearch looked
for coord in coords:
    plt.rcParams['lines.markersize'] = 2
    plt.plot(coord["learning_rate"], coord["dropout"], "ro")

plt.plot(best_coord["learning_rate"], best_coord["dropout"], "bo")
plt.savefig("data/figures/hyperparameter_contour_map_with_coords.png")

# Plot the bboxs
for bbox in bboxs:
    # Get the corners of the bbox
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    
    # Plot a rectangle
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    
    # Add the patch to the Axes
    plt.gca().add_patch(rect)
    
plt.savefig("data/figures/hyperparameter_contour_map_with_bboxs.png")