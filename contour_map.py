import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as patches
np.random.seed(0)


# Load the matrix
matrix = np.load("data/results/grid.npy")
coords = json.load(open("data/results/coords.json", "r"))
bboxs = json.load(open("data/results/bboxs.json", "r"))
best_coord = json.load(open("data/results/best_result.json", "r"))

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
plt.ylabel("Weight Decay")


# Rotate ticks
plt.xticks(rotation=45)
plt.title("Accuracy for Hyperparameter Values")
plt.savefig("data/figures/hyperparameter_contour_map.png")

# Plots the coords as dots to show where KDSearch looked
for coord in coords:
    plt.rcParams['lines.markersize'] = 2
    plt.plot(coord["hyperparameters"]["learning_rate"], coord["hyperparameters"]["weight_decay"], "ro")
    # Add the score next to it
    plt.text(coord["hyperparameters"]["learning_rate"], coord["hyperparameters"]["weight_decay"], round(coord["score"], 2), fontsize=8)

plt.plot(best_coord["hyperparameters"]["learning_rate"], best_coord["hyperparameters"]["weight_decay"], "bo")
plt.text(best_coord["hyperparameters"]["learning_rate"], best_coord["hyperparameters"]["weight_decay"], round(coord["score"], 2), fontsize=8)
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