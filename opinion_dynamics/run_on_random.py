import numpy as np
import matplotlib.pyplot as plt
from models.carpentras import CarpentrasModel
from models.deffuant import DeffuantModel
import utils

# Generate random initial opinions (between -1 and 1) for 1000 agents
initial_opinions = utils.create_random_opinion_distribution(N=1000, min_val=-1, max_val=1, seed=42)
snapshots = utils.run_model_and_collect_snapshots(DeffuantModel, initial_opinions, num_snapshots=10)
x,y = utils.get_xy_from_snapshots(snapshots)

# Plot all snapshots
num_snapshots = len(snapshots)
cols = 5  # Number of columns in the subplot grid
rows = (num_snapshots + cols - 1) // cols  # Number of rows

plt.figure(figsize=(15, 5 * rows))

for i, snapshot in enumerate(snapshots):
    plt.subplot(rows, cols, i + 1)
    plt.hist(snapshot, bins=100, range=(-1, 1), edgecolor='black', alpha=0.7)
    plt.title(f'Snapshot {i}')
    plt.xlabel('Opinion Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
# # Plot the results: initial vs updated opinions
# plt.figure(figsize=(10, 5))

# # Plot initial opinions
# plt.subplot(1, 2, 1)
# plt.hist(initial_opinions, bins=100, range=(-1, 1), edgecolor='black', alpha=0.7)
# plt.title('Initial Opinions')
# plt.xlabel('Opinion Value')
# plt.ylabel('Frequency')

# # Plot updated opinions
# plt.subplot(1, 2, 2)
# plt.hist(updated_opinions, bins=100, range=(-1, 1), edgecolor='black', alpha=0.7)
# plt.title('Updated Opinions')
# plt.xlabel('Opinion Value')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()