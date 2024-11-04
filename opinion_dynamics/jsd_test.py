import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

# Function to generate random probability distributions
def generate_random_distribution(n, size=100):
    dist = np.random.dirichlet(np.ones(size), size=n)
    return dist

# Function to compute JSD between pairs of distributions
def compute_jsd(d1, d2):
    return jensenshannon(d1, d2)**2  # JSD is the square of the Jensen-Shannon distance

# Generate random distributions
n_distributions = 12
distribution_size = 100
distributions = generate_random_distribution(n_distributions, distribution_size)

# Compute pairwise JSD
jsd_values = np.zeros((n_distributions, n_distributions))
for i in range(n_distributions):
    for j in range(n_distributions):
        if i != j:
            jsd_values[i, j] = compute_jsd(distributions[i], distributions[j])

# Plot histograms for each distribution and show JSD values
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Histograms of Distributions with JSD Scores', fontsize=16)

for i, ax in enumerate(axes.flatten()):
    # Plot histogram
    ax.bar(range(distribution_size), distributions[i], color='blue', alpha=0.7)
    ax.set_title(f"Distribution {i + 1}", fontsize=10)
    ax.set_xlim(-0.5, distribution_size - 0.5)
    
    # Display JSD scores for this distribution against all others
    jsd_str = ", ".join([f"JSD(D{i+1}, D{j+1}): {jsd_values[i,j]:.3f}" for j in range(n_distributions) if i != j])
    ax.text(0.5, -0.1, jsd_str, ha='center', va='top', transform=ax.transAxes, fontsize=8, wrap=True)

# Adjust layout to fit JSD text properly
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
