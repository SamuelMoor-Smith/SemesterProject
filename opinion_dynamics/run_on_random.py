import numpy as np
import matplotlib.pyplot as plt
from models.carpentras import CarpentrasModel
from models.deffuant import DeffuantModel

# Generate random initial opinions (between -1 and 1) for N agents
np.random.seed(42)  # Set a seed value
N = 1000  # Number of agents
initial_opinions = np.random.uniform(-1, 1, N)  # Random distribution between -1 and 1

# Reset the seed to None to return randomness to the system clock or entropy source
np.random.seed(None)

# Initialize and run the model
model = CarpentrasModel(initial_opinions, initial_opinions)
updated_opinions = model.run_model(initial_opinions)

# Plot the results: initial vs updated opinions
plt.figure(figsize=(10, 5))

# Plot initial opinions
plt.subplot(1, 2, 1)
plt.hist(initial_opinions, bins=100, range=(-1, 1), edgecolor='black', alpha=0.7)
plt.title('Initial Opinions')
plt.xlabel('Opinion Value')
plt.ylabel('Frequency')

# Plot updated opinions
plt.subplot(1, 2, 2)
plt.hist(updated_opinions, bins=100, range=(-1, 1), edgecolor='black', alpha=0.7)
plt.title('Updated Opinions')
plt.xlabel('Opinion Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()