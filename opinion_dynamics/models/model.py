import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.optimize import differential_evolution

class Model:
    def __init__(self, x, y, params=None):
        self.x = x
        self.y = y
        self.params = params or {}  # Parameters specific to the model

    def run_model(self, x):
        """
        Runs the opinion dynamics model. This should be overridden by each model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def calculate_score(self, x_new, y):
        """
        Compare the histograms of simulated and real data.
        """
        real_hist, _ = np.histogram(x_new, bins=10, range=(0, 1), density=True)
        sim_hist, _ = np.histogram(y, bins=10, range=(0, 1), density=True)
        stat = np.sum((real_hist - sim_hist) ** 2)
        return stat

    def train(self):
        """
        Train model by running through each dataset in x and comparing to y.
        """
        total_score = 0
        for i in range(len(self.x)):
            x_new = self.run_model(self.x[i])  # Override method
            score = self.calculate_score(x_new, self.y[i])
            total_score += score
        return total_score

    def set_params(self, param_values):
        """
        Set the model-specific parameters.
        """
        for param_name, param_value in zip(self.params.keys(), param_values):
            self.params[param_name] = param_value

    def get_bounds(self):
        """
        Get the parameter bounds for the model.
        This method should return a list of bounds for each parameter in self.params.
        """
        raise NotImplementedError("Subclasses should implement this method.")
