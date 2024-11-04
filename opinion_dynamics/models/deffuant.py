import numpy as np
from models.model import Model
import utils

class DeffuantModel(Model):

    def __init__(self, data, params=None, random_params=False, difference='wasserstein'):
        super().__init__(data, params=params, random_params=random_params, difference=difference)

    def run_model(self, x):
        """
        Args:
            x: Array of initial opinion values.
            mu: Convergence parameter (how much interactors converge together).
            epsilon: Confidence threshold (how close must interactors be to converge).
            iterations: How many iterations (scaled from 0 to 10000 but on 0-1 scale). 

        Returns:
            Updated opinion distribution from running x on the deffuant model.
        """
        p = self.params
        mu = p['mu'] * 0.5
        epsilon = p['epsilon'] * 0.7

        # np.random.seed(42)
        N = len(x)
        x_new = np.copy(x)  # Create a copy to avoid modifying x while iterating

        int_iterations = int(p['iterations'] * 1000) + 100

        random_pairs = utils.generate_multiple_random_pairs(N, int_iterations)

        # print(self.params["mu"])
        for idx in range(int_iterations):
            i, j = random_pairs[idx]
            while j == i:  # Ensure i and j are different
                j = np.random.randint(0, N)

            opinion_difference = abs(x_new[i] - x_new[j])
            if opinion_difference <= epsilon:
                update_to_i = mu * (x_new[j] - x_new[i])
                update_to_j = mu * (x_new[i] - x_new[j])
                # print (update_to_i, update_to_j)
                x_new[i] += update_to_i
                x_new[j] += update_to_j

        return x_new
    
    def get_default_params(self):
        return {
            'mu': 0.5,
            'epsilon': 0.2,
            'iterations': 0.5
        }