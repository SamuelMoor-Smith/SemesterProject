import numpy as np
from models.model import Model
import utils
from hyperopt import STATUS_OK  # Import Hyperopt

class HKAveragingModel(Model):
    def __init__(self, data, params=None, random_params=False, difference='wasserstein', method="arithmetic"):
        self.method = method
        super().__init__(data, params=params, random_params=random_params, difference=difference)

    def run_model(self, x):
        """
        Args:
            x: Array of initial opinion values.
            epsilon: Confidence threshold (how close must interactors be to converge).
            iterations: How many iterations

        Returns:
            Updated opinion distribution from running x on the HK averaging model.
        """
        p = self.params

        N = len(x)
        x_new = np.copy(x)  # Create a copy to avoid modifying x while iterating

        num_agents = int(p['agents'] * N)
        agents = utils.generate_multiple_random_nums(N, num_agents)

        # for _ in range(int_iterations):
            # Compute pairwise differences only once
        diffs = np.abs(x[:, None] - x[None, :])

        # Create a mask for elements within epsilon
        mask = diffs <= p['epsilon']

        # Update opinions with the mean of opinions within epsilon
        for agent in agents:
            close_opinions = x[mask[agent]]
            if close_opinions.size > 0:
                x_new[agent] = utils.calculate_mean(close_opinions, method=self.method)
        
        # # Check for convergence and exit if stabilized
        # if np.allclose(x, x_new, atol=1e-6):
        #     break
        
        # # Update x for the next iteration
        # x = np.copy(x_new)
        
        return x_new
    
    def get_default_params(self):
        return {
            'epsilon': 0.2,
            'agents': 0.5
        }
