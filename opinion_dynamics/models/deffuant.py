import numpy as np
from models.model import Model

class DeffuantModel(Model):
    def __init__(self, x, y):
        params = {
            'mu': 0.5,
            'epsilon': 0.2,
            'iterations': 1000000
        }
        super().__init__(x, y, params)

    def run_model(self, x):
        """
        Args:
            x: Opinion inputs
            mu: Convergence parameter (how much interactors converge together).
            epsilon: Confidence threshold (how close must interactors be to converge).
            iterations: How many iterations

        Returns:
            Updated opinion distribution from running x on the deffuant model.
        """
        N = len(x)
        x_new = np.copy(x)  # Create a copy to avoid modifying x while iterating

        for _ in range(int(self.params["iterations"])):
            i, j = np.random.choice(N, 2, replace=False)
            opinion_difference = abs(x_new[i] - x_new[j])
            if opinion_difference <= self.params["epsilon"]:
                update_to_i = self.params["mu"] * (x_new[j] - x_new[i])
                update_to_j = self.params["mu"] * (x_new[i] - x_new[j])
                # print (update_to_i, update_to_j)
                x_new[i] += update_to_i
                x_new[j] += update_to_j

        return x_new

    def get_bounds(self):
        return [(0, 1),    # mu bounds
                (0, 1),    # epsilon bounds
                (500, 1000)]  # iterations bounds