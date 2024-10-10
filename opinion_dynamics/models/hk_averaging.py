import numpy as np
from models.model import Model
from utils import calculate_mean

class HKAveragingModel(Model):
    def __init__(self, x, y, method="arithmetic"):
        self.method = method
        params = {
            'epsilon': 0.2,
            'iterations': 5
        }
        super().__init__(x, y, params)

    def run_model(self, x):
        """
        Args:
            x: Opinion inputs
            epsilon: Confidence threshold (how close must interactors be to converge).
            iterations: How many iterations

        Returns:
            Updated opinion distribution from running x on the HK averaging model.
        """
        N = len(x)
        x_new = np.copy(x)  # Create a copy to avoid modifying x while iterating

        for _ in range(int(self.params["iterations"])):
            for i in range(N):
                # Find all agents whose opinions are within epsilon of x[i]
                close_opinions = x[np.abs(x - x[i]) <= self.params["epsilon"]]

                # If there are any opinions within epsilon, update x[i] to their average
                if len(close_opinions) > 0:
                    x_new[i] = calculate_mean(close_opinions, method=self.method)
            
            x = x_new
        
        return x

    def get_bounds(self):
        return [(0, 1),            # epsilon bounds
                (1, 10)]       # iterations bounds