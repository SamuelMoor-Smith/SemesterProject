import numpy as np
from models.model import Model

class CarpentrasModel(Model):
    def __init__(self, x, y):

        # All have been doubled from paper because we have scale 1 - 10 instead of -10 - 10 -- not true rn
        params = {
            'mobility_max': 0.24,   # Max mobility for agents with certainty 0.1
            # 'mobility_min': 0.05,   # Min mobility for agents with certainty 1.0
            # 'flip_prob': 0.04,      # Probability of flipping the opinion
            # 'shift_amount': 0.03,   # Amount of opinion shift towards another agent
            'iterations': 10000      # Number of iterations
        }

        self.fixed_params = {
            'flip_prob': 0.04,
            'shift_amount': 0.03,
            'mobility_min': 0.05
        }

        super().__init__(x, y, params)
    
    def run_model(self, x):
        """
        Args:
            opinions: Array of initial opinion values.
            
        Returns:
            Updated opinion distribution after iterations.
        """
        N = len(x)
        x_new = np.copy(x)  # Create a copy to avoid modifying x while iterating

        for _ in range(int(self.params["iterations"])):
            # Select a random pair of agents
            i, j = np.random.choice(N, 2, replace=False)
            
            # 1. Agent i shifts their opinion with normally distributed random noise based on their certainty
            noise_sd = self.params["mobility_max"] - (self.params["mobility_max"] - self.fixed_params["mobility_min"]) * np.abs(x_new[i])
            x_new[i] += np.random.normal(0, noise_sd)
            
            # Ensure the opinion remains in the valid range [-1, 1]
            x_new[i] = self.keep_in_range(x_new[i])

            # 2. Agent i flips their opinion with a 4% chance (unless the sign already changed? does this matter?)
            if np.random.rand() < self.fixed_params["flip_prob"]:
                x_new[i] = -x_new[i]

            # 3. Agent i shifts their opinion by 0.03 in the direction of agent j's opinion
            x_new[i] += self.fixed_params["shift_amount"] * np.sign(x_new[j] - x_new[i])

            # Ensure the opinion remains in the valid range [-1, 1]
            x_new[i] = self.keep_in_range(x_new[i])

        return x_new
    
    def keep_in_range(self, x):
        if x > 1:
            return 1
        elif x < -1:
            return -1
        return x

    def get_bounds(self):
        return [
            (0, 1),
            # (0, 1),
            # (0, 1),
            # (0, 1),
            (0, 100000)
        ]