import numpy as np
from models.model import Model
import utils

class CarpentrasModel(Model):

    def __init__(self, data, params=None, random_params=False, difference='wasserstein'):
        super().__init__(data, params=params, random_params=random_params, difference=difference)
    
    def run_model(self, x):
        """
        Args:
            x: Array of initial opinion values.
            mobility_max: Maximum mobility value.
            mobility_min: Minimum mobility value.
            flip_prob: Probability of flipping an opinion.
            shift_amount: Amount to shift opinion towards another agent.
            iterations: Number of iterations to run the model for.
            
        Returns:
            Updated opinion distribution after iterations.
        """
        p = self.params
        shift_amount = 6*p['shift_amount'] / 100
        flip_prob = 8*p['flip_prob'] / 100
        mob_min = 10 * p['mobility_min'] / 100
        mob_max = 48 * p['mobility_max'] / 100

        N = len(x)
        # print(N)
        x_new = np.copy(x)  # Create a copy to avoid modifying x while iterating
        
        int_iterations = int(p['iterations'] * 20000)

        random_pairs = utils.generate_multiple_random_pairs(N, int_iterations)
        standard_noises = np.random.normal(0, 1, int_iterations)
        flip_draws = np.random.rand(int_iterations)

        if mob_max < mob_min:
            mob_max = mob_min

        mobility_range = mob_max - mob_min

        for idx in range(int_iterations):
            # Select a random pair of agents
            i, j = random_pairs[idx]
            while j == i:
                j = np.random.randint(0, N)
            
            # 1. Agent i shifts their opinion with normally distributed random noise based on their certainty
            noise_sd = mob_max - mobility_range * np.abs(x_new[i])
            x_new[i] += standard_noises[idx] * noise_sd
            
            # Keep opinion within range
            x_new[i] = np.clip(x_new[i], -1, 1)

            # 2. Agent i flips their opinion with a 4% chance (unless the sign already changed? does this matter?)
            if flip_draws[idx] < flip_prob:
                x_new[i] = -x_new[i]

            # 3. Agent i shifts their opinion by 0.03 in the direction of agent j's opinion
            x_new[i] += shift_amount * np.sign(x_new[j] - x_new[i])

            # Keep opinion within range
            x_new[i] = np.clip(x_new[i], -1, 1)

        return x_new
    
    def get_default_params(self):
        return {
            'mobility_max': 0.24,
            'mobility_min': 0.5,
            'flip_prob': 0.5,
            'shift_amount': 0.5,
            'iterations': 0.5
        }