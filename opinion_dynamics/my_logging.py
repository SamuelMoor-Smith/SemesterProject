from tqdm import tqdm  # Progress bar for tracking progress
import logging  # For logging the progress and parameter updates

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Create a logging callback for minimize optimizers
def logging_callback(xk):
    logging.info(f"Current parameter set: {xk}")

# Create a progress bar callback for differential evolution using tqdm
class TQDMProgress:
    def __init__(self, total):
        self.pbar = tqdm(total=total)

    def __call__(self, xk, convergence):
        self.pbar.update(1)
        logging.info(f"Convergence: {convergence}, Parameters: {xk}")

# Define the callback function
def bayesian_callback(res):
    # This gets called after every iteration of gp_minimize
    # You can add custom logging here
    print(f"Iteration {len(res.x_iters)}: Best Score so far: {res.fun}")
    print(f"Current Parameters: {res.x}")