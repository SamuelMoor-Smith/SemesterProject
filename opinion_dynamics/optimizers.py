from tqdm import tqdm  # Progress bar for tracking progress
from my_logging import logging_callback, TQDMProgress, bayesian_callback
from skopt import gp_minimize  # Import Bayesian Optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand  # Import Hyperopt
import time
import logging

# Set Hyperopt logger to display only errors
logger = logging.getLogger("hyperopt.tpe")
logger.setLevel(logging.ERROR)

# Define your optimizers in a dictionary
optimizers = {
    'hyperopt': lambda model, bounds: fmin(
        fn=model.hyperopt_objective,  # Objective function
        space={param: hp.uniform(param, bounds[i][0], bounds[i][1]) for i, param in enumerate(model.params.keys())},  # Hyperopt parameter space
        algo=tpe.suggest,  # Use TPE algorithm
        max_evals=300,  # Number of evaluations
        trials=Trials(),  # To store results
        show_progressbar=False
        # rstate=np.random.default_rng(42)  # Ensure reproducibility
    ),
    'hyperopt_with_params': lambda model, params, bounds: fmin(
        fn=model.hyperopt_objective,  # Objective function
        space={param: hp.uniform(param, bounds[i][0], bounds[i][1]) for i, param in enumerate(params)},  # Hyperopt parameter space
        algo=tpe.suggest,  # Use TPE algorithm
        max_evals=100,  # Number of evaluations
        trials=Trials(),  # To store results
        show_progressbar=False
        # rstate=np.random.default_rng(42)  # Ensure reproducibility
    ),
    # 'bayesian_optimization': lambda model, bounds: gp_minimize(
    #     model.objective_function, dimensions=bounds, n_calls=100, random_state=42,
    #     callback=[bayesian_callback]),  # Bayesian optimization
    # 'differential_evolution': lambda model, bounds: differential_evolution(
    #     model.objective_function, bounds, popsize=5, tol=0.01, maxiter=100, callback=TQDMProgress(total=100)),  # Progress bar for DE # too slow
    # 'nelder_mead': lambda model, bounds: minimize(
    #     model.objective_function, x0=np.mean(bounds, axis=1), method='Nelder-Mead', 
    #     callback=logging_callback, options={'maxiter': 100}),
    # 'powell': lambda model, bounds: minimize(
    #     model.objective_function, x0=np.mean(bounds, axis=1), method='Powell', 
    #     callback=logging_callback, options={'maxiter': 100}),
}

def get_optimizer(optimizer_name):
    """
    Get the optimizer function based on the name.
    """
    return optimizers[optimizer_name]