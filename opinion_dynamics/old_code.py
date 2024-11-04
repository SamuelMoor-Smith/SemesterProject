# def get_xy_from_snapshots(snapshots):
#     """
#     Create X and Y data from the opinion snapshots.
#     X is the data at time t, Y is the data at time t+1.
    
#     Args:
#         snapshots: List or array of opinion snapshots at different timesteps.
        
#     Returns:
#         X: List of arrays where each array is the opinions at a time step (input).
#         Y: List of arrays where each array is the opinions at the next time step (target).
#     """
#     X = []
#     Y = []
    
#     # Loop through snapshots and pair each one with the next
#     for i in range(len(snapshots) - 1):
#         X.append(snapshots[i])      # Current snapshot
#         Y.append(snapshots[i + 1])  # Next snapshot
    
#     return np.array(X), np.array(Y)


# def generate_random_pair(max_val):
#     """
#     Get two random agents from the list of agents. Ensure they are different.
#     """
#     i = np.random.randint(0, max_val)
#     j = np.random.randint(0, max_val)
#     while j == i:
#         j = np.random.randint(0, max_val)
#     return i, j

# def generate_multiple_random_nums(max_val, num_pairs):
#     """
#     Generate multiple random numbers.
#     """
#     return np.random.randint(0, max_val, int(num_pairs))

# Define your models
models = [
    # DeffuantModel(x, y),
    # # HKAveragingModel(x, y, "arithmetic"),
    # # HKAveragingModel(x, y, "geometric"),
    # # HKAveragingModel(x, y, "harmonic"),
    # CarpentrasModel(x, y),
]

# Define your optimizers in a dictionary
# optimizers = {
#     'hyperopt': lambda model, bounds: fmin(
#         fn=model.hyperopt_objective,  # Objective function
#         space={param: hp.uniform(param, bounds[i][0], bounds[i][1]) for i, param in enumerate(model.params.keys())},  # Hyperopt parameter space
#         algo=tpe.suggest,  # Use TPE algorithm
#         max_evals=500,  # Number of evaluations
#         trials=Trials(),  # To store results
#         # rstate=np.random.default_rng(42)  # Ensure reproducibility
#     ),
#     # 'bayesian_optimization': lambda model, bounds: gp_minimize(
#     #     model.objective_function, dimensions=bounds, n_calls=100, random_state=42,
#     #     callback=[bayesian_callback]),  # Bayesian optimization
#     # 'differential_evolution': lambda model, bounds: differential_evolution(
#     #     model.objective_function, bounds, popsize=5, tol=0.01, maxiter=100, callback=TQDMProgress(total=100)),  # Progress bar for DE # too slow
#     # 'nelder_mead': lambda model, bounds: minimize(
#     #     model.objective_function, x0=np.mean(bounds, axis=1), method='Nelder-Mead', 
#     #     callback=logging_callback, options={'maxiter': 100}),
#     # 'powell': lambda model, bounds: minimize(
#     #     model.objective_function, x0=np.mean(bounds, axis=1), method='Powell', 
#     #     callback=logging_callback, options={'maxiter': 100}),
# }