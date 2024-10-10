from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from ess.ess_file import ESSFile
from models.deffuant import DeffuantModel
from models.hk_averaging import HKAveragingModel
from models.carpentras import CarpentrasModel

# Load data

# lrscale - Placement on left right scale
# imwbcnt - Immigrants make country worse or better place to live
# imbgeco - Immigration bad or good for country's economy
# trstun - Trust in the United Nations
# trstplc - Trust in the police
# pplfair - Most people try to take advantage of you, or try to be fair

_ESSFILE = ESSFile('opinion_dynamics/ess/combined-sept26.csv', 'imbgeco')

x, y = _ESSFILE.get_xy(scale=5, adjust=1)
_ESSFILE.plot_y_data(y)

# Define your models
models = [
    # DeffuantModel(x, y),
    # HKAveragingModel(x, y, "arithmetic"),
    # HKAveragingModel(x, y, "geometric"),
    # HKAveragingModel(x, y, "harmonic"),
    CarpentrasModel(x, y),
    CarpentrasModel(x, y),
    CarpentrasModel(x, y),
    CarpentrasModel(x, y),
    CarpentrasModel(x, y)
]

def objective_function(param_values, model):
    """
    Objective function that gets optimized for a specific model.
    """
    model.set_params(param_values)
    return model.train()

# Optimize each model with differential evolution
for model in models:
    print(f"Running optimization for {model.__class__.__name__}...")
    
    # Get the bounds specific to the model
    bounds = model.get_bounds()
    
    # Run the optimizer
    # result = differential_evolution(objective_function, bounds, args=(model,))
    result = differential_evolution(objective_function, bounds, args=(model,), popsize=5, maxiter=100, tol=0.01)

    best_params = result.x
    
    print(f"Best parameters for {model.__class__.__name__}: {best_params}")
    
    # Run the model with the best parameters and plot the results
    for i in range(len(x)):
        x_new = model.run_model(x[i])
        
        # Plot comparison
        plt.hist(x_new, bins=100, alpha=0.5, label='Simulated')
        plt.hist(y[i], bins=100, alpha=0.5, label='Real')
        plt.xlabel("Opinion")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()