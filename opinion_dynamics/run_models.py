from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import numpy as np
from ess.ess_file import ESSFile
from models.deffuant import DeffuantModel
from models.hk_averaging import HKAveragingModel
from models.carpentras import CarpentrasModel
import utils
from tqdm import tqdm  # Progress bar for tracking progress
import time
import optimizers
import os

if __name__ == '__main__':

    # Run this simulation a bunch of times
    for i in range(10):

        # generate random initial opinions
        initial_opinions = utils.create_random_opinion_distribution(N=1000, min_val=0, max_val=1, seed=i)

        # Create a model with random parameters
        model = DeffuantModel([initial_opinions], difference='wasserstein', random_params=True)
        # model = HKAveragingModel([initial_opinions], difference='wasserstein', random_params=True, method="arithmetic")
        
        # Print out the parameters
        print(f"Initial parameters: {model.params}")

        # Run the model for 10 steps
        model.create_data_from_initial(n=10)

        # Run the same model for 10 steps 10 times with the same parameters
        # And retrieve the baseline score
        avg_score_base, std_dev_base = model.avg_score_and_std_for_n_steps_multiple_runs(n=10, num_runs=10, plot_first=True)
        print(f"Baseline score: {avg_score_base} +/- {std_dev_base}")

        # Now run the optimizer to find the best parameters
        bounds = model.get_bounds(num_params=3) # 3 for Deffuant, 2 for HK
        optimizer = optimizers.get_optimizer('hyperopt')

        model.set_params(model.generate_random_params())
        best_params = optimizer(model, bounds)

        print(f"Best parameters: {best_params}")

        # Set the best parameters
        model.set_params(best_params)

        # Run the model for 10 steps
        avg_score_opt, std_dev_opt = model.avg_score_and_std_for_n_steps_multiple_runs(n=10, num_runs=10, plot_first=True)
        print(f"Optimizer score: {avg_score_opt} +/- {std_dev_opt}")
        
        # save the following data to a file under full_optimizer_tests/{model.id}

        # If optimizer score is better, say so
        if avg_score_opt < avg_score_base:
            print("Optimizer improved the score!")

        else:
            # If optimizer score is worse, check if it's within 1 std dev of baseline
            if avg_score_opt < avg_score_base + std_dev_base:
                print("No significant difference detected.")
            else:
                print("Significant difference detected!")

        # Save the print statement to a file
        with open(f'deffuant_plots/num2/partial_optimizer_tests/{model.id}/results.txt', 'a') as f:
            f.write(f"Run {i}\n")
            f.write(f"Initial parameters: {model.generator_params}\n")
            f.write(f"Baseline score: {avg_score_base} +/- {std_dev_base}\n")
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Optimizer score: {avg_score_opt} +/- {std_dev_opt}\n")
            if avg_score_opt < avg_score_base:
                f.write("Optimizer improved the score!\n")
            else:
                if avg_score_opt < avg_score_base + std_dev_base:
                    f.write("No significant difference detected.\n")
                else:
                    f.write("Significant difference detected!\n")
            f.write("\n")
