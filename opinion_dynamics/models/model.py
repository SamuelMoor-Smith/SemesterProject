import numpy as np
import matplotlib.pyplot as plt
import models.utils.differences as diff
import models.utils.parallelization as parallelization
from hyperopt import STATUS_OK  # Import Hyperopt
import plotting.basic as my_plotting
import time
import random
import copy

class Model:
    def __init__(self, data, params=None, random_params=False, difference='wasserstein'):
        self.id = random.randint(0, 1000000)
        self.data = data
        self.init = data[0]
        self.generator_params = None
        self.params = params or self.get_default_params()
        self.plot = False
        self.difference = difference
        if random_params:
            self.params = self.generate_random_params()

    def get_default_params(self):
        """
        Get the default parameters for the model.
        This method should return a dictionary of default parameters.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def generate_random_params(self):
        """
        Generate random parameters for the model.
        """
        return {param: np.random.uniform(0, 1) for param in self.get_default_params().keys()}

    def run_model(self, x):
        """
        Runs the opinion dynamics model. This should be overridden by each model.
        x is the initial opionions
        """
        raise NotImplementedError("Subclasses should implement this method.") 

    def create_data_from_initial(self, n=10):
        """
        Run the model for n steps from the initial data. And return each step.
        """
        self.generator_params = copy.deepcopy(self.params)
        x_new = self.init
        for _ in range(1, n):
            x_new = self.run_model(x_new)
            self.data.append(x_new)
    
    def run_n_steps_from_initial_with_score(self, n=10):
        """
        Run the model for n steps from the initial data. And return each step and the score.
        """
        run_snapshots = [self.init]
        scores = [0]

        x_new = self.init
        for i in range(1, n):
            if i >= len(self.data):
                break
            x_new = self.run_model(x_new)
            run_snapshots.append(x_new)
            scores.append(diff.calculate_score(x_new, self.data[i], method=self.difference))
        
        if self.plot:
            self.plot_snapshots(run_snapshots, scores)
            self.plot = False

        return run_snapshots, scores
    
    def total_score_for_n_steps_one_run(self, n=10):
        """
        Calculate the total score for n steps from the initial data.
        """
        _, scores = self.run_n_steps_from_initial_with_score(n=n)
        return sum(scores)
    
    def avg_score_and_std_for_n_steps_multiple_runs(self, n=10, num_runs=10, parallel='sequential', plot_first=False):
        """
        Calculate the average score and std dev for n steps from the initial data over num_runs.
        """
        start_time = time.time()
        if plot_first:
            self.plot = True
        results = parallelization.get_total_score(parallel, self.total_score_for_n_steps_one_run, n, num_runs)
        avg_score = np.mean(results)
        std_dev = np.std(results)
        duration = time.time() - start_time
        # print(f"Execution time: {duration:.4f} seconds")
        return avg_score, std_dev

    def set_params(self, params):
        """
        Set the model-specific parameters.
        """
        for param_name, param_value in params.items():
            self.params[param_name] = float(param_value)

    def get_bounds(sel, num_params):
        """
        Get the parameter bounds for the model.
        This method should return a list of bounds for each parameter in self.params.
        """
        # raise NotImplementedError("Subclasses should implement this method.")
        bounds = []
        for _ in range(num_params):
            bounds.append((0.0, 1.0))
        return bounds
    
    def hyperopt_objective(self, params):
        """The objective function to minimize with Hyperopt."""
        self.set_params(params)
        avg_score, _ = self.avg_score_and_std_for_n_steps_multiple_runs(n=5, num_runs=7)
        return {'loss': avg_score, 'status': STATUS_OK}
    
    def plot_snapshots(self, run_snapshots, scores):
        my_plotting.plot_true_vs_simulated_snapshots(
            self.data, 
            run_snapshots, 
            model_name=self.__class__.__name__, 
            difference=self.difference, 
            save_plot=True, 
            save_path=f"carpentras_plots/num2/partial_optimizer_tests/{self.id}", 
            scores=scores, 
            params=self.params,
            generator_params=self.generator_params
        )
    
    
