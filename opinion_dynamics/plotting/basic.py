import matplotlib.pyplot as plt
import os
import time
import numpy as np

def plot_one_true_vs_simulated_snapshot(true_data, sim_data, score):
    """plots the true vs simulated snapshot side by side"""

    plt.figure(figsize=(15, 6))  # Adjusted figure size for compact layout

    plt.subplot(1, 2, 1)
    plt.hist(sim_data, bins=100, range=(-1, 1), edgecolor='black', alpha=0.7)
    plt.title(f'Simulated Snapshot\nJSD Score: {score:.3f}', fontsize=8)
    plt.xlabel('Opinion Value', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.subplot(1, 2, 2)
    plt.hist(true_data, bins=100, range=(-1, 1), edgecolor='black', alpha=0.7)
    plt.title(f'True Snapshot', fontsize=8)  # Reduced title font size
    plt.xlabel('Opinion Value', fontsize=8)  # Reduced label font size
    plt.ylabel('Frequency', fontsize=8)  # Reduced label font size
    plt.xticks(fontsize=7)  # Reduced tick label font size
    plt.yticks(fontsize=7)

    plt.tight_layout(pad=1.5)  # Adjusted padding for better fit
    plt.show()


def plot_true_vs_simulated_snapshots(true_data, sim_data, 
                                     model_name, 
                                     params,
                                     generator_params,
                                     scores,
                                     bins=100,
                                     plot_range=(0, 1),
                                     difference="wasserstein", 
                                     save_plot=False, 
                                     save_path="plots"):
    """plots the true vs simulated snapshots side by side"""

    num_snapshots = len(true_data)  # Number of snapshots (should be 10 in most cases)
    # print(f"Plotting {num_snapshots} snapshots...")

    cols = 4
    rows = (num_snapshots*2) // cols  # Adjusted rows calculation for 2 plots per snapshot

    plt.figure(figsize=(15, 15))  # Adjusted figure size for compact layout
    # write params in subtitle
    plt.suptitle(
        f"""
            Model: {model_name}
            Sim Params: {params}
            True Params: {generator_params}
            Total score sum:{sum(scores):.3f} and Std Dev: {np.std(scores):.3f}
        """
    )

    for i in range(num_snapshots):
        # Plot true data
        plt.subplot(rows, cols, i * 2 + 1)
        plt.hist(true_data[i], bins=bins, range=plot_range, edgecolor='black', alpha=0.7)
        plt.title(f'True Snapshot {i+1}')
        plt.xlabel('Opinion Value')
        plt.ylabel('Frequency')
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        # Plot simulated data (model run results)
        plt.subplot(rows, cols, i * 2 + 2)
        plt.hist(sim_data[i], bins=bins, range=plot_range, edgecolor='black', alpha=0.7)
        plt.title(f'Simulated Snapshot {i+1}\n{difference} Score: {scores[i]:.3f}')
        plt.xlabel('Opinion Value')
        plt.ylabel('Frequency')
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.tight_layout(pad=1.5)  # Adjusted padding for better fit

    if save_plot:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"true_vs_simulated_snapshots_{timestamp}.png"
        plt.savefig(os.path.join(save_path, filename))
        # print(f"Plot saved at {os.path.join(save_path, filename)}")
        plt.close()  # Close the plot to free up memory
    else:
        plt.show()

# Plot all the differences the average score for each epsilon value
def plot_average_score_and_std_dev(param_name, param_values, avg_scores, std_devs, difference, plot_dir):
    plt.figure(figsize=(8, 6))
    plt.title(f"Average Score for {difference} difference")
    plt.errorbar(param_values, avg_scores, yerr=std_devs, fmt='-o', label=f'{difference} difference')
    plt.xlabel(param_name)
    plt.ylabel("Average Score")
    # plt.show()

    # Generate a unique filename for each plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{difference}_{param_name}_{timestamp}.png"
    filepath = os.path.join(plot_dir, filename)
    
    # Save the plot to the file
    plt.savefig(filepath)
    print(f"Plot saved as {filepath}")
    
    plt.close()