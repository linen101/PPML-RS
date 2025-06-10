# try some dp noise on 1d data.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import HuberRegressor, LinearRegression
import os
import sys

from numpy.linalg import eigh, inv


module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from torrent.torrent import torrent_dp  # Import torrent module
from synthetic.synthetic_one_dimensional import generate_synthetic_dataset_one_d, strategic_corruption_scaled

def gaussian_mechanism(X, y, epsilon, delta, B_y, w_torrent):
    """
    Applies the Gaussian mechanism to least squares regression:
    M(X, y) = w_torrent + Gaussian noise
    """
    d,n = X.shape
    Sigma = X @ X.T
    # Step 2: Estimate lambda_min(Sigma)
    lambda_min = np.min(eigh(Sigma)[0])  # smallest eigenvalue

    # Step 3: Compute L2 sensitivity
    Delta_f = ((2 * np.sqrt(d) * B_y) / lambda_min) + (2 * d * np.sqrt(n * d) * B_y * np.sqrt(d)) / (lambda_min ** 2)

    # Step 4: Compute noise scale
    sigma = (np.sqrt(2 * np.log(2 / delta)) / epsilon) * Delta_f

    # Step 5: Sample noise from N(0, I_d)
    noise = np.random.randn(d, 1)

    # Step 6: Return private estimator
    beta_private = w_torrent + sigma * noise

    return beta_private

def run_tests_dp_1d(num_trials):
    # Generate Dataset with Intercept
    n = 100  # Number of samples
    sigma = 0.1
    alpha = 0.2  # 10% of data points are corrupted
    beta = alpha + 0.1
    epsilon = 0.1
    max_iters = 10
    dp_epsilons = [0.5, 0.7, 0.9, 1, 1.5, 3, 5]
    dp_delta = 1e-5
    dp_B_y = 1
    # assume y values bounded in [-1, 1]
    for _ in range(num_trials):
        X_aug, Y, w_star = generate_synthetic_dataset_one_d(n, sigma)

        # Apply strategic corruption
        Y_corrupted, corrupted_indices, w_corrupt = strategic_corruption_scaled(X_aug, Y, alpha)
        
        # create figure
        plt.figure(figsize=(8, 6))

        # original data (non-corrupted)
        plt.scatter(X_aug[1, :], Y, label="Original Data", color="black", alpha=0.8)

        # highlight corrupted locations
        plt.scatter(X_aug[1, corrupted_indices], Y[corrupted_indices], label="Corruption Locations",
                    color="darkmagenta", marker="v", edgecolors="black", s=150, alpha=0.8)

        # corrupted data points
        plt.scatter(X_aug[1, corrupted_indices], Y_corrupted[corrupted_indices], label="Strategically Corrupted Points",
                    color="blueviolet", marker="x", s=100)

        # generate test values
        x_vals = np.linspace(X_aug[1, :].min(), X_aug[1, :].max(), 100).reshape(1, -1)
        x_vals_aug = np.vstack([np.ones((1, x_vals.shape[1])), x_vals])  # Add intercept term

        # honest Model (True)
        y_vals_true = x_vals_aug.T @ w_star
        #dp_B_y = max((abs(y_vals_true)))
        plt.plot(x_vals.T, y_vals_true, label=f'True Model $(5x + 4)$', color="blue", linestyle="solid")
        
        # adversarial Model (-2x -10)
        y_vals_adv = x_vals_aug.T @ w_corrupt
        plt.plot(x_vals.T, y_vals_adv, label=f'Adversarial Model $(-2x - 15)$', color="mediumpurple", linestyle="solid")

        colors = cm.viridis(np.linspace(0, 1, len(dp_epsilons)))  
        line_styles = ['dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (1, 1))]  # as many as epsilons
        for i, dp_epsilon in enumerate(dp_epsilons):
            w_torrent, iter_count = torrent_dp(X_aug, Y_corrupted, beta, epsilon, dp_epsilon, dp_delta, max_iters)
            y_vals_torrent = x_vals_aug.T @ w_torrent
            plt.plot(x_vals.T, y_vals_torrent, label=f'Torrent ($\\epsilon = {dp_epsilon}$)', 
                    color=colors[i], 
                    linestyle=line_styles[i % len(line_styles)], 
                    linewidth=1.5)

        # Labels and legend
        plt.xlabel("Feature X")
        plt.ylabel("Label Y")
        plt.legend()
        plt.title(f"Strategic Corruption in 1D $(\\beta = {alpha})$")
        plt.grid(False)

        # Show plot
        plt.show()
        plt.close()


def run_tests_dp_1d_avg(num_trials):
    # Generate Dataset with Intercept
    n = 10000  # Number of samples
    sigma = 0.1
    alpha = 0.3  # 10% of data points are corrupted
    beta = alpha + 0.05
    epsilon = 0.1
    max_iters = 5
    dp_epsilons = [0.5, 0.7, 0.9, 1, 1.5, 3, 5]
    dp_delta = 1e-5
    dp_B_y = 1
    # assume y values bounded in [-1, 1]
    
    predictions = np.zeros((len(dp_epsilons), 100))  # shape: (7, 100)

    for _ in range(num_trials):
        X_aug, Y, w_star = generate_synthetic_dataset_one_d(n, sigma)

        # Apply strategic corruption
        Y_corrupted, corrupted_indices, w_corrupt = strategic_corruption_scaled(X_aug, Y, alpha)
        

        # generate test values
        x_vals = np.linspace(X_aug[1, :].min(), X_aug[1, :].max(), 100).reshape(1, -1)
        x_vals_aug = np.vstack([np.ones((1, x_vals.shape[1])), x_vals])  # Add intercept term

        # honest Model (True)
        y_vals_true = x_vals_aug.T @ w_star
        #dp_B_y = max((abs(y_vals_true)))
        
        for i, dp_epsilon in enumerate(dp_epsilons):
            w_torrent, iter_count = torrent_dp(X_aug, Y_corrupted, beta, epsilon, dp_epsilon, dp_delta, max_iters)
            y_vals_torrent = x_vals_aug.T @ w_torrent
            predictions[i] +=  y_vals_torrent.ravel()
            
    #get average
    predictions /= num_trials
    
    # create figure
    plt.figure(figsize=(8, 6))

    # original data (non-corrupted)
    plt.scatter(X_aug[1, :], Y, label="Original Data", color="black", alpha=0.8)

    # highlight corrupted locations
    plt.scatter(X_aug[1, corrupted_indices], Y[corrupted_indices], label="Corruption Locations",
                color="darkmagenta", marker="v", edgecolors="black", s=150, alpha=0.8)

    # corrupted data points
    plt.scatter(X_aug[1, corrupted_indices], Y_corrupted[corrupted_indices], label="Strategically Corrupted Points",
                color="blueviolet", marker="x", s=100)
    
    plt.plot(x_vals.T, y_vals_true, label=f'True Model $(5x + 4)$', color="blue", linestyle="solid")
    
    # adversarial Model (-2x -10)
    y_vals_adv = x_vals_aug.T @ w_corrupt
    plt.plot(x_vals.T, y_vals_adv, label=f'Adversarial Model $(-2x - 15)$', color="mediumpurple", linestyle="solid")

    
    colors = cm.viridis(np.linspace(0, 1, len(dp_epsilons)))  
    line_styles = ['dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (1, 1))]  # as many as epsilons
    for i, dp_epsilon in enumerate(dp_epsilons):
        plt.plot(x_vals.T, predictions[i], label=f'Torrent ($\\epsilon = {dp_epsilon}$)', 
                    color=colors[i], 
                    linestyle=line_styles[i % len(line_styles)], 
                    linewidth=1.5)
    # Labels and legend
    plt.xlabel("Feature X")
    plt.ylabel("Label Y")
    plt.legend()
    plt.title(f"Strategic Corruption in 1D $(\\beta = {alpha})$")
    plt.grid(False)

    # Show plot
    plt.show()
    plt.close()





# Run the tests with averaging
num_trials = 10
# Run the average version
run_tests_dp_1d_avg(num_trials)

#run_tests_dp_1d(num_trials)
