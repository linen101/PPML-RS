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
from torrent.torrent import torrent_dp, torrent, torrent_admm  # Import torrent module
from synthetic.synthetic_one_dimensional import generate_synthetic_dataset_one_d, strategic_corruption_scaled



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
    n = 100 # Number of samples
    sigma = 0.1
    alpha = 0.1  # 10% of data points are corrupted
    beta = alpha + 0.1
    epsilon = 0.01
    max_iters = 10
    dp_epsilons = [1, 10]
    dp_delta = 1e-5
    dp_B_y = 1
    # assume y values bounded in [-1, 1]
    
    predictions_torrent = np.zeros((len(dp_epsilons), n))  # e.g. shape: (7, 100)
    predictions_huber = np.zeros(n)  # e.g. shape: (100)
    for _ in range(num_trials):
        X_aug, Y, w_star = generate_synthetic_dataset_one_d(n, sigma)

        # Apply strategic corruption
        Y_corrupted, corrupted_indices, w_corrupt = strategic_corruption_scaled(X_aug, Y, alpha)
        
        # generate test values
        x_vals = np.linspace(X_aug[1, :].min(), X_aug[1, :].max(), n).reshape(1, -1)
        x_vals_aug = np.vstack([np.ones((1, x_vals.shape[1])), x_vals])  # Add intercept term

        # honest Model (True)
        y_vals_true = x_vals_aug.T @ w_star
        
        for i, dp_epsilon in enumerate(dp_epsilons):
            w_torrent, iter_count = torrent_dp(X_aug, Y_corrupted, beta, epsilon, dp_epsilon, dp_delta, max_iters)
            #w_torrent, iter_count = torrent(X_aug, Y_corrupted, beta, epsilon, max_iters)
            y_vals_torrent = x_vals_aug.T @ w_torrent
            predictions_torrent[i] +=  y_vals_torrent.ravel()
        
        huber = HuberRegressor(fit_intercept=False).fit(X_aug.T, Y_corrupted.ravel())
        w_huber_intercept = huber.coef_.reshape(-1, 1)  # Already includes intercept and slope
        #w_huber_intercept = np.array([[huber.intercept_], [huber.coef_[0]]])
        y_vals_huber = x_vals_aug.T @ w_huber_intercept   
        predictions_huber += y_vals_huber.ravel()
    
    #get average torrent
    predictions_torrent /= num_trials
    
    #get average huber
    predictions_huber /= num_trials
    
    
    
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
    plt.plot(x_vals.T, y_vals_adv, label=f'Adversarial Model $(4.5x + 4.5)$', color="mediumpurple", linestyle="solid")

    #torrent
    colors = cm.viridis(np.linspace(0, 1, len(dp_epsilons)))  
    line_styles = ['dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (1, 1))]  # as many as epsilons
    for i, dp_epsilon in enumerate(dp_epsilons):
        plt.plot(x_vals.T, predictions_torrent[i], label=f'Torrent ($\\epsilon = {dp_epsilon}$)', 
                    color=colors[i], 
                    linestyle=line_styles[i % len(line_styles)], 
                    linewidth=1.5)
    
    #huber
    plt.plot(x_vals.T, predictions_huber, label="Huber Regression", color="red", linestyle="dashdot")
    
    # Labels and legend
    plt.xlabel("Feature X")
    plt.ylabel("Label Y")
    plt.legend()
    plt.title(f"Strategic Corruption in 1D $(n= {n}, \\beta = {alpha})$")
    plt.grid(False)

    # Show plot
    plt.show()
    plt.close()


# Run the tests with averaging
num_trials = 2


# Run the average version
#run_tests_dp_1d_avg(num_trials)

run_tests_dp_1d(num_trials)