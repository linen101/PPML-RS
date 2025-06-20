import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.linear_model import HuberRegressor
import sys
import os
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from synthetic.strategic_corruptions import generate_synthetic_dataset, strategic_corruption_on_X, adversarial_corruption, rotate_w_arbitrary , sparse_noise_w_arbitrary , interpolate_w_arbitrary, strategic_corruption_scaled, rotate_w_partial
from realdata.real_data import load_and_process_gas_sensor_data
from plots.plots import plot_regression_errors_n, plot_regression_errors_d, plot_iterations_n, plot_iterations_d
from torrent.torrent import torrent_dp, torrent, torrent_admm_dp, split_matrix, split_matrix_Y, torrent_admm
from decimal import *
from sever.sever import sever
from stir.irls_regressors import irls_init, admm_huber

getcontext().prec = 4
markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed

def run_tests_dp(num_trials=10):
    # Define test parameters
    dp_epsilon = 1
    dp_delta = 1e-5
    n = 2000  # Number of samples
    dimension = 100
    alpha_init= 0.1
    beta = alpha_init + 0.1  # filter size
    d_values = [1, 2, 3]  # Different dimensions
    alpha_values = [0.1, 0.12, 0.15, 0.18, 0.2]  # Corruption rates
    sigma = 0.1  # Noise level
    test_perc = 0.2  # Test set percentage
    
    epsilon = 0.01  # Convergence threshold
    additive = 0.1
    multiplicative = 1
    # Initialize lists to store results
    w_errors_d_torrent = np.zeros(len(d_values))
    w_errors_d_huber = np.zeros(len(d_values))
    iters_d = np.zeros(len(d_values))
    iters_d_huber = np.zeros(len(d_values))
    
    w_errors_alpha_torrent = np.zeros(len(alpha_values))
    w_errors_alpha_huber = np.zeros(len(alpha_values))
    iters_alpha = np.zeros(len(alpha_values))
    iters_alpha_huber = np.zeros(len(alpha_values))
    m = 2
    rho=1
    admm_steps=10
    robust_rounds=5
    train_size = n - n*test_perc
    # Run multiple trials
    for t in range(num_trials):
        print(f'$ \ TRIAL= {t}$')
        
        for i, d in enumerate(d_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc)
            w_corrupt = multiplicative*w_star + additive
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha_init)
            
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            rho=1
            w_torrent, iter_count= torrent_admm_dp(X_parts, y_parts, beta, epsilon, rho, dp_epsilon, dp_delta, admm_steps, robust_rounds, w_star)
            #w_torrent, iter_count= torrent_dp(X_train, Y_cor, beta, epsilon, dp_epsilon, dp_delta, robust_rounds)
            w_errors_d_torrent[i] += np.linalg.norm(w_torrent - w_star)
            print(w_errors_d_torrent[i])
            iters_d[i] += iter_count
            print(iter_count)
        """
        for j, alpha in enumerate(alpha_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, dimension, sigma, test_perc)

            w_corrupt = multiplicative*w_star + additive

            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)
            
            # Run Torrent
            beta = alpha + 0.1  # filter size
            
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            rho=1
            w_torrent, iter_count= torrent_admm_dp(X_parts, y_parts, beta, epsilon, rho, dp_epsilon, dp_delta, admm_steps, robust_rounds, w_star)
            #w_torrent, iter_count= torrent_dp(X_train, Y_cor, beta, epsilon, dp_epsilon, dp_delta, robust_rounds)
            w_errors_alpha_torrent[j] += np.linalg.norm(w_torrent - w_star)
            print(w_errors_alpha_torrent[j])
            iters_alpha[j] += iter_count
            print(iter_count)
        """  
# Compute averages
    w_errors_d_torrent /= num_trials
    iters_d //= num_trials + 1 
    #w_errors_alpha_torrent /= num_trials
    #iters_alpha //= num_trials + 1 
    
###
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, w_errors_d_torrent, marker='o', linestyle="dashed", label='Torrent $w_{error}$', color='palevioletred')
    plt.xlabel('Dimension $(d)$', fontsize=14)
    plt.ylabel(r'$\| w - w^* \|_2$', fontsize=14)
    plt.title(f'Strategic Corruption ' f'$ \ (DP \\epsilon= {dp_epsilon}, n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.show()
    """
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, w_errors_alpha_torrent, marker='o', linestyle="dashed", label='Torrent $w_{error}$', color='palevioletred')
    plt.xlabel('Corruption Rate $(\\beta)$', fontsize=14)
    plt.ylabel(r'$\| w - w^* \|_2$', fontsize=14)
    plt.title(f'Strategic Corruption  ' f'$ \ (DP \\epsilon= {dp_epsilon}, n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.show()
    """
# Run the tests with averaging
num_trials = 10
run_tests_dp(num_trials)
