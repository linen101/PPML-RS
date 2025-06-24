import numpy as np
import matplotlib.pyplot as plt
import time
import math
import matplotlib.cm as cm

from sklearn.linear_model import HuberRegressor
import sys
import os
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from synthetic.strategic_corruptions import generate_synthetic_dataset, strategic_corruption_scaled
from synthetic.toy_dataset import generate_synthetic_dataset
from torrent.torrent import torrent_dp, torrent, torrent_admm_dp, split_matrix, split_matrix_Y, torrent_admm
from decimal import *


getcontext().prec = 4
markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed

def run_tests_torrent_dp_d(num_trials=10):
    # test size and noise params
    dp_epsilons = [0.5, 1, 2, 5, 10]  # Try different privacy levels
    dp_noise = [0.01, 0.05, 0.1, 0.2]
    dp_delta = 1e-5
    n = 1000
    d_values = [10, 50, 100]
    alpha_init = 0.1
    beta = alpha_init + 0.1
    sigma = 0.1     # standard deviation bounded noise
    test_perc = 0.1
    train_size = int(n * (1 - test_perc))    
    epsilon = 0.01  # convergence threshold
    
    # corruption
    additive = 10
    multiplicative = 10
    
    # admm params
    m = 2
    rho = 1
    admm_steps = 10
    robust_rounds = 5

    # errors will be shape: (len(dp_epsilons), len(d_values))
    w_errors_dp_d = np.zeros((len(dp_noise), len(d_values)))
    
    for t in range(num_trials):
        print(f'>>> Trial {t+1}/{num_trials}')
        for e, noise in enumerate(dp_noise):
            for i, d in enumerate(d_values):
                X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc)
                w_corrupt = multiplicative * w_star + additive
                Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha_init)

                X_parts = split_matrix(X_train, m, train_size)
                y_parts = split_matrix_Y(Y_cor, m, train_size)
                
                # add noise to see how much noise torrent can have.
                w_torrent, iter_count = torrent_admm_dp(X_parts, y_parts, beta, epsilon, rho, noise, dp_delta, admm_steps, robust_rounds, w_star)
                error = np.linalg.norm(w_torrent - w_star)
                w_errors_dp_d[e, i] += error
    
    # Average over trials
    w_errors_dp_d /= num_trials

    # Plot dimension
    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(dp_noise)))
    line_styles = ['dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (1, 1))]

    for e, noise in enumerate(dp_noise):
        errors = w_errors_dp_d[e]
        plt.plot(d_values, w_errors_dp_d[e], label=f'$noise = {noise}$', 
                 color=colors[e], linestyle=line_styles[e % len(line_styles)], linewidth=2)
        for x, y in zip(d_values, errors):
            plt.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9, color=colors[e])


    plt.xlabel('Dimension $(d)$', fontsize=14)
    plt.ylabel(r'$\| w - w^* \|_2$', fontsize=14)
    plt.title(f'Strategic Corruption with DP (n={n}, β={alpha_init}, σ={sigma})', fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    

def run_tests_torrent_dp_alpha(num_trials=10):
    # test size and noise params
    dp_epsilons = [0.5, 1, 2, 5, 10]  # Try different privacy levels
    dp_noise = [0.01, 0.05, 0.1, 0.2]
    dp_delta = 1e-5
    n = 100000
    dimension = 100
    alpha_values = [0.1, 0.2, 0.3, 0.4]
    sigma = 0.1
    test_perc = 0.1
    train_size = int(n * (1 - test_perc))
    epsilon = 0.01  # convergence threshold

    # corruption
    additive = 10
    multiplicative = 10
    
    # admm params
    m = 2
    rho = 1
    admm_steps = 10
    robust_rounds = 5
    
    # errors will be shape: (len(dp_epsilons), len(alpha_values))
    w_errors_dp_alpha = np.zeros((len(dp_noise), len(alpha_values)))

    for t in range(num_trials):
        print(f'>>> Trial {t+1}/{num_trials}')
        for e, noise in enumerate(dp_noise):
            for i, alpha in enumerate(alpha_values):
                X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, dimension, sigma, test_perc)
                w_corrupt = multiplicative * w_star + additive
                Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)

                X_parts = split_matrix(X_train, m, train_size)
                y_parts = split_matrix_Y(Y_cor, m, train_size)
                
                # add noise to see how much noise torrent can have.
                beta = alpha + 0.1
                w_torrent, iter_count = torrent_admm_dp(X_parts, y_parts, beta, epsilon, rho, noise, dp_delta, admm_steps, robust_rounds, w_star)
                error = np.linalg.norm(w_torrent - w_star)
                w_errors_dp_alpha[e, i] += error    

    # Average over trials
    w_errors_dp_alpha /= num_trials
    
    # Plot alpha
    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(dp_noise)))
    line_styles = ['dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (1, 1))]

    for e, noise in enumerate(dp_noise):
        errors = w_errors_dp_alpha[e]
        plt.plot(alpha_values, errors, label=f'$noise = {noise}$', 
                 color=colors[e], linestyle=line_styles[e % len(line_styles)], linewidth=2)
        for x, y in zip(alpha_values, errors):
            plt.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9, color=colors[e])


    plt.xlabel(f'Corruption fraction $\\beta', fontsize=14)
    plt.ylabel(r'$\| w - w^* \|_2$', fontsize=14)
    plt.title(f'Strategic Corruption with DP (n={n}, d={dimension}, σ={sigma})', fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
# Run the tests with averaging
num_trials = 5

#Run tests d
#run_tests_torrent_dp_d(num_trials)

#Run tests alpha
run_tests_torrent_dp_alpha(num_trials)