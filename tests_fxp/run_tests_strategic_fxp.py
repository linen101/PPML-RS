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
from synthetic.strategic_corruptions import  strategic_corruption_scaled
from synthetic.toy_dataset import generate_synthetic_dataset
from torrent.torrent import  torrent_admm, split_matrix, split_matrix_Y, torrent_admm_dp
from torrent.torrent_fxp import torrent_admm_fxp
from fixed_point.fixed_point_helpers import *
from plots.plots_fxp import plot_metric_vs_alpha_fxp, plot_metric_vs_d_fxp

markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed


def run_tests_fxp_d(num_trials=10):
    # Define test size and noise parameters
    n = 10000  # Number of samples
    alpha_init= 0.2
    beta = alpha_init + 0.1  # filter size
    d_values = [100, 200]  # Different dimensions
    sigma = 0.1  # Noise level
    test_perc = 0.2  # Test set percentage
    epsilon = 0.1  # Convergence threshold
    
    # Coruption strategies parameters
    additive = 10
    multiplicative = 10
    
    # Initialize lists to store results
    w_errors_d_torrent = np.zeros(len(d_values))
    w_errors_d_torrent_fxp = fxp(np.zeros(len(d_values)))
        
    # admm parameters
    m = 2
    rho = 1
    admm_steps = 5
    robust_rounds = 5
    modelz = 1
    train_size =  n - n*test_perc
    
    # method
    methods = ['Torrent', 'Torrent fxp']
    # Run multiple trials
    for _ in range(num_trials):
        
        for i, d in enumerate(d_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc)
            w_corrupt = additive + w_star * multiplicative
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha_init)

            # Run Torrent
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            w_torrent, _= torrent_admm(X_parts, y_parts, beta, epsilon, rho, admm_steps, robust_rounds, w_star)
            w_errors_d_torrent[i] += np.linalg.norm(w_torrent - w_star)
            print(w_errors_d_torrent[i])
            
            # Run Torrent fxp
            X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)

            w_torrent_fxp, _= torrent_admm_fxp(X_parts_fxp, y_parts_fxp, beta, epsilon, rho, admm_steps, robust_rounds, w_star)
            #print(w_torrent_fxp.info())
            w_errors_d_torrent_fxp[i] += np.linalg.norm(w_torrent_fxp - w_star)
            print(w_errors_d_torrent_fxp[i].info())
            ##
        
    # Compute averages
    w_errors_d_torrent /= num_trials
    w_errors_d_torrent_fxp /= num_trials
    w_errors_d = [w_errors_d_torrent, w_errors_d_torrent_fxp]
    print(f' TORRENT error: {w_errors_d_torrent}')
    print(f' TORRENT fxp error: {w_errors_d_torrent_fxp}')

    # Plot results
    plot_metric_vs_d_fxp(w_errors_d, methods, r'$\| w - w^* \|_2$',  d_values, alpha_init, sigma, n)
    
def run_tests_fxp_alpha(num_trials=10):
    # Define test ssize and noise parameters
    n = 1000  # Number of samples
    dimension = 10
    alpha_values = [ 0.1, 0.2, 0.3]  # Corruption rates
    sigma = 0.1  # Noise level
    test_perc = 0.2  # Test set percentage
    epsilon = 0.1  # Convergence threshold
    
    # Corruption strat parameters
    additive = 10
    multiplicative = 10
    
    # Initialize lists to store results
    w_errors_alpha_torrent = np.zeros(len(alpha_values))
    w_errors_alpha_torrent_fxp = fxp(np.zeros(len(alpha_values)))

    # admm parameters
    m = 2
    rho=1
    admm_steps=5
    robust_rounds=5
    modelz=1
    train_size =  n - n*test_perc
    
    # method
    methods = ['Torrent', 'Torrent fxp']
    # Run multiple trials
    for _ in range(num_trials):
        for j, alpha in enumerate(alpha_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, dimension, sigma, test_perc)
            w_corrupt = multiplicative*w_star + additive  
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)
            
            # Run Torrent
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            beta = alpha + 0.1  # filter size
            w_torrent, _ = torrent_admm_dp(X_parts, y_parts, beta, epsilon,  rho, 0.01, 0.00001, admm_steps, robust_rounds, w_star)
            w_errors_alpha_torrent[j] += np.linalg.norm(w_torrent - w_star)
            #print(w_errors_alpha_torrent[j])
            
            # Run Torrent fxp
            X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)

            # can also add dp noise here
            w_torrent_fxp, _= torrent_admm_fxp(X_parts_fxp, y_parts_fxp, beta, epsilon, rho, admm_steps, robust_rounds, w_star, 0.01)
            #print(w_torrent_fxp.info())
            w_errors_alpha_torrent_fxp[j] += np.linalg.norm(w_torrent_fxp - w_star)
            #print(w_errors_alpha_torrent_fxp[j].info())
            
    w_errors_alpha_torrent /= num_trials
    w_errors_alpha_torrent_fxp /= num_trials
    print(f' TORRENT error: {w_errors_alpha_torrent}')
    print(f' TORRENT fxp error: {w_errors_alpha_torrent_fxp}')
    w_errors_alpha = [w_errors_alpha_torrent, w_errors_alpha_torrent_fxp]
    
    # Plot results
    plot_metric_vs_alpha_fxp(w_errors_alpha, methods, r'$\| w - w^* \|_2$',  alpha_values, dimension, sigma, n)
    

# Run the tests with averaging
num_trials = 2
#run_tests_fxp_d(num_trials)
run_tests_fxp_alpha(num_trials)

