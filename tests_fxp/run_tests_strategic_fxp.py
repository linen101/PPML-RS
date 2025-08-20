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
from torrent.torrent_fxp import torrent_admm_fxp, torrent_admm_fxp_analyze_gauss
from fixed_point.fixed_point_helpers import *
from plots.plots_fxp import plot_metric_vs_alpha_fxp, plot_metric_vs_d_fxp

markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed


def run_tests_fxp_d(num_trials=10):
    # Define test size and noise parameters
    n = 10000  # Number of samples
    alpha_init= 0.2
    beta = alpha_init + 0.1  # filter size
    d_values = [10, 25, 50, 100]  # Different dimensions
    #d_values=[10]
    # dp noise accordingly
    dp_w = [0.0491542458, 0.2959143979, 1.184429552, 4.821505653]           # for n = 10000, ||w*|| > 1
    #dp_w = [0.01851400853, 0.06787019106, 0.1861504838, 0.5212434669]      #for n = 10000
    #dp_w = [0.00203186667, 0.00736297528, 0.01993442884, 0.05482502464]     # for n = 100000
    #dp_w = [0.005377744112, 0.03197006629, 0.1262511128, 0.5046469554]      # for n = 100000, ||w*|| > 1
    #dp_w = [0.0491542458] 
    
    # n= 10000 
    dp_noise_x = [86.87224608, 217.1806152, 434.3612304, 868.7224608]
    dp_noise_y = [274.7141631, 1085.903076, 3071.397715, 8687.224608]
    
    # n = 100000
    #dp_noise_x = [96.89610525, 242.2402631, 484.4805263, 968.9610525]
    #dp_noise_y = [306.412389, 1211.201316, 3425.794655, 9689.610525]
    
    #dp_noise_x = [86.87224608]    #n=10^5, d=10
    #dp_noise_y = [274.7141631]     #n=10^5, d=10
    sigma = 0.1  # Noise level
    test_perc = 0  # Test set percentage
    epsilon = 0.1  # Convergence thresholdc
    
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
            norm_w = np.linalg.norm((w_star))
            norm_w_inv = 1 / norm_w
            #print(f'w star : {w_star}')
            #print(f'w star norm: {norm_w}')
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha_init)

            # Run Torrent
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            # Use the dp parameter corresponding to this d
            #dp_w_val = dp_w[i]
            dp_noise_x_val = dp_noise_x[i]
            dp_noise_y_val = dp_noise_y[i]
            
            # Run Torrent fxp
            X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)
            
            w_torrent_fxp, _= torrent_admm_fxp(X_parts_fxp, y_parts_fxp, beta, epsilon, rho, admm_steps, robust_rounds, w_star, dp_w_val)
            w_errors_d_torrent_fxp[i] += np.linalg.norm(abs(w_torrent_fxp - w_star)) * norm_w_inv
            print(f' sum trials error DP OLS: {w_errors_d_torrent_fxp[i]}')
            
            w_torrent, _ = torrent_admm_fxp_analyze_gauss(X_parts_fxp, y_parts_fxp, beta, epsilon, rho, admm_steps, robust_rounds, w_star, dp_noise_x_val, dp_noise_y_val)
            w_errors_d_torrent[i] += (np.linalg.norm(abs(w_torrent - w_star)) * norm_w_inv)
            print(f' sum trials error Analyze Gauss: {w_errors_d_torrent[i]}')
        
    # Compute averages
    w_errors_d_torrent /= num_trials
    w_errors_d_torrent_fxp /= num_trials
    w_errors_d = [w_errors_d_torrent, w_errors_d_torrent_fxp]
    print(f' TORRENT analyze gauss error: {w_errors_d_torrent}')
    print(f' TORRENT fxp error: {w_errors_d_torrent_fxp}')

    # Plot results
    #plot_metric_vs_d_fxp(w_errors_d, methods, r'$\| w - w^* \|_2$',  d_values, alpha_init, sigma, n)
    
def run_tests_fxp_alpha(num_trials=10):
    # Define test ssize and noise parameters
    n = 10000  # Number of samples
    dimension = 10
    alpha_values = [ 0.1, 0.15, 0.2, 0.25, 0.3]  # Corruption rates
    sigma = 0.1  # Noise level
    test_perc = 0  # Test set percentage
    epsilon = 0.1  # Convergence threshold
    # dp noise accordingly
    #dp_w = [0.0491542458, 0.2959143979, 1.184429552, 4.821505653]           # for n = 10000, ||w*|| > 1
    #dp_w = [0.01851400853, 0.06787019106, 0.1861504838, 0.5212434669]      #for n = 10000
    #dp_w = [0.00203186667, 0.00736297528, 0.01993442884, 0.05482502464]     # for n = 100000
    #dp_w = [0.005377744112, 0.03197006629, 0.1262511128, 0.5046469554]      # for n = 100000, ||w*|| > 1
    dp_w = 0.0491542458  #n=10^5, d=10
    # 
    #dp_noise_x = [86.87224608, 217.1806152, 434.3612304, 868.7224608]
    #dp_noise_y = [274.7141631, 1085.903076, 3071.397715, 8687.224608]  # for n = 10000, ||w*|| > 1
    #dp_noise_x = [96.89610525, 242.2402631, 484.4805263, 968.9610525]   
    #dp_noise_y = [306.412389, 1211.201316, 3425.794655, 9689.610525]    
    dp_noise_x = 86.87224608    #n=10^5, d=10
    dp_noise_y = 274.7141631     #n=10^5, d=10
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
    methods = ['Torrent analyze gauss', 'Torrent fxp']
    # Run multiple trials
    for _ in range(num_trials):
        for j, alpha in enumerate(alpha_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, dimension, sigma, test_perc)
            w_corrupt = multiplicative*w_star + additive  
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)
            
            norm_w = np.linalg.norm((w_star))
            norm_w_inv = 1 / norm_w
            
            # Run Torrent
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            beta = alpha + 0.1  # filter size
            
            # Run Torrent analyze gauss fxp
            X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)
            
            w_torrent, _ = torrent_admm_fxp_analyze_gauss(X_parts_fxp, y_parts_fxp, beta, epsilon, rho, admm_steps, robust_rounds, w_star, dp_noise_x, dp_noise_y)
            w_errors_alpha_torrent[j] += np.linalg.norm(w_torrent - w_star)* norm_w_inv
            print(f' sum trials error analyze gauss: {w_errors_alpha_torrent[j]}')

            ## Run Torrent dp fxp
            w_torrent_fxp, _= torrent_admm_fxp(X_parts_fxp, y_parts_fxp, beta, epsilon, rho, admm_steps, robust_rounds, w_star, dp_w)
            w_errors_alpha_torrent_fxp[j] += np.linalg.norm(w_torrent_fxp - w_star)* norm_w_inv
            print(f' sum trials error DP OLS: {w_errors_alpha_torrent_fxp[j]}')
            
    w_errors_alpha_torrent /= num_trials
    w_errors_alpha_torrent_fxp /= num_trials
    print(f' TORRENT analyze gauss error: {w_errors_alpha_torrent}')
    print(f' TORRENT fxp error: {w_errors_alpha_torrent_fxp}')
    w_errors_alpha = [w_errors_alpha_torrent, w_errors_alpha_torrent_fxp]
    
    # Plot results
    plot_metric_vs_alpha_fxp(w_errors_alpha, methods, r'$\| w - w^* \|_2$',  alpha_values, dimension, sigma, n)
    

# Run the tests with averaging
num_trials = 10
#run_tests_fxp_d(num_trials)
run_tests_fxp_alpha(num_trials)

