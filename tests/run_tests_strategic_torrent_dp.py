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
from torrent.torrent import torrent_dp
from decimal import *
from sever.sever import sever
from stir.irls_regressors import irls_init, admm_huber

getcontext().prec = 4
markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed

def run_tests_huber(num_trials=10):
    # Define test parameters
    dp_epsilon = 100000
    dp_delta = 1e-5
    n = 10000  # Number of samples
    dimension = 200
    alpha_init= 0.1
    beta = alpha_init + 0.1  # filter size
    d_values = [10, 50, 100, 150]  # Different dimensions
    alpha_values = [0.1, 0.12, 0.15, 0.18, 0.2 ]  # Corruption rates
    sigma = 0.1  # Noise level
    test_perc = 0.2  # Test set percentage
    
    epsilon = 0.5  # Convergence threshold
    theta = np.pi  # Rotation
    variance = 0.1 #, interpolation
    mixing = 0.5 #interpolation
    additive = 0.9
    multiplicative = 0.1
    sparsity = 0.2
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
    robust_rounds=20
    modelz=1
    # Run multiple trials
    for _ in range(num_trials):
        for i, d in enumerate(d_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc)
            #w_corrupt = variance * np.random.randn(d, 1) #random model with covariance scaled by 100
            #w_corrupt = interpolate_w_arbitrary(w_star, mixing, variance) # interpolate between random model and w_star
            #w_corrupt = sparse_noise_w_arbitrary(w_star) # add sparse noise in arbitrary dimensions
            #w_corrupt = np.random.uniform(-10, 10, (d, 1))
            #w_corrupt =rotate_w_arbitrary(w_star) 
            w_corrupt = w_star + 100
            #w_corrupt = w_star * multiplicative
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha_init)
            
            w_torrent, iter_count= torrent_dp(X_train, Y_cor, beta, epsilon, dp_epsilon, dp_delta, robust_rounds)
            w_errors_d_torrent[i] += np.linalg.norm(w_torrent - w_star)
            print(w_errors_d_torrent[i])
            iters_d[i] += iter_count
            print(iter_count)
           

        for j, alpha in enumerate(alpha_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, dimension, sigma, test_perc)
            #w_corrupt = variance * np.random.randn(dimension, 1) #random model with covariance scaled by 100
            #w_corrupt = interpolate_w_arbitrary(w_star, mixing, variance) # interpolate between random model and w_star
            #w_corrupt = sparse_noise_w_arbitrary(w_star) # add sparse noise in arbitrary dimensions
            # w_corrupt = np.random.uniform(-10, 10, (dimension, 1))
            #w_corrupt = rotate_w_arbitrary(w_star)
            w_corrupt = w_star + additive
            #w_corrupt = w_star * multiplicative
            #w_corrupt = 1/w_star + 10   
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)
            
            # Run Torrent
            beta = alpha + 0.1  # filter size
            w_torrent, iter_count= torrent_dp(X_train, Y_cor, beta, epsilon, dp_epsilon, dp_delta, robust_rounds)
            w_errors_alpha_torrent[j] += np.linalg.norm(w_torrent - w_star)
            print(w_errors_alpha_torrent[j])
            iters_alpha[j] += iter_count
            print(iter_count)
            
# Compute averages
    w_errors_d_torrent /= num_trials
    iters_d //= num_trials + 1 
    w_errors_alpha_torrent /= num_trials
    iters_alpha //= num_trials + 1 
    
    

###
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, w_errors_d_torrent, marker='o', linestyle="dashed", label='Torrent $w_{error}$', color='palevioletred')
    plt.xlabel('Dimension $(d)$', fontsize=14)
    plt.ylabel(r'$\| w - w^* \|_2$', fontsize=14)
    #plt .title('Error vs. Dimension' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=8)
    plt.title(f'Rotation ' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, w_errors_alpha_torrent, marker='o', linestyle="dashed", label='Torrent $w_{error}$', color='palevioletred')
    plt.xlabel('Corruption Rate $(\\beta)$', fontsize=14)
    plt.ylabel(r'$\| w - w^* \|_2$', fontsize=14)
    #plt.title(f'$Error$ vs. Corruption Rate' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=8)
    plt.title(f'Rotation  ' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    #plt.title(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    #plt.title(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.show()

    
###
    plt.figure()
    plt.plot(d_values, iters_d, marker='o', color="palevioletred", linestyle="dashed", label='Torrent Iterations')
    #plt.plot(d_values, iters_d_huber, marker='x', color="teal", linestyle="dashed", label='Huber Iterations')
    plt.xlabel('Dimension $(d)$', fontsize=14)
    plt.ylabel(' Iterations', fontsize=14)
    #  y-axis increments by 1 
    y_max =  max(iters_d)
    plt.yticks(np.arange(1, y_max + 1, 1)) 
    #plt.title('Iterations vs. Dimension' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$', fontsize=8)
    plt.title(f'Rotation  ' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',fontsize=14)
    #plt.title(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',fontsize=14)
    #plt.title(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Multiplicative $\\gamma = {multiplicative} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    plt.grid(False)
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(alpha_values, iters_alpha, marker='o', color="palevioletred", linestyle="dashed", label='Iterations')
    #plt.plot(alpha_values, iters_alpha_huber, marker='x', color="teal", linestyle="dashed", label='Huber Iterations')
    plt.xlabel('Corruption Rate', fontsize=14)
    plt.ylabel(' Iterations', fontsize=14)
    #  y-axis increments by 1 
    y_max =  max(iters_alpha)
    plt.yticks(np.arange(1, y_max + 1, 1)) 
    #plt.title('Iterations vs. Corruption Rate' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=8)
    plt.title(f'Rotation  ' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    #plt.title(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    #plt.title(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, \ d={dimension}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Multiplicative $\\gamma = {multiplicative} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    plt.grid(False)
    plt.show()

# Run the tests with averaging
num_trials = 1
run_tests_huber(num_trials)
