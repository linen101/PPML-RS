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
from synthetic.strategic_corruptions import rotate_w_arbitrary , sparse_noise_w_arbitrary , interpolate_w_arbitrary, strategic_corruption_scaled, rotate_w_partial
from synthetic.toy_dataset import generate_synthetic_dataset, corrupt_dataset
from realdata.real_data import load_and_process_gas_sensor_data
from torrent.torrent import  torrent, torrent_ideal, torrent_admm, split_matrix, split_matrix_Y
from decimal import *
from stir.irls_regressors import irls_init, admm_huber
getcontext().prec = 4
markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed

def cosine_similarity(w1, w2):
    return np.dot(w1, w2.T)[0, 0] / (np.linalg.norm(w1) * np.linalg.norm(w2))

def run_tests_huber_d(num_trials=10):
    # Define test size and noise parameters
    n = 2000  # Number of samples
    alpha_init= 0.4
    beta = alpha_init + 0.1  # filter size
    d_values = [10, 50, 100, 300, 500]  # Different dimensions
    sigma = 0.1  # Noise level
    test_perc = 0.2  # Test set percentage
    epsilon = 0.1  # Convergence threshold
    
    # Coruption strategies parameters
    theta = np.pi  # Rotation
    variance = 0.1 #, interpolation
    mixing = 0.5 #interpolation
    additive = 10
    multiplicative = 0.1
    sparsity = 0.2
    
    # Initialize lists to store results
    w_errors_d_torrent = np.zeros(len(d_values))
    w_errors_d_huber = np.zeros(len(d_values))
    iters_d = np.zeros(len(d_values))
    iters_d_huber = np.zeros(len(d_values))
    
    # admm parameters
    m = 2
    rho=1
    admm_steps=10
    robust_rounds=10
    modelz=1
    train_size =  n - n*test_perc
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
            #Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha_init)
            Y_cor = corrupt_dataset(X_train, Y_train, w_star, alpha_init, sigma)
            ## 
            # Run Torrent
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            rho=1
            w_torrent, iter_count= torrent_admm(X_parts, y_parts, beta, epsilon, rho, admm_steps, robust_rounds, w_star)
            w_errors_d_torrent[i] += np.linalg.norm(w_torrent - w_star)
            print(w_errors_d_torrent[i])
            iters_d[i] += iter_count
            print(iter_count)
            # Run Huber Regression
            #huber = HuberRegressor( epsilon=1.35, max_iter=10, alpha=0.1, warm_start=False, fit_intercept=False, tol=0.1).fit(X_train.T, Y_cor.ravel())
            #w_huber = huber.coef_
            #w_huber = w_huber.reshape(-1,1)
            w_huber, _ = irls_init(X_train, Y_cor, delta=1.345, max_iterations=robust_rounds, tol=epsilon, scheme='HUBER', sigma=sigma, w_star=w_star)
            ##_huber, it_huber = admm_huber(X_train, Y_cor, delta=1.345, rho=1.0, max_iter=1000, tol=0.2, w_star=w_star)
            w_errors_d_huber[i] += np.linalg.norm(w_huber - w_star)
            #iters_d_huber[i] = it_huber
            ##
        
    # Compute averages
    w_errors_d_torrent /= num_trials
    w_errors_d_huber /= num_trials
    iters_d = (iters_d // num_trials) + 1
    
    # show scaled plots together for error with dimension
    ###############################
    
    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 5), gridspec_kw={'height_ratios': [1, 1]})

    # Bottom plot (for small values)
    ax1.plot(d_values, w_errors_d_torrent, marker='o', color = 'palevioletred', linestyle="dashed", label='Torrent')
    ax1.set_ylim(0, max(w_errors_d_torrent)+0.05)  # Adjust for small values
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.tick_bottom()
    ax1.tick_params(labeltop=False)  
    #ax1.legend()

    # Top plot (for large values)
    ax2.plot(d_values, w_errors_d_huber, marker='s', color='teal', linestyle="dashed", label='Huber')
    ax2.set_ylim(0, max(w_errors_d_huber)+1)  # Adjust for large values
    ax2.spines['bottom'].set_visible(False)
    ax2.xaxis.tick_top()
    ax2.tick_params(labeltop=False)
    
    
    # diagonal break marks
    d = .015  
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (1-d, 1+d), **kwargs)  
    ax1.plot((1 - d, 1 + d), (1-d, 1+d), **kwargs)  

    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d, +d), (- d, + d), **kwargs)  
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  
    
    
    fig.supylabel(r'$\| w - w^* \|_2$', fontsize=14)
    fig.supxlabel('Dimension $(d)$')
    #plt .title('Error vs. Dimension' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=8)
    #plt.title(f'Rotation Corruption Strategy' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #fig.suptitle(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    fig.suptitle(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)    
    #fig.suptitle(f'Multiplicative $\\gamma = {multiplicative} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #fig.suptitle(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #fig.suptitle(f'Rotation ' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',fontsize=14)
    fig.legend(loc='center', ncol=2, fontsize=12)
    plt.show()
            
    # Plot results
    
    ### dimension
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, w_errors_d_torrent, marker='o', linestyle="dashed", label='Torrent $w_{error}$', color='palevioletred')
    plt.plot(d_values, w_errors_d_huber, marker='s', linestyle="dashed", label='Huber $w_{error}$', color='teal')
    # Annotate |ws_torrent - w_star|_2 values
    #for i, d in enumerate(d_values):
    #    plt.annotate(f'{w_errors_d_torrent[i]:.2f}', (d, w_errors_d_torrent[i]), 
    #                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7, color='blue')
    plt.xlabel('Dimension $(d)$', fontsize=14)
    plt.ylabel(r'$\| w - w^* \|_2$', fontsize=14)
    #plt.title(f'Multiplicative $\\gamma = {multiplicative} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=8)
    #plt.title(f'Rotation ' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    plt.title(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.show()
    
        
    ### iterations
    plt.figure()
    plt.plot(d_values, iters_d, marker='o', color="palevioletred", linestyle="dashed", label='Torrent Iterations')
    #plt.plot(d_values, iters_d_huber, marker='x', color="teal", linestyle="dashed", label='Huber Iterations')
    plt.xlabel('Dimension $(d)$', fontsize=14)
    plt.ylabel(' Iterations', fontsize=14)
    #  y-axis increments by 1 
    y_max =  max(iters_d)
    plt.yticks(np.arange(1, y_max + 1, 1)) 
    #plt.title('Iterations vs. Dimension' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$', fontsize=8)
    #plt.title(f'Rotation  ' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',fontsize=14)
    #plt.title(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',fontsize=14)
    #plt.title(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    plt.title(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Multiplicative $\\gamma = {multiplicative} $' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=14)
    plt.grid(False)
    plt.legend()
    plt.show()
    
def run_tests_huber_alpha(num_trials=10):
    # Define test ssize and noise parameters
    n = 1000  # Number of samples
    dimension = 10
    alpha_values = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65 ]  # Corruption rates
    sigma = 0.1  # Noise level
    test_perc = 0.2  # Test set percentage
    epsilon = 0.1  # Convergence threshold
    
    # Corruption strat parameters
    theta = np.pi  # Rotation
    variance = 0.1 #, interpolation
    mixing = 0.5 #interpolation
    additive = 10
    multiplicative = 10
    sparsity = 0.2
    
    # Initialize lists to store results
    w_errors_alpha_torrent = np.zeros(len(alpha_values))
    w_errors_alpha_huber = np.zeros(len(alpha_values))
    iters_alpha = np.zeros(len(alpha_values))
    iters_alpha_huber = np.zeros(len(alpha_values))
    
    # admm parameters
    m = 2
    rho=1
    admm_steps=10
    robust_rounds=10
    modelz=1
    train_size =  n - n*test_perc
    # Run multiple trials
    for _ in range(num_trials):
        for j, alpha in enumerate(alpha_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, dimension, sigma, test_perc)
            #w_corrupt = variance * np.random.randn(dimension, 1) #random model with covariance scaled by 100
            #w_corrupt = interpolate_w_arbitrary(w_star, mixing, variance) # interpolate between random model and w_star
            #w_corrupt = sparse_noise_w_arbitrary(w_star) # add sparse noise in arbitrary dimensions
            # w_corrupt = np.random.uniform(-10, 10, (dimension, 1))
            #w_corrupt = rotate_w_arbitrary(w_star)
            w_corrupt = multiplicative*w_star + additive
            #w_corrupt = w_star * multiplicative
            #w_corrupt = 1/w_star + 10   
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)
            #Y_cor = corrupt_dataset(X_train, Y_train, w_star, alpha, sigma)
            
            # Run Torrent
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            rho=1
            beta = alpha + 0.1  # filter size
            w_torrent, iter_count = torrent_admm(X_parts, y_parts, beta, epsilon, rho, admm_steps, robust_rounds, w_star)
            w_errors_alpha_torrent[j] += np.linalg.norm(w_torrent - w_star)
            print(w_errors_alpha_torrent[j])
            iters_alpha[j] += iter_count
            
            # Run Huber Regression
            #huber = HuberRegressor( epsilon=1.35, max_iter=500, alpha=0.1, warm_start=False, fit_intercept=False, tol=0.1).fit(X_train.T, Y_cor.ravel())
            #w_huber = huber.coef_
            #w_huber = w_huber.reshape(-1,1)
            w_huber, _ = irls_init(X_train, Y_cor, delta=1.345, max_iterations=robust_rounds, tol=epsilon, scheme='HUBER', sigma=sigma, w_star=w_star)
            #w_huber, it_huber_a = admm_huber(X_train, Y_cor, delta=1.345, rho=1.0, max_iter=1000, tol=0.2, w_star=w_star)
            w_errors_alpha_huber[j] += np.linalg.norm(w_huber - w_star)
            #iters_alpha_huber[j] = it_huber_a
    w_errors_alpha_torrent /= num_trials
    w_errors_alpha_huber /= num_trials
    iters_alpha = (iters_alpha // num_trials) + 1

    # show scaled plots
    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 5), gridspec_kw={'height_ratios': [1, 1]})

    # Bottom plot (for small values)
    ax1.plot(alpha_values, w_errors_alpha_torrent, marker='o', linestyle="dashed", label='Torrent', color='palevioletred')
    ax1.set_ylim(0, max(w_errors_alpha_torrent)+0.05)  # Adjust for small values
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.tick_bottom()
    ax1.tick_params(labeltop=False)  
    #ax1.legend()

    # Top plot (for large values)
    ax2.plot(alpha_values, w_errors_alpha_huber, marker='s', linestyle="dashed", label='Huber', color='teal')
    ax2.set_ylim(0, max(w_errors_alpha_huber)+1)  # Adjust for large values
    ax2.spines['bottom'].set_visible(False)
    ax2.xaxis.tick_top()
    ax2.tick_params(labeltop=False)
    
    
    # diagonal break marks
    d = .015  
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (1-d, 1+d), **kwargs)  
    ax1.plot((1 - d, 1 + d), (1-d, 1+d), **kwargs)  

    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d, +d), (- d, + d), **kwargs)  
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  
    
    
    fig.supylabel(r'$\| w - w^* \|_2$', fontsize=14)
    fig.supxlabel(f'Corrruption $(\\beta)$')
    #plt .title('Error vs. Dimension' f'$ \ (n={n}, \\beta={alpha_init}, \\sigma={sigma})$',  fontsize=8)
    #fig.suptitle(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    #fig.suptitle(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    #fig.suptitle(f'Rotation  ' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',fontsize=14)
    fig.suptitle(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    #fig.suptitle(f'Multiplicative $\\gamma = {multiplicative} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    fig.legend(loc='center', ncol=2, fontsize=12)
    plt.show()
        
    
    ### errors
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, w_errors_alpha_torrent, marker='o', linestyle="dashed", label='Torrent $w_{error}$', color='palevioletred')
    plt.plot(alpha_values, w_errors_alpha_huber, marker='s', linestyle="dashed", label='Huber $w_{error}$', color='teal')
    # Annotate |ws_torrent - w_star|_2 values
    #for i, alpha in enumerate(alpha_values):
    #    plt.annotate(f'{w_errors_alpha_torrent[i]:.2f}', (alpha, w_errors_alpha_torrent[i]), 
    #                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7, color='blue')
    plt.xlabel('Corruption Rate $(\\beta)$', fontsize=14)
    plt.ylabel(r'$\| w - w^* \|_2$', fontsize=14)
    #plt.title(f'$Error$ vs. Corruption Rate' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=8)
    #plt.title(f'Rotation  ' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    plt.title(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    #plt.title(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.show()

    
    ### iterations

    plt.figure()
    plt.plot(alpha_values, iters_alpha, marker='o', color="palevioletred", linestyle="dashed", label='Iterations')
    #plt.plot(alpha_values, iters_alpha_huber, marker='x', color="teal", linestyle="dashed", label='Huber Iterations')
    plt.xlabel('Corruption Rate', fontsize=14)
    plt.ylabel(' Iterations', fontsize=14)
    #  y-axis increments by 1 
    y_max =  max(iters_alpha)
    plt.yticks(np.arange(1, y_max + 1, 1)) 
    #plt.title('Iterations vs. Corruption Rate' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=8)
    #plt.title(f'Rotation  ' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    #plt.title(f'Interpolation $\\eta = {variance}, \ \\lambda = {mixing} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$', fontsize=14)
    #plt.title(f'Sparsity $\\eta = {variance}, \ p = {sparsity} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    plt.title(f'Additive $\\gamma = {additive} $' f'$ \ (n={n}, \ d={dimension}, \\sigma={sigma})$',  fontsize=14)
    #plt.title(f'Multiplicative $\\gamma = {multiplicative} $' f'$ \ (n={n}, d={dimension}, \\sigma={sigma})$',  fontsize=14)
    plt.grid(False)
    plt.show()    

# Run the tests with averaging
num_trials = 10
run_tests_huber_d(num_trials)
#run_tests_huber_alpha(num_trials)

