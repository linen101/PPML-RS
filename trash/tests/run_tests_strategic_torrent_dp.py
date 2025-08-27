import numpy as np
import matplotlib.pyplot as plt
import time
import math
import matplotlib.cm as cm
from scipy.stats import iqr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import HuberRegressor, LinearRegression
import sys
import os
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from synthetic.strategic_corruptions import strategic_corruption_scaled
from synthetic.toy_dataset import generate_synthetic_dataset
from torrent.torrent import torrent_dp, torrent, torrent_admm_dp, split_matrix, split_matrix_Y, torrent_admm
from decimal import *
from plots.plots_dp import plot_metric_vs_d, plot_metric_vs_alpha

""" The error is measured on the test set. We measure the
root mean square error (RMSE) normalized by the range of the tar-
get value, between the actual room temperature and the predicted
room temperature"""

getcontext().prec = 4
markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed



def cosine_similarity(w1, w2):
    return np.dot(w1, w2.T)[0, 0] / (np.linalg.norm(w1) * np.linalg.norm(w2))

def prediction_error(X, Y, w):
    iqry = iqr(Y)
    mse = mean_squared_error(Y, X.T @ w)
    rmse = (mse ** 0.5)
    rmse = rmse / iqry
    return rmse

def prediction_error_mae(X, Y, w):
    mae = mean_absolute_error(Y, X.T @ w)
    return mae

def run_tests_torrent_dp_d(num_trials=10):
    # test size and noise params
    dp_epsilons = [0.5, 1, 2, 5, 10]  # Try different privacy levels
    dp_noise = [100, 200]
    dp_delta = 1e-4
    n = 10000
    d_values = [10, 50, 100]
    alpha_init = 0.1
    beta = alpha_init + 0.1
    sigma = 0.1     # standard deviation bounded noise
    test_perc = 0.2
    train_size = int(n * (1 - test_perc))    
    epsilon = 0.01  # convergence threshold
    
    # corruption
    additive = 10
    multiplicative = 10
    
    # admm params
    m = 2
    rho = 1
    admm_steps = 5
    robust_rounds = 5

    # errors will be shape: (len(dp_epsilons), len(d_values))
    w_errors_dp_d = np.zeros(( len(d_values),len(dp_noise)))
    pred_errors_dp_d = np.zeros_like(w_errors_dp_d)
    pred_errors_mae_dp_d = np.zeros_like(w_errors_dp_d)
    
    for t in range(num_trials):
        print(f'>>> Trial {t+1}/{num_trials}')
        for i, d in enumerate(d_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc)
            w_corrupt = multiplicative*w_star + additive
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha_init)
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            #compute ols to compare errors
            ols = LinearRegression(fit_intercept=False).fit(X_train.T, Y_train.ravel())
            w_ols = ols.coef_.reshape(-1, 1)  # Already includes intercept and slope
            error_ols = np.linalg.norm(w_ols - w_star)
            print(f'     OLS error: {error_ols}')
            mae_ols = prediction_error_mae(X_test, Y_test, w_ols)
            #print(f'     OLS MAE: {mae_ols}')
            rmse_ols = prediction_error(X_test, Y_test, w_ols)
            #print(f'     OLS RMSE: {rmse_ols}')
            for e, noise in enumerate(dp_noise): 
                print(f'>>>>>> DP noise: {noise}')               
                # add noise to see how much noise torrent can have.
                w_torrent, iter_count = torrent_admm_dp(X_parts, y_parts, beta, epsilon, rho, noise, dp_delta, admm_steps, robust_rounds, w_star)
                
                # compute errors
                param_error = np.linalg.norm(w_torrent - w_star)
                pred_error = prediction_error(X_test, Y_test, w_torrent)
                pred_error_mae = prediction_error_mae(X_test, Y_test, w_torrent)
                
                print(f'          TORRENT error: {param_error}')
                
                w_errors_dp_d[i, e] += param_error
                pred_errors_dp_d[i, e] += pred_error
                pred_errors_mae_dp_d[i, e] += pred_error_mae
    
    # Average over trials
    w_errors_dp_d /= num_trials
    print(f' TORRENT error: {w_errors_dp_d}')
    pred_errors_dp_d /= num_trials
    pred_errors_mae_dp_d /= num_trials

    plot_metric_vs_d(w_errors_dp_d.T, r'$\| w - w^* \|_2$', dp_noise, d_values, alpha_init, sigma, n)
    #plot_metric_vs_d(pred_errors_dp_d.T, r'$rmse( y - Xw )/iqr $', dp_noise, d_values,alpha_init, sigma, n)
    #plot_metric_vs_d(pred_errors_mae_dp_d.T, r'$mae( Xw - Xw^* )$',dp_noise, d_values, alpha_init, sigma, n)


def run_tests_torrent_dp_alpha(num_trials=10):
    # test size and noise params
    dp_epsilons = [0.5, 1, 2, 5, 10]  # Try different privacy levels
    dp_noise = [0.1, 0.5, 1, 2, 5, 10]
    dp_delta = 1e-4
    n = 1000
    dimension = 10
    alpha_values = [ 0.4, 0.5, 0.6, 0.7]
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
    w_errors_dp_alpha = np.zeros((len(alpha_values), len(dp_noise)))
    for t in range(num_trials):
        print(f'>>> Trial {t+1}/{num_trials}')
        for i, alpha in enumerate(alpha_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, dimension, sigma, test_perc)
            w_corrupt = multiplicative*w_star + additive
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)
            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            
            #compute ols to compare errors
            ols = LinearRegression(fit_intercept=False).fit(X_train.T, Y_train.ravel())
            w_ols = ols.coef_.reshape(-1, 1)  # Already includes intercept and slope
            error_ols = np.linalg.norm(w_ols - w_star)/np.linalg.norm(w_star)
            print(f'     OLS error: {error_ols}')
            mae_ols = prediction_error_mae(X_test, Y_test, w_ols)
            #print(f'     OLS MAE: {mae_ols}')
            rmse_ols = prediction_error(X_test, Y_test, w_ols)
            #print(f'     OLS RMSE: {rmse_ols}')
            for e, noise in enumerate(dp_noise): 
                print(f'>>>>>> DP noise: {noise}')  
                
                # add noise to see how much noise torrent can have.
                beta = alpha + 0.1
                w_torrent, iter_count = torrent_admm_dp(X_parts, y_parts, beta, epsilon, rho, noise, dp_delta, admm_steps, robust_rounds, w_star)
                error = np.linalg.norm(w_torrent - w_star)/np.linalg.norm(w_star)
                print(f'          TORRENT error: {error}')
                
                w_errors_dp_alpha[i, e] += error  

    # Average over trials
    w_errors_dp_alpha /= num_trials
    print(f' TORRENT error: {w_errors_dp_alpha}')
    
    plot_metric_vs_alpha(w_errors_dp_alpha.T, r'$\| w - w^* \|_2/\|w^*\|$', dp_noise, alpha_values, dimension, sigma, n)

# Run the tests with averaging
num_trials = 4

#Run tests d
run_tests_torrent_dp_d(num_trials)

#Run tests alpha
#run_tests_torrent_dp_alpha(num_trials)