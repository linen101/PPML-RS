
from sklearn.preprocessing import normalize
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing,fetch_species_distributions, fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import pandas as pd

module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
 
from synthetic.strategic_corruptions import  adversarial_corruption
from synthetic.toy_dataset import generate_synthetic_dataset, corrupt_dataset
from plots.plots import plot_regression_errors_n, plot_regression_errors_d, plot_iterations_n, plot_iterations_d
from torrent.torrent import torrent, torrent_admm_dp, split_matrix, split_matrix_Y, torrent_admm_ag
from torrent.torrent_fxp import split_matrix_fxp, torrent_admm_fxp_analyze_gauss
from fixed_point.fixed_point_helpers import *
from decimal import *
import seaborn as sns
from sklearn.linear_model import HuberRegressor, LinearRegression
from realdata.real_data import load_and_process_gas_sensor_data, load_and_process_energy_data
getcontext().prec = 4
markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed


# -------------------
# Run function
# -------------------
def run(X_train, Y_train, X_test, Y_test, beta):

    dp_X = 7.55
    dp_Y = 7.55
    

    # Normalize rows of X
    d, n = X_train.shape
    
    # OLS solution
    w_linear = np.linalg.inv(X_train @ X_train.T) @ (X_train @ Y_train)
    norm_w = (np.linalg.norm(w_linear))
    norm_w_inv = 1 / norm_w
    #norm_w_inv = fxp(norm_w_inv)

    # Apply adversarial corruption
    Y_cor, _ = adversarial_corruption(X_train, Y_train, alpha=beta, beta=1)

    # Split into parties
    X_parts = split_matrix(X_train, 2, n)
    y_parts = split_matrix_Y(Y_cor, 2, n)
    X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)

    # TORRENT regression
    
    w_torrent, _ = torrent_admm_fxp_analyze_gauss(
        X_parts_fxp, y_parts_fxp, beta=beta, epsilon=0.1, rho=1, admm_steps=5, rounds=5, wstar=None, dp_noise_x=dp_X, dp_noise_y=dp_Y)
    
    '''
    w_torrent, _ = torrent_admm_ag(
        X_parts, y_parts, beta=beta,
        epsilon=0.1, rho=1, admm_steps=5, rounds=5,
        wstar=None, dp_X=dp_X, dp_y=dp_Y
    )
    '''
    
    #w_torrent, _ = torrent(X_train, Y_cor, beta, epsilon=0.1, max_iters=5)
    # Predictions
    Y_pred_test_linear = X_test.T @ w_linear
    X_test = fxp(X_test)
    Y_pred_test_torrent = np.matmul(X_test.T,w_torrent)

    # Error
    fxp(w_linear)
    error = fxp(np.linalg.norm(w_torrent - w_linear))*norm_w_inv        #cast to fxo through norm
    print("OLS is:", w_linear)
    print("Torrent is:", w_torrent)
    print("Error is:", error.info())
    return error, Y_pred_test_linear, Y_pred_test_torrent

# -------------------
# Experiment
# -------------------
def run_experiment(betas, runs=1):
    avg_errors, std_errors = [], []
    all_linear_preds, all_torrent_preds = {}, {}

    for beta in betas:
        run_errors, linear_preds, torrent_preds = [], [], []

        for irun in range(runs):
            
            # -------------------
            # Load dataset
            # -------------------
            X_train, X_test, Y_train, Y_test = load_and_process_energy_data(test_percentage=0.2, i=irun)
            X_train = X_train.T
            X_test = X_test.T
            Y_train = Y_train.reshape(-1, 1)
            Y_test = Y_test.reshape(-1, 1)
            X_train = normalize(X_train, axis=0)
            X_test = normalize(X_test, axis=0)
            Y_train = normalize(Y_train, norm='max', axis=0) 
            Y_test = normalize(Y_test, norm='max', axis=0)
            # -------------------
            # experiment
            # -------------------
            error, Y_pred_lin, Y_pred_tor = run(X_train, Y_train, X_test, Y_test, beta)
            run_errors.append(error)
            linear_preds.append(Y_pred_lin)
            torrent_preds.append(Y_pred_tor)

        avg_errors.append(np.mean(run_errors))
        print("Average Errors:", avg_errors)
        std_errors.append(np.std(run_errors))

        # Average predictions
        all_linear_preds[beta] = np.mean(linear_preds, axis=0)
        all_torrent_preds[beta] = np.mean(torrent_preds, axis=0)

        
    # 2. Scatter plots (OLS vs TORRENT)
    for beta in betas:
        plt.figure(figsize=(14, 8))
        plt.scatter(Y_test, all_linear_preds[beta], alpha=0.7, color='blue', marker='o', label='OLS ($\\beta=0$)')
        plt.scatter(Y_test, all_torrent_preds[beta], alpha=0.9, color='violet', marker='v', label=f'TRIP* ($\\beta={beta}$)')
        plt.xlabel("Actual", fontsize=25)
        plt.ylabel("Predicted", fontsize=25)
        plt.title(f"Actual vs. Predicted (Average over 10 Runs), β={beta}", fontsize=25)
        plt.legend( fontsize=25)
        plt.grid(False)
        # Save figure (PNG, high resolution)
        plt.savefig(f"scatter_beta_{beta}_energy.png", dpi=300, bbox_inches="tight")
        plt.show()
    return avg_errors, std_errors, all_linear_preds, all_torrent_preds

# -------------------
# Run + Plots
# -------------------
betas = [0.1,  0.2, 0.3,  0.4]
runs = 2
avg_errors, std_errors, avg_linear_preds, avg_torrent_preds = run_experiment(
     betas, runs=runs
)

# 1. Error vs Beta with std shading
plt.figure(figsize=(14, 8))
plt.plot(betas, avg_errors, marker='o', linestyle='-', color='purple', label="Error")
plt.xlabel(r"$\beta$", fontsize=25)
plt.ylabel(r'Error $\|w^* - \hat{w}\| / \|w^*\|$', fontsize=25)
plt.title("TORRENT Error vs. β", fontsize=25)
plt.legend(fontsize=25)
plt.grid(False)
plt.savefig(f"error_beta_energy.png", dpi=300, bbox_inches="tight")
plt.show()


print("Averaged Errors:", avg_errors)
