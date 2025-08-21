from ucimlrepo import fetch_ucirepo
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
from synthetic.strategic_corruptions import  strategic_corruption_scaled, adversarial_corruption
from synthetic.toy_dataset import generate_synthetic_dataset
from torrent.torrent import  torrent_admm, split_matrix, split_matrix_Y, torrent_admm_dp
from torrent.torrent_fxp import torrent_admm_fxp, torrent_admm_fxp_analyze_gauss
from fixed_point.fixed_point_helpers import *
from plots.plots_fxp import plot_metric_vs_alpha_fxp, plot_metric_vs_d_fxp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed

def run_tests_fxp_real():
    # Fetch the dataset
    test_perc = 0.2
    bike_sharing = fetch_ucirepo(id=275)  # Bike Sharing dataset

    # Extract features and targets
    X_df = bike_sharing.data.features
    y_df = bike_sharing.data.targets
    n ,d = X_df.shape
    # Use only 'temp' as feature
    X = X_df[['temp']].values  # shape (n_samples, 1)

    # Response: bike count, normalized to [0,1]
    if 'cnt' in y_df.columns:
        y_raw = y_df['cnt']
    else:
        y_raw = y_df.iloc[:,0]
    y = (y_raw )
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_perc, random_state=42
    )

    # Convert y to numpy arrays and flatten (1D) for sklearn
    y_train = np.array(y_train).ravel()
    y_test  = np.array(y_test).ravel()

    # Transpose X to match your internal pipeline (d × n)
    X_train_t = X_train.T  # shape (d, n_train)
    y_train_t = y_train.reshape(1, -1)  # shape (1, n_train)

    X_test_t  = X_test.T   # shape (d, n_test)

    print("X_train shape (d x n):", X_train_t.shape)
    print("y_train shape (1 x n):", y_train.shape)
    print("X_test shape (d x n):", X_test_t.shape)
    print("y_test shape (1 x n):", y_test.shape)

    # Fit Huber Regressor using sklearn (expects n_samples x n_features)
    huber = LinearRegression().fit(X_train, y_train)
    w_huber = huber.coef_

    # Compute predictions on test set
    Y_pred_test_huber = X_test @ w_huber  # shape (n_samples,)

    # Compute error metrics
    corange = y_train.max() - y_train.min() 
    mae_huber = mean_absolute_error(y_test, Y_pred_test_huber)
    mse_huber = mean_squared_error(y_test, Y_pred_test_huber)
    rmse_huber = (mse_huber ** 0.5) / corange  # normalized RMSE

    print("mae:", mae_huber)
    print("mse:", mse_huber)
    print("rmse:", rmse_huber)
    # corruptions
    train_size =  n - n*test_perc
    beta = 0.2
    Y_cor, _ = adversarial_corruption(X_train_t, y_train_t.T, alpha=0.1, beta=100)
    X_parts = split_matrix(X_train_t, 2, train_size)
    y_parts = split_matrix_Y(Y_cor, 2, train_size)
    w_torrent, _= torrent_admm(X_parts, y_parts, beta=0.4, epsilon=0.1, rho=1, admm_steps=5, rounds=5)

    # Compute predictions on test set
    Y_pred_test_torrent = X_test @ w_torrent  # shape (n_samples,)

    # Compute error metrics
    corange = y_train.max() - y_train.min() 
    mae_torrent = mean_absolute_error(y_test, Y_pred_test_torrent)
    mse_torrent = mean_squared_error(y_test, Y_pred_test_torrent)
    rmse_torrent = (mse_torrent ** 0.5) / corange  # normalized RMSE
    print("mae:", mae_torrent)
    print("mse:" ,mse_torrent)
    print("rmse:", rmse_torrent)
    w_inv = np.linalg.norm(w_huber)
    print("error:", np.linalg.norm(w_torrent[0] - w_huber)*w_inv)
    
      
    plt.figure(figsize=(8, 6))
    #plt.scatter(y_test, Y_pred_test_huber, alpha=0.7, color='blue', marker='o', label=f'OLS ($\\beta=0$')
    plt.scatter(y_test, Y_pred_test_torrent, alpha=0.5, color='violet', marker='v', label=f'Torrent ($\\beta={beta}$)')


    plt.xlabel("Actual Student Attendance")
    plt.ylabel("Predicted Student Attendance")
    plt.title("Actual vs. Predicted Student Attendance (Corrupted vs. Clean Dataset)")
    plt.legend()
    plt.show()

run_tests_fxp_real()
