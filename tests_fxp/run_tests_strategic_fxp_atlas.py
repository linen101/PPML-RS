import pandas as pd
import numpy as np
import os, sys
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from synthetic.strategic_corruptions import  strategic_corruption_scaled, adversarial_corruption
from synthetic.toy_dataset import generate_synthetic_dataset
from torrent.torrent import  torrent_admm, split_matrix, split_matrix_Y, torrent_admm_dp
import matplotlib.pyplot as plt

def run_tests_opportunity_atlas(csv_path):
    """
    Linear regression (Huber) predicting relative college attendance
    from parent's income percentile using Opportunity Atlas CSV.
    """
    test_perc = 0.5
    
    # Load CSV
    df = pd.read_csv(csv_path)
    # Drop rows with missing X or y
    #df= df.dropna(subset=['par_income_bin', 'attend_level_sat', 'rel_apply_sat', 'rel_apply', 'rel_att_cond_app_sat', 'rel_attend'])
    df= df.dropna(subset=[ 'par_rank_black_pooled_mean', 'frac_below_median_black_pooled', 'frac_years_xw_black_pooled'])
    # Feature: parent's income percentile
    # Normalize and add intercept column (bias)
    #X = df[['par_income_bin']].values
    X = df[['par_rank_black_pooled_mean', 'frac_below_median_black_pooled', 'frac_years_xw_black_pooled']].values   # normalize to [0,1]
    X = np.hstack([X, np.ones((X.shape[0], 1))])   # add intercept term    
    # Response: relative college attendance
    
    #y = df[['rel_attend']].values
    y = df['working_black_pooled_p50'].values # normalize to [0,1]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_perc, random_state=42
    )
    
    # Convert y to numpy arrays (flatten for sklearn)
    y_train = np.array(y_train).ravel()
    y_test  = np.array(y_test).ravel()
    
    # Transpose X to match d × n convention
    X_train_t = X_train.T  # shape (d, n_train)
    y_train_t = y_train.reshape(1, -1)  # shape (1, n_train)
    X_test_t  = X_test.T   # shape (d, n_test)
    
    print("X_train shape (d x n):", X_train_t.shape)
    print("y_train shape (1 x n):", y_train.shape)
    print("X_test shape (d x n):", X_test_t.shape)
    print("y_test shape (1 x n):", y_test.shape)
    
    # Fit Huber Regressor
    huber = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    w_huber = huber.coef_
    
    # Predictions on test set
    Y_pred_test_huber = X_test @ w_huber  # shape (n_samples,)
    
    # Compute error metrics
    corange = y_train.max() - y_train.min()
    mae_huber = mean_absolute_error(y_test, Y_pred_test_huber)
    mse_huber = mean_squared_error(y_test, Y_pred_test_huber)
    rmse_huber = (mse_huber ** 0.5) / corange  # normalized RMSE
    
    print("mae:", mae_huber)
    print("mse:", mse_huber)
    print("rmse:", rmse_huber)
    print("Huber coefficients:", w_huber)
    w_inv = 1 / np.linalg.norm(w_huber)
    
    # corruptions
    n ,d = X.shape
    train_size =  n - n*test_perc
    beta = 0
    Y_cor, _ = adversarial_corruption(X_train_t, y_train_t.T, alpha=beta, beta=100)
    X_parts = split_matrix(X_train_t, 2, train_size)
    y_parts = split_matrix_Y(Y_cor, 2, train_size)
    w_torrent, _= torrent_admm_dp(X_parts, y_parts, beta=beta, epsilon=0.1, rho=1, dp_epsilon=0.1, dp_delta=-0.001, admm_steps=5, rounds=5)

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
    print("Torrent coefficients:", w_torrent)
    print("error:", np.linalg.norm(w_torrent[0] - w_huber)*w_inv)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, Y_pred_test_huber, alpha=0.7, color='blue', marker='o', label=f'OLS ($\\beta=0$')
    plt.scatter(y_test, Y_pred_test_torrent, alpha=0.5, color='violet', marker='v', label=f'Torrent ($\\beta={beta}$)')
    plt.xlabel("Actual ")
    plt.ylabel("Predicted ")
    plt.title("Actual vs. Predicted (Corrupted vs. Clean Dataset)")
    plt.legend()
    plt.show()


   

# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of your script
csv_path = os.path.join(script_dir, "cz_outcomes.csv")
run_tests_opportunity_atlas(csv_path)
