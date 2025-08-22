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
from torrent.torrent_fxp import torrent_admm_fxp, torrent_admm_fxp_analyze_gauss
from fixed_point.fixed_point_helpers import *
import matplotlib.pyplot as plt

def read_from_file(csv_name, feat, lab, test_perc, intercept=0):
    csv_path = os.path.join(script_dir, csv_name)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=feat + [lab])
    X = df[feat].values   # normalize to [0,1]
    y = df[[lab]].values
    if intercept:
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # shape (n, 2)
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_perc, random_state=42
    )
    # Transpose X to match d × n convention
    X_train = X_train.T  # shape (d, n_train)
    X_test  = X_test.T   # shape (d, n_test)
    y_train = y_train.reshape(-1, 1)  # shape (1, n_train)
    y_test = y_test.reshape(-1, 1)  # shape (1, n_train)
    return X_train, X_test, y_train, y_test

def read_from_multiple_files(csv_names, feat, lab, suffixes, test_perc, intercept=0):
    
    csv_path = np.empty(len(csv_names), dtype=object)
    df = np.empty(len(csv_names), dtype=object)
    for i, name in enumerate(csv_names):
        csv_path[i] = os.path.join(script_dir, name)
        df[i] = pd.read_csv(csv_path[i])
        
    merged_df = df[0]
    for i, name in enumerate(df):
        merged_df = pd.merge(merged_df, df[i+1], on='cz', suffixes=(suffixes[i],suffixes[i+1]))
        if (i == len(df)-2):
            break

    merged_df = merged_df.dropna(subset=feat + [lab])
    # Feature matrix (X)
    X = merged_df[feat].values  # shape (n, 1)
    for i, name in enumerate(feat):
        X[:, i] = X[:, i] / 10
    if intercept:
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # shape (n, 2)

    # Response variable (y)
    y = merged_df[lab].values # shape (n,)
    y = y/10000
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_perc, random_state=42
    )
    # Transpose X to match d × n convention
    X_train = X_train.T  # shape (d, n_train)
    X_test  = X_test.T   # shape (d, n_test)
    y_train = y_train.reshape(-1, 1)  # shape (1, n_train)
    y_test = y_test.reshape(-1, 1)  # shape (1, n_train)

    return X_train, X_test, y_train, y_test


def run_tests_opportunity_atlas(X_train, X_test, y_train, y_test):
    num_runs = 5   # how many independent runs
    beta = 0.2
    dp_w = 0

    # store errors & coefficients for plotting
    all_linear_preds = []
    all_torrent_preds = []
    all_errors = []

    d, n = X_train.shape

    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")

        # Fit Linear Model (OLS in fixed point)
        w_linear = fxp(np.empty((d,1)))
        w_linear = np.matmul(np.linalg.inv(np.matmul(fxp(X_train),fxp(X_train.T))), np.matmul(fxp(X_train),fxp(y_train))) 
        w_linear_dp = w_linear + dp_w*np.random.randn(d,1)

        norm_w = np.linalg.norm(w_linear)
        norm_w_inv = 1 / norm_w

        # Predictions on test set
        Y_pred_test_linear = np.matmul(X_test.T,w_linear)  # shape (n_samples,)

        # Apply adversarial corruption
        Y_cor, _ = adversarial_corruption(X_train, y_train, alpha=beta, beta=10)

        # Split into parties
        X_parts = split_matrix(X_train, 2, n)
        y_parts = split_matrix_Y(Y_cor, 2, n)
        X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)

        # Run Torrent
        w_torrent, _ = torrent_admm_fxp(X_parts_fxp, y_parts_fxp, beta=beta, epsilon=0.1, rho=1, admm_steps=5, rounds=5, wstar=None, dp_w=0)
        # Compute predictions on test set
        Y_pred_test_torrent = np.matmul(X_test.T, w_torrent)  # shape (n_samples,)
        # Log
        print("Linear coefficients:", w_linear.T)
        print("Torrent coefficients:", w_torrent.T)
        error = np.linalg.norm(w_torrent - w_linear)
        print("Error:", error)

        # save results
        all_linear_preds.append(Y_pred_test_linear)
        all_torrent_preds.append(Y_pred_test_torrent)
        all_errors.append(error)


    # === Plotting after all runs ===
    plt.figure(figsize=(8, 6))

    # Plot OLS
    for run in range(num_runs):
        plt.scatter(y_test, all_linear_preds[run],
                    alpha=0.3, color='blue', marker='o',
                    label='OLS ($\\beta=0$)' if run == 0 else "")

    # Plot Torrent
    for run in range(num_runs):
        plt.scatter(y_test, all_torrent_preds[run],
                    alpha=0.3, color='violet', marker='v',
                    label=f'Torrent ($\\beta={beta}$)' if run == 0 else "")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs. Predicted (Over {num_runs} Runs, Corrupted vs. Clean Dataset)")
    plt.legend()
    plt.show()

    # Error summary
    print("\nAverage error over runs:", np.mean(all_errors))
    print("Std of error:", np.std(all_errors))
   

# Example usage
test_perc = 0.4
script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of your script

#X_train, X_test, y_train, y_test = read_from_file("cz_outcomes.csv",['par_rank_pooled_pooled_mean', 'frac_below_median_pooled_pooled'], 'working_pooled_pooled_p50', test_perc)
#run_tests_opportunity_atlas(X_train, X_test, y_train, y_test)

csv_names = ["shown_cz_d_frac_college_graduates.csv", "shown_cz_d_med_hh_inc.csv", "shown_cz_d_poverty_rate.csv", "shown_cz_d_share_non_white.csv"]
suffixes = ["_college",  "_hh", "_poverty", "_population"]
feat = ['Change_in_Fraction_of_College_Graduates', 'Change_in_Median_Household_Income', 'Change_in_Poverty_Rate']
lab = 'Change_in_Fraction_of_Non-White_Population'
X_train, X_test, y_train, y_test = read_from_multiple_files(csv_names, feat, lab, suffixes, test_perc=0.2, intercept=0)
run_tests_opportunity_atlas(X_train, X_test, y_train, y_test)

