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
    n , d = X.shape
    for i in range(n):
        X[i] = X[i]
    return X, y

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
        X[:, i] = X[:, i]
        #X.T[i] = X.T[i]/np.linalg.norm(X.T[i])
    if intercept:
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # shape (n, 2)
    n , d = X.shape
    for i in range(n):
        X[i] = X[i]/np.linalg.norm(X[i])
          
    # Response variable (y)
    y = merged_df[lab].values # shape (n,)
    y = y/10000
    

    return X,y


def run_tests_opportunity_atlas(X, y, beta, num_runs=2):
    dp_w = 0.1641895591
    dp_X = 7.55
    dp_Y = 7.55
    all_errors = []
    all_linear_preds = []
    all_torrent_preds = []

    for run in range(num_runs):
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_perc, random_state=42+run
        )
        X_train, X_test = X_train.T, X_test.T
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
        d, n = X_train.shape
         # Display results
        print("Shapes:")
        print("X_train:", X_train.shape)  # (features, samples)
        print("y_train:", y_train.shape)  # (samples, 1)
        print("X_test:", X_test.shape)  # (features, samples)
        print("y_test:", y_test.shape)  # (samples,)
        
        # Linear regression baseline
        w_linear = np.linalg.inv(X_train @ X_train.T)@(X_train @ y_train)
        #w_linear = LinearRegression(fit_intercept=False).fit(X_train.T, y_train).coef_
        #w_linear = w_linear.T
        norm_w = np.linalg.norm(w_linear)
        norm_w_inv = 1 / norm_w
        norm_w_inv = fxp(norm_w_inv)
        # Apply adversarial corruption
        Y_cor, _ = adversarial_corruption(X_train, y_train, alpha=beta, beta=10)

        # Split into parties + convert to fxp
        X_parts = split_matrix(X_train, 2, n)
        y_parts = split_matrix_Y(Y_cor, 2, n)
        X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)
        
        # Torrent ADMM (fxp)
        w_torrent, _ = torrent_admm_fxp_analyze_gauss(
            X_parts_fxp, y_parts_fxp, beta=beta,
            epsilon=0.1, rho=1, admm_steps=5, rounds=5,
            wstar=None, dp_noise_x=dp_X, dp_noise_y=dp_Y
        )
        '''

        w_torrent, _ = torrent_admm_dp(
            X_parts, y_parts, beta=beta,
            epsilon=0.1, rho=1, admm_steps=5, rounds=5,
            wstar=None, dp_X=dp_X, dp_y=dp_Y
        )
        '''
        

        # Predictions
        Y_pred_test_linear = np.matmul(X_test.T, w_linear)
        X_test = fxp(X_test)
        Y_pred_test_torrent = np.matmul(X_test.T, w_torrent)

        # Error
        w_linear = fxp(w_linear)
        error = fxp(np.linalg.norm(w_torrent - w_linear.T) )
        all_errors.append(error)
        print("OLS is:", w_linear)
        print("Torrent is:", w_torrent)
        print("Error is:", error.info())
        
        all_linear_preds.append(Y_pred_test_linear)
        all_torrent_preds.append(Y_pred_test_torrent)

    # === Scatter plot (averaged) ===
    sum_linear = np.zeros_like(all_linear_preds[0])
    sum_torrent = np.zeros_like(all_torrent_preds[0])
    for run in range(num_runs):
        sum_linear += all_linear_preds[run]
        sum_torrent += all_torrent_preds[run]

    avg_linear_preds = sum_linear / num_runs
    avg_torrent_preds = sum_torrent / num_runs
    plt.figure(figsize=(8, 6))
    #plt.ylim(0, 1)
    #plt.xscale("log")
    #plt.yscale("log")
    plt.scatter(y_test, avg_linear_preds, alpha=0.7, color='blue', marker='o', label='OLS ($\\beta=0$)')
    plt.scatter(y_test, avg_torrent_preds, alpha=0.9, color='violet', marker='v', label=f'Torrent ($\\beta={beta}$)')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs. Predicted (Average over 10 Runs), β={beta}")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"scatter_beta_{beta}_atlas.png", dpi=300, bbox_inches="tight")
    plt.show()

    return np.mean(all_errors), np.std(all_errors)


def plot_errors_vs_beta(X, y, betas, num_runs=2):
    avg_errors = []
    std_errors = []

    for beta in betas:
        avg_err, std_err = run_tests_opportunity_atlas(X, y, beta, num_runs=num_runs)
        avg_errors.append(avg_err)
        std_errors.append(std_err)

    # Line plot
    plt.figure(figsize=(8, 6))
    plt.plot(betas, avg_errors, marker='o', linestyle='-', color='purple', label="Error")
    plt.fill_between(betas,
                     np.array(avg_errors) - np.array(std_errors),
                     np.array(avg_errors) + np.array(std_errors),
                     alpha=0.2, color='purple')
    
    
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r'Error $\|w^* - \hat{w}\| / \|w^*\|$')
    plt.title("TRIP Error vs. β")
    plt.legend()
    plt.grid(False)
    plt.show()
    plt.savefig(f"error_beta_atlas.png", dpi=300, bbox_inches="tight")
    return avg_errors, std_errors


# === Example usage ===
test_perc = 0.2
script_dir = os.path.dirname(os.path.abspath(__file__))
'''
csv_names = [
    "shown_cz_d_med_hh_inc.csv",
    "shown_cz_d_frac_college_graduates.csv",
    "shown_cz_d_poverty_rate.csv",
    "shown_cz_d_share_non_white.csv"
]
suffixes = ["_hh", "_college", "_poverty", "_population"]
feat = [
    'Change_in_Fraction_of_Non-White_Population',
    'Change_in_Fraction_of_College_Graduates',
    'Change_in_Poverty_Rate'
]
lab = 'Change_in_Median_Household_Income'
'''

csv_names = [
    "shown_cz_kfr_rP_gP_pall.csv",
    "shown_cz_hours_yr_rP_gP_pall.csv",
    "shown_cz_wageflex_rank_rP_gP_pall.csv",
]
suffixes = ["_Week", "_Wage", "_poverty", "_population"]
feat = [
    'Hours_Worked_Per_Week_at_Age_35_rP_gP_pall',
    'Hourly_Wage_$/hour_at_Age_35_rP_gP_pall',
]
lab = 'Household_Income_at_Age_35_rP_gP_pall'
X, y = read_from_file("cz_outcomes.csv"
                      ,[
                        #'par_rank_pooled_pooled_mean',
                         #'hours_wk_pooled_pooled_mean',
                         'work_26_pooled_pooled_mean'
                         ],
                         'working_pooled_pooled_mean', test_perc, intercept=0)
#X, y = read_from_multiple_files(csv_names, feat, lab, suffixes, test_perc=test_perc, intercept=0)

#betas = [0.1, 0.15, 0.2, 0.25]
betas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
avg_errors, std_errors = plot_errors_vs_beta(X, y, betas, num_runs=10)

print("Betas:", betas)
print("Average Errors:", avg_errors)
print("STD Errors:", std_errors)
