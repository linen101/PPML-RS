# try some dp noise on 1d data.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import HuberRegressor, LinearRegression
import os
import sys
from numpy.linalg import eigh, inv


module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from torrent.torrent import torrent_intermediate, torrent # Import torrent module
from synthetic.synthetic_one_dimensional import generate_synthetic_dataset_one_d, strategic_corruption_scaled
def run_tests_dp_1d_avg_ols(num_trials):
    # Generate Dataset with Intercept
    n = 100 # Number of samples
    sigma = 0.1
    alpha = 0.1  # 10% of data points are corrupted
    beta = alpha + 0.1
    epsilon = 0.01
    max_iters = 10
    # assume y values bounded in [-1, 1]
    
    predictions_ols = np.zeros(n)  # e.g. shape: (100)
    for _ in range(num_trials):
        X_aug, Y, w_star = generate_synthetic_dataset_one_d(n, sigma)

        # Apply strategic corruption
        Y_corrupted, corrupted_indices, w_corrupt = strategic_corruption_scaled(X_aug, Y, alpha)
        
        # generate test values
        x_vals = np.linspace(X_aug[1, :].min(), X_aug[1, :].max(), n).reshape(1, -1)
        x_vals_aug = np.vstack([np.ones((1, x_vals.shape[1])), x_vals])  # Add intercept term

        # honest Model (True)
        y_vals_true = x_vals_aug.T @ w_star
        
        ols = LinearRegression(fit_intercept=False).fit(X_aug.T, Y.ravel())
        w_ols = ols.coef_.reshape(-1, 1)  # Already includes intercept and slope
        #w_huber_intercept = np.array([[huber.intercept_], [huber.coef_[0]]])
        y_vals_ols = x_vals_aug.T @ w_ols  
        predictions_ols += y_vals_ols.ravel()
    
    
    #get average ols
    predictions_ols /= num_trials
    
    
    
    # create figure
    plt.figure(figsize=(8, 6))

    # original data (non-corrupted)
    plt.scatter(X_aug[1, :], Y, label="Original Data", color="black", alpha=0.8)

    #highlight corrupted locations
    plt.scatter(X_aug[1, corrupted_indices], Y[corrupted_indices], label="Corruption Locations",
                color="darkmagenta", marker="v", edgecolors="black", s=150, alpha=0.8)

    # corrupted data points
    plt.scatter(X_aug[1, corrupted_indices], Y_corrupted[corrupted_indices], label="Strategically Corrupted Points",
                color="blueviolet", marker="x", s=100)
    
    plt.plot(x_vals.T, y_vals_true, label=f'True Model', color="blue", linestyle="solid")
    
    # adversarial Model (-2x -10)
    y_vals_adv = x_vals_aug.T @ w_corrupt
    plt.plot(x_vals.T, y_vals_adv, label=f'Adversarial Model $(4.5x + 4.5)$', color="red", linestyle="dotted")
    
    #ols
    plt.plot(x_vals.T, predictions_ols, label="OLS Regression", color="mediumpurple", linestyle="dashdot")
    
    # Labels and legend
    plt.xlabel("Feature X")
    plt.ylabel("Label Y")
    plt.legend()
    #plt.title(f"Strategic Corruption in 1D $(n= {n}, \\beta = {alpha})$")
    plt.grid(False)

    # Show plot
    plt.show()
    plt.close()


# Run the tests with averaging
num_trials = 10

#Run OLS
run_tests_dp_1d_avg_ols(num_trials)
