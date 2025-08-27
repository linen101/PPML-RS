# try some dp noise on 1d data.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
import os
import sys

from numpy.linalg import eigh, inv


module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from torrent.torrent import torrent # Import torrent module
from synthetic.synthetic_one_dimensional import generate_synthetic_dataset_one_d, strategic_corruption_scaled

def main():
    # Generate Dataset with Intercept
    n = 50  # Number of samples
    sigma = 1
    alpha = 0.2  # 10% of data points are corrupted
    beta = alpha + 0.1
    epsilon = 0.1
    max_iters = 20
    dp_epsilon = 0.1
    dp_delta = 1e-5
    dp_B_y = 1
    # assume y values bounded in [-1, 1]
    for _ in range(10):
        X_aug, Y, w_star = generate_synthetic_dataset_one_d(n, sigma)

        # Apply strategic corruption
        Y_corrupted, corrupted_indices, w_corrupt = strategic_corruption_scaled(X_aug, Y, alpha)
        """
        print(f"X_aug shape: {X_aug.shape}")  # Should be (2, n)
        print(f"w_star shape: {w_star.shape}")  # Should be (2, 1)
        print(f"w_corrupt shape: {w_corrupt.shape}")  # Should be (2, 1)
        print(f"Y_corrupted shape: {Y_corrupted.shape}")  # Should be (n, 1)
        """
        
        # create figure
        plt.figure(figsize=(8, 6))

        # original data (non-corrupted)
        plt.scatter(X_aug[1, :], Y, label="Original Data", color="black", alpha=0.8)

        # highlight corrupted locations
        plt.scatter(X_aug[1, corrupted_indices], Y[corrupted_indices], label="Corruption Locations",
                    color="darkmagenta", marker="v", edgecolors="black", s=150, alpha=0.8)

        # corrupted data points
        plt.scatter(X_aug[1, corrupted_indices], Y_corrupted[corrupted_indices], label="Strategically Corrupted Points",
                    color="blueviolet", marker="x", s=100)

        # compute least squares estimate (handling intercept correctly)
        w_approx = np.linalg.pinv(X_aug.T) @ Y_corrupted  # FIXED

        # generate test values
        x_vals = np.linspace(X_aug[1, :].min(), X_aug[1, :].max(), 100).reshape(1, -1)
        x_vals_aug = np.vstack([np.ones((1, x_vals.shape[1])), x_vals])  # Add intercept term

        # honest Model (True)
        y_vals_true = x_vals_aug.T @ w_star
        #dp_B_y = max((abs(y_vals_true)))
        plt.plot(x_vals.T, y_vals_true, label=f'True Model $(5x + 4)$', color="blue", linestyle="solid")

        # corrupted model (OLS Fit)
        #y_vals_corrupt = x_vals_aug.T @ w_approx
        #plt.plot(x_vals.T, y_vals_corrupt, label="OLS Fit (Corrupted Data)", color="darkviolet", linestyle="dotted")

        # adversarial Model (-2x -10)
        y_vals_adv = x_vals_aug.T @ w_corrupt
        plt.plot(x_vals.T, y_vals_adv, label=f'Adversarial Model $(-2x - 15)$', color="mediumpurple", linestyle="dotted")

        # huber Regression (handling intercept)
        huber = HuberRegressor(fit_intercept=False).fit(X_aug.T, Y_corrupted.ravel())
        w_huber_intercept = huber.coef_.reshape(-1, 1)  # Already includes intercept and slope
        #w_huber_intercept = np.array([[huber.intercept_], [huber.coef_[0]]])
        y_vals_huber = x_vals_aug.T @ w_huber_intercept
        plt.plot(x_vals.T, y_vals_huber, label="Huber Regression", color="slategrey", linestyle="dashdot")

        # Torrent Model
        w_torrent, iter_count = torrent(X_aug, Y_corrupted, beta, epsilon, max_iters)
        y_vals_torrent = x_vals_aug.T @ w_torrent
        plt.plot(x_vals.T, y_vals_torrent, label='Torrent Regression', color="magenta", linestyle="dashdot", linewidth=1.5)

        # Labels and legend
        plt.xlabel("Feature X")
        plt.ylabel("Label Y")
        plt.legend()
        plt.title(f"Strategic Corruption in 1D $(\\beta = {alpha})$")
        plt.grid(False)

        # Show plot
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
