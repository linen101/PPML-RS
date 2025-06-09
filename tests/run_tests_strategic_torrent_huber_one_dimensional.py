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
from torrent.torrent import torrent_dp  # Import torrent module
#from admm.admm import admm
def generate_synthetic_dataset_one_d(n, sigma):
    """
    Creates a synthetic dataset where X and Y follow a linear model with a known intercept.
    """
    # Set the true model: Y = 5x + 4
    w_star = np.array([[4], [5]])  # Intercept = 4, Slope = 5

    # Generate features X
    X = np.random.randn(1, n) + 1.2 # Shifted distribution

    # Add intercept term (row of ones)
    X_aug = np.vstack([np.ones((1, n)), X]) 

    # Generate noise
    epsilon = sigma * np.random.randn(n, 1)

    # Compute labels Y
    Y = np.dot(X_aug.T, w_star) + epsilon

    return X_aug, Y, w_star

def strategic_corruption_scaled(X_aug, Y, alpha, base_scaling=1.0, influence_factor=0.5):
    """
    Applies adversarial corruption to Y to shift it toward an adversarial model: Y = -2x -10.
    """
    d, n = X_aug.shape

    # Set adversarial model: Y = 5x -15
    w_corrupt = np.array([[-15], [-2]])  # Intercept = -15, Slope = -2

    # Compute ideal labels under the adversarial model
    Y_corrupt_ideal = np.dot(X_aug.T, w_corrupt)

    # Compute L2 distances from the adversarial model
    corruption_magnitude = np.abs(Y - Y_corrupt_ideal)

    # Select alpha * n samples that need the least change
    num_corrupted_samples = int(alpha * n)
    corrupted_indices = np.argsort(corruption_magnitude.flatten())[:num_corrupted_samples]

   # Adaptive scaling factor: More aggressive when alpha is small
    scaling_factor = (base_scaling + (influence_factor / alpha))

    # Apply corruption with adaptive amplification
    Y_corrupted = Y.copy()
    Y_adversarial = Y_corrupt_ideal[corrupted_indices] + scaling_factor * (Y_corrupt_ideal[corrupted_indices] - Y[corrupted_indices])
    Y_corrupted[corrupted_indices] = Y_adversarial

    return Y_corrupted, corrupted_indices, w_corrupt


def gaussian_mechanism(X, y, epsilon, delta, B_y, w_torrent):
    """
    Applies the Gaussian mechanism to least squares regression:
    M(X, y) = w_torrent + Gaussian noise
    """
    d,n = X.shape
    Sigma = X @ X.T
    # Step 2: Estimate lambda_min(Sigma)
    lambda_min = np.min(eigh(Sigma)[0])  # smallest eigenvalue

    # Step 3: Compute L2 sensitivity
    Delta_f = ((2 * np.sqrt(d) * B_y) / lambda_min) + (2 * d * np.sqrt(n * d) * B_y * np.sqrt(d)) / (lambda_min ** 2)

    # Step 4: Compute noise scale
    sigma = (np.sqrt(2 * np.log(2 / delta)) / epsilon) * Delta_f

    # Step 5: Sample noise from N(0, I_d)
    noise = np.random.randn(d, 1)

    # Step 6: Return private estimator
    beta_private = w_torrent + sigma * noise

    return beta_private

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
        #huber = HuberRegressor(fit_intercept=False).fit(X_aug.T, Y_corrupted.ravel())
        #w_huber_intercept = huber.coef_.reshape(-1, 1)  # Already includes intercept and slope

        #w_huber_intercept = np.array([[huber.intercept_], [huber.coef_[0]]])
        #y_vals_huber = x_vals_aug.T @ w_huber_intercept
        #plt.plot(x_vals.T, y_vals_huber, label="Huber Regression", color="slategrey", linestyle="dashdot")

        # Torrent Model
        w_torrent, iter_count = torrent_dp(X_aug, Y_corrupted, beta, epsilon, dp_epsilon, dp_delta, max_iters)
        #w_dp = gaussian_mechanism(X_aug, Y_corrupted, dp_epsilon, dp_delta, dp_B_y, w_torrent)
        #print(f"w_dp shape: {w_dp.shape}")  # Should be (2, 1)
        # Plot dp and normal
        y_vals_torrent = x_vals_aug.T @ w_torrent
        #y_vals_dp = x_vals_aug.T @ w_dp

        plt.plot(x_vals.T, y_vals_torrent, label='Torrent Regression', color="magenta", linestyle="dashdot", linewidth=1.5)
        #plt.plot(x_vals.T, y_vals_dp, label='DP Torrent', color="red", linestyle="dashed", linewidth=1.5)


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
