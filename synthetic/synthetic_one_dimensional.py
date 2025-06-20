     
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
import os
import sys
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
    

def strategic_corruption_scaled(X_aug, Y, alpha, base_scaling=1.0, influence_factor=0.05):
    """
    Applies adversarial corruption to Y to shift it toward an adversarial model: Y = -2x -10.
    """
    d, n = X_aug.shape

    # Set adversarial model: Y = -5x -10
    w_corrupt = np.array([[4.5], [4.5]])  # Intercept = -10, Slope = -5

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
    

def generate_synthetic_dataset_one_d(n, sigma):
    """
    Creates a synthetic dataset where X and Y follow a linear model with a known intercept.
    """
    # Set the true model: Y = 5x + 4
    w_star = np.array([[15], [5]])  # Intercept = 4, Slope = 5

    # Generate features X
    X = 2*np.random.randn(1, n) + 2  # Shifted distribution

    # Add intercept term (row of ones)
    X_aug = np.vstack([np.ones((1, n)), X]) 

    # Generate noise
    epsilon = sigma * np.random.randn(n, 1)

    # Compute labels Y
    Y = np.dot(X_aug.T, w_star) + epsilon

    return X_aug, Y, w_star