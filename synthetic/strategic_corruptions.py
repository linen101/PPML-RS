     
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
import os
import sys
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from torrent.torrent import torrent, torrent_ideal

markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed

def generate_synthetic_dataset_one_d(n, d, sigma):
    """
    Creates a synthetic dataset where X and Y have linear dependence.
    """
    # True model weights
    w_star = np.random.randn(1, 1)

    # Generate sub-Gaussian features X
    #shape_param = 0.5  # Weibull shape parameter
    #X = np.sign(np.random.randn(d, n)) * np.random.weibull(shape_param, size=(d, n))
    X = np.random.randn(1, n) + 10
    # Generate noise
    epsilon = sigma * np.random.randn(n, 1)

    # Compute labels Y
    Y = np.dot(X.T, w_star) + epsilon

    return X, Y, w_star


def generate_synthetic_dataset(n, d, sigma, test_percentage=0.2):
    """
    Creates a toy dataset.

    Parameters:
    - n: Number of samples
    - d: Dimensions of the samples
    - sigma: Variance of sub-Gaussian Noise
    - test_percentage: Percentage of data to be used for testing (default is 0.2)
    
    Returns:
    - X_train: Training feature vectors matrix with dimensions [d, n_train]
    - Y_train: Training labels matrix with dimensions [n_train, 1]
    - X_test: Testing feature vectors matrix with dimensions [d, n_test]
    - Y_test: Testing labels matrix with dimensions [n_test, 1]
    - w_star: True model matrix with dimensions [d, 1]
    """
    # Generate true model w* as a matrix with dimensions [d, 1]
    # The ciefficients w_{star_i} are sampled from a Gaussian distribution with standard deviation
    w_star = np.random.randn(d, 1) 

    # Generate feature vectors X with dimensions [d, n] 
    # X are sampled from a Gaussian distribution with standard deviation
    X = np.random.randn(d, n) + 10

    # Generate Custom Heavy-tailed Sub-Gaussian Distribution
    #    This uses a bounded, symmetrized Weibull distribution (which can be sub-Gaussian).
    #shape_param = 0.5  # Weibull shape parameter (controls tail behavior)   
    #X = np.sign(np.random.randn(d, n)) * np.random.weibull(shape_param, size=(d, n))

    # Generate sub-Gaussian noise epsilon
    epsilon = sigma * np.random.randn(n, 1)

    # Generate labels Y using the model and noise
    Y = np.dot(X.transpose(), w_star) + epsilon

    # Split data into training and testing sets
    n_test = int(n * test_percentage)
    n_train = n - n_test

    X_train = X[:, :n_train]
    Y_train = Y[:n_train, :]

    X_test = X[:, n_train:]
    Y_test = Y[n_train:, :]

    return X_train, Y_train, X_test, Y_test, w_star

def rotate_w_partial(w, num_axes=10):
    d = w.shape[0]
    assert num_axes <= d, "num_axes must be <= dimension of w"

    # Select `num_axes` random axes
    indices = np.random.choice(d, num_axes, replace=False)

    # Generate an orthogonal matrix for only those axes
    Q = np.eye(d)
    sub_matrix = np.random.randn(num_axes, num_axes)
    Q[np.ix_(indices, indices)], _ = np.linalg.qr(sub_matrix)

    # Apply rotation
    return Q @ w



"""_rotation strategy below exmple in 2d_
    theta = np.pi / 6  # Rotate by 30 degrees
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta), np.cos(theta)]])
w_corrupt = rotation_matrix @ w_star  # Applies a 2D rotation (extendable to higher dimensions)

"""
def rotate_w_arbitrary(w ):
    """
    Rotates the vector w in arbitrary dimensions by theta radians 
    around a randomly chosen orthogonal axis.

    Parameters:
    - w: Original weight vector of shape (d, 1)
    - theta: Rotation angle in radians

    Returns:
    - w_rotated: The rotated weight vector of shape (d, 1)
    """
    d = w.shape[0]

    # Generate a random orthogonal matrix via QR decomposition
    random_matrix = np.random.rand(d, d)
    Q, _ = np.linalg.qr(random_matrix)  # Q is an orthogonal rotation matrix

    # Apply  rotation
    w_rotated =   Q @ w

    return w_rotated



""" add sparsity
for example  in                 2d
sparsity = 0.3  # 30% of w_star entries will have noise
perturbation = (np.random.rand(*w_star.shape) < sparsity) * np.random.uniform(-2, 2, w_star.shape)
w_corrupt = w_star + perturbation

"""
def sparse_noise_w_arbitrary(w, sparsity=0.3, noise_scale=1.5):
    """
    Applies sparse perturbations to w_star in arbitrary dimensions.

    Parameters:
    - w: Original weight vector of shape (d, 1)
    - sparsity: Fraction of elements to modify (between 0 and 1)
    - noise_scale: Magnitude of noise applied to modified elements

    Returns:
    - w_noisy: Perturbed weight vector of shape (d, 1)
    """
    d = w.shape[0]
    
    # Generate a binary mask: 1 where noise is applied, 0 elsewhere
    mask = np.random.rand(d, 1) < sparsity
    
    # Generate sparse noise (random uniform perturbation)
    perturbation = mask * (np.random.uniform(-noise_scale, noise_scale, (d, 1)) + 5)

    # Apply perturbation
    w_noisy = w + perturbation

    return w_noisy


"""_summary_
w_rand = np.random.randn(*w_star.shape)  # Completely different model
mixing_ratio = 0.3  # 30% towards w_rand, 70% remains w_star
w_corrupt = (1 - mixing_ratio) * w_star + mixing_ratio * w_rand

"""

def interpolate_w_arbitrary(w, mixing_ratio=0.5, variance=1.5):
    """
    Interpolates between w_star and a randomly generated model in arbitrary dimensions.

    Parameters:
    - w: Original weight vector of shape (d, 1)
    - mixing_ratio: Fraction of blending (0 = no corruption, 1 = fully random)

    Returns:
    - w_interpolated: The interpolated weight vector of shape (d, 1)
    """
    d = w.shape[0]

    # Generate a completely random model
    w_rand = variance*np.random.randn(d, 1) + 5

    # Blend between the original and random model
    w_interpolated = (1 - mixing_ratio) * w - mixing_ratio * w_rand

    return w_interpolated

def strategic_corruption_one_d(X, Y, w_star, w_corrupt, alpha):
    """
    Applies adversarial corruption to Y such that the new dataset appears closer to w_corrupt.
    """
    d, n = X.shape

    # Compute ideal labels under the adversarial model w_corrupt
    Y_corrupt_ideal = np.dot(X.T, w_corrupt)

    # Compute l2/ abs[] distances from desired corruption model
    corruption_magnitude = np.abs(Y - Y_corrupt_ideal)

    # Select the alpha * n samples that need the least change to match w_corrupt
    num_corrupted_samples = int(alpha * n)
    corrupted_indices = np.argsort(corruption_magnitude.flatten())[:num_corrupted_samples]

    # Apply corruption
    Y_corrupted = Y.copy()
    Y_corrupted[corrupted_indices] = Y_corrupt_ideal[corrupted_indices]

    return Y_corrupted, corrupted_indices

def strategic_corruption_scaled(X, Y, w_star, w_corrupt, alpha, base_scaling=1.0, influence_factor=0.5 ):
    """
    Applies adversarial corruption to Y such that the new dataset appears closer to w_corrupt.
    When the corruption fraction is small, the adversarial points are moved further to maximize influence.

    Parameters:
    - X: (d, n) Feature matrix
    - Y: (n,) Original labels
    - w_star: (d,) True model parameters
    - w_corrupt: (d,) Adversarial model parameters
    - alpha: Fraction of points to corrupt (0 < alpha <= 1)
    - base_scaling: Minimum scaling factor (default=1.0, meaning no extra push when alpha is large)
    - influence_factor: Controls how much more adversarial points are pushed when alpha is small

    Returns:
    - Y_corrupted: (n,) Corrupted labels
    - corrupted_indices: Indices of corrupted points
    """
    d, n = X.shape

    # Compute ideal labels under the adversarial model w_corrupt
    Y_corrupt_ideal = np.dot(X.T, w_corrupt)

    # Compute L2 distances from the desired corruption model
    corruption_magnitude = np.abs(Y - Y_corrupt_ideal)

    # Select the alpha * n samples that need the least change to match w_corrupt
    num_corrupted_samples = int(alpha * n)
    corrupted_indices = np.argsort(corruption_magnitude.flatten())[:num_corrupted_samples]

    # Adaptive scaling factor: More aggressive when alpha is small
    scaling_factor = (base_scaling + (influence_factor / alpha))

    # Apply corruption with adaptive amplification
    Y_corrupted = Y.copy()
    Y_adversarial = Y_corrupt_ideal[corrupted_indices] + scaling_factor * (Y_corrupt_ideal[corrupted_indices] - Y[corrupted_indices])
    Y_corrupted[corrupted_indices] = Y_adversarial

    return Y_corrupted, corrupted_indices



def strategic_corruption_on_X(X, Y, w_star, w_corrupt, alpha, X_noise_level=0.1, adaptive=True, extreme_outliers=False):
    """
    Applies adversarial corruption to both Y (labels) and X (covariates) in a way that makes the dataset appear
    closer to w_corrupt while maintaining plausible feature structure.

    Parameters:
    - X: Features matrix with dimensions [d, n]
    - Y: Labels matrix with dimensions [n, 1]
    - w_star: True model weights
    - w_corrupt: Target corruption model weights
    - alpha: Fraction of samples to corrupt (0 <= alpha <= 1)
    - X_noise_level: Strength of corruption noise added to X
    - adaptive: If True, scales noise inversely with feature variance
    - extreme_outliers: If True, adds targeted extreme outliers to high-variance features

    Returns:
    - X_corrupted: Corrupted feature matrix
    - Y_corrupted: Corrupted labels matrix
    - corrupted_indices: Indices of corrupted samples
    """
    d, n = X.shape

    # Compute ideal labels under the adversarial model w_corrupt
    Y_corrupt_ideal = np.dot(X.T, w_corrupt)

    # Compute distances from desired corruption model
    corruption_magnitude = np.abs(Y - Y_corrupt_ideal)

    # Select alpha * n samples that need the least change to match w_corrupt
    num_corrupted_samples = int(alpha * n)
    corrupted_indices = np.argsort(corruption_magnitude.flatten())[:num_corrupted_samples]

    # Apply corruption to labels with controlled interpolation
    lambda_values = np.random.uniform(0.5, 1.0, size=num_corrupted_samples).reshape(-1, 1)
    Y_corrupted = Y.copy()
    Y_corrupted[corrupted_indices] = (1 - lambda_values) * Y[corrupted_indices] + lambda_values * Y_corrupt_ideal[corrupted_indices]

    # Adaptive noise scaling based on feature variance
    X_corrupted = X.copy()
    feature_variance = np.var(X, axis=1, keepdims=True)  # Variance per feature

    for idx in corrupted_indices:
        perturbation = X_noise_level * np.random.randn(d, 1)
        if adaptive:
            perturbation *= np.maximum(0.1, 1 / (feature_variance + 1e-6))  # Scale noise inversely with variance
        
        if extreme_outliers and np.random.rand() < 0.1:  # Add extreme corruption with 10% probability
            perturbation *= np.random.uniform(500, 10000)  # Multiply by a large factor

        X_corrupted[:, idx] += perturbation.flatten()  # Corrupt only selected indices

    return X_corrupted, Y_corrupted, corrupted_indices

def strategic_corruption_on_X_adaptive(X, Y, w_star, w_corrupt, alpha):
    """
    Applies adversarial corruption to both Y (labels) and X (covariates) in a way that makes the dataset appear
    closer to w_corrupt with the smaller pertubation.

    Parameters:
    - X: Features matrix with dimensions [d, n]
    - Y: Labels matrix with dimensions [n, 1]
    - w_star: True model weights (shape [d, 1])
    - w_corrupt: Target corruption model weights (shape [d, 1])
    - alpha: Fraction of samples to corrupt (0 <= alpha <= 1)

    Returns:
    - X_corrupted: Corrupted feature matrix
    - Y_corrupted: Corrupted labels matrix
    - corrupted_indices: Indices of corrupted samples
    """
    d, n = X.shape  # Feature dimension and number of samples
    num_corrupt = int(alpha * n)  # Number of samples to corrupt
    
    # Compute the direction of corruption
    delta_w = w_corrupt - w_star
    
    # Compute perturbation magnitude for each sample
    perturbation_magnitudes = np.abs(np.dot(X.T, delta_w))
    
    # Select indices that need the smallest perturbation
    corrupted_indices = np.argsort(perturbation_magnitudes[0])[:num_corrupt]
    
    # Copy original data
    X_corrupted = X.copy()
    Y_corrupted = Y.copy()
    
    # Apply corruption
    for idx in corrupted_indices:
        X_corrupted[:, idx] += delta_w.flatten() * np.dot(delta_w.flatten(), X[:, idx])  # Modify feature vectors
        Y_corrupted[idx, 0] = np.dot(w_corrupt.T, X_corrupted[:, idx])  # Modify labels
    
    return X_corrupted, Y_corrupted, corrupted_indices


# Function to apply adversarial corruption strategy,  modifying a percentage of the dataset
def adversarial_corruption(X, Y, alpha=0.1, beta=10):
    """
    Corrupts a fraction of the dataset to force the modeprint (i)l to predict Y = 0.

    Parameters:
    - X: Feature matrix (d, n)
    - Y: Labels (n, 1)
    - alpha: Fraction of samples to corrupt (0 <= alpha <= 1)
    - beta: Strength of adversarial corruption for Y

    Returns:
    - X_corrupted: Feature matrix after corruption
    - Y_corrupted: Labels after corruption
    - corrupted_indices: Indices of corrupted samples
    """
    d, n = X.shape

    # Select fraction of points to corrupt
    num_corrupted_samples = int(alpha * n)
    corrupted_indices = np.random.choice(n, size=num_corrupted_samples, replace=False)

    # Compute X_corrupted using the formula X_bad = (1 / (alpha * n_bad)) * Y^T X
    X_corrupted = X.copy()
    X_corrupted[:, corrupted_indices] = (1 / (beta * num_corrupted_samples)) * (X @ Y)

    # Corrupt Y
    Y_corrupted = Y.copy()
    Y_corrupted[corrupted_indices] = -beta  # Force Y to be a constant negative value

    return X_corrupted, Y_corrupted, corrupted_indices


def main():
    # Generate Dataset
    n, d = 10, 1  # 1D for visualization
    sigma = 0.2
    alpha = 0.3 # 30% of data points are corrupted

    X, Y, w_star = generate_synthetic_dataset_one_d(n, d, sigma)

    # Define an adversarial corruption model
    w_corrupt = (-1/w_star)
    
    # Apply strategic corruption
    Y_corrupted, corrupted_indices = strategic_corruption_scaled(X, Y, w_star, w_corrupt, alpha)

    # Visualization
    plt.figure(figsize=(8, 6))

    # Original data (non-corrupted)
    plt.scatter(X.T, Y, label="Original Data", color="teal", alpha=0.6)

    # Highlight the original positions of corrupted points in orange
    plt.scatter(X.T[corrupted_indices], Y[corrupted_indices], label="Corruption Locations", 
                color="orange", marker="o", edgecolors="black", s=100, alpha=0.8)

    # Corrupted data points in red (final corrupted positions)
    plt.scatter(X.T[corrupted_indices], Y_corrupted[corrupted_indices], label="Strategically Corrupted Points", 
                color="mediumvioletred", marker="x", s=100)

    # Lighter arrows to indicate corruption direction
    plt.quiver(X.T[corrupted_indices], Y[corrupted_indices], 
               np.zeros_like(Y[corrupted_indices]), 
               (Y_corrupted - Y)[corrupted_indices], 
               angles='xy', scale_units='xy', scale=1, color='red', alpha=0.8, width=0.0002)

    # Compute the corrupted model
    dot_X = X.dot(X.transpose())
    dot_inv = np.linalg.pinv(dot_X).dot(X)
    w_approx = dot_inv.dot(Y_corrupted)

    # True model line
    x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_vals_true = x_vals @ w_star 
    plt.plot(x_vals, y_vals_true, label=r'True Model Predictions ($\langle x, w^* \rangle $)', 
             color="teal", linestyle="solid")

    # Corrupted model line
    y_vals_corrupt = x_vals @ w_approx
    plt.plot(x_vals, y_vals_corrupt, label=r'OLS Model Predictions ($\langle x, w_{OLS} \rangle $)', 
             color="darkviolet", linestyle="dotted")
    
    # Adversarial model line
    y_vals_adv = x_vals @ w_corrupt
    plt.plot(x_vals, y_vals_adv, label=r'Adversarial Model Predictions ($\langle x, w_{Adv} \rangle $)', 
             color="peru", linestyle="dotted")
    
    #class HuberRegressor(*, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05)

    huber = HuberRegressor( epsilon=1.35, max_iter=100, alpha=1, warm_start=False, fit_intercept=False, tol=0.1).fit(X.T, Y_corrupted.ravel())
    w_huber = huber.coef_
    y_vals_huber = x_vals @ w_huber
    plt.plot(x_vals, y_vals_huber, label=r'Huber ($\langle x, w_{Huber} \rangle $)', 
             color="green", linestyle="dashdot")
    
    w_torrent, iter_count, _ = torrent_ideal(X, Y_corrupted, alpha+0.1, epsilon=0.1, max_iters=20, w_star=w_star, w_iter=True)
    y_vals_tor = x_vals @ w_torrent
    plt.plot(x_vals, y_vals_tor, label=r'Torrent ($\langle x, w_{Torrent} \rangle $)', 
             color="red", linestyle="dotted")
    # Labels and legend
    plt.xlabel("Feature X")
    plt.ylabel("Label Y")
    plt.legend()
    plt.title(f"Strategic Decision of Corruption Locations $(\\alpha = {alpha})$")
    plt.grid(False)

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
