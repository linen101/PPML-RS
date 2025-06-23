# toy_dataset.py

     
import numpy as np
import matplotlib.pyplot as plt



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
    
    #w_star = np.random.uniform(low=-10, high=10, size=(d, 1))
    w_star /= np.linalg.norm(w_star)

    # Generate feature vectors X with dimensions [d, n] 
    # X are sampled from a Gaussian distribution with standard deviation
    X = 10*np.random.randn(d, n)

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

def corrupt_dataset(X, Y, w_star, alpha, sigma, return_cor_indices=False):
    """
    Corrupts a percentage of the dataset.

    Parameters:
    - X: Features matrix with dimensions [d, n]
    - Y: Labels matrix with dimensions [n, 1]
    - alpha: Corruption percentage (0 <= alpha <= 1)
    - sigma: Variance of sub-Gaussian Noise

    Returns:
    - Corrupted Labels matrix Y
    """

    # Get the dimensions
    d, n = X.shape


    # Generate a sparse corruption vector b
    b = np.zeros((n, 1))
    num_corrupted_features = int(alpha * n)
    corrupted_feature_indices = np.random.choice(np.arange(n), size=num_corrupted_features, replace=True)
    maxY = np.max(Y)
    minY = np.min(Y)
    b[corrupted_feature_indices] = np.random.uniform(maxY - minY)    # change to uniform error max - min (y) bc subgaussian is very small -> random.uniform(a, b)

    # Corrupt the labels Y with b
    Y_corrupted = Y + b
    #epsilon = sigma * np.random.randn(n, 1)
    #Y_corrupted = np.dot(X.transpose(), w_star) + b + epsilon
    #Y_corrupted = np.dot(X.transpose(), w_star) + b 
    if return_cor_indices==False:
        return Y_corrupted
    else:
        return Y_corrupted, corrupted_feature_indices

