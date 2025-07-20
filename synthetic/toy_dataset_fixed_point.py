import numpy as np
import sys
import os
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from fixed_point.fixed_point_helpers import *    
from torrent.torrent import split_matrix, split_matrix_Y
from torrent.torrent_admm_fx import torrent_admm

def generate_synthetic_dataset_fxp(n, d, sigma):
    w_star = np.random.uniform(-10, 10, size=(d, 1))
    w_star /= np.linalg.norm(w_star)
    w_star = fxp_array(w_star)

    X = fxp_array(np.random.randn(d, n))
    epsilon = fxp_array(sigma * np.random.randn(n, 1))

    X_T = X.T
    Y_linear = fxp_matmul(X_T, w_star)

    Y = np.empty_like(Y_linear)
    for i in range(n):
        Y[i, 0] = Y_linear[i, 0] + epsilon[i, 0]

    return X, Y, w_star


def corrupt_dataset_fxp(X, Y, w_star, alpha, sigma, return_cor_indices=False):
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
    maxY = fxp(np.max(Y))
    minY = fxp(np.min(Y))
    b[corrupted_feature_indices] = fxp_array(np.random.uniform(maxY - minY))    # change to uniform error max - min (y) bc subgaussian is very small -> random.uniform(a, b)

    # Corrupt the labels Y with b
    Y_corrupted = Y + b
    #epsilon = sigma * np.random.randn(n, 1)
    #Y_corrupted = np.dot(X.transpose(), w_star) + b + epsilon
    #Y_corrupted = np.dot(X.transpose(), w_star) + b 
    if return_cor_indices==False:
        return Y_corrupted
    else:
        return Y_corrupted, corrupted_feature_indices

"""
# test
X, Y, w_star = generate_synthetic_dataset_fxp(100, 5, sigma=0.1)
print("Y[0] (fixed-point):", Y[0, 0], "≈", Y[0, 0].get_val())
print(" w_star (float):", [elem[0] for elem in w_star])

Xdist = split_matrix(X, 3, 100)
Ydist = split_matrix_Y(Y, 3, 100)

w,_ = torrent_admm(Xdist, Ydist,  0.1, 0.1, 1, 5, rounds = 2, wstar= None)
print(" w_approx (float):", [elem[0] for elem in w])


Y, indx = corrupt_dataset_fxp(X, Y, w_star, 0.1, 0.1, return_cor_indices=True)
print("Y[indx] (fixed-point):", Y[indx[0],0], "≈", Y[indx[0], 0].get_val())
print(indx)
"""