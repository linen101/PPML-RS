# irls_regressors.py


import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def huber_proximal(z_tilde, delta, rho):
    """Proximal operator for the Huber loss"""
    abs_z = np.abs(z_tilde)
    z_new = np.zeros_like(z_tilde)

    # Case 1: Quadratic region (Huber behaves like squared loss)
    mask1 = abs_z <= delta
    z_new[mask1] = z_tilde[mask1] / (1 + 1/rho)

    # Case 2: Linear region (Huber behaves like absolute loss)
    mask2 = abs_z > delta
    z_new[mask2] = np.sign(z_tilde[mask2]) * np.maximum(abs_z[mask2] - delta / rho, 0)

    return z_new

def admm_huber(X, y, delta, rho, max_iter, tol, w_star):
    """
    ADMM for Huber regression.
    
    Parameters:
    - X: (d, n) Feature matrix
    - y: (n, 1) Target vector
    - delta: Huber loss threshold
    - rho: ADMM penalty parameter
    - max_iter: Maximum number of iterations
    - tol: Convergence tolerance
    
    Returns:
    - w: Estimated coefficients
    """
    d, n = X.shape
    w = np.zeros((d, 1))

    z = np.zeros((n,1))  # Initialize z (shape n,)
    u = np.zeros((n, 1))
 

    # Precompute X^T X and X^T y for efficiency
    XTy = X @ y  # Shape (d,1)
    XTX = X @ X.T + rho * np.eye(d)  # Regularized system (shape d,d)
    
    for it in range(max_iter):
        # w-update: Solve (X X^T + rho I) w = X y + X (z - u)
        w_new = np.linalg.solve(XTX, XTy + X @ (z - u))


        # z-update: Apply the proximal operator for Huber loss
        z_tilde = X.T @ w_new - y + u

        z_new = huber_proximal(z_tilde, delta, rho)

        # u-update: Dual ascent
        u_new = u + (X.T @ w_new - y - z_new)/rho  # Keep shape (n,)

        # Convergence check
        if  np.linalg.norm(w_new - w_star) < tol:
            break
        
        # Update variables
        w, z, u = w_new, z_new, u_new

    return w, it



# Calculate the weights used in the diagonal matrix based on the user choice
def weight_function(e, sigma ,delta, w):
    if w == 'OLS' :
        e[:] = 1
        return e
    elif w == 'HUBER' :
        if sigma == 0:
            k = delta 
        else:
            k = delta /sigma 
        return np.minimum(k / e, 1.0) 
    elif w == 'BISQUARE':
        if sigma == 0:
            k = delta 
        else:
            k = 4.685 * sigma  
        return np.maximum((1 - (e / k)**2),0)**2
    elif w == 'TRUNCATED':
        return 1 / np.maximum(e, delta)
    else:
        raise ValueError("Invalid weighting scheme. Choose from 'OLS', 'HUBER', 'BISQUARE', 'TRUNCATED'")


# Implementing M-estimators (ols, huber, bisquare, truncated)
def irls(X, y, weights, delta, max_iterations, tol, scheme, sigma, w_star=None):
    """ 
    Parameters:
    - Training data: {X, y}
    - initial model weights w^0 : weights
    - tuning parameter for huber, bisquare, truncated: delta
    - maximum number of iterations for convergence:  max_iterations
    - tol: error tolerance between estimated and gold model. in truncated irls this changes in every irls call
    - scheme: the scheme used to assign "significance" on absolute errors
    - sigma: the variance of gaussian error in data, used to calculate the tuning parameters of huber, bisquare
    - real model: w_star, is used to compute the ideal convergence
    Returns:
    - estimated weights: w
    - number of iterations: iterations
    """ 
    iteration = 0
    for iteration in range(max_iterations):
        
        # Step 1: Compute error weights
        dot_prod = X.transpose().dot(weights)
        residuals = dot_prod - y            #y - wx
        abs_residuals = np.abs(residuals)
        # choose one of the 'OLS', 'HUBER', 'BISQUARE', 'TRUNCATED' weighting schemes
        s =  weight_function(abs_residuals, sigma , delta, scheme) 

        # Step 2: Solve weighted least squares problem
        # Update parameters using the weighted least squares solution
        S = np.diagflat(s)
        # XSX^T
        X_weighted = X.dot(S).dot(X.T)
        #print(f"X_weighted {scheme} = {X_weighted}")
        # (XSX^T)^-1
        X_weighted_inv = np.linalg.inv(X_weighted)
        # (XSX^T)^-1 (XSy)
        weights_new = X_weighted_inv.dot(X).dot(S).dot(y)

        # Print the current iteration and parameters
        #print(f"Iteration {iteration + 1}: weights = {weights.flatten()}")
        
        if w_star is not None:
            if np.linalg.norm(weights_new - w_star) <= tol :
                weights = weights_new
                return weights, iteration
        else:
            if np.linalg.norm(weights_new - weights) <= tol :
                weights = weights_new
                return weights, iteration
        
        weights = weights_new
    return weights, iteration    

def irls_init(X, y, delta, max_iterations, tol, scheme, sigma, w_star=None):
    rows, columns = X.shape
    w = np.zeros(rows)
    weights = w.reshape(-1,1)
    weights_new, iters = irls(X, y, weights, delta, max_iterations, tol, scheme, sigma, w_star)
    return weights_new, iters

def truncated_irls(X, y, M, max_iterations, tolerance, eta, w_star=None):
    """ 
    Parameters:
    - Training data: {X, y}
    - tuning parameter when assigning "significance" on errors: M
    - maximum number of iterations for convergence:  max_iterations
    - tolerance: global error tolerance between estimated and gold model. do not confuse with "tol" which changes in every outer iteration
    - eta: how much is increased the tuning parameter in each iteration
    - real model: w_star, is used to compute the ideal convergence
    Returns:
    - estimated weights: w
    - number of iterations: iterations
    """ 
    i = 0
    iterations_inner = 0
    iterations_sum = 0
    tol = 2.0/eta*M
    rows, columns = X.shape
    w = np.zeros(rows)
    weights = w.reshape(-1,1)
    for i in range(max_iterations):
        if w_star is not None:
            weights_new, iterations_inner = irls(X, y, weights, M, max_iterations, tolerance, 'TRUNCATED', 0, w_star)
        else:
            weights_new, iterations_inner = irls(X, y, weights, M, max_iterations, tol, 'TRUNCATED', 0)  
        iterations_sum = iterations_sum + 1 + iterations_inner
        M = eta*M
        tol = 2.0/M
        if w_star is not None:
            if np.linalg.norm(weights_new - w_star) <= tolerance :
                weights = weights_new
                return weights, iterations_sum
        weights = weights_new    
    return weights, iterations_sum    