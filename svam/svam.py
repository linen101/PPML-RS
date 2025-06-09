# SVAM

import numpy as np
import math

def svam(X, y,  w_est, beta, ksi, max_iters, tol, w_star=None):
    """ 
    Parameters:
    - Training data: {X, y}
    - w_est: initial model estimate
    - beta: variance
    - ksi: step to increase variance
    - maximum number of iterations for convergence:  max_iters
    - tol: error tolerance between estimated and gold model. 
    - real model: w_star, is used to compute the ideal convergence
    Returns:
    - estimated weights: w
    - number of iterations in the ideal setting: iterations
    """ 
    iteration = 0
    for iteration in range(max_iters):
        
        dot_prod = X.transpose().dot(w_est)
        residuals = dot_prod - y            #y - wx
        
        # exponential weight to use in the weighted least squares
        s =  np.exp(-beta/2 * np.square(residuals))
        # Step 2: Solve weighted least squares problem
        # Update parameters using the weighted least squares solution
        S = np.diagflat(s)
        X_weighted = X.dot(S).dot(X.T)
        #print(f"X_weighted {scheme} = {X_weighted}")
        X_weighted_inv = np.linalg.inv(X_weighted)
        w_est = X_weighted_inv.dot(X).dot(S).dot(y)

        # Print the current iteration and parameters
        #print(f"Iteration {iteration + 1}: weights = {weights.flatten()}")
        if w_star is not None:
            if np.linalg.norm(w_est - w_star) <= tol :
                return w_est, iteration
     
        beta = ksi * beta
          
    return w_est, iteration