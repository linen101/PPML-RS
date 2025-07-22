# TORRENT

import numpy as np
import math
from numpy.linalg import eigh, inv

def update_ridge_regr(X, y, S, alpha):
    """ 
    Parameters:
    - Training data: {X, y}
    - alpha: regularization parameter
    - S: active set of datapoints
    Returns:
    - w: estimated weights
    """ 
    # keep only points in S
    X_s = X[:, list(S)]
    y_s = y[list(S)]
    
    #(XX^T), (Xy)
    # Compute intermediate matrices
    XXT = np.matmul(X_s, X_s.T)
    Xy = np.matmul(X_s, y_s)
    
     #(XX^T + aI)^{-1}(Xy)
    # Compute ridge regression coefficients
    n_features = X_s.shape[0]

   
    inv= np.linalg.inv(XXT + alpha * np.eye(n_features))

    w = np.matmul(inv, Xy)
    
    return w

def update_gd(X,y, w, S, eta, r):
    X_new = X[:, list(S)]
    y_new = y[list(S)]
    dot = X_new.transpose().dot(w)
    e = y_new - dot
    g = X_new.dot(e)
    # perform gd with fixed step size eta
    return (w - eta * g) 
    

def ht_test_quantiles(r, k):
    """
    Parameters:
    - r : Input vector.
    - k : Number of elements to keep.

    Returns:
    - subset_indices (set): Set of indices from the original vector r that we keep.
    """
    
    # Create a modified vector by sorting the elements of r in ascending order of their magnitude
    sorted_indices = sorted(range(len(r)), key=lambda i: abs(r[i]))
    
    # get og index of order statistic r_(k-1)
    q=sorted_indices[k-1]

    # get (k-1)-quantile
    q_value = abs(r[q])
    
    # Keep only the first k elements of the modified vector
    subset_indices = {i for i in sorted_indices[:k]}  
    #print(q_value)
    
    return subset_indices, q_value


def torrent_ideal(X, y, beta, epsilon, max_iters, w_star, w_iter=False):  
    """ 
    Parameters:
    - Training data: {X, Y}
    - step length size: \eta
    - thresholding parameter: beta
    - tolerance: epsilon
    - real model: w_star
    Returns:
    - estimated weights: w
    """ 
    # initialize parameters w_0 = 0, S_0 = [n], t = 0, r_0 = y
    iteration_ideal = 0
    d, n = X.shape
    S = np.arange(n)  # Create an array with values from 0 to n-1
    w = np.zeros(d)
    w = w.reshape(-1,1)
    w_per_iteration = []
    while np.linalg.norm(abs(w - w_star)) > epsilon :
        w = update_fc(X,y, S)
        w_per_iteration.append(w.copy())
        # Compute dot product <w,x>
        dot_prod = X.transpose().dot(w)
        # Compute residuals r
        r = dot_prod - y            #y - wx
        # Keep only 1-beta datapoints for the next iteration
        S = hard_thresholding(r,math.ceil((1-beta)*n))
        iteration_ideal = iteration_ideal + 1
        if iteration_ideal > max_iters:
            break
    if w_iter:
        return w, iteration_ideal, w_per_iteration  
    else:
        return w, iteration_ideal 


def torrent_intermediate(X, y, beta, epsilon, max_iters=10):

def torrent_rg(X, y,  beta, epsilon, max_iters=10, ridge=10):
    """ 
    Parameters:
    - Training data: {X, Y}
    - step length size: \eta
    - thresholding parameter: beta
    - tolerance: epsilon
    Returns:
    - estimated weights: w
    """ 
    # initialize parameters w_0 = 0, S_0 = [n], t = 0, r_0 = y
    iteration = 0 
    iteration_ideal = 0
    r = y
    abs_r = np.abs(r)
    d, n = X.shape
    S = np.arange(n)  # Create an array with values from 0 to n-1
    w = np.zeros(d)
    w = w.reshape(-1,1)
    while np.linalg.norm(r[list(S)]) > epsilon :
        w = update_ridge_regr(X,y, S, ridge)
        # Compute dot product <w,x>
        dot_prod = X.transpose().dot(w)
        # Compute residuals r
        r = dot_prod - y            #y - wx
        # Keep only 1-beta datapoints for the next iteration
        S = hard_thresholding(r,math.ceil((1-beta)*n))
        iteration = iteration + 1
        if iteration > max_iters:
            break
    return w, iteration  

def torrent_test_threshold(X, y,  beta, epsilon, max_iters=10):
    """ 
    Parameters:
    - Training data: {X, Y}
    - step length size: \eta
    - thresholding parameter: beta
    - tolerance: epsilon
    Returns:
    - estimated weights: w
    - q-th ranked value in the list of residual errors: q
    """ 
    # initialize parameters w_0 = 0, S_0 = [n], t = 0, r_0 = y
    iteration = 0 
    iteration_ideal = 0
    r = y
    abs_r = np.abs(r)
    d, n = X.shape
    S = np.arange(n)  # Create an array with values from 0 to n-1
    w = np.zeros(d)
    w = w.reshape(-1,1)
    q = np.empty(max_iters+1, dtype=object)
    while np.linalg.norm(r[list(S)]) > epsilon :
        w = update_fc(X,y, S)
        # Compute dot product <w,x>
        dot_prod = X.transpose().dot(w)
        # Compute residuals r
        r = dot_prod - y            #y - wx
        # Keep only 1-beta datapoints for the next iteration
        #print(r)
        S, q[iteration] = ht_test_quantiles(r,math.ceil((1-beta)*n))
        iteration = iteration + 1
        if iteration > max_iters:
            break
    return w, q


def torrent_S(X, y,  beta, epsilon, max_iters=10):
    """ 
    Robust Regression using TORRENT
    
    Parameters:
    - Training data: {X, Y}
    - step length size: \eta
    - thresholding parameter: beta
    - tolerance: epsilon
    Returns:
    - estimated weights: w
    - list of computed Ss across iterations: Sarray
    """ 
    # initialize parameters w_0 = 0, S_0 = [n], t = 0, r_0 = y
    iteration = 0
    r = y
    abs_r = np.abs(r)
    d, n = X.shape
    S = np.arange(n)  # Create an array with values from 0 to n-1
    w = np.zeros(d)
    w = w.reshape(-1,1)
    Sarray = np.empty(max_iters, dtype=object)
    while np.linalg.norm(r[list(S)]) > epsilon :
        w = update_fc(X,y, S)
        # Compute dot product <w,x>
        dot_prod = X.transpose().dot(w)
        # Compute residuals r
        r = dot_prod - y            #y - wx
        # Keep only 1-beta datapoints for the next iteration
        S = hard_thresholding(r,math.ceil((1-beta)*n))
        Sarray[iteration] =  np.array(list(S))
        iteration = iteration + 1
        if iteration > max_iters-1:
            break
    return w, Sarray

def torrent_S_with_corruption(X, y, beta, epsilon, gamma, corrupted_indices, max_iters=10):
    """ 
    Robust regression using TORRENT with corruption injected in each iteration,
    to test the resilience of TORRENT if an Adversarial Party escapes correctness proofs 
    in the private setting.
    
    Parameters:
    - X : Input feature matrix.
    - y : Target vector.
    - beta : Thresholding parameter.
    - epsilon : Convergence tolerance.
    - gamma : Percentage of points in S to replace with corrupted indices.
    - corrupted_indices : Set of indices considered as corrupted.
    - max_iters : Maximum number of iterations.
    
    Returns:
    - Estimated weights: w
    - List of computed Ss across iterations: Sarray
    """ 
    iteration = 0
    r = y
    abs_r = np.abs(r)
    d, n = X.shape
    S = np.arange(n)  # Create an array with values from 0 to n-1
    w = np.zeros(d).reshape(-1, 1)
    Sarray = np.empty(max_iters, dtype=object)
    
    while np.linalg.norm(r[list(S)]) > epsilon:
        # Update the model parameters
        w = update_fc(X, y, S)
        
        # Compute residuals
        dot_prod = X.transpose().dot(w)
        r = dot_prod - y
        
        # Hard thresholding to keep the cleanest points
        S = hard_thresholding(r, math.ceil((1 - beta) * n))
        
        # Inject corruption into the active set
        num_to_replace = math.ceil(gamma * len(S))  # Number of points to replace
        if num_to_replace > 0:
            S = inject_corruption(S, corrupted_indices, num_to_replace)
        
        # Save the active set for this iteration
        Sarray[iteration] = np.array(list(S))
        iteration += 1
        
        if iteration > max_iters - 1:
            break
    
    return w, Sarray

def inject_corruption(S, corrupted_indices, num_to_replace):
    """
    Replace a percentage of indices in S with corrupted indices.
    
    Parameters:
    - S : Set of active indices.
    - corrupted_indices : Set of known corrupted indices.
    - num_to_replace : Number of indices to replace with corrupted indices.
    
    Returns:
    - Modified set S.
    """
    S = list(S)
    corrupted_to_add = list(corrupted_indices)
    
    # Shuffle the corrupted indices for random selection
    np.random.shuffle(corrupted_to_add)
    
    # Replace elements in S with corrupted indices
    for _ in range(num_to_replace):
        if len(corrupted_to_add) > 0:
            corrupt_index = corrupted_to_add.pop()
            replace_index = np.random.choice(S)  # Randomly pick an element from S to replace
            S[S.index(replace_index)] = corrupt_index
    
    return set(S)

# Torrent algorithm with residual tracking
def torrent_with_residuals(X, y, beta, epsilon, corrupted_indices, max_iters=10):
    """
    Runs the Torrent algorithm while tracking residual scores over iterations.

    Returns:
    - w: Estimated weights
    - residuals_per_iteration: List of residuals per iteration
    - selected_indices_per_iteration: List of selected indices per iteration
    """ 
    iteration = 0 
    d, n = X.shape
    S = np.arange(n)  # Initial set includes all points
    w = np.zeros((d, 1))
    residuals_per_iteration = []

    r = y.copy()

    while  np.linalg.norm(r[list(S)]) > epsilon:
        w = update_fc(X, y, S)
        r = X.T @ w - y  # Compute residuals

        # Store residuals for visualization
        residuals_per_iteration.append(r.copy())

        # Hard thresholding to retain (1-beta) fraction of points
        S = hard_thresholding(r, math.ceil((1 - beta) * n))
        iteration += 1

        if iteration >= max_iters:
            break

    return w, residuals_per_iteration

    