# TORRENT

import numpy as np
import math
#import pyproximal
#from pyproximal.optimization.primal import ADMML2
#from pyproximal.proximal import ProxL2, ProxL1

# HELPERS
def split_matrix(X, m, n):
    """
    Splits the matrix X into m parts along the COLUMNS.
    
    Parameters:
    X (np.ndarray): The input matrix of shape (d, n).
    m (int): The number of parts to split the matrix into.
    
    Returns:
    list: A list of matrices, each of shape (d, n_i) where sum(n_i) = n.
    """
    if m > n:
        raise ValueError("The number of parts m cannot be greater than the number of columns n")
    
    # Calculate the sizes of each part
    part_sizes = [n // m] * m
    remainder = n % m
    for i in range(remainder):
        part_sizes[i] += 1

    # Split the matrix
    splits = np.cumsum(part_sizes)[:-1]
    X_parts = np.hsplit(X, splits)
    
    joined_matrix = np.empty(m, dtype=object)
    
    for i, part in enumerate(X_parts):
        joined_matrix[i] = part
    
    return joined_matrix

def split_matrix_Y(X, m, n):
    """
    Splits the matrix X into m parts along the COLUMNS.
    
    Parameters:
    X (np.ndarray): The input matrix of shape (d, n).
    m (int): The number of parts to split the matrix into.
    
    Returns:
    list: A list of matrices, each of shape (d, n_i) where sum(n_i) = n.
    """
    if m > n:
        raise ValueError("The number of parts m cannot be greater than the number of columns n")
    
    # Calculate the sizes of each part
    part_sizes = [n // m] * m
    remainder = n % m
    for i in range(remainder):
        part_sizes[i] += 1

    # Split the matrix
    splits = np.cumsum(part_sizes)[:-1]
    X_parts = np.split(X, splits)
    
    joined_matrix = np.empty(m, dtype=object)
    
    for i, part in enumerate(X_parts):
        joined_matrix[i] = part
    
    return joined_matrix

def precomp(X, y, S, Ainit, binit):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        S (_type_): _description_
        Xinit (_type_): _description_
        yinit (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Compute SUPPLEMENT matrices
    XSXT, XSy = supplcomp(X, y, S)
    
    Ahat =  Ainit - XSXT
    A = np.linalg.inv(Ahat)
    b = binit - XSy   
    return A, b

def initialcomp(X, y, rho):
    features = X.shape[0]
    
    #(XX^T), (Xy)
    # Compute intermediate matrices
    XXT = np.matmul(X, X.T) + rho/2 * np.eye(features)
    Xy = np.matmul(X, y)
    
    return XXT, Xy

def supplcomp(X, y, S):
    features = X.shape[0]
    
    #(XX^T), (Xy)
    # Compute intermediate matrices
    XSXT = np.matmul(np.matmul(X, S), X.T )
    XSy = np.matmul(np.matmul(X,S), y)
    
    return XSXT, XSy
    
# ADMM
def admm_initcomp(X, y, S, Ainit, binit, z=None):
    """Initialize variables for ADMM."""
    parties = X.shape[0]
    d, n = X[0].shape

    # Initialize variables
    u = np.array([np.zeros((d, 1)) for _ in range(parties)], dtype=object)
    w = np.array([np.zeros((d, 1)) for _ in range(parties)], dtype=object)

    # Copy S (avoid modifying the input)
    S_new = np.array([np.diagflat(1 - np.diagonal(S[i])) for i in range(parties)])

    # If z is None, initialize as zero vector
    z = np.zeros((d, 1)) if z is None else z.reshape(-1, 1)

    # Compute A, b for each party
    A = np.empty(parties, dtype=object)
    b = np.empty(parties, dtype=object)

    for i in range(parties):
        A[i], b[i] = precomp(X[i], y[i], S_new[i], Ainit[i], binit[i])

    return A, b, w, z, u

def admm(X, y, S, rho, k, Ainit, binit, z = None):
    """_ consensus admm 
    where 
    multiple parties compute local models
    under the constrain that
    their local models are close to each other (converge to a global model)
    and this global model has small squared error on their data
    _

    Args:
        X (np.marray): m dimensional matrix of the local data samples of each P_i (n,d matrices)
        y (np.marray): m dimensional matrix of the local labels (n matrices)
        S (np.marray): m dimensional matrix of the local  weights (diagonal square matrices)
        rho (int): penalty on the divergence of local and global models
        k (int): umber of iteratiosn
        
    Returns:
            _type_: _description_    
    """
    # get number of parties, features
    parties = X.shape[0]
    d, n = X[0].shape
    
    # Initializations
    A, b, w, z, u  = admm_initcomp(X, y, S, Ainit, binit, z)
    
    for i in range(k):
        znew = np.zeros(d)
        znew = znew.reshape(-1,1)
        for j in range(parties):
            w[j] = np.matmul(A[j], b[j] + rho/2 * z - 1/2 * u[j] )
            #print (w[j])
            znew = znew + w[j]
            #print(znew)  
        znew = znew / parties
        for j in range(parties):
            u[j] = u[j] + rho *(w[j] - znew)
        z = znew    
    return z       

# OLS
def ols(X, y, S, rho):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        S (_type_): _description_
        rho (_type_): _description_
    """
    features = X.shape[0]
    
    #(XX^T), (Xy)
    # Compute intermediate matrices
    XSXT, XSy = initialcomp(X, y, S)
    
    #(XX^T + rho/2 I)^{-1}
    # Compute inverse
    XSXTinv = np.linalg.inv(XSXT + rho/2 * np.eye(features))
    
    return XSXTinv, XSy

# TORRENT HELPERS
def q_quantile(r, q):
    """_summary_

    Args:
        r (_type_): _description_
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Flatten all the matrices in r into a single array
    rvalues = np.concatenate([matrix.flatten() for matrix in r])
    
    # Compute the q-quantile of the flattened array
    qquant = np.quantile(rvalues, q)
    
    return qquant

def hard_thresholding(r, q, S):
    """_summary_

    Args:
        r (_type_): _description_
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get number of parties
    m = r.shape[0]
    
    # compute q-quantile of residual errors
    quant = q_quantile(r, q)
    
    for i in range(m):
        S[i] = np.array([1 if value < q else 0 for value in r[i]])
        S[i] = np.diagflat(S[i])

    return S

def hard_thresholding_admm(r, q, S):
    """_summary_

    Args:
        r (_type_): _description_
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get number of parties
    m = r.shape[0]
    
    # compute q-quantile of residual errors
    quant = q_quantile(r, q)
    
    for i in range(m):
        S[i] = np.array([1 if value < q else 0 for value in r[i]])
        S[i] = np.diagflat(S[i])

    return S

def hard_thresholding(r, k):
    """
    Parameters:
    - r : Input vector.
    - k : Number of elements to keep.

    Returns:
    - subset_indices (set): Set of indices from the original vector r that we keep.
    """
    
    # Create a modified vector by sorting the elements of r in ascending order of their magnitude
    sorted_indices = sorted(range(len(r)), key=lambda i: abs(r[i]))
    #print(sorted_indices)
    q=sorted_indices[k-1]
    #print(sorted_indices[k-1])
    #print(r[q])
    
    # Keep only the first k elements of the modified vector
    subset_indices = {i for i in sorted_indices[:k]}  
    #print(subset_indices)
    
    return subset_indices

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

def update_fc(X,y, S):
    X_s = X[:, list(S)]
    y_s = y[list(S)]
    dot_X = X_s.dot(X_s.transpose())
    dot_inv = np.linalg.pinv(dot_X).dot(X_s)
    w = dot_inv.dot(y_s)
    return (w) 

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
    
    
def torrent(X, y,  beta, epsilon, max_iters=10):
    
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
        w = update_fc(X,y, S)
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

def torrent_admm(X, y,  beta, epsilon, rho, admm_steps, rounds = 10, wstar= None, modelz = None):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        beta (_type_): _description_
        epsilon (_type_): _description_
        rho (_type_): _description_
        admm_steps (_type_): _description_
        rounds (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    # get number of parties
    m = X.shape[0]
    
    # create empty A, b, S matrices
    Ainit = np.empty(m, dtype=object)
    binit = np.empty(m, dtype=object)
    S = np.empty(m, dtype=object)
    
    # create empty matrices to compute the residual error for each party
    dot_prod = np.empty(m, dtype=object)
    r = np.empty(m, dtype=object)

    # get number of features
    d,_ = X[0].shape
    
    # initialize parameters w_0 = 0, S_0 = [n]
    w = np.zeros(d)
    w = w.reshape(-1,1)

    n = 0
    for i in range(m):
        _,ni = X[i].shape
        S[i] = np.diagflat(np.ones(ni))
        n = X[i].shape[0] + n
        Ainit[i], binit[i] = initialcomp(X[i], y[i], rho) 

    for ro in range(rounds) :
        if modelz is None:
            w = admm(X, y, S, rho, admm_steps, Ainit, binit)
        else:
            w = admm(X, y, S, rho, admm_steps, Ainit, binit, w)  
        if np.linalg.norm(abs(w - wstar)) < epsilon:  
            break         
        for i in range(m):
            # Compute dot product <w,x>
            dot_prod[i] = np.matmul(X[i].T,w)
            # Compute residuals r
            r[i] = abs(dot_prod[i] - y[i])            #y - wx
        S = hard_thresholding_admm(r, 1-beta, S)       
    return w,ro

def torrent_dp(X, y,  beta, epsilon, dp_epsilon, dp_delta, max_iters=10):
    
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
    r = y
    abs_r = np.abs(r)
    d, n = X.shape
    S = np.arange(n)  # Create an array with values from 0 to n-1
    w = np.zeros(d)
    w = w.reshape(-1,1)
    
    # DP noise
    sigma = (np.sqrt(2 * np.log(2 / dp_delta)) / dp_epsilon) * 1
    dp_noise = sigma * np.random.randn(d, 1)
    while np.linalg.norm(r[list(S)]) > epsilon :
        w = update_fc(X,y, S) + dp_noise
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