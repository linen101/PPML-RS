# TORRENT

import numpy as np
import math
from numpy.linalg import eigh, inv
import random

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
    n = int(n)
    m = int(m)
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
    n = int(n)
    m = int(m)
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

# TORRENT 

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
 
def update_fc(X,y, S):
    X_s = X[:, list(S)]
    y_s = y[list(S)]
    dot_X = X_s.dot(X_s.transpose())
    dot_inv = np.linalg.pinv(dot_X).dot(X_s)
    w = dot_inv.dot(y_s)
    return (w) 

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
    iteration = 0
    r = y
    d, n = X.shape
    S = np.arange(n)
    w = np.zeros((d, 1))

    w_history = [w.copy()]
    S_history = [S.copy()]  # Record initial full set

    while np.linalg.norm(r[list(S)]) > epsilon:
        w = update_fc(X, y, S)
        r = X.T @ w - y
        S = hard_thresholding(r, math.ceil((1 - beta) * n))
        w_history.append(w.copy())
        S_history.append(S.copy())  # Track S
        iteration += 1
        if iteration >= max_iters:
            break

    return w, iteration, w_history, S_history

#TORRENT ADMM

def count_smaller_than(residuals, m):
    count = int(0)
    for i, residual in enumerate(residuals):
        #def _(i):
        c = residual < m
        count = count + c
    return int(count)

def count_greater_than(residuals, m):
    count = int(0)
    for i, residual in enumerate(residuals):
        #def _(i):
        c = residual > m
        count = count + c
    return int(count)
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

def dist_quantile(r, q):
    #print(f'residualstor:{r}')
    n = sum(ri.size for ri in r)
    #print(f'samples:{n}')
    
    m = len(r)
    #print(f'parties:{m}')
    
    quantile = int(np.ceil(q * (n)))
        
    # min positive value in domain
    step = 2**(-16)
    #print(f'step:{step}')
    alpha = step
    
    # max value in domain
    beta = 100
    less_than_player = np.zeros(m, dtype=int)
    greater_than_player = np.zeros(m, dtype=int)

    for i in range(16):
        less_than_sum = int(0)
        greater_than_sum = int(0)
        qvalue = ((alpha + beta )/2) 
        #print(qvalue)
        for j in range(m):
            less_than_player[j] = count_smaller_than(r[j], qvalue) 
            greater_than_player[j] = count_greater_than(r[j], qvalue) 
            less_than_sum += less_than_player[j]
            greater_than_sum += greater_than_player[j]  
        if (less_than_sum >= quantile):
            beta = qvalue - step

        if (greater_than_sum >= n - quantile + 1):
            alpha = qvalue + step  

        if ((less_than_sum <= quantile-1) & (greater_than_sum <= n - quantile)):
            print('Quantile found')
            print(qvalue)
            break     
    return(qvalue)
def compute_rank(residuals, x):
    # local computation of the rank of a subrange
    # residuals is the local residuals of party i
    count = int(0)
    for i, residual in enumerate(residuals):
        #def _(i):
        c = bool(residual < x)
        #print(f'c is {c}')
        count = count + c
    return int(count)

def compute_weights(r, quantile, alpha, beta, m, parties, n, dp_e):
    u = int(0)
    for i in range(parties):
        rank = compute_rank(r[i], m)
        u += rank
    if (u < quantile):
        #print(f'(upper range) Number of elements less than {m} is {u}')
        wl_beta = 1
        wu_alpha = (math.exp(dp_e*(u - quantile)))
    else:
        #print(f'(lower range) Number of elements less than {m} is {u}')
        wl_beta = (math.exp(dp_e*(quantile - u)))
        wu_alpha = 1
    return (wu_alpha, wl_beta)            
        
def select_range(w_alpham, w_mbeta, alpha, beta, m, step):
    M = (np.zeros(2))  
    M[0] = w_alpham
    M[1] = w_mbeta + w_alpham
    t = random.uniform(0+step,1+step)
    #print(f't is :{t}')
    #t = random.getrandbits(32)
    r = M[1] * t
    #print (f'r is: {r.info()}')
    il = 0
    iu = 1
    while (il < iu):
        #im = math.floor((il+iu)/2)
        c = M[0] < r
        if (c==1):
            il = 1
        else:
            iu = 0   
    if (il==0):
        #print(f'choose lower')
        return (alpha, m-step)
    else:       
        return (m+step, beta)
    
def dp_dist_quantile(r, q, dp_e=1):
    n = sum(ri.size for ri in r)    
    parties = len(r)    
    quantile = int(np.ceil(q * (n)))
    
    bit_precision_dec = 16
    bit_precision_total = 32
    
    # step in domain (e.g. in integers step=1)
    step = 2 ** (-bit_precision_dec)
    
    # min positive value in domain
    alpha = step
    # max value in domain
    #beta = fxp(2**(bit_precision_total - bit_precision_dec -1))
    beta = 100
    for i in range(bit_precision_total):
        # suggested q rank element
        m = (alpha + beta)/2
        #print(m)
        # weights matrix
        w_alpham, w_mbeta = compute_weights(r, quantile, alpha, beta, m, parties, n, dp_e)
        alpha, beta = select_range(w_alpham, w_mbeta, alpha, beta, m, step)
    return(m)

def hard_thresholding_admm(r, q):
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
    quant = dp_dist_quantile(r, q)
    #print(f'Torrent quantile:{quant}')
    S = np.empty(m, dtype=object)
    for i in range(m):
        S[i] = np.array([1 if value < quant else 0 for value in r[i]])
        S[i] = np.diagflat(S[i])

    return S
def torrent_admm(X, y,  beta, epsilon, rho, admm_steps, rounds = 10, wstar= None):
    """_summary_

    Args:
        X (_type_): containing parties feature vectors
        y (_type_): containing parties responses
        beta (_type_): corruption rate
        epsilon (_type_): model recovery target error
        rho (_type_): admm rho
        admm_steps (_type_): admm rounss
        rounds (int, optional): torrent rounds

    Returns:
        _type_: model, rounds
    """
    # get number of parties
    m = X.shape[0]
    
    # create empty A, b, S matrices
    A = np.empty(m, dtype=object)
    b = np.empty(m, dtype=object)
    S = np.empty(m, dtype=object)
    
    # create empty matrices to compute the residual error for each party
    dot_prod = np.empty(m, dtype=object)
    r = np.empty(m, dtype=object)

    # get number of features
    d,_ = X[0].shape
    
    # initialize parameters w_0 = 0, S_0 = [n]
    w = np.zeros(d)
    w = w.reshape(-1,1)

    for i in range(m):
        _,ni = X[i].shape
        S[i] = np.diagflat(np.ones(ni))

    for ro in range(rounds) :
        w = admm(X, y, S, rho, admm_steps)
        #print(np.linalg.norm(abs(w - wstar)) )
        #if wstar is not None:
            #if np.linalg.norm(abs(w - wstar)) < epsilon:  
            #    print("here!")
            #    break         
        for i in range(m):
            # Compute dot product <w,x>
            dot_prod[i] = np.matmul(X[i].T,w)
            # Compute residuals r
            r[i] = abs(dot_prod[i] - y[i])            #y - wx
        S = hard_thresholding_admm(r, 1-beta)      
    return w,ro

def admm(X, y, S, rho, k):
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
    d, _ = X[0].shape
    # Initialize variables
    u = np.array([np.zeros((d, 1)) for _ in range(parties)], dtype=object)
    w = np.array([np.zeros((d, 1)) for _ in range(parties)], dtype=object)
    z = np.zeros((d, 1))

    # initialize A, b for each party
    A = np.empty(parties, dtype=object)
    b = np.empty(parties, dtype=object)
    
    for i in range(k):
        znew = np.zeros((d,1))
        #znew = znew.reshape(-1,1)
        for j in range(parties):
            Ahat = X[j]@ S[j]@ X[j].T + rho/2 * np.eye(d)
            A[j] = np.linalg.inv(Ahat)
            b[j] = X[j] @ S[j] @ y[j]
            w[j] = A[j] @ ( b[j] + rho/2 * z - 1/2 * u[j] )
            #print (w[j])
            znew = znew + w[j]
            #print(znew)  
        znew = znew / parties
        #print("z intermediate:", [elem[0] for elem in znew])
        for j in range(parties):
            u[j] = u[j] + rho *(w[j] - znew)
        z = znew    
    return z       

#TORRENT DP 


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

def torrent_dp(X, y,  beta, epsilon, dp_epsilon, dp_delta, max_iters):
    
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
    for iterations in range(max_iters):
        w = update_fc(X,y, S) 
        w = w + dp_noise
        #w = gaussian_mechanism(X, y, dp_epsilon, dp_delta, 1, w)
        # Compute dot product <w,x>
        dot_prod = X.transpose().dot(w)
        # Compute residuals r
        r = dot_prod - y            #y - wx
        # Keep only 1-beta datapoints for the next iteration
        S = hard_thresholding(r,math.ceil((1-beta)*n))
    w = w - dp_noise
    return w, iteration

def torrent_admm_dp(X, y,  beta, epsilon, rho, admm_steps, rounds = 10, wstar= None, dp_X=0, dp_y=0):
    """_summary_

    Args:
        X (_type_): containing parties feature vectors
        y (_type_): containing parties responses
        beta (_type_): corruption rate
        epsilon (_type_): model recovery target error
        rho (_type_): admm rho
        admm_steps (_type_): admm rounss
        rounds (int, optional): torrent rounds

    Returns:
        _type_: model, rounds
    """
    # get number of parties
    m = X.shape[0]
    
    # create empty A, b, S matrices
    A = np.empty(m, dtype=object)
    b = np.empty(m, dtype=object)
    S = np.empty(m, dtype=object)
    
    # create empty matrices to compute the residual error for each party
    dot_prod = np.empty(m, dtype=object)
    r = np.empty(m, dtype=object)

    # get number of features
    d,_ = X[0].shape
    
    # initialize parameters w_0 = 0, S_0 = [n]
    w = np.zeros(d)
    w = w.reshape(-1,1)
    
    for i in range(m):
        _,ni = X[i].shape
        S[i] = np.diagflat(np.ones(ni))
    iteration=0
    
  
    for iteration in range(rounds): 
            w = admm_analyze_gauss(X, y, S, rho, admm_steps, dp_X, dp_y) 
            #w = admm(X, y, S, rho, admm_steps) + sigma*np.random.randn(d, 1)
            for i in range(m):
                # Compute dot product <w,x>
                dot_prod[i] = np.matmul(X[i].T,w)
                # Compute residuals r
                r[i] = abs(dot_prod[i] - y[i])            #y - wx
            S = hard_thresholding_admm(r, 1-beta) 
            iteration= iteration+1      
    #w = admm(X, y, S, rho, admm_steps)
    #print(np.linalg.norm(abs(w - wstar)) )
    return w,iteration

def admm_analyze_gauss(X, y, S, rho, k, dp_X, dp_y):
    """_ consensus admm 
   with analyze gauss from Dwork14
    _   
    """
    # get number of parties, features
    parties = X.shape[0]
    d, _ = X[0].shape
    # Initialize variables
    u = np.array([np.zeros((d, 1)) for _ in range(parties)], dtype=object)
    w = np.array([np.zeros((d, 1)) for _ in range(parties)], dtype=object)
    z = np.zeros((d, 1))

    # initialize A, b for each party
    A = np.empty(parties, dtype=object)
    b = np.empty(parties, dtype=object)
    
    dp_noisex =  dp_X*np.random.randn(d**2, 1)
    dp_noisex = dp_noisex.flatten().reshape(d,d)
    dp_noisey =  dp_y*np.random.randn(d, 1)
    for i in range(k):
        znew = np.zeros((d,1))
        #znew = znew.reshape(-1,1)
        for j in range(parties):
            Ahat = X[j]@ S[j]@ X[j].T + rho/2 * np.eye(d) + dp_noisex
            A[j] = np.linalg.inv(Ahat)
            b[j] = X[j] @ S[j] @ y[j] + dp_noisey
            w[j] = A[j] @ ( b[j] + rho/2 * z - 1/2 * u[j] )
            #print (w[j])
            znew = znew + w[j]
            #print(znew)  
        znew = znew / parties
        for j in range(parties):
            u[j] = u[j] + rho *(w[j] - znew)
        z = znew    
    return z 
