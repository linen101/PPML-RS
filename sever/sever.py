# sever.py

import numpy as np
import numpy.matlib as npm
import numpy.random as npr

import scipy as sc
import scipy.stats.mstats as scstatsm

import math
"""
We are assuming samples x \in R^{d x n}, 
that is data X is a [d, n] matrix,
labels y is a [n,1] matrix, 
weights w is a [d,1] matrix,
gradients g is a [d,n] matrix,
sum of gradients is a [1,d] matrix
and centered gradients is also a [d,n] matrix
"""


def ridge_regr(X, y, alpha):
    """ 
    Parameters:
    - Training data: {X, y}
    - alpha: regularization parameter
    Returns:
    - w: estimated weights
    """ 
    
    #(XX^T), (Xy)
    # Compute intermediate matrices
    XXT = np.matmul(X, X.T)
    Xy = np.matmul(X, y)
    
     #(XX^T + aI)^{-1}(Xy)
    # Compute ridge regression coefficients
    n_features = X.shape[0]
    w = np.matmul(np.linalg.inv(XXT + alpha * np.eye(n_features)), Xy)
    
    return w

def ridge_regr_gd(X, y, alpha=1.0, max_iterations=20, tol=0.0001, w_star=None):
    """ 
    Parameters:
    - Training data: {X, y}
    - max_iterations : maximum number of iterations for convergence
    - alpha: regularization parameter
    - tol: error tolerance between estimated and gold model. in truncated irls this changes in every irls call
    - w_star: real model, is used to compute the ideal convergence
    Returns:
    - w: estimated weights
    """ 
    flag = True
    iteration = 0
    # Initialize weights
    w = np.zeros(X.shape[0])
    w = w.reshape(-1,1)
    while iteration < max_iterations & flag:
        # Compute the gradient
        gradient = -2 * np.matmul(X, (y - np.dot(X.T, w))) + 2 * alpha * w
        
        # Update the weights using the gradient
        w_new = w - 0.001 * gradient  # adjust the step size (??) (0.001 in this case)
        
        # Check for convergence
        if np.linalg.norm(w_new - w) < tol:
            print("Converged at iteration:", iteration)
            flag = False
        
        # Update weights
        w = w_new
        
        iteration += 1
    
    return w

# Example usage:
# w = ridge_regr(X_train, y_train, alpha=0.5)

def compute_grads(X,y, w, r):
    """ 
    Parameters:
    - {X, y}: Training data
    - w: Estimated weights
    - r: regularization parameter
    Returns:
    - g [d,n] : gradients matrix
    """ 
    n = X.shape[1]
    
    # y - x^T*w, y: [n,1]
    losses = y - np.matmul(X.T,w)
    
    # [n,1] --> [n,n]
    losses = np.diagflat(losses)
    
    # x (y - x^T*w ), losses: [d,n]
    g_no_reg = np.matmul(X,losses)
    
    # r*w, reg : [d,n]
    reg = r * npm.repmat(w, 1, n)
    
    # x (y - x^T*w ) + rw : [d,n]
    g = g_no_reg + reg
    
    return g

def compute_average_grads(g):
    n = g.shape[1]
    
    # Σ_{i ín S} (1/n)[xi (xi^T - yi)] , m:[1,d]
    m = np.sum(g, axis=1) / n
    return m

def compute_centered_grads(g,m):  
    n = g.shape[1]  
    
    # [xi (xi^T - yi)] - Σ_{i ín S} (1/n)[xi (xi^T - yi)], g: : [d,n]
    g_cen = g - npm.repmat(m, n, 1).T 
    return g_cen

def filter(S, scores, p):
    """_summary_
    in each iteration remove the top p fraction of outliers
    according to the scores scores_i.
    Args:
        S (_type_): set of active datapoints' indices. e.g S=[1,2,3,6,8,10] 
        scores (_type_): score matrix for each datapoint, scores[n]
        p (_type_): how many outliers to remove
    """
    #Remove the p fraction of largest scores
    #If this would remove all the data, remove nothing
    
    # 1-p quantile equals z means that (1-p)% of scores are less than or equal to z
    quantile = scstatsm.mquantiles(scores, 1 - p, alphap=0.5, betap=0.5)
    
    # if quantile <= 0 that means that this condition: 
    # scores[i] < 1.0 will remove all points
    # since scores[i]/ quantile
    # and we dont want that!
    if quantile > 0:
        #print(quantile)
        scores = scores / quantile
        indices = [i for i in range(len(S)) if scores[i] < 1.0]
        #print(indices)
    else:
        scores = scores / np.max(scores)
        indices = S
   
    # Print quantile value and resulting scores
    #print("Quantile:", quantile)
    #print("Filtered Scores:", scores)
    
    return indices


def filter_HT(S, scores, k):
    """
    check "practical considerations" section in the paper
    Parameters:
    - S: list of indices    
    - scores : Input vector.
    - k : Number of elements to keep.

    Returns:
    - list_indices : List of indices from the original vector scores that we keep.
    """
    
    # Create a modified vector by sorting the elements of scores in ascending order of their magnitude
    # BUT this will sort only the elements of scores that correspond to the indices in the list S.
    sorted_indices = sorted(range(len(S)), key=lambda i: abs(scores[i]))
    
    # Keep only the first k elements of the modified vector
    list_indices = [i for i in sorted_indices[:k]]
    
    return list_indices    

def power_iter_method(X):
    """computes the top right singular vector, 
    top left singular vector and singular value
    of the matrix X.

    Args:
        X (_type_): [d,n] data matrix

    Returns:
        [d,1], real, [n,1]: top right sing vec, singular value, top left sing vec
    """
    flag = True
    d,n = X.shape
    right0 = npr.randn(n, 1)
    #s = log(4 log(2n/δ)/εδ)/2λ
    eps = 1e-6
    #delta = 0.01
    #l = 0.1
    #s = math.log(4 * math.log (2*n / delta) / eps * delta)/ 2 * l
    while flag:
    #for i in range(int(s)):
        
        #x_i = A.TAx_{i−1}
        right = np.dot(np.dot(X.T, X),right0)
        right /=  np.linalg.norm(right)
        if np.linalg.norm(right - right0)<= eps:
            flag = False
        right0 = right
    Xr = np.dot(X, right)
    sigma = np.linalg.norm(Xr)
    left = Xr / sigma
    return (left, sigma, right)

def compute_outlier_scores(g,v):
    """ computes outlier scores 
    project centered gradients
    onto the direction of the top left singular vector
    of the matrix G of centered gradients.

    Args:
        g (_type_): [d,|S|] matrix of centered gradients
        v (_type_): top left singular vector [d,1]

    Returns:
        _type_: _description_
    """
    s = np.matmul(g.T,v)
    s = s**2
    return s

def sever(X, y, S, r, a, iters):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        S (_type_): _description_
        r (_type_): _description_
        a (_type_): _description_
        iters (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(iters):
        #print(i)
        d,n = X.shape
        X = X[:, S]
        y = y[ S]
        w = ridge_regr(X, y , r)
        #print(w)
    
        g = compute_grads(X, y, w, r)
        #print(g)
    
        m = compute_average_grads(g)
        #print(m)  
    
        G = compute_centered_grads(g,m)
        #print(G)
    
        L,s,R = power_iter_method(X)
        #print(L)  # we want that because we have X: [d,n]
    
        #print(R) 
    
        #print(s)
    
        t = compute_outlier_scores(G,L)
    
        #print(t)
        # each time removing the
        # p = ε/2 fraction of points with largest score.
        #print(len(S), len(t), math.ceil((1-a/2)*len(S)))
        S = filter_HT(S, t, math.ceil((1-a/2)*len(S)))
   
    return w
    
    
    
def severq(X, y, S, r, a, iters):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        S (_type_): _description_
        r (_type_): _description_
        a (_type_): _description_
        iters (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(iters):
        #print(i)
        d,n = X.shape
        X = X[:, S]
        y = y[ S]
        w = ridge_regr(X, y , r)
        #print(w)
    
        g = compute_grads(X, y, w, r)
        #print(g)
    
        m = compute_average_grads(g)
        #print(m)  
    
        G = compute_centered_grads(g,m)
        #print(G)
    
        L,s,R = power_iter_method(X)
        #print(L)  # we want that because we have X: [d,n]
    
        #print(R) 
    
        #print(s)
    
        t = compute_outlier_scores(G,L)
    
        #print(t)
        # each time removing the
        # p = ε/2 fraction of points with largest score.
        
        S = filter(S, t, (a/2))
   
    return w
    
        