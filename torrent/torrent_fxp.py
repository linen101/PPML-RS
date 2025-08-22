#TORRENT ADMM
import numpy as np
import sys
import os
import math
import random
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from fixed_point.fixed_point_helpers import *    
from synthetic.toy_dataset import *
#from torrent import split_matrix, split_matrix_Y, torrent_admm, q_quantile


def count_smaller_than(residuals, m):
    count = int(0)
    for i, residual in enumerate(residuals):
        #def _(i):
        c = bool(residual < m)
        count = count + c
    return int(count)

def count_greater_than(residuals, m):
    count = int(0)
    for i, residual in enumerate(residuals):
        #def _(i):
        c = residual > m
        count = count + c
    return int(count)

def fxp_dist_quantile(r, q):
    n = sum(ri.size for ri in r)    
    m = len(r)    
    quantile = int(np.ceil(q * (n)))
    
    bit_precision_dec = FXP_CONFIG.get("n_frac", 0)
    bit_precision_total = FXP_CONFIG.get("n_word", 0)
    # min positive value in domain
    step = fxp(2 ** (-bit_precision_dec))
    alpha = step
    # alpha = 0
    # max value in domain
    beta = fxp(2**(bit_precision_total - bit_precision_dec -1))
    #beta = 10
    less_than_player = np.zeros(m, dtype=int)
    greater_than_player = np.zeros(m, dtype=int)

    for i in range(bit_precision_total):
        less_than_sum = int(0)
        greater_than_sum = int(0)
        qvalue = (alpha + beta)/2
        #print(qvalue)
        for j in range(m):
            less_than_player[j] = count_smaller_than(r[j], qvalue) 
            greater_than_player[j] = count_greater_than(r[j], qvalue) 
            less_than_sum += less_than_player[j]
            greater_than_sum += greater_than_player[j]
        #print(less_than_sum)
        #print(greater_than_sum)    
        if (less_than_sum >= quantile):
            beta = qvalue - step
            #print('less')
            #print(qvalue)
        if (greater_than_sum >= n - quantile + 1):
            alpha = qvalue + step  
            #print('more')
            #print(qvalue)
        if ((less_than_sum <= quantile-1) & (greater_than_sum <= n - quantile)):
            #print('Quantile found')
            #print(qvalue)
            break     
    return(qvalue)

def fxp_quantile(r, q):
    #print(f'residuals1:{r}')
    n = sum(ri.size for ri in r)
    #print(n)
    rvalues = fxp(np.empty(n))
    m = len(r)
    ctr = 0
    for i in range (m):
        ni = r[i].shape[0]
        r_flt = r[i].flatten()
        #print(r_flt)
        #print(ni)
        for j in range (ni):
            rvalues[j+ctr] = r_flt[j]
            #print(rvalues[j+ctr])
        ctr+=ni
    # Sort and compute quantile index
    #print(rvalues)
    sorted_rvalues = np.sort(rvalues)
    #print(sorted_rvalues)
    idx = int(np.ceil(q * (n-1)))
    qvalue = sorted_rvalues[idx]
    #print(q.info())
    return(qvalue)

## DP with FXP 

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
        w_beta = fxp(1)
        w_alpha = (math.exp(dp_e*(u - quantile))) # divide by n here to normalize
        #print(f'(upper range) Number of elements less than {m} is {u} and weight is: {wu_alpha} and fxp weight is {fxp(wu_alpha)}')
    else:
        #print(f'(lower range) Number of elements less than {m} is {u}')
        w_beta = (math.exp(dp_e*(quantile - u))) # divide by n here to normalize
        w_alpha = fxp(1)
        #print(f'(lower range) Number of elements less than {m} is {u} and weight is: {wl_beta} and fxp weight is {fxp(wl_beta)}')
    return (w_alpha, w_beta)            
        
def select_range(w_alpham, w_mbeta, alpha, beta, m, step):
    w_total = w_mbeta + w_alpham
    t = random.uniform(0+step,1+step)
    #print(f't is :{t}')
    w_total_random = w_total * t
    #print (f'r is: {r.info()}')
    c = w_alpham < w_total_random
    if (c==1):
        #print(f'choose upper')
        return (m+step, beta)
    else:
        #print(f'choose lower')
        return (alpha, m-step)
          
def dp_fxp_dist_quantile(r, q, dp_e=1):
    n = sum(ri.size for ri in r)    
    parties = len(r)    
    quantile = int(np.ceil(q * (n)))
    
    bit_precision_dec = FXP_CONFIG.get("n_frac", 0)
    bit_precision_total = FXP_CONFIG.get("n_word", 0)
    
    # step in domain (e.g. in integers step=1)
    step = fxp(2 ** (-bit_precision_dec))
    
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
        if (alpha==beta):
            m = alpha
            break
    return(m)
"""
# ---- EXAMPLE ----
# Create example fixed-point matrices
np.random.seed(42)
mat1 = np.random.uniform(low=0, high=100, size=20)  # 3x1 matrix
#mat1 = np.random.randn(3,1)  # 2x1 matrix

mat1_fxp= fxp(mat1)
mat2 = np.random.randn(10,1)  # 2x1 matrix
mat2_fxp= fxp(mat2)
r = [mat1_fxp, mat2_fxp]
q1 = np.zeros(100)
q2 = np.zeros(100)
q3 = np.zeros(100)
q1_sum = 0
q2_sum = 0
q3_sum = 0
for i in range (100):
    q1[i] = fxp_quantile(r, 0.5)
    q2[i] = fxp_dist_quantile(r, 0.5)
    q3[i] = dp_fxp_dist_quantile(r, 0.5)
    q1_sum += q1[i]
    q2_sum += q2[i]
    q3_sum += q3[i]

q1_sum /= 100
q2_sum /= 100
q3_sum /= 100  
print(f'fxp quantile: {q1_sum}, fxp distributeed: {q2_sum}, dp fxp distributeed: {q3_sum}')
"""
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
    quant = dp_fxp_dist_quantile(r, q)
    #print(f'Torrent fxp quantile:{quant}')
    S = np.empty(m, dtype=object)
    for i in range(m):
        S[i] = np.array([1 if value < quant else 0 for value in  r[i].flatten()])
        S[i] = np.diagflat(S[i])

    return S

def admm_fxp(X, y, S, rho, k):
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
    u = (np.array([(np.zeros((d, 1))) for _ in range(parties)], dtype=object))
    w = (np.array([(np.zeros((d, 1))) for _ in range(parties)], dtype=object))
    z = (np.zeros((d, 1)))

    # initialize A, b for each party
    A = np.empty(parties, dtype=object)
    b = np.empty(parties, dtype=object)
    #D = np.empty(parties, dtype=object)
    
    for i in range(k):
        znew = fxp(np.zeros((d, 1)))
        #znew = znew.reshape(-1,1)
        for j in range(parties):
            #print("X[j].shape =", X[j].shape, "S[j].shape =", S[j].shape, "j=", j, "i=", i)
            D = np.matmul(X[j], S[j])
            
            Ahat = np.matmul(D, X[j].T)
            A[j] = np.linalg.inv(Ahat)
            b[j] = np.matmul(D, y[j])
            w[j] = np.matmul(A[j], b[j] + rho/2 * z - 1/2 * u[j] )
            #print (w[j])
            znew = znew + w[j]
            #print(znew)  
        znew = znew / parties
        #print("z intermediate:",  znew)
        for j in range(parties):
            u[j] = u[j] + rho  *(w[j] - znew)
        z = znew    
    return z   

def torrent_admm_fxp(X, y,  beta, epsilon, rho, admm_steps, rounds, wstar, dp_w):
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

    if wstar is not None:
        norm_w = np.linalg.norm((wstar))
        norm_w_inv = 1/norm_w

    # get number of parties
    m = X.shape[0]
    
    # create empty S matrix
    S = np.empty(m, dtype=object)
    
    # create empty matrices to compute the residual error for each party
    dot_prod = (np.empty(m, dtype=object))
    r = (np.empty(m, dtype=object))

    # get number of features
    d,_ = X[0].shape
    
    #n = size(X)
    
    # initialize parameters w_0 = 0, S_0 = [n]
    w = fxp(np.zeros((d, 1)))
    iteration=0
    for i in range(m):
        _,ni = X[i].shape
        S[i] = np.diagflat(np.ones(ni))

    for ro in range(rounds) :
        w = admm_fxp(X, y, S, rho, admm_steps)
        #w = admm_fxp(X, y, S, rho, admm_steps) + fxp(dp_w*np.random.randn(d, 1))
        if wstar is not None:
            error= np.linalg.norm((w - wstar))
            error_norm = error * norm_w_inv
            print(f'DP ols error is: {error_norm}' )
            #    break         
        for i in range(m):
            # Compute dot product <w,x>
            dot_prod[i] = np.matmul(X[i].T,w)
            # Compute residuals r
            r[i] =  abs(y[i] - dot_prod[i]) #y - wx
            #print(f'res in tor: {r[i]}')
        S = hard_thresholding_admm(r, 1-beta)  
        iteration = iteration + 1     
    return w,ro

def torrent_admm_fxp_analyze_gauss(X, y,  beta, epsilon, rho, admm_steps, rounds, wstar, dp_noise_x, dp_noise_y):
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
    norm_w = np.linalg.norm((wstar))
    norm_w_inv = 1 / norm_w
    #print(f'dp noise is: {dp_noise}')
    # get number of parties
    m = X.shape[0]
    
    # create empty S matrix
    S = np.empty(m, dtype=object)
    
    # create empty matrices to compute the residual error for each party
    dot_prod = (np.empty(m, dtype=object))
    r = (np.empty(m, dtype=object))

    # get number of features
    d,_ = X[0].shape
    
    #n = size(X)
    
    # initialize parameters w_0 = 0, S_0 = [n]
    w = fxp(np.zeros((d, 1)))
    iteration=0
    for i in range(m):
        _,ni = X[i].shape
        S[i] = np.diagflat(np.ones(ni))

    for ro in range(rounds) :
        #w = admm_fxp(X, y, S, rho, admm_steps)
        w = admm_fxp_analyze_gauss(X, y, S, rho, admm_steps, dp_noise_x, dp_noise_y)
        if wstar is not None:
            #print(f'fxp unormalised analyze gauss error is: {np.linalg.norm((w - wstar))}' )
            print(f'fxp  analyze gauss error is: {np.linalg.norm((w - wstar))* norm_w_inv }' )
        #    if np.linalg.norm((w - wstar))* norm_w_inv < epsilon:  
        #        break         
        for i in range(m):
            # Compute dot product <w,x>
            dot_prod[i] = np.matmul(X[i].T,w)
            # Compute residuals r
            r[i] = abs(dot_prod[i] - y[i])          #y - wx
            #print(f'res in tor: {r[i]}')
        S = hard_thresholding_admm(r, 1-beta)  
        iteration = iteration + 1     
    return w,ro

def admm_fxp_analyze_gauss(X, y, S, rho, k, dp_noise_x, dp_noise_y):
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
    u = (np.array([(np.zeros((d, 1))) for _ in range(parties)], dtype=object))
    w = (np.array([(np.zeros((d, 1))) for _ in range(parties)], dtype=object))
    z = (np.zeros((d, 1)))

    # initialize A, b for each party
    A = np.empty(parties, dtype=object)
    b = np.empty(parties, dtype=object)
    #D = np.empty(parties, dtype=object)
    dp_noisex =  dp_noise_x*np.random.randn(d**2, 1)
    dp_noisex = dp_noisex.flatten().reshape(d,d)
    dp_noisey =  dp_noise_y*np.random.randn(d, 1)
    for i in range(k):
        znew = fxp(np.zeros((d, 1)))
        #znew = znew.reshape(-1,1)
        for j in range(parties):
            #print("X[j].shape =", X[j].shape, "S[j].shape =", S[j].shape, "j=", j, "i=", i)
            D = np.matmul(X[j], S[j])
            
            Ahat = np.matmul(D, X[j].T)  + dp_noisex
            A[j] = np.linalg.inv(Ahat)
            b[j] = np.matmul(D, y[j]) + dp_noisey
            w[j] = np.matmul(A[j], b[j] + rho/2 * z - 1/2 * u[j] )
            #print (w[j])
            znew = znew + w[j]
            #print(znew)  
        znew = znew / parties
        #print("z intermediate:",  znew)
        for j in range(parties):
            u[j] = u[j] + rho  *(w[j] - znew)
        z = znew    
    return z   


np.random.seed(42)
m, d, n, test_percentage, sigma = 3, 2, 50, 0.1, 0.1  # 3 parties, 2 features, 5 samples each
"""
# Example1
X = np.empty(m, dtype=object)
y = np.empty(m, dtype=object)
true_w = np.array([[1.5], [-2.0]])
for i in range(m):
    Xi = np.random.randn(d, n)
    noise = 0.05 * np.random.randn(n, 1)
    yi = Xi.T @ true_w + noise
    X[i] = fxp_array(Xi)
    y[i] = fxp_array(yi.flatten())

w, rounds = torrent_admm(X, y, beta=0.2, epsilon=0.1, rho=1, admm_steps=5, rounds=5, wstar=fxp_array(true_w))
print("Recovered w (float):", [elem[0] for elem in w])
print("Rounds:", rounds)
"""
"""
# Example2
X, y, X_test, y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_percentage)

X_fxp = fxp(X)
y_fxp = fxp(y)

Xdist = split_matrix(X, m, n)
ydist = split_matrix_Y(y, m, n)
#Xdist_fxp = split_matrix(X_fxp, m, n)
#ydist_fxp = split_matrix_Y(y_fxp, m, n)
Xdist_fxp = (np.empty(len(Xdist), dtype=object))
ydist_fxp = (np.empty(len(ydist), dtype=object))

for i in range(len(Xdist)):
    Xdist_fxp[i] = fxp(Xdist[i])
    ydist_fxp[i] = fxp(ydist[i])

print("torrent fpx:")
w, rounds = torrent_admm_fxp(Xdist_fxp, ydist_fxp, beta=0.1, epsilon=0.1, rho=1, admm_steps=5, rounds=5)

print("torrent:")
w, rounds = torrent_admm(Xdist, ydist, beta=0.1, epsilon=0.1, rho=1, admm_steps=5, rounds=5)
"""

