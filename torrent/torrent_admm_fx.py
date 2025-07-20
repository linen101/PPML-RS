#TORRENT ADMM
import numpy as np
import sys
import os
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from fixed_point.fixed_point_helpers import *    
from synthetic.toy_dataset import *
from torrent import split_matrix, split_matrix_Y, torrent_admm, q_quantile


def fxp_quantile(r, q):
    n = sum(ri.size for ri in r)
    print(n)
    rvalues = np.empty(n)
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
    idx = int(np.ceil(q * (n - 1)))
    q = sorted_rvalues[idx]
    #print(q.info())
    return(q)

# ---- EXAMPLE ----
"""

# Create example fixed-point matrices
np.random.seed(42)
mat1 = np.random.randn(3,1)  # 3x1 matrix
mat1_fxp= fxp(mat1)
#print(mat1_fxp.info())
mat2 = np.random.randn(3,1)  # 2x1 matrix
mat2_fxp= fxp(mat2)
r = [mat1_fxp, mat2_fxp]
q = fxp_quantile(r, 0.5, 6)
print(q)
r = [mat1, mat2]
#print(r)
print(abs(mat1))

print(mat1_fxp)
mat1_flat_fxp = mat1_fxp.flatten()
mat1_flat_fxp_abs = abs(mat1_flat_fxp)
mat1_sort= np.sort(mat1_flat_fxp_abs)
print(mat1_sort)
r_fxp = [mat1_fxp, mat2_fxp]  # list of matrices
r_fxp_flat = [mat.flatten() for mat in r_fxp]
print(r_fxp_flat)
X=np.matmul(mat1_fxp, mat2_fxp)
print(X)
Xinv= np.linalg.inv(X)
np.concatenate(r)
#r_fxp_flat_concat = np.concatenate(r_fxp_flat)
#print(r_fxp_flat_concat)

#print(r_fxp)
# Compute median (0.5 quantile)
#smedian = q_quantile(r, q=0.5)
#median_fxp = fxp_quantile(r_fxp, q=0.5, n_frac=16)

#print("median:", median)
#print("Fixed-point median (as Fxp):", median_fxp)
#print("Underlying fixed-point integer value:", median_fxp.val)
#print("Float equivalent:", median_fxp.get_val())
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
    quant = fxp_quantile(r, q)
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
    u = fxp(np.array([(np.zeros((d, 1))) for _ in range(parties)], dtype=object))
    w = fxp(np.array([(np.zeros((d, 1))) for _ in range(parties)], dtype=object))
    z = fxp(np.zeros((d, 1)))

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
        print("z intermediate:",  znew)
        for j in range(parties):
            u[j] = u[j] + rho  *(w[j] - znew)
        z = znew    
    return z   

def torrent_admm_fxp(X, y,  beta, epsilon, rho, admm_steps, rounds = 10, wstar= None):
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

    for i in range(m):
        _,ni = X[i].shape
        S[i] = fxp(np.diagflat(np.ones(ni)))

    for ro in range(rounds) :
        w = admm_fxp(X, y, S, rho, admm_steps)
        #print(np.linalg.norm(abs(w - wstar)) )
        if wstar is not None:
            if np.linalg.norm(abs(w - wstar)) < epsilon:  
                break         
        for i in range(m):
            # Compute dot product <w,x>
            dot_prod[i] = np.matmul(X[i].T,w)
            # Compute residuals r
            r[i] = abs(dot_prod[i] - y[i])          #y - wx
        S = hard_thresholding_admm(r, 1-beta)       
    return w,ro


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
