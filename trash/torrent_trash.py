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
        znew = np.zeros((d,1))
        #znew = znew.reshape(-1,1)
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
        S = hard_thresholding_admm(r, 1-beta)       
    return w,ro
def torrent_admm_dp(X, y,  beta, epsilon, rho, dp_epsilon, dp_delta, admm_steps, rounds = 10, wstar= None):
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
    iteration = 0
    # DP noise
    sigma = (np.sqrt(2 * np.log(2 / dp_delta)) / dp_epsilon) * 1
    dp_noise = sigma * np.random.randn(d, 1)
    while np.linalg.norm(abs(w - wstar)) > epsilon:
        if iteration > rounds:
            w = admm(X, y, S, rho, admm_steps, Ainit, binit) 
            break
        else:
            w = admm(X, y, S, rho, admm_steps, Ainit, binit) + dp_noise       
        for i in range(m):
            # Compute dot product <w,x>
            dot_prod[i] = np.matmul(X[i].T,w)
            # Compute residuals r
            r[i] = abs(dot_prod[i] - y[i])            #y - wx
        S = hard_thresholding_admm(r, 1-beta, S)  
        iteration = iteration + 1     
    return w,iteration