#irls test corruption rates

import numpy as np

from synthetic.toy_dataset import generate_synthetic_dataset, corrupt_dataset, define_print_params
from stir.irls_regressors import irls_init, truncated_irls
from plots.plots import plot_regression_errors_a, plot_iterations_a
from torrent.torrent import torrent, torrent_ideal
from svam.svam import svam
from sever.sever import sever

def test_corruptions(a_values, n, d, alpha, sigma, test_perc, delta, m_estimators, method_labels, max_iterations, error_tolerance, M ,eta, iters_sever=4, w_star=None):
    # Initialize iteration parameter
    i = 0    
    
    # Initialize weight  arrays
    w_star = np.empty(len(a_values), dtype=object)
    w_robust_M = np.empty((len(a_values), len(m_estimators)), dtype=object)
    w_robust_STIR = np.empty(len(a_values), dtype=object)
    w_robust_STIR_ideal = np.empty(len(a_values), dtype=object)
    w_TORRENT = np.empty(len(a_values), dtype=object)
    w_TORRENT_ideal = np.empty(len(a_values), dtype=object)
    w_svam = np.empty(len(a_values), dtype=object)
    w_svam_ideal = np.empty(len(a_values), dtype=object)
    w_sever = np.empty(len(a_values), dtype=object)
    
    # Initialize  error arrays
    error_M = np.empty((len(a_values), len(m_estimators)), dtype=object)
    error_STIR = np.empty(len(a_values), dtype=object)
    error_STIR_ideal = np.empty(len(a_values), dtype=object)
    error_TORRENT = np.empty(len(a_values), dtype=object)
    error_TORRENT_ideal = np.empty(len(a_values), dtype=object)
    error_svam = np.empty(len(a_values), dtype=object)
    error_svam_ideal = np.empty(len(a_values), dtype=object)
    error_sever = np.empty(len(a_values), dtype=object)
    
    # Initialize iterations arrays for each method
    iters_M = np.empty((len(a_values), len(m_estimators)), dtype=object)
    iters_STIR = np.empty(len(a_values), dtype=object)
    iters_TORRENT = np.empty(len(a_values), dtype=object)
    iters_svam = np.empty(len(a_values), dtype=object)
    
    #Initialize ideal iterations arrays for each method
    iters_M_ideal = np.empty((len(a_values), len(m_estimators)), dtype=object)
    iters_STIR_ideal = np.empty(len(a_values), dtype=object)
    iters_TORRENT_ideal = np.empty(len(a_values), dtype=object)
    iters_svam_ideal = np.empty(len(a_values), dtype=object)
    
    
    for a in a_values:
        X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc)   
    
        # Manipulate percentage a of data 
        Y_cor = corrupt_dataset(X_train, Y_train, w_star, a_values[i], sigma)
    
        # robust estimate of ground truth model w_star using an M-estimator inside IRLS.
        for j in range(len(m_estimators)):
            w_robust_M[i,j], iters_M[i,j] = irls_init(X_train, Y_cor, delta, max_iterations, error_tolerance, m_estimators[j], sigma)
            error_M[i,j] = np.linalg.norm(w_star - w_robust_M[i,j])
        print(f"M {error_M}")
        
        # initialize weights
        rows, columns = X_train.shape
        w = np.zeros(rows)
        weights = w.reshape(-1,1)
        
        #### STIR ####
        # robust estimate of ground truth model w_star using a stagewise truncated IRLS.
        Μ = np.linalg.norm(weights - w_star)
        w_robust_STIR[i], iters_STIR[i] = truncated_irls(X_train, Y_cor, M, max_iterations, error_tolerance, eta)
        
        # STIR error between estimated and gold model
        error_STIR[i] = np.linalg.norm(w_star - w_robust_STIR[i])
        print(f"STIR {error_STIR}")
        
        
        ### TORRENT ###
        # robust estimate of ground truth model w_star using a ols with hard thresholding.
        beta = a + 0.1
        w_TORRENT[i], iters_TORRENT[i] = torrent(X_train, Y_cor, beta, error_tolerance, max_iterations)
        
        # TORRENT error between estimated and gold model
        error_TORRENT[i] = np.linalg.norm(w_star - w_TORRENT[i])
        print(f"torrent {error_TORRENT}")
        
        ### SVAM ###
        #svam test
        ksi = 1.1           # step to increase variance
        beta_svam = 0.05     # variance
        epsilon = 1e-6      # tolerance
        K = 20              # maximum iterations
       
        # robust estimate of ground truth model w_star using svam.
        w_svam[i], iters_svam[i]  = svam(X_train, Y_cor, weights, beta_svam, ksi, max_iterations, epsilon )
        
        # SVAM error between estimated and gold model
        error_svam[i] = np.linalg.norm(w_star - w_svam[i])
        print(f"svam {error_svam}")
        
        
        # SEVER
        # parameters needed
        ridge_sever = 0 # ridge regression parameter
        d_sever,n_sever = X_train.shape
        S = [j for j in range(0, n_sever)]
    
        # sever model computed
        w_sever[i] = sever(X_train, Y_cor, S, ridge_sever, a_values[i], iters_sever)
        
        # sever error between estimated and gold model
        error_sever[i] = np.linalg.norm(w_star - w_sever[i])
        print(f"sever {error_sever}")
        
        ### IDEAL SETTING ###
        # count iterations until convergence knowing the gold model
        w_robust_STIR_ideal[i], iters_STIR_ideal[i] = truncated_irls(X_train, Y_cor, M, max_iterations, error_tolerance, eta, w_star)
        error_STIR_ideal[i] = np.linalg.norm(w_star - w_robust_STIR_ideal[i])
        w_TORRENT_ideal[i], iters_TORRENT_ideal[i] = torrent_ideal(X_train, Y_cor, beta, error_tolerance, max_iterations, w_star)
        error_TORRENT_ideal[i] = np.linalg.norm(w_star - w_TORRENT_ideal[i])
        w_svam_ideal[i], iters_svam_ideal[i] = svam(X_train, Y_cor, weights, beta_svam, ksi, max_iterations, epsilon, w_star )
        error_svam_ideal[i] = np.linalg.norm(w_star - w_svam_ideal[i])
        i = i + 1
    
    #print error diagrams
    #errors =  np.vstack(([error_M[:, i] for i in range(len(m_estimators))], error_STIR, error_TORRENT, error_STIR_ideal))
    errors =  np.vstack(( error_STIR, error_TORRENT, error_svam, error_sever))
    method_labels_new = method_labels 
    plot_regression_errors_a(errors, n, d, a_values, sigma, method_labels, error_tolerance, M ,eta)
    
    '''
    #print iteration diagrams
    #iters =  np.vstack(([iters_M[:, i] for i in range(len(m_estimators))], iters_STIR, iters_TORRENT ))
    iters =  np.vstack(( iters_STIR, iters_TORRENT , iters_svam, iters_STIR_ideal, iters_TORRENT_ideal, iters_svam_ideal))
    plot_iterations_a(iters, n, d, a_values, sigma, method_labels_new, error_tolerance, M ,eta)
    '''
    