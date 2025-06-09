#torrent test samples

import numpy as np

from synthetic.toy_dataset import generate_synthetic_dataset, corrupt_dataset, define_print_params
from stir.irls_regressors import irls_init, truncated_irls
from plots.plots import plot_regression_errors_a,  plot_iterations_a, add_ideal
from torrent.torrent import torrent, torrent_ideal

def test_corruptions_torrent(n, d, a_values, sigma, test_perc, delta, m_estimators, method_labels, max_iterations, error_tolerance, M ,eta, w_star=None):
    # Initialize weight and error arrays and iteration parameter
    i = 0    
    Y_cor = np.empty(len(a_values), dtype=object)
    error_TORRENT = np.empty(len(a_values), dtype=object)
    w_TORRENT = np.empty(len(a_values), dtype=object)
    error_TORRENT_ideal = np.empty(len(a_values), dtype=object)
    w_TORRENT_ideal = np.empty(len(a_values), dtype=object)    

    
    # Initialize iterations arrays for each method

    iters_TORRENT = np.empty(len(a_values), dtype=object)
    iters_TORRENT_ideal = np.empty(len(a_values), dtype=object)
    
    #Initialize ideal iterations arrays for each method

    iters_TORRENT_ideal = np.empty(len(a_values), dtype=object)
    
    for a in a_values:
        X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc)   
    
        # Manipulate percentage a of data 
        Y_cor[i] = corrupt_dataset(X_train, Y_train, w_star, a_values[i], sigma)
    
        beta = a_values[i] + 0.1
        w_TORRENT[i], iters_TORRENT[i] = torrent(X_train, Y_cor[i], beta, error_tolerance)
        
        # TORRENT error between estimated and gold model
        error_TORRENT[i] = np.linalg.norm(w_star - w_TORRENT[i])
        print(f"Torrent {error_TORRENT}")
        # count iterations knowing gold model
        w_TORRENT_ideal[i], iters_TORRENT_ideal[i] = torrent_ideal(X_train, Y_cor[i], beta, error_tolerance, w_star)
        
        error_TORRENT_ideal[i] = np.linalg.norm(w_star - w_TORRENT_ideal[i])
        
        i = i + 1
    
  
    #print error diagrams
    errors =  np.vstack((error_TORRENT, error_TORRENT_ideal))
    method_labels_new = ['TORRENT', 'TORRENT_ideal']
   
    plot_regression_errors_a(errors, n, d, a_values, sigma, method_labels_new)
    
    #print iteration diagrams
    iters =  np.vstack(( iters_TORRENT, iters_TORRENT_ideal ))
    plot_iterations_a(iters, n, d, a_values, sigma, method_labels_new)  
