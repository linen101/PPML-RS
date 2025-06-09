from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing,fetch_species_distributions, fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
import pandas as pd

module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
 
from synthetic.strategic_corruptions import generate_synthetic_dataset, strategic_corruption_on_X, adversarial_corruption, rotate_w_arbitrary , sparse_noise_w_arbitrary , interpolate_w_arbitrary, strategic_corruption_scaled, rotate_w_partial
from synthetic.toy_dataset import generate_synthetic_dataset, corrupt_dataset
from plots.plots import plot_regression_errors_n, plot_regression_errors_d, plot_iterations_n, plot_iterations_d
from torrent.torrent import hard_thresholding, torrent, torrent_ideal, torrent_S, torrent_S_with_corruption, torrent_with_residuals
from decimal import *
from sever.sever import sever
import seaborn as sns
from sklearn.linear_model import HuberRegressor, LinearRegression
from realdata.real_data import load_and_process_gas_sensor_data, load_and_process_energy_data
from realdata.pet_finder import load_and_process_pet_finder_data
getcontext().prec = 4
markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed


def run_test_real_dataset(num_trials=3):
    # Load the  dataset
    # Split dataset into training and test sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = load_and_process_energy_data( test_percentage=0.2)
    #X_train, X_test, Y_train, Y_test = load_and_process_pet_finder_data(data_folder="pet_finder")
    # Transpose X_train to match required shape [d, n]
    X_train = X_train.T  # Shape (features, samples)
    X_test = X_test.T


    # Reshape Y_train to ensure it is a column vector
    Y_train = Y_train.reshape(-1, 1)  # Shape (m, 1)
    Y_test = Y_test.reshape(-1, 1)  # Shape (m, 1)
    
    # Compute w_star using the pseudo-inverse 
    #w_star = np.linalg.pinv(X_train@X_train.T) @ (X_train@Y_train)  # (d, n) * (n, 1) →  (d, 1)
    
    huber=HuberRegressor( epsilon=1.35, max_iter=100, alpha=0.1, warm_start=False, fit_intercept=False, tol=0.1).fit(X_train.T, Y_train.ravel())
    w_star = huber.coef_.reshape(-1, 1)


    # Display results
    print("Shapes:")
    print("X_train:", X_train.shape)  # (features, samples)
    print("Y_train:", Y_train.shape)  # (samples, 1)
    print("X_test:", X_test.shape)  # (features, samples)
    print("Y_test:", Y_test.shape)  # (samples,)
    print("w_star shape:", w_star.shape)  # (1, features+1)
    
    
    # Define test parameters
    d, n = X_train.shape
    alpha_values = [ 0.25]  # Corruption rates
    sigma = 0.1  # Noise level
    test_perc = 0.1  # Test set percentage
    
    epsilon = 1e-6  # Convergence threshold
   
    rmse_alpha = np.zeros(len(alpha_values))
    rmse_huber_alpha = np.zeros(len(alpha_values))
    mae_alpha = np.zeros(len(alpha_values))
    mae_huber_alpha = np.zeros(len(alpha_values))
    
    iters_alpha = np.zeros(len(alpha_values))
    
    exec_times_alpha = np.zeros(len(alpha_values))
    convergence_data = {}
   
    corange = Y_train.max() - Y_train.min() 
    # Run multiple trials
    for _ in range(num_trials):
        for i,alpha in enumerate(alpha_values):
            beta = alpha + 0.1  # filter size
            start_time = time.time()
            
            #Y_cor = corrupt_dataset(X_train, Y_train, w_star, alpha, sigma=0)
            w_corrupt = 100+(w_star)
            w_corrupt.reshape(-1,1)
            Y_cor, Y_ind = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)
            # run torrent
            w, it = torrent(X_train, Y_cor, beta, epsilon, max_iters=25)
            #exec_times_alpha.append(time.time() - start_time)
            
            # run huber
            huber = HuberRegressor().fit(X_train.T, Y_cor.ravel())
            w_huber = huber.coef_
            
            
            # Compute predictions on the test set
            Y_pred_test = X_test.T @ w  # Predicted values for Y_test
            Y_pred_test_huber = X_test.T @ w_huber  # Predicted values for Y_test

            # Compute error metrics Torrent
            mae = mean_absolute_error(Y_test, Y_pred_test)
            mse = mean_squared_error(Y_test, Y_pred_test)
            rmse = (mse ** 0.5) / corange  # Root Mean Squared Error divided by the range, wishing for rmse 0.1 -0.2
            rmse_alpha[i] += rmse
            mae_alpha[i] += mae
            
            #Compute error metrics Huber
            mae_huber = mean_absolute_error(Y_test, Y_pred_test_huber)
            mse_huber = mean_squared_error(Y_test, Y_pred_test_huber)
            rmse_huber = (mse_huber ** 0.5) / corange  # Root Mean Squared Error divided by the range, wishing for rmse 0.1 -0.2
            rmse_huber_alpha[i] += rmse_huber
            mae_huber_alpha[i] += mae_huber
            
          
            # Display metrics not so usefull for real dataset.
            # w_error = np.linalg.norm(w - w_star.ravel())
            # print("error:", w_error)
            # w_errors_alpha.append(w_error)
            iters_alpha[i] += it
            #convergence_data[f'alpha={alpha}'] = w_per_iteration

            # Plot predictions vs. actual
            #plt.scatter(Y_test, Y_pred_test, alpha=0.7, colorizer='violet')
            #plt.xlabel("Actual Addoption Speed")
            #plt.ylabel("Predicted Addoption Speed")
            #plt.title("Actual vs. Predicted")
            #plt.show()
          # Create combined scatter plot
          
            ols_model = LinearRegression()
            ols_model.fit(X_train.T, Y_train.ravel())
            Y_pred_ols = ols_model.predict(X_test.T)
            plt.figure(figsize=(8, 6))
            plt.scatter(Y_test, Y_pred_ols, alpha=0.7, color='blue', marker='o', label=f'OLS ($\\beta=0$')
            plt.scatter(Y_test, Y_pred_test, alpha=0.5, color='violet', marker='v', label=f'Torrent ($\\beta={alpha}$)')


            plt.xlabel("Actual Temperature")
            plt.ylabel("Predicted Temperature")
            plt.title("Actual vs. Predicted Temperature (Corrupted vs. Clean Dataset)")
            plt.legend()
            plt.show()
                    
  
    rmse_alpha /= num_trials
    rmse_huber_alpha /= num_trials
    mae_alpha /= num_trials
    mae_huber_alpha /= num_trials
    iters_alpha //= num_trials + 1 
    
      
            # Display results
    print("\nTest Set Error Metrics:")
    print(f"Mean Absolute Error (MAE) Torrent, Huber : {mae_alpha}, {mae_huber_alpha}")
    print(f"Root Mean Squared Error (RMSE) Torrent, Huber : {rmse_alpha}, {rmse_huber_alpha}")

    # Plot  error scaling withn corruption rate
    
    plt.figure()
    plt.plot(alpha_values, rmse_alpha, marker='o',  color="peru", linestyle="dashed" , label='$Torrent$')
    plt.plot(alpha_values, rmse_huber_alpha, marker='o',  color="teal", linestyle="dashed" , label='$Huber$')
    plt.xlabel('Corruption Rate' r'$(\alpha)$' , fontsize=8)
    plt.ylabel(r'RMSE $(y_{test} - \hat{y})$')
    plt.title('$Error$ vs. Corruption Rate' f'$ \ (n={n}, d={d})$',  fontsize=8)
    plt.legend()
    plt.grid(False)
    plt.show()
    
   
    
    plt.figure()
    plt.plot(alpha_values, iters_alpha, marker='o', color="peru", linestyle="dashed" ,  label='Iterations')
    plt.xlabel('Corruption Rate' , fontsize=8)
    plt.ylabel('Iterations', fontsize=8)
    plt.title('Iterations vs. Corruption Rate' f'$ \ (n={n}, d={d})$', fontsize=8)
    #  y-axis increments by 1 
    y_max =  max(iters_alpha)
    plt.yticks(np.arange(1, y_max + 1, 1))  
    plt.grid(False)
    plt.show()
    
    
    """
    plt.figure()
    plt.plot(alpha_values, exec_times_alpha, marker='o',  color="blue", linestyle="dashed" , label='Execution Time')
    plt.xlabel('Corruption Rate ($alpha$)', fontsize=8)
    plt.ylabel('Time (s)', fontsize=8)
    plt.title('Execution Time vs. Corruption Rate' f'$ \ (n={n}, d={d})$', fontsize=8)
    plt.grid(False)
    plt.show()
    """


# Run the tests and generate plots
run_test_real_dataset()
