import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed

import sys
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)

def add_ideal(labels):
    return ['ideal ' + label for label in labels]

def plot_regression_errors_n(errors, sample_sizes, dimensionality, corruption_rate, data_error_variance, method_labels, error_tolerance=None, M=None, eta=None):
    # Plotting
    plt.figure(figsize=(10, 6))
    flat_errors = np.hstack(errors)  # Flatten the list of error arrays to get overall min and max
    for i, error in enumerate(errors):
        marker = markers[i % len(markers)]  # Cycling through markers
        w_star_w_error = np.abs(error)
        plt.plot(sample_sizes, w_star_w_error, marker=marker, label=method_labels[i])
        
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Error (|w^* - w|)')
    plt.legend()
    plt.grid(True)
    plt.xticks(sample_sizes)

    # Setting y-axis limits based on percentiles to reduce the effect of outliers
    lower_percentile = np.percentile(flat_errors, 1)  # 5th percentile
    upper_percentile = np.percentile(flat_errors, 99)  # 95th percentile
    y_margin = 0.1 * (upper_percentile - lower_percentile)  # Add a 10% margin
    plt.ylim(lower_percentile - y_margin, upper_percentile + y_margin)

    title = f'Regression Errors vs. Number of Samples (d: {dimensionality}, a: {corruption_rate}, σ: {data_error_variance})'
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.95)
    
    # Save the plot
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../Figures/Samples', '{}-{}-{}-{}.png'.format('error_samples', f'd:{dimensionality}', f'a:{corruption_rate}', f'σ:{data_error_variance}'))
    plt.savefig(save_path)
    plt.show

    
    
def plot_regression_errors_d(errors, sample_size, dimensionality, corruption_rate, data_error_variance, method_labels, error_tolerance, M , eta):
        # Plotting
    plt.figure(figsize=(10, 6))
    flat_errors = np.hstack(errors)  # Flatten the list of error arrays to get overall min and max
    for i, error in enumerate(errors):
        marker = markers[i % len(markers)]  # Cycling through markers
        w_star_w_error = np.abs(error)
        plt.plot(dimensionality, w_star_w_error, marker = marker, label=method_labels[i])

    #plt.title('Regression Errors vs. Dimensionality')
    plt.xlabel('Number of Dimensions (d)')
    plt.ylabel('Error (|w^* - w|)')
    plt.legend()
    plt.grid(True)
    
    # Setting y-axis limits based on percentiles to reduce the effect of outliers
    lower_percentile = np.percentile(flat_errors, 1)  # 5th percentile
    upper_percentile = np.percentile(flat_errors, 99)  # 95th percentile
    y_margin = 0.1 * (upper_percentile - lower_percentile)  # Add a 10% margin
    plt.ylim(lower_percentile - y_margin, upper_percentile + y_margin)

    
    # Set the x-ticks to only the values of d dimensions
    plt.xticks(dimensionality)
    title = f'Regression Errors vs. Dimensionality  ( n: {sample_size}, a: {corruption_rate}, σ: {data_error_variance})'
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.95)  # y parameter adjusts the vertical position of the title
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../Figures/Dimensions', '{}-{}-{}-{}.png'.format('error_dimensions', f'd:{sample_size}', f'a:{corruption_rate}', f'σ:{data_error_variance}'))
    plt.savefig(save_path)
    #    plt.gca().set_xticks(sample_sizes)

def plot_regression_errors_a(errors, sample_size, dimensionality, corruption_rates, data_error_variance, method_labels, error_tolerance, M ,eta):
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, error in enumerate(errors):
        w_star_w_error = np.abs(error)
        marker = markers[i % len(markers)]  # Cycling through markers
        plt.plot(corruption_rates, w_star_w_error, marker= marker, label=method_labels[i])
        
    #plt.title('Regression Errors vs. Number of Samples')
    plt.xlabel('Corruption rates (α)')
    plt.ylabel('Error (|w^* - w|)')
    plt.legend()
    plt.grid(True)
     # Set the x-ticks to only the values of n samples
    plt.xticks(corruption_rates)

    # Set the same distance between each tick on the x-axis
    #plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))  


    title = f'Regression Errors vs. Corruption rate ( n: {sample_size}, d: {dimensionality},  σ: {data_error_variance})'
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.95)  # y parameter adjusts the vertical position of the title
    # Save the plot inside the "Figures" folder
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../Figures/Corruptions', '{}-{}-{}-{}.png'.format('error_corruptions', f'd:{sample_size}',  f'd:{dimensionality}', f'σ:{data_error_variance}'))
    plt.savefig(save_path)
    
def plot_regression_errors_sigma(errors, sample_size, dimensionality, corruption_rate, data_error_variances, method_labels, error_tolerance, M ,eta):
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, error in enumerate(errors):
        w_star_w_error = np.abs(error)
        marker = markers[i % len(markers)]  # Cycling through markers
        plt.plot(data_error_variances, w_star_w_error, marker= marker, label=method_labels[i])
        
    #plt.title('Regression Errors vs. Number of Samples')
    plt.xlabel('Data error variance (σ)')
    plt.ylabel('Error (|w^* - w|)')
    plt.legend()
    plt.grid(True)
     # Set the x-ticks to only the values of n samples
    plt.xticks(data_error_variances)

    # Set the same distance between each tick on the x-axis
    #plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))  


    title = f'Regression Errors vs. Data error variance ( n: {sample_size}, d: {dimensionality},  α: {corruption_rate}, tolerance: {error_tolerance}, tuning parameter: {M} , step: {eta})'
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.95)  # y parameter adjusts the vertical position of the title
    # Save the plot inside the "Figures" folder
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../Figures/Noise', '{}-{}-{}-{}.png'.format('error_sigma', f'd:{sample_size}', f'd:{dimensionality}', f'a:{corruption_rate}'))
    plt.savefig(save_path)    


def plot_iterations_n(iters, sample_sizes, dimensionality, corruption_rate, data_error_variance, method_labels, error_tolerance, M ,eta):
    # Plotting
    plt.figure(figsize=(10, 6))
    
    for i, iters in enumerate(iters):
        marker = markers[i % len(markers)]  # Cycling through markers
        plt.plot(sample_sizes, iters, marker= marker, label=method_labels[i])
        
    #plt.title('Iterations until Convergence vs. Number of Samples')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Iterations')
    plt.legend()
    plt.grid(True)
     # Set the x-ticks to only the values of n samples
    plt.xticks(sample_sizes)

    # Set the same distance between each tick on the x-axis
    #plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))  


    title = f'Iterations vs. Number of Samples (d: {dimensionality}, a: {corruption_rate},σ: {data_error_variance}, tolerance: {error_tolerance}, tuning parameter: {M} , step: {eta})'
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.95)  # y parameter adjusts the vertical position of the title
    # Save the plot inside the "Figures" folder
    save_path = os.path.join('Figures/Iterations/Samples', '{}-{}-{}-{}.png'.format('iteration_samples', f'd:{dimensionality}', f'a:{corruption_rate}', f'σ:{data_error_variance}'))
    plt.savefig(save_path)
    
def plot_iterations_d(iters, sample_sizes, dimensionality, corruption_rate, data_error_variance, method_labels, error_tolerance, M ,eta):
        # Plotting
    plt.figure(figsize=(10, 6))
    for i, iters in enumerate(iters):
        marker = markers[i % len(markers)]  # Cycling through markers
        plt.plot(dimensionality, iters, marker = marker, label=method_labels[i])
        
    #plt.title('Iterations until Convergence vs. Dimensions')
    plt.xlabel('Dimensions (d)')
    plt.ylabel('Iterations')
    plt.legend()
    plt.grid(True)
     # Set the x-ticks to only the values of n samples
    plt.xticks(dimensionality)

    # Set the same distance between each tick on the x-axis
    #plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))  


    title = f'Iterations vs. Dimensionality (n: {sample_sizes}, α: {corruption_rate},σ: {data_error_variance}, tolerance: {error_tolerance}, tuning parameter: {M} , step: {eta})'
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.95)  # y parameter adjusts the vertical position of the title
    # Save the plot inside the "Figures" folder
    save_path = os.path.join('Figures/Iterations/Dimensions', '{}-{}-{}-{}.png'.format('iteration_dimensions', f'n:{sample_sizes}', f'a:{corruption_rate}', f'σ:{data_error_variance}'))
    plt.savefig(save_path)   
    
def plot_iterations_a(iters, sample_size, dimensionality, corruption_rates, data_error_variance, method_labels, error_tolerance, M ,eta):
        # Plotting
    plt.figure(figsize=(10, 6))
    for i, iters in enumerate(iters):
        marker = markers[i % len(markers)]  # Cycling through markers
        plt.plot(corruption_rates, iters, marker=marker, label=method_labels[i])
        
    #plt.title('Iterations until Convergence vs. Dimensions')
    plt.xlabel('Corruption rates (a)')
    plt.ylabel('Iterations')
    plt.legend()
    plt.grid(True)
     # Set the x-ticks to only the values of n samples
    plt.xticks(corruption_rates)

    # Set the same distance between each tick on the x-axis
    #plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))  


    title = f'Iterations vs.Corruption rate (n: {sample_size}, d: {dimensionality},σ: {data_error_variance}, tolerance: {error_tolerance}, tuning param: {M} , step: {eta})'
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.95)  # y parameter adjusts the vertical position of the title
    # Save the plot inside the "Figures" folder
    save_path = os.path.join('Figures/Iterations/Corruptions', '{}-{}-{}-{}.png'.format('iteration_corruptions', f'n:{sample_size}', f'd:{dimensionality}', f'σ:{data_error_variance}'))
    plt.savefig(save_path)       
    
def plot_iterations_sigma(iters, sample_size, dimensionality, corruption_rate, data_error_variances, method_labels, error_tolerance, M ,eta):
        # Plotting
    plt.figure(figsize=(10, 6))
    for i, iters in enumerate(iters):
        marker = markers[i % len(markers)]  # Cycling through markers
        plt.plot(data_error_variances, iters, marker= marker, label=method_labels[i])
        
    #plt.title('Iterations until Convergence vs. Dimensions')
    plt.xlabel('data error variances (σ)')
    plt.ylabel('Iterations')
    plt.legend()
    plt.grid(True)
     # Set the x-ticks to only the values of n samples
    plt.xticks(data_error_variances)

    # Set the same distance between each tick on the x-axis
    #plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))  


    title = f'Iterations vs. data error variance (n: {sample_size}, d: {dimensionality}, α: {corruption_rate}, tolerance: {error_tolerance}, tuning param: {M} , step: {eta})'
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.95)  # y parameter adjusts the vertical position of the title
    # Save the plot inside the "Figures" folder
    save_path = os.path.join('Figures/Iterations/Noise', '{}-{}-{}-{}.png'.format('iteration_sigma', f'n:{sample_size}',  f'd:{dimensionality}', f'a:{corruption_rate}'))
    plt.savefig(save_path)           