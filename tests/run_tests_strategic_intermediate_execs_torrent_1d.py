# try some dp noise on 1d data.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import HuberRegressor, LinearRegression
import os
import sys
from numpy.linalg import eigh, inv


module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from torrent.torrent import torrent_intermediate, torrent # Import torrent module
from synthetic.synthetic_one_dimensional import generate_synthetic_dataset_one_d, strategic_corruption_scaled


def run_tests_1d_intermediate(num_trials):
    n = 100
    sigma = 0.1
    alpha = 0.2
    beta = alpha + 0.1
    epsilon = 0.01
    max_iters = 5

    all_preds_per_iter = [[] for _ in range(max_iters + 1)]  # includes initial w=0

    for k in range(num_trials):
        X_aug, Y, w_star = generate_synthetic_dataset_one_d(n, sigma)
        Y_corrupted, corrupted_indices, w_corrupt = strategic_corruption_scaled(X_aug, Y, alpha)

        x_vals = np.linspace(X_aug[1, :].min(), X_aug[1, :].max(), n).reshape(1, -1)
        x_vals_aug = np.vstack([np.ones((1, x_vals.shape[1])), x_vals])

        y_vals_true = x_vals_aug.T @ w_star
        y_vals_adv = x_vals_aug.T @ w_corrupt

        w_torrent, iter_count, w_history, S_history = torrent_intermediate(X_aug, Y_corrupted, beta, epsilon, max_iters)

        for i, w_iter in enumerate(w_history):
            y_iter_pred = (x_vals_aug.T @ w_iter).ravel()
            all_preds_per_iter[i].append(y_iter_pred)

            if k == 0:
                # Store only the first trial’s data and intermediate states
                X_aug_plot = X_aug.copy()
                Y_plot = Y.copy()
                corrupted_indices_plot = corrupted_indices.copy()
                Y_corrupted_plot = Y_corrupted.copy()
                y_vals_true_plot = y_vals_true
                y_vals_adv_plot = y_vals_adv
                x_vals_plot = x_vals.T

                w_history_plot = w_history  # Store history from trial 0
                S_history_plot = S_history


        for i, (w, S) in enumerate(zip(w_history_plot, S_history_plot)):
            y_vals_iter = x_vals_aug.T @ w
    
            #plt.figure(figsize=(8, 6))
            #plt.plot(x_vals_plot, y_vals_true_plot, label="True Model", color="blue")
            #plt.plot(x_vals_plot, y_vals_iter, label=f"Torrent Iteration: {i}", color="magenta", linestyle="dashdot")
            plt.plot(x_vals_plot, y_vals_iter, color="magenta", linestyle="dashdot")
            
            # Only show points in S
            S_array = np.array(list(S))  # Make sure S is a NumPy array
            corrupted_in_S = np.intersect1d(S_array, corrupted_indices_plot)
            clean_in_S = np.setdiff1d(S_array, corrupted_indices_plot)
            # Show original uncorrupted locations for the corrupted points kept in S
            plt.scatter(X_aug_plot[1, corrupted_in_S], Y_plot[corrupted_in_S],
                label="Original Value (Kept Corrupted)", color="white", marker="v",
                edgecolors="grey", s=150, alpha=0.8)

            #plt.scatter(X_aug_plot[1, clean_in_S], Y_corrupted_plot[clean_in_S], 
                       # label="Kept Clean", color="black", alpha=0.8)
            #plt.scatter(X_aug_plot[1, corrupted_in_S], Y_corrupted_plot[corrupted_in_S],
                       # label="Kept Corrupted", color="blueviolet", marker="x", s=80)
            plt.scatter(X_aug_plot[1, clean_in_S], Y_corrupted_plot[clean_in_S], 
                        color="black", alpha=0.8)
            plt.scatter(X_aug_plot[1, corrupted_in_S], Y_corrupted_plot[corrupted_in_S],
                         color="blueviolet", marker="x", s=80)           
            
            #plt.xlabel("Feature X")
            #plt.ylabel("Label Y")
            #plt.title(f"Torrent Iteration {i} – Points in $S$")
            #plt.legend()
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            # Remove frame (all spines)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            plt.show()


        
# Run the tests with averaging
num_trials = 1
run_tests_1d_intermediate(num_trials)        