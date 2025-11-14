import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.linear_model import HuberRegressor
import sys
import os
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)
from synthetic.strategic_corruptions import  strategic_corruption_scaled
from synthetic.toy_dataset import generate_synthetic_dataset
from torrent.torrent import  torrent_admm, split_matrix, split_matrix_Y, torrent_admm_dp
from torrent.torrent_fxp import torrent_admm_fxp, torrent_admm_fxp_analyze_gauss
from fixed_point.fixed_point_helpers import *
from plots.plots_fxp import plot_metric_vs_alpha_fxp, plot_metric_vs_d_fxp

markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed


def run_tests_fxp_d(num_trials=10):
    # Parameters
    alpha_init = 0.2
    beta = alpha_init + 0.1
    d_values = [10, 25, 50, 100]
    sigma = 0.1
    test_perc = 0.01
    epsilon = 0.1
    additive = 10
    multiplicative = 10
    m = 2
    rho = 1
    admm_steps = 5
    robust_rounds = 5
    modelz = 1

    # Define configurations for different n
    configs = {
        10000: {
            "dp_w": [0.0491542458, 0.2959143979, 1.184429552, 4.821505653],
            "dp_noise_x": [86.87224608, 217.1806152, 434.3612304, 868.7224608],
            "dp_noise_y": [274.7141631, 1085.903076, 3071.397715, 8687.224608]
        },
        100000: {
            "dp_w": [0.005377744112, 0.03197006629, 0.1262511128, 0.5046469554],
            "dp_noise_x": [96.89610525, 242.2402631, 484.4805263, 968.9610525],
            "dp_noise_y": [306.412389, 1211.201316, 3425.794655, 9689.610525]
        }
    }

    results = {}

    for n, params in configs.items():
        print(f"\n=== Running for n = {n} ===")
        train_size = n - int(n * test_perc)

        dp_w_list = params["dp_w"]
        dp_noise_x_list = params["dp_noise_x"]
        dp_noise_y_list = params["dp_noise_y"]

        # store trial errors for mean/var computation
        errors_dp = (np.zeros((num_trials, len(d_values))))
        errors_gauss = (np.zeros((num_trials, len(d_values))))

        for trial in range(num_trials):
            for i, d in enumerate(d_values):
                # generate dataset
                X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc, i=trial)
                w_corrupt = additive + w_star * multiplicative
                norm_w = np.linalg.norm(w_star)
                norm_w_inv = 1 / norm_w

                Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha_init)

                X_parts = split_matrix(X_train, m, train_size)
                y_parts = split_matrix_Y(Y_cor, m, train_size)

                dp_w_val = dp_w_list[i]
                dp_noise_x_val = dp_noise_x_list[i]
                dp_noise_y_val = dp_noise_y_list[i]

                # FXP input
                X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)
                #w_star_fxp = fxp(w_star)

                # --- DP OLS style ---
                w_torrent_fxp, _ = torrent_admm_fxp(
                    X_parts_fxp, y_parts_fxp, beta, epsilon, rho,
                    admm_steps, robust_rounds, wstar=None, dp_w=dp_w_val
                )
                error_torrent = (np.linalg.norm((w_torrent_fxp).get_val() - w_star)) * norm_w_inv    # cast to fxp
                print("d :", d)
                print("Error DP: ", error_torrent)
                errors_dp[trial, i] = (error_torrent)

                # --- Analyze Gaussian noise ---
                w_torrent, _ = torrent_admm_fxp_analyze_gauss(
                    X_parts_fxp, y_parts_fxp, beta, epsilon, rho,
                    admm_steps, robust_rounds, wstar=None,
                    dp_noise_x=dp_noise_x_val, dp_noise_y=dp_noise_y_val
                )
                error_gauss = (np.linalg.norm((w_torrent).get_val() - w_star)) * norm_w_inv      # cast to fxp
                print("Error AG: ", error_gauss)
                errors_gauss[trial, i] = (error_gauss)
                
        # Compute mean + variance
        mean_dp = np.mean(errors_dp, axis=0)
        var_dp = np.var(errors_dp, axis=0)
        mean_gauss = np.mean(errors_gauss, axis=0)
        var_gauss = np.var(errors_gauss, axis=0)

        results[n] = {
            "mean_dp": mean_dp,
            "var_dp": var_dp,
            "mean_gauss": mean_gauss,
            "var_gauss": var_gauss,
            "d_values": d_values
        }

        # Print summary for this n
        print("\nResults for n =", n)
        for i, d in enumerate(d_values):
            print(f"d={d}:  DP -> mean {mean_dp[i]:.6f}, var {var_dp[i]:.6f} | "
                  f"Gauss -> mean {mean_gauss[i]:.6f}, var {var_gauss[i]:.6f}")

    return results


# Run
#results = run_tests_fxp_d(num_trials=5)

def run_tests_fxp_alpha(num_trials=10):
    # Parameters
    n = 10000  # Number of samples
    dimension = 10
    alpha_values = [0.1,  0.2,  0.3, 0.4]  # Corruption rates
    sigma = 0.1
    test_perc = 0.01
    epsilon = 0.1

    # DP params (fixed for this experiment)
    dp_w = 0.0491542458   # n=10^4, d=10
    dp_noise_x = 86.87224608
    dp_noise_y = 274.7141631

    # Corruption strategy
    additive = 10
    multiplicative = 10

    # ADMM parameters
    m = 2
    rho = 1
    admm_steps = 5
    robust_rounds = 5

    train_size = n - int(n * test_perc)

    # Store trial errors: shape (num_trials, len(alpha_values))
    errors_gauss = (np.zeros((num_trials, len(alpha_values))))
    errors_dp = (np.zeros((num_trials, len(alpha_values))))

    for trial in range(num_trials):
        for j, alpha in enumerate(alpha_values):
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(
                n, dimension, sigma, test_perc, i=trial
            )
            w_corrupt = multiplicative * w_star + additive
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)

            norm_w = np.linalg.norm(w_star)
            norm_w_inv = 1 / norm_w

            X_parts = split_matrix(X_train, m, train_size)
            y_parts = split_matrix_Y(Y_cor, m, train_size)
            beta = alpha + 0.1

            # Convert to fxp
            X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)

            # --- Torrent analyze gauss ---
            w_torrent, _ = torrent_admm_fxp_analyze_gauss(
                X_parts_fxp, y_parts_fxp, beta, epsilon, rho,
                admm_steps, robust_rounds, wstar=None,
                dp_noise_x=dp_noise_x, dp_noise_y=dp_noise_y
            )
            error_gauss = (np.linalg.norm(w_torrent.get_val() - w_star)) * norm_w_inv
            errors_gauss[trial, j] = (error_gauss)
            print("Error AG: ", error_gauss)
            # --- Torrent DP fxp ---
            w_torrent_fxp, _ = torrent_admm_fxp(
                X_parts_fxp, y_parts_fxp, beta, epsilon, rho,
                admm_steps, robust_rounds, wstar=None, dp_w=dp_w
            )
            error_dp = (np.linalg.norm(w_torrent_fxp.get_val() - w_star)) * norm_w_inv
            errors_dp[trial, j] = error_dp
            print("Error DP: ", error_dp)
    # Compute mean and variance
    mean_gauss = np.mean(errors_gauss, axis=0)
    var_gauss = np.var(errors_gauss, axis=0)
    mean_dp = np.mean(errors_dp, axis=0)
    var_dp = np.var(errors_dp, axis=0)

    results = {
        "alpha_values": alpha_values,
        "mean_gauss": mean_gauss,
        "var_gauss": var_gauss,
        "mean_dp": mean_dp,
        "var_dp": var_dp
    }

    # Print summary
    print("\nResults for different alpha values:")
    for j, alpha in enumerate(alpha_values):
        print(
            f"alpha={alpha:.2f}:  "
            f"Gauss -> mean {mean_gauss[j]:.6f}, var {var_gauss[j]:.6f} | "
            f"DP -> mean {mean_dp[j]:.6f}, var {var_dp[j]:.6f}"
        )

    return results


# Example usage
#results_alpha = run_tests_fxp_alpha(num_trials=5)

def run_tests_fxp_noise(num_trials=5):
    """
    Runs experiments for different dimensions and noise (sigma) values.
    Example setup:
      - d = 10, sigma ∈ [0.1, 0.2, 0.5, 0.7, 1]
      - d = 25, sigma ∈ [0.1, 0.2, 0.5, 0.7, 1]
      - d = 50, sigma ∈ [0.1, 0.2, 0.5, 0.7, 1]
      - d = 75, sigma ∈ [0.1, 0.2, 0.5, 0.7, 1]
      - d = 100, sigma ∈ [0.1, 0.2, 0.5, 0.7, 1]
    """
    n = 10000
    test_perc = 0.01
    epsilon = 0.1
    additive = 10
    multiplicative = 10
    m = 2
    rho = 1
    admm_steps = 5
    robust_rounds = 5
    alpha = 0.2
    beta = alpha + 0.1
    sigma = 0.1

    # Dimensions and noise levels to test
    dp_delta = 0.00001
    d_values = [10, 25, 50, 75, 100]
    noise_values = [0.001]
        
    results = {}

    for d in d_values:
        print(f"\n=== Running for d = {d} ===")
        results[d] = {}
        for dp_noise in noise_values:
            print(f"\n--- epsilon = {dp_noise} ---")
            # DP params (approximation for n=10^4)
            # DP params (approximation for n=10^4)
            lambda_min = n - 0.1
            dp_sens = (1/(lambda_min**2)) * d * math.sqrt(d)* (n * math.sqrt(d) + d * math.sqrt(n)) + (1/lambda_min)*d
            dp_w = math.sqrt(2*math.log(1.25/dp_delta))*dp_sens/dp_noise

            dp_sens_x = d
            dp_sens_y = d*math.sqrt(d)
            dp_noise_x = math.sqrt(2*math.log(1.25/dp_delta))*dp_sens_x/dp_noise
            dp_noise_y = math.sqrt(2*math.log(1.25/dp_delta))*dp_sens_y/dp_noise


            errors_dp = np.zeros(num_trials)
            errors_gauss = np.zeros(num_trials)

            for trial in range(num_trials):
                # Generate dataset
                X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(
                    n, d, sigma, test_perc, i=trial
                )

                w_corrupt = multiplicative * w_star + additive
                Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)
                norm_w_inv = 1 / np.linalg.norm(w_star)

                X_parts = split_matrix(X_train, m, n - int(n * test_perc))
                y_parts = split_matrix_Y(Y_cor, m, n - int(n * test_perc))
                X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)

                # --- DP fixed-point TORRENT ---
                w_torrent_fxp, _ = torrent_admm_fxp(
                    X_parts_fxp, y_parts_fxp, beta, epsilon, rho,
                    admm_steps, robust_rounds, wstar=None, dp_w=dp_w, eEM=dp_noise/16
                )
                err_dp = np.linalg.norm(w_torrent_fxp.get_val() - w_star) 
                errors_dp[trial] = err_dp

                # --- Gaussian fixed-point TORRENT ---
                w_torrent_gauss, _ = torrent_admm_fxp_analyze_gauss(
                    X_parts_fxp, y_parts_fxp, beta, epsilon, rho,
                    admm_steps, robust_rounds, wstar=None,
                    dp_noise_x=dp_noise_x, dp_noise_y=dp_noise_y, eEM=dp_noise/16
                )
                err_gauss = np.linalg.norm(w_torrent_gauss.get_val() - w_star) 
                errors_gauss[trial] = err_gauss

                print(f"Trial {trial+1}: DP error = {err_dp:.12f}, Gauss error = {err_gauss:.12f}")

            # Aggregate
            mean_dp = np.mean(errors_dp)
            var_dp = np.var(errors_dp)
            mean_gauss = np.mean(errors_gauss)
            var_gauss = np.var(errors_gauss)

            results[d][dp_noise] = {
                "mean_dp": mean_dp,
                "var_dp": var_dp,
                "mean_gauss": mean_gauss,
                "var_gauss": var_gauss
            }

            print(f" dp_noise={dp_noise:.12f}, DP mean={mean_dp:.12f}, var={var_dp:.12f} | "
                  f"Gauss mean={mean_gauss:.12f}, var={var_gauss:.12f}")

    return results
results_noise = run_tests_fxp_noise(num_trials=5)