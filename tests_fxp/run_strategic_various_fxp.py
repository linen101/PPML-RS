import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

# Import project modules
module_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)

from synthetic.strategic_corruptions import (
    rotate_w_arbitrary,
    sparse_noise_w_arbitrary,
    strategic_corruption_scaled,
    interpolate_w_arbitrary
    
)
from synthetic.toy_dataset import generate_synthetic_dataset
from torrent.torrent import torrent_admm, split_matrix, split_matrix_Y, torrent_admm_dp
from torrent.torrent_fxp import torrent_admm_fxp, torrent_admm_fxp_analyze_gauss
from fixed_point.fixed_point_helpers import *
#dp_values=[0.005377744112, 0.03197006629, 0.1262511128, 0.5046469554],


# ========== Helper: Run experiment ========== #
def run_experiment(
    mode="dimension",
    corruption_fn=None,
    num_trials=1,
    n=2000,
    d_values=[10, 25, 50, 100],
    alpha_values=[0.1, 0.2, 0.3, 0.4],
    dp_values = [0.00203186667,0.00736297528,0.01993442884,0.05482502464],
    sigma=0.1,
    test_perc=0.2,
    epsilon=0.1,
):
    """
    mode: 'dimension' or 'alpha'
    corruption_fn: function (w_star, d) -> w_corrupt
    """
    results = []

    if mode == "dimension":
            # Zip dimension and dp values
        x_dp_pairs = list(zip(d_values, dp_values))
    else:
        x_dp_pairs = [(alpha_values[i], dp_values[0]) for i in range(len(alpha_values))]  # dp fixed

    for x, dp_w in x_dp_pairs:
        error_accum=fxp(0)
        avg_error=fxp(0)
        #error_accum = 0.0
        #avg_error = 0.0
        for trial in range(num_trials):
            d = x if mode == "dimension" else d_values[0]   # fix d in alpha experiments
            alpha = x if mode == "alpha" else 0.3           # fix corruption rate in dimension experiments

            # Generate data
            X_train, Y_train, X_test, Y_test, w_star = generate_synthetic_dataset(n, d, sigma, test_perc, trial)

            # Choose corrupted model
            w_corrupt = corruption_fn(w_star, d)

            # Apply corruption
            Y_cor, _ = strategic_corruption_scaled(X_train, Y_train, w_star, w_corrupt, alpha)

            # Run Torrent (fixed-point)
            m = 2
            X_parts = split_matrix(X_train, m, int(n*(1-test_perc)))
            y_parts = split_matrix_Y(Y_cor, m, int(n*(1-test_perc)))
            X_parts_fxp, y_parts_fxp = split_matrix_fxp(X_parts, y_parts)
            beta = alpha + 0.1
            '''
            w_torrent, _ = torrent_admm_dp(
                X_parts, y_parts, beta, epsilon, rho=1,
                admm_steps=5, rounds=5, wstar=None, dp_w=dp_w
            )
            '''
            w_torrent, _ = torrent_admm_fxp(
                X_parts_fxp, y_parts_fxp, beta, epsilon, rho=1,
                admm_steps=5, rounds=5, wstar=None, dp_w=dp_w
            )
            
            error = fxp(0)
           
            # Normalized error
            norm_w = np.linalg.norm((w_star))
            norm_w_inv = 1 / norm_w
            norm_w_inv = fxp(norm_w_inv)
            error = (np.linalg.norm(w_torrent - w_star) )
            print(f"[{mode}] {x=} dp_w={dp_w} trial {trial+1}: error={error.info()}")
            error_accum += error
        
        num_trials_inv = 1 / num_trials
        #num_trials_inv = fxp(num_trials_inv)
        avg_error = error_accum * num_trials_inv
        print(f"[{mode}] {x=} dp_w={dp_w} averaged error={avg_error}\n")
        results.append(avg_error)

    return [x for x, _ in x_dp_pairs], results

# ========== Define corruption strategies ========== #
def corruption_additive(w_star, d):
    return w_star + 10

def corruption_multiplicative(w_star, d):
    return 10 * w_star

def corruption_rotation(w_star, d):
    return rotate_w_arbitrary(w_star)

def corruption_sparse(w_star, d):
    return sparse_noise_w_arbitrary(w_star)

def corruption_interpolation(w_star, d):
    return interpolate_w_arbitrary(w_star)


strategies = {
    "Additive": corruption_additive,
    "Multiplicative": corruption_multiplicative,
    "Rotation": corruption_rotation,
    "Sparse": corruption_sparse,
    "Interpolation": corruption_interpolation
}


# ========== Run and plot experiments ========== #
def plot_results(n=2000, fixed_d=25, fixed_beta=0.3, fixed_dp=0.00736297528):
    palette = sns.color_palette("colorblind", len(strategies))
    '''
    # 1. Error vs Dimension
    plt.figure(figsize=(14,8))
    for (name, fn), color in zip(strategies.items(), palette):
        d_vals, errors = run_experiment(mode="dimension", corruption_fn=fn, n=n, alpha_values=[fixed_beta])
        print("Errors d:", errors)
        plt.plot(d_vals, errors, marker='o', linestyle="--", label=name, color=color, linewidth=2)

    plt.xlabel("Dimension (d)", fontsize=25)
    plt.ylabel(r"$\| w - w^* \|_2/\|w^*\|_2$", fontsize=25)
    plt.title(f"Trip* Error vs Dimension\n( β={fixed_beta})", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(False)
    plt.savefig(f"error_trip-beta-{fixed_beta}-synthetic.png", dpi=300, bbox_inches="tight")
    plt.show()
    '''
    # 2. Error vs Corruption Rate
    plt.figure(figsize=(14,8))
    for (name, fn), color in zip(strategies.items(), palette):
        alpha_vals, errors = run_experiment(mode="alpha", corruption_fn=fn, n=n, d_values=[fixed_d], dp_values=[fixed_dp])
        print("Errors β:", errors)
        plt.plot(alpha_vals, errors, marker='s', linestyle="--", label=name, color=color, linewidth=2)

    plt.xlabel("Corruption Rate (β)", fontsize=25)
    plt.ylabel(r"$\| w - w^* \|_2/\|w^*\|_2$", fontsize=25)
    plt.title(f"Trip* Error vs Corruption Rate\n( d={fixed_d})", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(False)
    plt.savefig(f"error_trip-d-{fixed_d}-synthetic.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_results()