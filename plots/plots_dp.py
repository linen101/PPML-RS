import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


markers = ['o', 'v', 's', 'p', 'x', 'h']  # Add more if needed
# Plot metrics d
def plot_metric_vs_d(errors_matrix, ylabel, dp_noise, d_values, alpha_init, sigma, n):
    # Plot dimension
    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(dp_noise)))
    line_styles = ['dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (1, 1))]
    for e, noise in enumerate(dp_noise):
        errors = errors_matrix[e]
        plt.plot(d_values, errors, label=f'$noise = {noise}$',
                 color=colors[e], linestyle=line_styles[e % len(line_styles)], linewidth=2)
        for x, y in zip(d_values, errors):
            plt.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9, color=colors[e])
    plt.xlabel('Dimension $(d)$', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'Strategic Corruption with DP (n={n}, β={alpha_init}, σ={sigma})', fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# Plot metrics alpha
def plot_metric_vs_alpha(errors_matrix, ylabel, dp_noise, alpha_values, dimension, sigma, n):
    # Plot dimension
    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(dp_noise)))
    line_styles = ['dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (1, 1))]
    for e, noise in enumerate(dp_noise):
        errors = errors_matrix[e]
        plt.plot(alpha_values, errors, label=f'$noise = {noise}$',
                 color=colors[e], linestyle=line_styles[e % len(line_styles)], linewidth=2)
        for x, y in zip(alpha_values, errors):
            plt.text(x, y + 0.0005, f'{y:.3f}', ha='center', va='bottom', fontsize=9, color=colors[e])
    plt.xlabel(f'Corruption $(\\beta)$', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'Strategic Corruption with DP (n={n}, d={dimension}, σ={sigma})', fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()