import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Keep this (Correct way to config SVG fonts) ---
plt.rcParams["svg.fonttype"] = "none"

from tol_colors import tol_cmap, tol_cset

cmap = tol_cset("bright")
from scipy.stats import beta


def rho(x: float, bandwidth: float) -> float:
    """
    Calculates the rho function for boundary correction as defined in Chen (1999).
    """
    h_squared = bandwidth**2
    term2_sqrt_arg = 4 * h_squared**2 + 6 * h_squared + 2.25 - x**2 - x / bandwidth
    if term2_sqrt_arg < 0:
        term2_sqrt_arg = 0.0
    return (2 * h_squared + 2.5) - np.sqrt(term2_sqrt_arg)


def get_beta_params_f2(x, b):
    """
    Returns alpha, beta parameters for the Modified Beta Kernel (Type 2)
    based on the location of evaluation point x.
    """
    # Define boundary width
    boundary_limit = 2 * b

    if x < boundary_limit:
        # Left Boundary Region: [0, 2b)
        # K(rho(x,b), (1-x)/b)
        a = rho(x, b)
        beta_param = (1 - x) / b

    elif x > (1 - boundary_limit):
        # Right Boundary Region: (1-2b, 1]
        # K(x/b, rho(1-x, b))
        a = x / b
        beta_param = rho(1 - x, b)

    else:
        # Interior Region: [2b, 1-2b]
        # K(x/b, (1-x)/b)
        a = x / b
        beta_param = (1 - x) / b

    return a, beta_param


def plot_chen_f2_kernels():
    # Support for the plot
    t = np.linspace(0, 1, 1000)

    # Bandwidth (h in your paper)
    h = 0.2

    # Evaluation points to demonstrate adaptivity
    # 0.0 (Boundary), 0.15 (Transition), 0.5 (Interior)
    eval_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, x_eval in enumerate(eval_points):
        a, b_param = get_beta_params_f2(x_eval, h)

        # Check for validity (alpha, beta must be > 0)
        if a > 0 and b_param > 0:
            y = beta.pdf(t, a, b_param)
            label = f"$x={x_eval}$ ($\\alpha={a:.2f}, \\beta={b_param:.2f}$)"
            ax.plot(t, y, lw=2, label=label, color=cmap[i])
        else:
            print(f"Invalid parameters for x={x_eval}: a={a}, b={b_param}")

    # ax.set_title(f'Beta Kernel Shapes ($\hat{{f}}_2$) for $h={h}$')
    ax.set_xlabel("t (Data Domain)")
    ax.set_ylabel("Beta kernels $K^*_{x,h}(t)$")

    # # Highlight the boundary region
    # ax.axvline(2*h, color='gray', linestyle='--', alpha=0.5)
    # ax.text(h, ax.get_ylim()[1]*0.1, 'Boundary\nRegion', ha='center', alpha=0.6)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR = "plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(f"{OUTPUT_DIR}/kernel_shape_plot.pdf")
    plt.show()


if __name__ == "__main__":
    plot_chen_f2_kernels()
