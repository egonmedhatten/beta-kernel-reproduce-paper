#!/usr/bin/env python3
"""
Generates a JCGS-compliant plot of the Ablation Study.
Uses color for the online version, but relies on distinct linestyles 
and markers so it remains 100% readable in black-and-white print.
Plots the Delta (Δ) LSCV (Proposed - Baseline) across sample sizes.
"""

import sys
from pathlib import Path

# Adjust path if your _paths.py is located differently
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, PLOTS_DIR

import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_CSV = str(DATA_DIR / "ablation_study" / "ablation_results.csv")
OUTPUT_PDF = str(PLOTS_DIR / "ablation_delta_plot.pdf")

MODELS = ["MODEL_A", "MODEL_B", "MODEL_C"]
PROPOSED = "MODEL_D"
MODEL_NAMES = {
    "MODEL_A": "Baseline: Var Only",
    "MODEL_B": "Baseline: Var + Skew",
    "MODEL_C": "Baseline: Var + Kurt",
}

# --- The "Dual-Format" Style Dictionary ---
# Colors for online visibility, distinct linestyles/markers for B&W print readability.
# Using Paul Tol's vibrant colorblind-safe hex codes.
STYLES = {
    "MODEL_A": {"color": "#EE6677", "linestyle": "dotted", "marker": "o"},  # Red
    "MODEL_B": {"color": "#228833", "linestyle": "dashed", "marker": "s"},  # Green
    "MODEL_C": {"color": "#4477AA", "linestyle": "dashdot", "marker": "^"}, # Blue
}

def main():
    print("\n===========================================================")
    print("--- Generating Dual-Format Ablation Plot ---")
    print("===========================================================\n")

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}.")
        return

    # Drop NaNs and group by distribution and n to get medians
    lscv_cols = [f"{m}_lscv" for m in MODELS + [PROPOSED]]
    df_clean = df.dropna(subset=lscv_cols)
    grouped = df_clean.groupby(['distribution', 'n'])[lscv_cols].median().reset_index()

    distributions = grouped['distribution'].unique()
    
    # Set up a 2x2 grid for the 4 distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        dist_data = grouped[grouped['distribution'] == dist]
        n_vals = dist_data['n']
        prop_med = dist_data[f"{PROPOSED}_lscv"]

        for m in MODELS:
            base_med = dist_data[f"{m}_lscv"]
            # Delta = Proposed - Baseline. Negative means Proposed is better.
            delta = prop_med - base_med
            
            ax.plot(
                n_vals, 
                delta, 
                label=MODEL_NAMES[m], 
                color=STYLES[m]["color"], 
                linestyle=STYLES[m]["linestyle"], 
                marker=STYLES[m]["marker"],
                markersize=6,
                linewidth=2
            )

        # Add a reference line at y=0 (where Proposed == Baseline)
        ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.6, zorder=0)
        
        ax.set_title(f"Distribution: {dist}")
        ax.set_xscale('log') # Log scale for n is usually best
        ax.set_xticks([50, 100, 250, 500, 1000, 2000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, linestyle=':', alpha=0.6)

    # Global formatting
    fig.supxlabel("Sample Size (n)", y=0.02)
    fig.supylabel("Delta Median LSCV (Proposed - Baseline)", x=0.02)
    
    # Single legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
    print(f"Successfully generated dual-format plot: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()