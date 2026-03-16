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

from _plot_styles import setup_theme, ABLATION_STYLES, SHOW_SUPTITLE, FIGURE_WIDTH_FULL

setup_theme()

# --- Configuration ---
INPUT_CSV = str(DATA_DIR / "ablation_study" / "ablation_results.csv")
OUTPUT_PDF = str(PLOTS_DIR / "ablation_delta_plot.pdf")

MODELS = ["MODEL_A", "MODEL_B", "MODEL_C"]
PROPOSED = "MODEL_D"

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
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_FULL, 5.5), sharex=True)
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
            style = ABLATION_STYLES[m]
            
            ax.plot(
                n_vals, 
                delta, 
                label=style["label"], 
                color=style["color"], 
                linestyle=style["linestyle"], 
                marker=style["marker"],
                markersize=7,
                linewidth=1.8
            )

        # Add a reference line at y=0 (where Proposed == Baseline)
        ax.axhline(0, color='gray', linestyle='-', linewidth=1.2, alpha=0.6, zorder=0)
        
        ax.set_title(dist)
        ax.set_xscale('log') # Log scale for n is usually best
        ax.set_xticks([50, 100, 250, 500, 1000, 2000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, linestyle=':', alpha=0.6)

    # Global formatting
    fig.supxlabel("Sample Size ($n$)", y=0.02)
    fig.supylabel(r"$\Delta$ Median LSCV (Proposed $-$ Baseline)", x=0.02)
    
    # Single legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
    print(f"Successfully generated dual-format plot: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()