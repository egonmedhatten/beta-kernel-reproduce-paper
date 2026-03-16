"""Visualization of Experiment 2 real-world density estimation results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, PLOTS_DIR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import warnings
import os

from KDE import BetaKernelKDE
from KDE_Gauss import GaussianKDE

from _plot_styles import setup_theme, METHOD_STYLES, SHOW_SUPTITLE, FIGURE_WIDTH_FULL

setup_theme()

# --- Data Mappings ---

DATASET_TITLES = {
    "PctKids2Par": "Children with Two Parents (%)",
    "PctPopUnderPov": "Population Under Poverty (%)",
    "PctVacantBoarded": "Vacant & Boarded Housing (%)",
}

method_rename_map = {
    "BETA_ROT": "Beta (Ref)",
    "BETA_LSCV": "Beta (LSCV)",
    "LOGIT_SILV": "Logit (Silverman)",
    "LOGIT_LSCV": "Logit (LSCV)",
    "REFLECT_SILV": "Reflect (Silverman)",
    "REFLECT_LSCV": "Reflect (LSCV)",
}


METHOD_CONFIG = {
    "BETA_ROT": {
        "class": BetaKernelKDE,
        "init_args": {"bandwidth": "MISE_rule", "verbose": 0},
    },
    "BETA_LSCV": {
        "class": BetaKernelKDE,
        "init_args": {"bandwidth": "LSCV", "verbose": 0},
    },
    "LOGIT_SILV": {
        "class": GaussianKDE,
        "init_args": {"bandwidth": "silverman", "method": "logit", "verbose": 0},
    },
    "LOGIT_LSCV": {
        "class": GaussianKDE,
        "init_args": {"bandwidth": "LSCV", "method": "logit", "verbose": 0},
    },
    "REFLECT_SILV": {
        "class": GaussianKDE,
        "init_args": {"bandwidth": "silverman", "method": "reflect", "verbose": 0},
    },
    "REFLECT_LSCV": {
        "class": GaussianKDE,
        "init_args": {"bandwidth": "LSCV", "method": "reflect", "verbose": 0},
    },
}


def fetch_data():
    try:
        communities_crime = fetch_ucirepo(id=183)
        X = communities_crime.data.features
        datasets = {
            "PctKids2Par": X["PctKids2Par"].dropna().values,
            "PctPopUnderPov": X["PctPopUnderPov"].dropna().values,
            "PctVacantBoarded": X["PctVacantBoarded"].dropna().values,
        }
        return datasets
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def main():
    print("Loading results...")
    try:
        df_summary = pd.read_csv(
            str(DATA_DIR / "experiment2" / "experiment_2_summary.csv")
        )
    except FileNotFoundError:
        print("Error: CSV not found.")
        return

    datasets = fetch_data()
    if datasets is None:
        return

    dataset_names = list(datasets.keys())

    # Single-column figure: full width, 3 stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH_FULL, 8.5), sharex=True)

    # Evaluation grid
    x_plot = np.linspace(0, 1, 1000, endpoint=True)

    print("Generating plots...")
    for i, data_name in enumerate(dataset_names):
        ax = axes[i]
        data_vector = datasets[data_name]
        df_data = df_summary[df_summary["dataset"] == data_name]

        # 1. Plot Histogram (Stepfilled is cleaner for background data)
        ax.hist(
            data_vector,
            bins=40,
            density=True,
            histtype="stepfilled",  # Cleaner than bars
            alpha=0.2,  # Subtle background
            color="black",
            edgecolor="none",
            label="Data Histogram",
            zorder=1,
        )

        # 2. Plot Methods
        for _, row in df_data.iterrows():
            method_name = row["method"]
            bandwidth = row["bandwidth"]

            if method_name not in METHOD_CONFIG:
                continue

            config = METHOD_CONFIG[method_name]
            label_text = method_rename_map.get(method_name, method_name)
            style = METHOD_STYLES.get(label_text, {})
            color = style.get("color", "black")
            linestyle = style.get("linestyle", "-")
            marker = style.get("marker", None)
            # Proposed method gets a thicker line and higher z-order
            is_proposed = method_name == "BETA_ROT"
            lw = 2.2 if is_proposed else 1.5
            zorder = 10 if is_proposed else 5
            alpha = 1.0 if is_proposed else 0.85

            try:
                kde = config["class"](**config["init_args"])
                kde.bandwidth = bandwidth
                kde.fit(data_vector)
                pdf_plot = kde.pdf(x_plot)

                ax.plot(
                    x_plot,
                    pdf_plot,
                    label=label_text,
                    color=color,
                    linestyle=linestyle,
                    linewidth=lw,
                    zorder=zorder,
                    alpha=alpha,
                    marker=marker,
                    markevery=80,
                    markersize=6,
                )

            except Exception as e:
                print(f"Skipping {method_name}: {e}")

        # 3. Axis Styling per subplot
        readable_title = DATASET_TITLES.get(data_name, data_name)
        ax.set_title(readable_title, fontweight="bold", pad=10)

        # Add subtle grid
        ax.grid(True, linestyle=":", alpha=0.6, color="gray", linewidth=0.7)

        # Ensure 0 and 1 are strictly enforced limits
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

    # Set X-label only on the bottom plot
    axes[-1].set_xlabel("Feature Value (Normalized)")
    fig.supylabel("Density")

    # Create a unified legend
    # Get handles/labels from the first plot (assuming all plots have same methods)
    handles, labels = axes[0].get_legend_handles_labels()

    # Reorder legend to put Histogram first, then Beta Ref, then others
    # This ensures logical reading order
    order_map = {lbl: i for i, lbl in enumerate(labels)}

    # Define priority: Histogram -> Beta (Ref) -> Others
    def get_priority(lbl):
        if "Histogram" in lbl:
            return 0
        if "Beta (Ref)" in lbl:
            return 1
        return 2

    # Sort handles and labels
    sorted_pairs = sorted(
        zip(handles, labels), key=lambda x: (get_priority(x[1]), x[1])
    )
    handles, labels = zip(*sorted_pairs)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
    )

    plt.tight_layout()

    # Add extra space at bottom for legend
    plt.subplots_adjust(bottom=0.12)

    OUTPUT_DIR = str(PLOTS_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"{OUTPUT_DIR}/experiment_2_visual_fits.pdf"
    plt.savefig(
        output_filename, bbox_inches="tight"
    )  # bbox_inches='tight' ensures legend isn't cut off
    print(f"\nSuccessfully saved figure to {output_filename}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
