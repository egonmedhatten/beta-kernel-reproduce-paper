#!/usr/bin/env python3
"""
Generates the Transposed Aggregated Ablation Table (LaTeX-Safe).

This script computes the overall mean and median LSCV scores, win rates,
and significance asterisks, outputting a formatted transposed table
(metrics as rows, distributions as columns) for the paper.
"""

import sys
from pathlib import Path

# Adjust path if your _paths.py is located differently
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, TABLES_DIR

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# --- Configuration ---
INPUT_CSV = str(DATA_DIR / "ablation_study" / "ablation_results.csv")
MODELS = ["MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D"]
MODEL_NAMES = {
    "MODEL_A": "Var Only",
    "MODEL_B": "Var+Skew",
    "MODEL_C": "Var+Kurt",
    "MODEL_D": "Proposed",
}


def significance_stars(p):
    """Return asterisk string for significance level."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "$^{***}$"
    elif p < 0.01:
        return "$^{**}$"
    elif p < 0.05:
        return "$^{*}$"
    return ""


def main():
    print("\n===========================================================")
    print("--- Running Transposed Ablation Table Script ---")
    print("===========================================================\n")

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}. Please ensure the path is correct.")
        return

    lscv_cols = [f"{m}_lscv" for m in MODELS]
    df_clean = df.dropna(subset=lscv_cols)

    distributions = df_clean["distribution"].unique()

    # We build a dictionary where keys are Metric names,
    # and values are lists of results corresponding to each distribution column.
    metrics_dict = {f"{MODEL_NAMES[m]} LSCV": [] for m in MODELS}
    metrics_dict.update(
        {
            f"Win Rate (vs {MODEL_NAMES[m]})": []
            for m in ["MODEL_A", "MODEL_B", "MODEL_C"]
        }
    )

    for dist in distributions:
        dist_data = df_clean[df_clean["distribution"] == dist]
        d_scores = dist_data["MODEL_D_lscv"]

        # 1. Compute means and medians
        means = {m: dist_data[f"{m}_lscv"].mean() for m in MODELS}
        medians = {m: dist_data[f"{m}_lscv"].median() for m in MODELS}
        best_median = min(medians.values())

        # 2. Compute p-values for non-reference models vs MODEL_D
        pvals = {}
        for m in ["MODEL_A", "MODEL_B", "MODEL_C"]:
            m_scores = dist_data[f"{m}_lscv"]
            diffs = d_scores - m_scores
            if np.all(diffs == 0):
                pvals[m] = 1.0
            else:
                _, pvals[m] = wilcoxon(m_scores, d_scores)

        # 3. Format LSCV rows as mean (median) with bolding and asterisks
        for m in MODELS:
            mean_str = f"{means[m]:.4f}"
            median_str = f"{medians[m]:.4f}"
            stars = significance_stars(pvals[m]) if m != "MODEL_D" else ""
            cell = f"{mean_str} ({median_str})"
            if abs(medians[m] - best_median) < 1e-9:
                cell = f"\\textbf{{{cell}}}"
            metrics_dict[f"{MODEL_NAMES[m]} LSCV"].append(f"{cell}{stars}")

        # 4. Format Win Rates
        for m in ["MODEL_A", "MODEL_B", "MODEL_C"]:
            m_scores = dist_data[f"{m}_lscv"]
            win_rate = (d_scores < m_scores).mean()
            metrics_dict[f"Win Rate (vs {MODEL_NAMES[m]})"].append(
                f"{win_rate * 100:.1f}\\%"
            )

    # Convert to DataFrame (Rows = Metrics, Columns = Distributions)
    df_out = pd.DataFrame(metrics_dict, index=distributions).T
    df_out.index.name = "Metric"

    print("--- Final Ablation Summary (Console View) ---")
    # For the console view, temporarily strip LaTeX syntax so it's readable
    console_df = df_out.copy()
    for col in console_df.columns:
        console_df[col] = (
            console_df[col].str.replace("\\%", "%").str.replace("$<$", "<")
        )
    print(console_df.to_string())

    print("\n--- LaTeX Table Output (For Supplementary Material) ---")
    # Generate the LaTeX string using pandas 3.0+ Styler
    latex_table = df_out.style.to_latex(
        caption="Ablation study of fallback heuristic components across 6,000 trials per distribution. Displaying mean LSCV scores with median in parentheses (lower is better; bold indicates best median), win rates, and significance of Wilcoxon signed-rank tests comparing the proposed rule against simpler parameterizations. Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
        label="tab:ablation_study_aggregated",
        hrules=True,
    )

    # Inject standard LaTeX table formatting
    latex_table = latex_table.replace(
        "\\begin{table}\n", "\\begin{table}[ht]\n\\centering\n"
    )

    # Inject a midrule to visually separate the LSCV scores from the Win Rate rows
    latex_table = latex_table.replace(
        "\nWin Rate (vs Var Only)", "\n\\midrule\nWin Rate (vs Var Only)"
    )

    print(latex_table)

    # By removing caption and label, Pandas only generates the \begin{tabular} block
    latex_tabular = df_out.style.to_latex(hrules=True)

    # Inject a midrule to visually separate the LSCV scores from the Win Rate rows
    latex_tabular = latex_tabular.replace(
        "\nWin Rate (vs Var Only)", "\n\\midrule\nWin Rate (vs Var Only)"
    )

    try:
        with open(TABLES_DIR / "ablation_table.tex", "w") as f:
            f.write(
                "% Suggested caption: Ablation study of fallback heuristic components across 6,000 trials per distribution. Mean LSCV scores (median in parentheses); lower is better. Bold indicates the best median per distribution. Win rates of the proposed rule. Significance of Wilcoxon signed-rank tests: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.\n"
            )
            f.write(latex_tabular)
        print(
            f"\nSuccessfully saved LaTeX table to: {TABLES_DIR / 'ablation_table.tex'}"
        )
    except Exception as e:
        print(f"\nError saving LaTeX file: {e}")


if __name__ == "__main__":

    main()
