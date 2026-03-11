#!/usr/bin/env python3
"""
Analysis script for the Beta Reference Rule Ablation Study.

This script reads the raw ablation results, computes robust summary statistics
(Median LSCV scores), and performs pairwise Wilcoxon signed-rank tests
comparing the proposed Model D against Models A, B, and C.

It outputs a clean summary to the console and generates a LaTeX table
ready to be pasted into the Supplementary Material.
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# --- Configuration ---
INPUT_CSV = "data/ablation_study/ablation_results.csv"
MODELS = ["MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D"]
MODEL_NAMES = {
    "MODEL_A": "Var Only",
    "MODEL_B": "Var+Skew",
    "MODEL_C": "Var+Kurt",
    "MODEL_D": "Proposed",
}


def main():
    print("--- Loading and Cleaning Data ---")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}. Please ensure the path is correct.")
        return

    # Identify the LSCV columns
    lscv_cols = [f"{m}_lscv" for m in MODELS]

    # Drop rows where any model failed to produce a valid LSCV score.
    # We must do this to ensure paired non-parametric tests are valid.
    initial_len = len(df)
    df_clean = df.dropna(subset=lscv_cols)
    dropped = initial_len - len(df_clean)
    print(
        f"Loaded {initial_len} trials. Dropped {dropped} trials due to LSCV non-convergence/NaNs.\n"
    )

    summary_records = []

    # We aggregate across all sample sizes (n) for maximum statistical power,
    # but group by the specific 'hard' distributions.
    distributions = df_clean["distribution"].unique()

    print("--- Running Statistical Tests (Wilcoxon Signed-Rank) ---")
    for dist in distributions:
        dist_data = df_clean[df_clean["distribution"] == dist]
        n_trials = len(dist_data)

        row = {"Distribution": dist, "Trials": n_trials}

        # 1. Compute Median LSCV for each model
        for m in MODELS:
            # We want the lowest (most negative) LSCV score
            row[f"{MODEL_NAMES[m]} Median LSCV"] = dist_data[f"{m}_lscv"].median()

        # 2. Perform Wilcoxon tests: Model D vs others
        d_scores = dist_data["MODEL_D_lscv"]

        for m in ["MODEL_A", "MODEL_B", "MODEL_C"]:
            m_scores = dist_data[f"{m}_lscv"]

            # Win Rate: How often did Model D get a lower LSCV than Model M?
            win_rate = (d_scores < m_scores).mean()
            row[f"Proposed Win Rate vs {MODEL_NAMES[m]}"] = f"{win_rate:.1%}"

            # Wilcoxon test (two-sided)
            # If all differences are exactly zero (models performed identically), wilcoxon throws a warning/error.
            diffs = d_scores - m_scores
            if np.all(diffs == 0):
                pval = 1.0
            else:
                _, pval = wilcoxon(m_scores, d_scores)

            # Format p-value with significance stars
            stars = (
                "***"
                if pval < 0.001
                else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            )
            row[f"p-value vs {MODEL_NAMES[m]}"] = f"{pval:.3e}{stars}"

        summary_records.append(row)

    # --- Create Output DataFrame ---
    summary_df = pd.DataFrame(summary_records)

    # Set the distribution as the index for a cleaner display
    summary_df.set_index("Distribution", inplace=True)

    print("\n--- Final Ablation Summary ---")
    # Print a transposed version to the console so it's easy to read
    print(summary_df.T.to_string())

    print("\n--- LaTeX Table Output (For Supplementary Material) ---")
    # Generate a simple LaTeX table
    latex_table = summary_df.style.format(precision=4).to_latex(
        caption="Ablation study of fallback heuristic components. Displaying median LSCV scores (lower is better) and Wilcoxon signed-rank test p-values comparing the proposed rule against simpler parameterizations.",
        label="tab:ablation_study",
        hrules=True,
    )
    print(latex_table)


if __name__ == "__main__":
    main()
