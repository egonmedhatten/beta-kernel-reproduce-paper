#!/usr/bin/env python3
"""
Generates the Transposed Aggregated Ablation Table (LaTeX-Safe).

This script computes the overall median LSCV scores and Wilcoxon signed-rank 
p-values, outputting a perfectly formatted, highly readable transposed table 
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
    metrics_dict = {
        "Var Only Med. LSCV": [],
        "Var+Skew Med. LSCV": [],
        "Var+Kurt Med. LSCV": [],
        "Proposed Med. LSCV": [],
        "Win Rate (vs Var Only)": [],
        "p-value (vs Var Only)": [],
        "Win Rate (vs Var+Skew)": [],
        "p-value (vs Var+Skew)": [],
        "Win Rate (vs Var+Kurt)": [],
        "p-value (vs Var+Kurt)": [],
    }

    for dist in distributions:
        dist_data = df_clean[df_clean["distribution"] == dist]
        d_scores = dist_data["MODEL_D_lscv"]
        
        # 1. Format Medians (bold the lowest per distribution)
        medians = {m: dist_data[f"{m}_lscv"].median() for m in MODELS}
        best_med = min(medians.values())
        for m in MODELS:
            med = medians[m]
            formatted = f"{med:.4f}"
            if abs(med - best_med) < 1e-9:
                formatted = f"\\textbf{{{formatted}}}"
            metrics_dict[f"{MODEL_NAMES[m]} Med. LSCV"].append(formatted)
            
        # 2. Format Statistical Tests
        for m in ["MODEL_A", "MODEL_B", "MODEL_C"]:
            m_scores = dist_data[f"{m}_lscv"]
            
            # Win Rate (escaped for LaTeX)
            win_rate = (d_scores < m_scores).mean()
            metrics_dict[f"Win Rate (vs {MODEL_NAMES[m]})"].append(f"{win_rate * 100:.1f}\\%")
            
            # Wilcoxon test
            diffs = d_scores - m_scores
            if np.all(diffs == 0):
                pval = 1.0
            else:
                _, pval = wilcoxon(m_scores, d_scores)
                
            # Formatted p-values with math-mode less-than signs
            # Clean p-values without redundant significance stars
            if pval < 0.001:
                pval_str = "$<$0.001"
            else:
                pval_str = f"{pval:.3f}"
                
            metrics_dict[f"p-value (vs {MODEL_NAMES[m]})"].append(pval_str)

    # Convert to DataFrame (Rows = Metrics, Columns = Distributions)
    df_out = pd.DataFrame(metrics_dict, index=distributions).T
    df_out.index.name = "Metric"

    print("--- Final Ablation Summary (Console View) ---")
    # For the console view, temporarily strip LaTeX syntax so it's readable
    console_df = df_out.copy()
    for col in console_df.columns:
        console_df[col] = console_df[col].str.replace("\\%", "%").str.replace("$<$", "<")
    print(console_df.to_string())

    print("\n--- LaTeX Table Output (For Supplementary Material) ---")
    # Generate the LaTeX string using pandas 3.0+ Styler
    latex_table = df_out.style.to_latex(
        caption="Ablation study of fallback heuristic components across 6,000 trials per distribution. Displaying median LSCV scores (lower is better), win rates, and Wilcoxon signed-rank test p-values comparing the proposed rule against simpler parameterizations.",
        label="tab:ablation_study_aggregated",
        hrules=True
    )
    
    # Inject standard LaTeX table formatting
    latex_table = latex_table.replace("\\begin{table}\n", "\\begin{table}[ht]\n\\centering\n")
    
    # Inject a midrule to visually separate the LSCV scores from the Statistical tests
    latex_table = latex_table.replace("\nWin Rate (vs Var Only)", "\n\\midrule\nWin Rate (vs Var Only)")
    
    print(latex_table)

    # By removing caption and label, Pandas only generates the \begin{tabular} block
    latex_tabular = df_out.style.to_latex(hrules=True)
    
    # Inject a midrule to visually separate the LSCV scores from the Statistical tests
    latex_tabular = latex_tabular.replace(
        "\nWin Rate (vs Var Only)", 
        "\n\\midrule\nWin Rate (vs Var Only)"
    )

    try:
        with open(TABLES_DIR / "ablation_table.tex", "w") as f:
            f.write(latex_tabular)
        print(f"\nSuccessfully saved LaTeX table to: {TABLES_DIR / 'ablation_table.tex'}")
    except Exception as e:
        print(f"\nError saving LaTeX file: {e}")

if __name__ == "__main__":

    main()