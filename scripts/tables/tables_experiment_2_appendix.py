"""Generate appendix tables for Experiment 2 (p-values and log-likelihood)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, TABLES_DIR

import pandas as pd
import numpy as np
import os


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


def format_p_value(p):
    """Format p-value: <0.001 or specific value."""
    if pd.isna(p):
        return "-"
    if p < 0.001:
        return "$<0.001$"
    else:
        return f"{p:.3f}"


def format_value(v, best_v, is_min_best=True, decimals=4):
    """Format metric value, bolding the best."""
    if pd.isna(v):
        return "-"

    # Check if this is the best value (within tolerance)
    is_best = False
    if is_min_best:
        if v <= best_v + 1e-9:
            is_best = True
    else:  # Max is best (for Log-Likelihood)
        if v >= best_v - 1e-9:
            is_best = True

    str_v = f"{v:.{decimals}f}"
    if is_best:
        return f"\\textbf{{{str_v}}}"
    return str_v


def get_method_macro(method_name):
    """Map CSV method names to user-defined LaTeX macros."""
    mapping = {
        "BETA_ROT": r"\rott",
        "BETA_LSCV": r"\blscvt",
        "LOGIT_SILV": r"\lsilvt",
        "LOGIT_LSCV": r"\llscvt",
        "REFLECT_SILV": r"\rsilvt",
        "REFLECT_LSCV": r"\rlscvt",
    }
    return mapping.get(method_name, method_name.replace("_", r"\_"))


def generate_appendix_tables(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # Define method order matching your main text
    target_methods = [
        "BETA_ROT",
        "BETA_LSCV",
        "LOGIT_SILV",
        "LOGIT_LSCV",
        "REFLECT_SILV",
        "REFLECT_LSCV",
    ]
    df = df[df["method"].isin(target_methods)]

    datasets = df["dataset"].unique()

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Table D.1: LSCV Significance (vs BETA_ROT)
    # ---------------------------------------------------------
    d1_path = os.path.join(output_dir, "table_d1_lscv_pvalues.tex")
    with open(d1_path, "w") as f:
        f.write(
            "% Suggested caption: LSCV scores for real-world datasets. Bold indicates the best score per dataset. Significance of Wilcoxon signed-rank tests vs.\\ the reference method: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.\n"
        )
        f.write(r"\begin{tabular}{llc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(
            r"\textbf{Dataset} & \textbf{Method} & \textbf{LSCV Score} \\ \hline" + "\n"
        )

        for dataset in datasets:
            subset = df[df["dataset"] == dataset].copy()

            # Find best LSCV score (min) for bolding
            best_lscv = subset["lscv_score"].min()

            # Sort by method order
            subset["method"] = pd.Categorical(
                subset["method"], categories=target_methods, ordered=True
            )
            subset = subset.sort_values("method")

            # Multirow for dataset name
            f.write(f"\\multirow{{6}}{{*}}{{\\textit{{{dataset}}}}} \n")

            for _, row in subset.iterrows():
                method_macro = get_method_macro(row["method"])

                # LSCV score with bolding
                lscv_val = row["lscv_score"]
                if pd.isna(lscv_val):
                    lscv_str = "-"
                else:
                    lscv_str = format_value(lscv_val, best_lscv, is_min_best=True)
                    # Add significance asterisks for non-reference methods
                    if row["method"] != "BETA_ROT":
                        p_col = "density_p_value_wilcoxon (BETA_ROT >)"
                        if p_col in row.index and pd.notna(row[p_col]):
                            lscv_str += significance_stars(row[p_col])

                f.write(f" & {method_macro} & {lscv_str} \\\\\n")

            f.write(r"\hline" + "\n")

        f.write(r"\end{tabular}" + "\n")
    print(f"Generated {d1_path}")

    # ---------------------------------------------------------
    # Table D.2: Log-Likelihood Results with mean (median) from per-fold data
    # ---------------------------------------------------------
    per_fold_path = str(
        DATA_DIR / "experiment2" / "per_fold" / "experiment_2_per_fold_results.csv"
    )
    try:
        df_folds = pd.read_csv(per_fold_path)
        df_folds = df_folds[df_folds["method"].isin(target_methods)]
        has_per_fold = True
    except FileNotFoundError:
        print(
            f"Warning: Per-fold data not found at {per_fold_path}, using summary values."
        )
        has_per_fold = False

    d2_path = os.path.join(output_dir, "table_d2_loglikelihood.tex")
    with open(d2_path, "w") as f:
        f.write(
            "% Suggested caption: Mean log-likelihood (median in parentheses) from cross-validated held-out folds. Higher is better. Significance of Wilcoxon signed-rank tests vs.\\ the reference method: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.\n"
        )
        f.write(r"\begin{tabular}{llc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(
            r"\textbf{Dataset} & \textbf{Method} & \textbf{Log-Likelihood} \\ \hline"
            + "\n"
        )

        for dataset in datasets:
            subset = df[df["dataset"] == dataset].copy()

            # Sort
            subset["method"] = pd.Categorical(
                subset["method"], categories=target_methods, ordered=True
            )
            subset = subset.sort_values("method")

            # Compute mean and median from per-fold data if available
            if has_per_fold:
                fold_subset = df_folds[df_folds["dataset"] == dataset]
                fold_stats = fold_subset.groupby("method")["log_likelihood"].agg(
                    ["mean", "median"]
                )
            else:
                fold_stats = None

            # Best Log-Likelihood (Max is best) - use mean from fold data if available
            if fold_stats is not None and not fold_stats.empty:
                best_ll = fold_stats["mean"].max()
            else:
                best_ll = subset["log_likelihood"].max()

            # Multirow
            f.write(f"\\multirow{{6}}{{*}}{{\\textit{{{dataset}}}}} \n")

            for _, row in subset.iterrows():
                method_macro = get_method_macro(row["method"])
                method_name = row["method"]

                # Log Likelihood - mean (median) from per-fold data
                if fold_stats is not None and method_name in fold_stats.index:
                    ll_mean = fold_stats.loc[method_name, "mean"]
                    ll_median = fold_stats.loc[method_name, "median"]
                    mean_str = format_value(
                        ll_mean, best_ll, is_min_best=False, decimals=2
                    )
                    ll_str = f"{mean_str} ({ll_median:.2f})"
                else:
                    ll_val = row["log_likelihood"]
                    ll_str = format_value(
                        ll_val, best_ll, is_min_best=False, decimals=2
                    )

                # Significance asterisks for non-reference methods
                if method_name != "BETA_ROT":
                    p_col = "loglik_p_value_wilcoxon (BETA_ROT >)"
                    if p_col in row.index and pd.notna(row[p_col]):
                        ll_str += significance_stars(row[p_col])

                f.write(f" & {method_macro} & {ll_str} \\\\\n")

            f.write(r"\hline" + "\n")

        f.write(r"\end{tabular}" + "\n")
    print(f"Generated {d2_path}")


if __name__ == "__main__":
    csv_file = str(DATA_DIR / "experiment2" / "experiment_2_summary.csv")
    output_folder = str(TABLES_DIR)
    os.makedirs(output_folder, exist_ok=True)

    generate_appendix_tables(csv_file, output_folder)
