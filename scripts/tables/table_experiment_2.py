"""Generate the main LaTeX table for Experiment 2 results."""

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


def format_time(t):
    """Format time: <0.0001 as <0.0001, small as 0.xxxx, large as 12.3"""
    if t < 0.0001:
        return "$<0.0001$"
    elif t < 1.0:
        return f"{t:.4f}"
    else:
        return f"{t:.1f}"


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


def generate_latex_file(csv_path, output_path):
    df = pd.read_csv(csv_path)

    target_methods = [
        "BETA_ROT",
        "BETA_LSCV",
        "LOGIT_SILV",
        "LOGIT_LSCV",
        "REFLECT_SILV",
        "REFLECT_LSCV",
    ]
    df = df[df["method"].isin(target_methods)]

    # Load per-fold data for mean (median) heldout density
    per_fold_path = str(DATA_DIR / "experiment2" / "per_fold" / "experiment_2_per_fold_results.csv")
    try:
        df_folds = pd.read_csv(per_fold_path)
        df_folds = df_folds[df_folds["method"].isin(target_methods)]
        fold_stats = df_folds.groupby(["dataset", "method"])["mean_heldout_density"].agg(["mean", "median"])
        has_folds = True
    except FileNotFoundError:
        print(f"Warning: Per-fold data not found at {per_fold_path}")
        has_folds = False

    datasets = df["dataset"].unique()

    with open(output_path, "w") as f:
        # Suggested caption as LaTeX comment
        f.write("% Suggested caption: Experiment 2 results on real-world datasets. LSCV scores (lower is better), mean heldout density with median in parentheses (higher is better), computation time, and fallback rate. Bold indicates the best value per dataset. Significance of Wilcoxon signed-rank tests vs.\\ the reference method: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.\n")
        # Write the table content (no begin/end table, no caption)
        f.write(r"\begin{tabular}{lcccccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(
            r"\textbf{Dataset} & \textbf{Method} & \textbf{LSCV Score} & \textbf{Heldout Density} & \textbf{Time (s)} & \textbf{Fallback Rate} \\ \hline"
            + "\n"
        )

        for dataset in datasets:
            subset = df[df["dataset"] == dataset].copy()

            # Find best LSCV score (min) and Time (min) for bolding
            best_lscv = subset["lscv_score"].min()
            best_time = subset["comp_time_sec"].min()

            # Find best heldout density (max is best)
            if has_folds:
                ds_fold_stats = fold_stats.loc[dataset] if dataset in fold_stats.index.get_level_values(0) else None
                best_density = ds_fold_stats["mean"].max() if ds_fold_stats is not None else None
            else:
                ds_fold_stats = None
                best_density = None

            # Pretty print dataset name with multirow - Adjusted row count to 6
            f.write(f"\\multirow{{6}}{{*}}{{\\textit{{{dataset}}}}} \n")

            # Order methods: ROT first, then Benchmarks
            method_order = target_methods
            subset["method"] = pd.Categorical(
                subset["method"], categories=method_order, ordered=True
            )
            subset = subset.sort_values("method")

            for _, row in subset.iterrows():
                method_macro = get_method_macro(row["method"])

                # Bandwidth
                h_str = f"{row['bandwidth']:.4f}"

                # LSCV (Bold if best, within tolerance) + significance asterisks
                lscv_val = row["lscv_score"]
                if abs(lscv_val - best_lscv) < 1e-6:
                    lscv_str = f"\\textbf{{{lscv_val:.4f}}}"
                else:
                    lscv_str = f"{lscv_val:.4f}"

                # Add significance asterisks for non-reference methods
                if row["method"] != "BETA_ROT":
                    p_col = "density_p_value_wilcoxon (BETA_ROT >)"
                    if p_col in row.index and pd.notna(row[p_col]):
                        lscv_str += significance_stars(row[p_col])

                # Heldout Density mean (median) from per-fold data
                if ds_fold_stats is not None and row["method"] in ds_fold_stats.index:
                    d_mean = ds_fold_stats.loc[row["method"], "mean"]
                    d_median = ds_fold_stats.loc[row["method"], "median"]
                    # Bold if best mean density
                    if best_density is not None and abs(d_mean - best_density) < 1e-9:
                        density_str = f"\\textbf{{{d_mean:.3f}}} ({d_median:.3f})"
                    else:
                        density_str = f"{d_mean:.3f} ({d_median:.3f})"
                    # Asterisks for non-reference
                    if row["method"] != "BETA_ROT":
                        p_col = "density_p_value_wilcoxon (BETA_ROT >)"
                        if p_col in row.index and pd.notna(row[p_col]):
                            density_str += significance_stars(row[p_col])
                else:
                    density_str = "-"

                # Time (Bold if best)
                time_val = row["comp_time_sec"]
                time_str = format_time(time_val)
                if abs(time_val - best_time) < 1e-6:
                    time_str = f"\\textbf{{{time_str}}}"

                # Fallback Rate Logic
                if row["method"] == "BETA_ROT":
                    rate = row["is_fallback_cv_mean"] * 100
                    fallback_str = f"{rate:.0f}\\%"
                else:
                    fallback_str = "-"

                f.write(
                    f" & {method_macro} & {lscv_str} & {density_str} & {time_str} & {fallback_str} \\\\\n"
                )

            f.write(r"\hline" + "\n")

        f.write(r"\end{tabular}" + "\n")

    print(f"Successfully generated table code in: {output_path}")


if __name__ == "__main__":
    # Run with your specific filename
    latex_output_path = str(TABLES_DIR)
    os.makedirs(latex_output_path, exist_ok=True)
    generate_latex_file(
        str(DATA_DIR / "experiment2" / "experiment_2_summary.csv"),
        f"{latex_output_path}/experiment_2_table.tex",
    )
