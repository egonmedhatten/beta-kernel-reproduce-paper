"""Generate LaTeX summary tables from Experiment 1 simulation results.

Loads the raw output from ``experiment_1_parallell.py``, splits data into
"nice", "hard", and "bimodal" distribution groups, and generates separate
publication-ready LaTeX tables for LSCV scores, ISE scores, and computation
times. Also produces detailed appendix tables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, TABLES_DIR

import pandas as pd
import numpy as np
import warnings
import re
import os
from scipy import stats
from tabulate import tabulate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print(f"--- Running with pandas version: {pd.__version__} ---")


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


# --- LaTeX Macro Definitions ---
LATEX_MACRO_MAP = {
    "BETA_ROT": r"\rott",
    "BETA_LSCV": r"\blscvt",
    "BETA_ISE": r"\biset",
    "BETA_ORACLE": r"\oraclet",
    "REFLECT_SILV": r"\rsilvt",
    "REFLECT_LSCV": r"\rlscvt",
    "REFLECT_ISE": r"\riset",
    "LOGIT_SILV": r"\lsilvt",
    "LOGIT_LSCV": r"\llscvt",
    "LOGIT_ISE": r"\liset",
}


def load_raw_data(file_path):
    """Loads the raw, unaggregated simulation data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    if df.empty:
        print("The CSV file is empty.")
        return None
    print(f"Loaded {len(df)} rows (all trials) from '{file_path}'.")
    return df


def format_dist_names_for_index(index_df):
    """Converts index names to pretty format for tables."""
    dist_level = index_df.index.get_level_values("distribution")
    new_dist_index = []
    for dist_name in dist_level:
        match = re.match(r"B\((\d+\.?\d*), (\d+\.?\d*)\)", dist_name)
        if match:
            new_dist_index.append(f"B({match.group(1)}, {match.group(2)})")
            continue
        match = re.match(r"NT\((\d+\.?\d*), (\d+\.?\d*)\)", dist_name)
        if match:
            var = float(match.group(2))
            new_dist_index.append(f"NT({match.group(1)}, {var:.2f})")
            continue
        if dist_name == "BIMODAL":
            new_dist_index.append("Bimodal")
            continue
        new_dist_index.append(dist_name)
    index_df.index = pd.MultiIndex.from_tuples(
        zip(new_dist_index, index_df.index.get_level_values("n")),
        names=["Distribution", "N"],
    )
    return index_df


def get_stats_for_metric(df_raw, methods, ref_method, metric_name, metric_col):
    """
    Calculates the mean, median, and p-value for a *single* metric
    across all methods.
    """
    stats_data = {}

    for method in methods:
        col_full_name = f"{method}_{metric_col}"

        mean_val = np.nan
        median_val = np.nan
        if col_full_name in df_raw.columns:
            mean_val = df_raw[col_full_name].mean()
            median_val = df_raw[col_full_name].median()

        p_val_raw = np.nan  # Raw numeric p-value

        if method != ref_method:
            ref_col_name = f"{ref_method}_{metric_col}"
            if (
                ref_col_name not in df_raw.columns
                or col_full_name not in df_raw.columns
            ):
                pass
            else:
                ref_series = df_raw[ref_col_name]
                method_series = df_raw[col_full_name]
                paired_df = pd.concat([ref_series, method_series], axis=1).dropna()

                if not paired_df.empty:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        _, p_val = stats.wilcoxon(
                            paired_df.iloc[:, 0], paired_df.iloc[:, 1]
                        )
                    p_val_raw = p_val

        stats_data[method] = {
            "mean": mean_val,
            "median": median_val,
            "p_val_raw": p_val_raw,
        }

    return stats_data


# --- NEW: Function to create a single-metric table ---
def create_metric_table(
    data_groups: dict,
    methods: list,
    ref_method: str,
    metric_name: str,
    metric_col: str,
    latex_file: str,
    include_hard_dist: bool = True,
    caption: str = "",
):
    """
    Generates a single, pivoted table for one metric (e.g., LSCV).
    """
    print("\n\n" + "=" * 80)
    print(f"--- MAIN PUBLICATION TABLE: {metric_name.upper()} Scores ---")
    print("=" * 80)

    # 1. Get stats for each group
    group_names = ["'Nice' Distributions", "'Bimodal' Distribution"]
    all_stats = {
        "'Nice' Distributions": get_stats_for_metric(
            data_groups["nice"], methods, ref_method, metric_name, metric_col
        ),
        "'Bimodal' Distribution": get_stats_for_metric(
            data_groups["bimodal"], methods, ref_method, metric_name, metric_col
        ),
    }
    if include_hard_dist:
        all_stats["'Hard' Distributions"] = get_stats_for_metric(
            data_groups["hard"], methods, ref_method, metric_name, metric_col
        )
        group_names.append("'Hard' Distributions")

    # 2. Build the DataFrame
    is_time = metric_name == "Time"

    # First, get the mean and median scores
    df_mean = pd.DataFrame.from_dict(
        {
            group: {m: s["mean"] for m, s in stats.items()}
            for group, stats in all_stats.items()
        },
        orient="index",
    ).T
    df_mean = df_mean[group_names]  # Ensure column order

    df_median = pd.DataFrame.from_dict(
        {
            group: {m: s["median"] for m, s in stats.items()}
            for group, stats in all_stats.items()
        },
        orient="index",
    ).T
    df_median = df_median[group_names]

    # Find bests for bolding: use median for non-time, mean for time
    if is_time:
        bests = {col: df_mean[col].idxmin() for col in df_mean.columns}
    else:
        bests = {col: df_median[col].idxmin() for col in df_median.columns}

    # Second, build the final string DataFrame
    df_formatted = pd.DataFrame(index=methods, columns=group_names)
    for group in group_names:
        for method in methods:
            mean = all_stats[group][method]["mean"]
            median = all_stats[group][method]["median"]
            p_raw = all_stats[group][method]["p_val_raw"]

            if pd.isna(mean):
                df_formatted.loc[method, group] = "-"
                continue

            if is_time:
                # Time: report only mean, no median, no significance, no bolding
                df_formatted.loc[method, group] = f"{mean:.4f}"
            else:
                # Non-time: mean (median), bold entire cell based on best median
                cell = f"{mean:.4f} ({median:.4f})"
                if method == bests.get(group):
                    cell = f"\\textbf{{{cell}}}"
                stars = significance_stars(p_raw) if method != ref_method else ""
                df_formatted.loc[method, group] = f"{cell}{stars}"

    # --- 3. Create Markdown Version ---
    df_md = df_formatted.copy()
    df_md = df_md.map(lambda x: str(x).replace(r"\textbf{", "**").replace("}", "**"))
    df_md.index.name = "Method"
    print(tabulate(df_md, headers="keys", tablefmt="pipe", stralign="center"))

    # --- 4. Create LaTeX Version ---
    df_latex = df_formatted.copy()

    df_latex.index = df_latex.index.map(lambda m: LATEX_MACRO_MAP.get(m, m))

    df_latex.index.name = "Method"

    col_format = "l" + "c" * len(group_names)  # e.g., lccc

    latex_string = df_latex.to_latex(
        escape=False,
        column_format=col_format,
        header=True,
        na_rep="-",
    )

    try:
        with open(latex_file, "w") as f:
            if caption:
                f.write(f"% Suggested caption: {caption}\n")
            f.write(latex_string)
        print(f"\nSuccessfully saved LaTeX table to: {latex_file}")
    except Exception as e:
        print(f"\nError saving LaTeX file: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = str(DATA_DIR / "experiment1" / "simulation_results_full.csv")

    latex_output_path = str(TABLES_DIR)
    os.makedirs(latex_output_path, exist_ok=True)

    method_order = [
        "BETA_ROT",
        "BETA_LSCV",
        "BETA_ISE",
        "BETA_ORACLE",
        "LOGIT_SILV",
        "LOGIT_LSCV",
        "LOGIT_ISE",
        "REFLECT_SILV",
        "REFLECT_LSCV",
        "REFLECT_ISE",
    ]

    raw_data = load_raw_data(file_path)

    if raw_data is not None:

        hard_dists = ["B(0.5, 0.5)", "B(0.8, 2.5)", "B(1.5, 1.5)"]
        bimodal_dist = "BIMODAL"

        df_hard = raw_data[raw_data["distribution"].isin(hard_dists)].copy()
        df_bimodal = raw_data[raw_data["distribution"] == bimodal_dist].copy()
        df_nice = raw_data[
            ~raw_data["distribution"].isin(hard_dists)
            & (raw_data["distribution"] != bimodal_dist)
        ].copy()

        print(
            f"\nSplitting analysis: {len(df_hard)} 'Hard' trials, {len(df_bimodal)} 'Bimodal' trials, {len(df_nice)} 'Nice' trials."
        )

        data_groups = {"nice": df_nice, "bimodal": df_bimodal, "hard": df_hard}

        create_metric_table(
            data_groups,
            method_order,
            "BETA_ROT",
            "LSCV",
            "lscv_score",
            os.path.join(latex_output_path, "main_table_lscv.tex"),
            include_hard_dist=True,
            caption="Mean LSCV scores (median in parentheses) across distribution groups. Bold indicates the best median per group. Significance of Wilcoxon signed-rank tests vs.\\ the reference method: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
        )

        create_metric_table(
            data_groups,
            method_order,
            "BETA_ROT",
            "ISE",
            "ise_score",
            os.path.join(latex_output_path, "main_table_ise.tex"),
            include_hard_dist=False,  # ISE is not valid for 'hard'
            caption="Mean ISE scores (median in parentheses) across distribution groups. Bold indicates the best median per group. Significance of Wilcoxon signed-rank tests vs.\\ the reference method: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
        )

        create_metric_table(
            data_groups,
            method_order,
            "BETA_ROT",
            "Time",
            "comp_time",
            os.path.join(latex_output_path, "main_table_time.tex"),
            include_hard_dist=True,
            caption="Mean computation times in seconds across distribution groups.",
        )
