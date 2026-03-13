"""Export aggregated Experiment 1 results to a supplementary CSV file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_DIR, SUPPLEMENTARY_DIR

import pandas as pd
import numpy as np
import os


def format_dist_name(dist_name):
    """Standardizes distribution names for the CSV."""
    if dist_name == "BIMODAL":
        return "Bimodal"
    # Add other formatting if needed, or leave as raw string
    return dist_name


def export_experiment_1_csv():
    # 1. Configuration
    input_file = str(DATA_DIR / "experiment1" / "simulation_results_full.csv")
    output_file = str(SUPPLEMENTARY_DIR / "supplementary_experiment_1_results.csv")

    # Check if input exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Reading raw data from {input_file}...")
    df = pd.read_csv(input_file)

    # 2. Define the columns we want to summarize
    methods = [
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

    # Create dictionary of aggregations
    # We want Mean and Standard Deviation for LSCV, ISE, and Time
    agg_dict = {}

    for method in methods:
        # LSCV
        if f"{method}_lscv_score" in df.columns:
            agg_dict[f"{method}_lscv_score"] = ["mean", "std"]
        # ISE
        if f"{method}_ise_score" in df.columns:
            agg_dict[f"{method}_ise_score"] = ["mean", "std"]
        # Time
        if f"{method}_comp_time" in df.columns:
            agg_dict[f"{method}_comp_time"] = ["mean", "std"]

    # Add Fallback rate if available (only need mean)
    if "BETA_ROT_is_fallback" in df.columns:
        agg_dict["BETA_ROT_is_fallback"] = ["mean"]

    # 3. Group and Aggregate
    print("Aggregating data (Mean & Std)...")
    # Group by Distribution and Sample Size
    df_grouped = df.groupby(["distribution", "n"]).agg(agg_dict)

    # 4. Flatten MultiIndex Columns
    # Current cols are like ('BETA_ROT_lscv_score', 'mean')
    # We want 'BETA_ROT_LSCV_Mean'
    new_cols = []
    for col_tuple in df_grouped.columns:
        metric_raw = col_tuple[0]
        stat = col_tuple[1].capitalize()  # 'mean' -> 'Mean'

        # Clean up metric name
        if "lscv_score" in metric_raw:
            metric = "LSCV"
            method = metric_raw.replace("_lscv_score", "")
        elif "ise_score" in metric_raw:
            metric = "ISE"
            method = metric_raw.replace("_ise_score", "")
        elif "comp_time" in metric_raw:
            metric = "Time_Sec"
            method = metric_raw.replace("_comp_time", "")
        elif "is_fallback" in metric_raw:
            metric = "Fallback_Rate"
            method = "BETA_ROT"
        else:
            metric = metric_raw
            method = ""

        new_cols.append(f"{method}_{metric}_{stat}")

    df_grouped.columns = new_cols

    # 5. Reset Index and Format
    df_grouped = df_grouped.reset_index()
    df_grouped["distribution"] = df_grouped["distribution"].apply(format_dist_name)

    # Rename grouping columns for clarity
    df_grouped.rename(
        columns={"distribution": "Distribution", "n": "Sample_Size"}, inplace=True
    )

    # 6. Save
    print(f"Saving to {output_file}...")
    df_grouped.to_csv(output_file, index=False)
    print("Done!")


if __name__ == "__main__":
    export_experiment_1_csv()
