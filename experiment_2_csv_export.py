import pandas as pd
import os


def export_experiment_2_csv():
    # 1. Configuration
    input_file = "data/experiment2/experiment_2_summary.csv"
    output_file = "supplementary-results/supplementary_experiment_2_results.csv"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)

    # 2. Filter Methods (matching your generate_latex_file list)
    target_methods = [
        "BETA_ROT",
        "BETA_LSCV",
        "LOGIT_SILV",
        "LOGIT_LSCV",
        "REFLECT_SILV",
        "REFLECT_LSCV",
    ]
    df = df[df["method"].isin(target_methods)].copy()

    # 3. Select and Rename Columns
    # Mapping raw columns to pretty headers
    cols_to_keep = {
        "dataset": "Dataset",
        "method": "Method",
        "lscv_score": "LSCV_Score",
        "log_likelihood": "Log_Likelihood",
        "comp_time_sec": "Computation_Time_Sec",
        "bandwidth": "Selected_Bandwidth",
        "is_fallback_cv_mean": "Fallback_Rate",
        # Include P-values if they exist in summary, based on your appendix script
        "density_p_value_wilcoxon (BETA_ROT >)": "P_Value_LSCV_vs_ROT",
        "loglik_p_value_wilcoxon (BETA_ROT >)": "P_Value_LogLik_vs_ROT",
    }

    # Filter columns that actually exist in the CSV
    existing_cols = {k: v for k, v in cols_to_keep.items() if k in df.columns}

    df_clean = df[list(existing_cols.keys())].rename(columns=existing_cols)

    # 4. Format Values (Optional cleaning)
    df_clean.sort_values(by=["Dataset", "Method"], inplace=True)

    # 5. Save
    print(f"Saving to {output_file}...")
    df_clean.to_csv(output_file, index=False)
    print("Done!")


if __name__ == "__main__":
    export_experiment_2_csv()
