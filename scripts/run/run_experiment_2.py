"""Experiment 2: Real-world density estimation on UCI crime data.

Compares six kernel density estimators on three variables from the UCI
Communities and Crime dataset. Uses true leave-one-out cross-validation
for the LSCV score and 10x10 repeated CV for log-likelihood tests.

Methods compared:
    - Fast rules: BETA_ROT, LOGIT_SILV, REFLECT_SILV
    - Slow LSCV:  BETA_LSCV, LOGIT_LSCV, REFLECT_LSCV

Statistical significance is assessed via Wilcoxon signed-rank tests
against the proposed BETA_ROT rule.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import REPO_ROOT, DATA_DIR, PLOTS_DIR

import os
import time
import warnings

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import beta, truncnorm, wilcoxon
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from ucimlrepo import fetch_ucirepo

from KDE import BetaKernelKDE
from KDE_Gauss import GaussianKDE

# --- Global Settings ---
N_FOLDS = 10
N_REPETITIONS = 10
RND_GEN = np.random.default_rng(2025)
# A small value to clip PDFs before np.log to avoid -inf
EPSILON = 1e-300

# --- Competitor Method Definitions ---
# The 6 practical methods for the real-world experiment
METHODS_TO_RUN = {
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

# %%
# =============================================================================
# HELPER FUNCTIONS (from Experiment 1)
# =============================================================================


def lscv_score(kde, n_folds=None):
    """
    Calculates the LSCV score for a *fitted* KDE instance using true
    leave-one-out cross-validation for Term 2. This is O(n^2).
    """
    bandwidth = kde.bandwidth
    data = kde.data_
    kde_class = kde.__class__

    kde_init_kwargs = {}
    if hasattr(kde, "method"):
        kde_init_kwargs["method"] = kde.method

    if bandwidth is None or bandwidth <= 0 or not np.isfinite(bandwidth):
        return np.nan
    if data is None or len(data) == 0:
        return np.nan
    n = len(data)

    # Term 1: integral of f_hat^2
    try:
        integrand_sq = lambda x: kde.pdf(x) ** 2
        term1, _ = quad(integrand_sq, 0, 1, limit=50, epsabs=1e-4)
        if not np.isfinite(term1):
            term1 = np.nan
    except Exception:
        term1 = np.nan

    # Term 2: LOO sum
    term2_sum = 0

    # This loop is O(n^2)
    print("\nCalculating true LOO score... (This may take a while)")
    try:
        # Create a mask for all data points
        all_indices = np.arange(n)

        for i in tqdm(range(n), desc="LSCV LOO"):
            # Get the i-th data point (the one to leave out)
            x_i = data[i]

            # Create the LOO training set by masking the i-th point
            X_loo = data[all_indices != i]

            if len(X_loo) == 0:
                continue

            # Fit a new KDE on the LOO data (n-1 points)
            kde_loo = kde_class(bandwidth=bandwidth, **kde_init_kwargs)
            kde_loo.fit(X_loo)

            # Evaluate the PDF at the left-out point
            pdf_val_i = kde_loo.pdf(x_i)

            if np.isfinite(pdf_val_i):
                term2_sum += pdf_val_i

        term2 = 2 * (term2_sum / n)
        if not np.isfinite(term2):
            term2 = np.nan

    except Exception as e:
        print(f"Warning: LOO calculation failed. Error: {e}")
        term2 = np.nan

    if np.isnan(term1) or np.isnan(term2):
        return np.nan

    return term1 - term2


# %%
# =============================================================================
# EXPERIMENT 2 CORE LOGIC
# =============================================================================


def run_comparison(data, data_name):
    """
    Runs the full 6-method comparison on a single real-world dataset.

    Returns two DataFrames:
    1. A summary DataFrame with aggregated scores.
    2. A per-fold DataFrame (for t-tests).
    """
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT ON: {data_name} (n={len(data)})")
    print("=" * 80)

    # --- 1. Fit on FULL data (for comp_time and LSCV score) ---
    print("Fitting all 6 methods on full dataset...")
    fitted_kdes = {}
    results_list = []

    for method_name, config in tqdm(METHODS_TO_RUN.items(), desc="Fitting methods"):
        try:
            kde = config["class"](**config["init_args"])

            start_time = time.perf_counter()
            kde.fit(data)
            comp_time = time.perf_counter() - start_time

            if kde.bandwidth is None or not np.isfinite(kde.bandwidth):
                raise Exception("Bandwidth selection failed.")

            fitted_kdes[method_name] = kde

            # Calculate LSCV score (on full data) using the new
            # true LOO function. This will be SLOW.
            score_lscv = lscv_score(kde)

            is_fallback_full = False
            if method_name == "BETA_ROT":
                is_fallback_full = getattr(kde, "is_fallback", False)

            results_list.append(
                {
                    "dataset": data_name,
                    "method": method_name,
                    "bandwidth": kde.bandwidth,
                    "comp_time_sec": comp_time,
                    "lscv_score": score_lscv,
                    "is_fallback_full": is_fallback_full,
                }
            )

        except Exception as e:
            print(f"!! FAILED (full fit): {method_name}. Error: {e}")
            results_list.append(
                {
                    "dataset": data_name,
                    "method": method_name,
                    "bandwidth": np.nan,
                    "comp_time_sec": np.nan,
                    "lscv_score": np.nan,
                    "is_fallback_full": np.nan,
                }
            )

    df_summary = pd.DataFrame(results_list)

    # --- 2. Calculate 10x10 Repeated CV Metrics ---
    print(
        f"\nCalculating {N_REPETITIONS}x{N_FOLDS} Repeated CV Metrics for all 6 methods..."
    )
    fold_results = []

    for rep in tqdm(range(N_REPETITIONS), desc="CV Repetitions"):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RND_GEN.integers(1000))

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
            X_train, X_test = data[train_idx], data[test_idx]

            for method_name, config in METHODS_TO_RUN.items():
                try:
                    kde_fold = config["class"](**config["init_args"])
                    kde_fold.fit(X_train)  # Re-fit selector on the training fold

                    if kde_fold.bandwidth is None or not np.isfinite(
                        kde_fold.bandwidth
                    ):
                        raise Exception("Bandwidth selection failed.")

                    # Evaluate on the left-out fold
                    pdf_values = kde_fold.pdf(X_test)

                    # Metric 1: Log-Likelihood (for predictive power)
                    pdf_values_log = np.clip(pdf_values, EPSILON, 1.0)  # Avoid log(0)
                    mean_log_lik = np.mean(np.log(pdf_values_log))

                    # Metric 2: Mean Held-Out Density (for LSCV-optimality)
                    # This is just mean(f_hat_fold(X_test))
                    mean_density = np.mean(pdf_values)

                    is_fallback_fold = False
                    if method_name == "BETA_ROT":
                        is_fallback_fold = getattr(kde_fold, "is_fallback", False)

                    fold_results.append(
                        {
                            "dataset": data_name,
                            "repetition": rep + 1,
                            "fold": fold_idx + 1,
                            "method": method_name,
                            "log_likelihood": mean_log_lik,
                            "mean_heldout_density": mean_density,
                            "is_fallback": is_fallback_fold,
                        }
                    )

                except Exception as e:
                    fold_results.append(
                        {
                            "dataset": data_name,
                            "repetition": rep + 1,
                            "fold": fold_idx + 1,
                            "method": method_name,
                            "log_likelihood": np.nan,
                            "mean_heldout_density": np.nan,
                            "is_fallback": np.nan,
                        }
                    )

    df_per_fold = pd.DataFrame(fold_results)

    # Aggregate scores and merge
    df_agg_summary = (
        df_per_fold.groupby("method")[
            ["log_likelihood", "mean_heldout_density", "is_fallback"]
        ]
        .mean()
        .reset_index()
    )
    df_agg_summary.rename(columns={"is_fallback": "is_fallback_cv_mean"}, inplace=True)

    df_summary = df_summary.merge(df_agg_summary, on="method", how="left")

    return df_summary, df_per_fold, fitted_kdes


def plot_densities(fitted_kdes, data, data_name, output_dir=str(PLOTS_DIR)):
    """
    Plots the density estimates for each method and saves the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate a smooth range of x-values for plotting
    x_min, x_max = 0.0, 1.0
    x_plot = np.linspace(x_min, x_max, 1000)

    plt.figure(figsize=(10, 6))

    # Plot histogram of the raw data for context
    plt.hist(data, bins=30, density=True, alpha=0.5, label="Data Histogram")

    for method_name, kde_model in fitted_kdes.items():
        if kde_model is not None and hasattr(kde_model, "pdf"):
            try:
                pdf_values = kde_model.pdf(x_plot)
                plt.plot(
                    x_plot,
                    pdf_values,
                    label=f"{method_name} (h={kde_model.bandwidth:.4f})",
                )
            except Exception as e:
                print(
                    f"Warning: Could not plot {method_name} for {data_name}. Error: {e}"
                )
        else:
            print(
                f"Warning: KDE model for {method_name} is not fitted or has no pdf method."
            )

    plt.title(f"Density Estimates for {data_name}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.xlim(x_min, x_max)

    plot_filename = os.path.join(output_dir, f"density_plot_{data_name}.pdf")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved density plot for {data_name} to {plot_filename}")


def main():
    """
    Main function to fetch data and run all experiments.
    """
    # 1. Fetch the data
    try:
        print("Fetching UCI Communities and Crime dataset...")
        communities_crime = fetch_ucirepo(id=183)
        X = communities_crime.data.features
        print("Data fetched successfully.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # 2. Define the datasets we want to test
    datasets_to_test = {
        "PctKids2Par": X["PctKids2Par"].dropna().values,
        "PctPopUnderPov": X["PctPopUnderPov"].dropna().values,
        "PctVacantBoarded": X["PctVacantBoarded"].dropna().values,
    }

    all_summary_results = []
    all_fold_results = []

    # 3. Run the comparison for each dataset
    for data_name, data_vector in datasets_to_test.items():
        data_to_run = data_vector
        if len(data_to_run) < 200:
            print(f"\nSkipping {data_name}: Not enough data (n={len(data_to_run)})")
            continue

        df_summary, df_per_fold, fitted_kdes = run_comparison(data_to_run, data_name)
        all_summary_results.append(df_summary)
        all_fold_results.append(df_per_fold)

        # Plot and save density estimates
        plot_densities(fitted_kdes, data_to_run, data_name)

    # --- 4. Consolidate and Save All Results ---
    if not all_summary_results:
        print("\nNo experiments were run. Exiting.")
        return

    df_final_summary = pd.concat(all_summary_results, ignore_index=True)
    df_final_folds = pd.concat(all_fold_results, ignore_index=True)

    # --- 5. Run Statistical Tests ---
    print("\n" + "=" * 80)
    print("Running Wilcoxon Signed-Rank Tests (vs. BETA_ROT)...")

    test_results = []
    baseline_method = "BETA_ROT"  # Your "proposed" method

    df_pivot_metrics = df_final_folds.pivot_table(
        index=["dataset", "repetition", "fold"],
        columns="method",
        values=["log_likelihood", "mean_heldout_density"],
    )

    for method_name in METHODS_TO_RUN:
        if method_name == baseline_method:
            continue

        for dataset_name in df_final_summary["dataset"].unique():
            baseline_score_lscv = df_final_summary.loc[
                (df_final_summary["dataset"] == dataset_name)
                & (df_final_summary["method"] == baseline_method),
                "lscv_score",
            ].values[0]

            method_score_lscv = df_final_summary.loc[
                (df_final_summary["dataset"] == dataset_name)
                & (df_final_summary["method"] == method_name),
                "lscv_score",
            ].values[0]

            # Get the per-fold data for this dataset
            fold_data = df_pivot_metrics.loc[dataset_name]

            p_val_loglik = np.nan
            p_val_density = np.nan
            n_samples = N_FOLDS * N_REPETITIONS

            try:
                # Test 1: LogLik (Higher is better)
                baseline_scores_loglik = fold_data[
                    ("log_likelihood", baseline_method)
                ].dropna()
                method_scores_loglik = fold_data[
                    ("log_likelihood", method_name)
                ].dropna()

                if (
                    len(baseline_scores_loglik) == n_samples
                    and len(method_scores_loglik) == n_samples
                ):
                    # Test: baseline_scores > method_scores
                    loglik_test = wilcoxon(
                        baseline_scores_loglik - method_scores_loglik,
                        alternative="greater",
                    )
                    p_val_loglik = loglik_test.pvalue

                # Test 2: Mean Held-Out Density (Higher is better)
                baseline_scores_density = fold_data[
                    ("mean_heldout_density", baseline_method)
                ].dropna()
                method_scores_density = fold_data[
                    ("mean_heldout_density", method_name)
                ].dropna()
                if (
                    len(baseline_scores_density) == n_samples
                    and len(method_scores_density) == n_samples
                ):
                    # Test: baseline_scores > method_scores
                    density_test = wilcoxon(
                        baseline_scores_density - method_scores_density,
                        alternative="greater",
                    )
                    p_val_density = density_test.pvalue

            except Exception as e:
                print(
                    f"Warning: Wilcoxon test failed for {method_name} on {dataset_name}: {e}"
                )

            test_results.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "lscv_BETA_ROT_wins": baseline_score_lscv
                    < method_score_lscv,  # Lower is better
                    "loglik_p_value_wilcoxon (BETA_ROT >)": p_val_loglik,
                    "density_p_value_wilcoxon (BETA_ROT >)": p_val_density,
                }
            )

    df_test_results = pd.DataFrame(test_results)

    # Merge p-values back into the main summary
    df_final_summary = df_final_summary.merge(
        df_test_results, on=["dataset", "method"], how="left"
    )

    # --- 6. Save to CSV ---
    summary_file = str(DATA_DIR / "experiment2" / "experiment_2_summary.csv")
    per_fold_file = str(
        DATA_DIR / "experiment2" / "per_fold" / "experiment_2_per_fold_results.csv"
    )
    os.makedirs(str(DATA_DIR / "experiment2"), exist_ok=True)
    os.makedirs(str(DATA_DIR / "experiment2" / "per_fold"), exist_ok=True)

    # Re-order columns to put new fallback columns in a logical place
    all_cols = list(df_final_summary.columns)
    # Pop the new columns
    is_fallback_full = all_cols.pop(all_cols.index("is_fallback_full"))
    is_fallback_cv_mean = all_cols.pop(all_cols.index("is_fallback_cv_mean"))
    # Re-insert them after 'bandwidth'
    df_final_summary = df_final_summary.reindex(
        columns=all_cols[:3] + [is_fallback_full, is_fallback_cv_mean] + all_cols[3:]
    )

    df_final_summary.to_csv(summary_file, index=False)
    df_final_folds.to_csv(per_fold_file, index=False)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2 COMPLETE.")
    print(f"Summary table saved to '{summary_file}'")
    print(f"Per-fold results saved to '{per_fold_file}'")
    print("\n--- Final Summary Table (Mean Scores) ---")
    print(df_final_summary.to_markdown(index=False, floatfmt=".4f"))
    print("=" * 80)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    main()
