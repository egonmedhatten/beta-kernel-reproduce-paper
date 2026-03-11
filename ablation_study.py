#!/usr/bin/env python3
"""
Ablation Study for the Beta Reference Rule Fallback Heuristic.

This script tests four variations of the fallback scaling factor (C)
on boundary-concentrated and bimodal distributions to rigorously justify
the chosen functional form, as requested by Reviewer 1.

Models tested:
- Model A: Variance only
- Model B: Variance + Skewness
- Model C: Variance + Excess Kurtosis
- Model D: Proposed Rule (Variance + Skewness + Excess Kurtosis)
"""

import concurrent.futures
import os
import time
import warnings

import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from scipy.stats import beta
from KDE import BetaKernelKDE

# =============================================================================
# ABLATION MODELS (Subclassing to cleanly override the fallback)
# =============================================================================


class BetaKDE_ModelA(BetaKernelKDE):
    """Model A: Naive Baseline (Variance only)."""

    def _calculate_hybrid_fallback(self, a, b) -> float:
        s = np.sqrt(self.variance(a, b))
        n = self.n_samples_
        return 1e-5 if s == 0 else s * (n ** (-0.4))


class BetaKDE_ModelB(BetaKernelKDE):
    """Model B: Asymmetry Penalty (Variance + Skewness)."""

    def _calculate_hybrid_fallback(self, a, b) -> float:
        s = np.sqrt(self.variance(a, b))
        sk = self.skewness(a, b)
        n = self.n_samples_
        C = s / (1 + abs(sk))
        return 1e-5 if s == 0 else C * (n ** (-0.4))


class BetaKDE_ModelC(BetaKernelKDE):
    """Model C: Spikiness Penalty (Variance + Excess Kurtosis)."""

    def _calculate_hybrid_fallback(self, a, b) -> float:
        s = np.sqrt(self.variance(a, b))
        kurt = self.kurtosis(a, b)
        n = self.n_samples_
        C = s / (1 + abs(kurt))
        return 1e-5 if s == 0 else C * (n ** (-0.4))


# Model D is the standard BetaKernelKDE, no subclass needed.

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

N_REPLICATIONS = 1000
SAMPLE_SIZES = [50, 100, 250, 500, 1000, 2000]
OUTPUT_CSV_FILE = "data/ablation_study/ablation_results.csv"
MASTER_SEED = 2026
MAX_WORKERS = 5  # 32

# We only test the distributions where the fallback heuristic is actually triggered.
DISTRIBUTIONS = {
    "B(0.5, 0.5)": lambda n, rng: rng.beta(0.5, 0.5, size=n),
    "B(0.8, 2.5)": lambda n, rng: rng.beta(0.8, 2.5, size=n),
    "B(1.5, 1.5)": lambda n, rng: rng.beta(1.5, 1.5, size=n),
    "BIMODAL": lambda n, rng: np.concatenate(
        [
            rng.beta(10, 30, size=n // 2),
            rng.beta(30, 10, size=n - (n // 2)),
        ]
    ),
}

METHODS = {
    "MODEL_A": BetaKDE_ModelA,
    "MODEL_B": BetaKDE_ModelB,
    "MODEL_C": BetaKDE_ModelC,
    "MODEL_D": BetaKernelKDE,  # The proposed full rule
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

from sklearn.model_selection import KFold
from scipy.integrate import quad


def lscv_score(kde, n_folds=10):
    """Calculates the universal LSCV score for a fitted KDE."""
    if kde.bandwidth is None or not np.isfinite(kde.bandwidth) or len(kde.data_) == 0:
        return np.nan

    n = len(kde.data_)
    try:
        term1, _ = quad(lambda x: kde.pdf(x) ** 2, 0, 1, limit=50, epsabs=1e-4)
    except Exception:
        return np.nan

    term2_sum = 0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    try:
        for train_idx, test_idx in kf.split(kde.data_):
            X_train, X_test = kde.data_[train_idx], kde.data_[test_idx]
            kde_fold = kde.__class__(bandwidth=kde.bandwidth)
            kde_fold.fit(X_train)
            pdf_values = kde_fold.pdf(X_test)
            term2_sum += np.sum(pdf_values[np.isfinite(pdf_values)])
        term2 = 2 * (term2_sum / n)
    except Exception:
        return np.nan

    return term1 - term2


# =============================================================================
# WORKER LOGIC
# =============================================================================


def run_ablation_trial(job_args):
    """Worker function to run a single trial across all 4 ablation models."""
    dist_name, n, trial, child_seed_seq = job_args
    rng = np.random.default_rng(child_seed_seq)
    data = DISTRIBUTIONS[dist_name](n, rng)

    results = {"distribution": dist_name, "n": n, "trial": trial}

    for method_name, KDEClass in METHODS.items():
        try:
            # Fit using the MISE_rule which triggers the fallback for these dists
            start_time = time.perf_counter()
            kde = KDEClass(bandwidth="MISE_rule", verbose=0)
            kde.fit(data)
            fit_time = time.perf_counter() - start_time

            # Evaluate
            score_lscv = lscv_score(kde)

            results[f"{method_name}_h"] = kde.bandwidth
            results[f"{method_name}_lscv"] = score_lscv
            results[f"{method_name}_time"] = fit_time
            results[f"{method_name}_fallback_triggered"] = getattr(
                kde, "is_fallback", False
            )

        except Exception as e:
            results[f"{method_name}_h"] = np.nan
            results[f"{method_name}_lscv"] = np.nan
            results[f"{method_name}_time"] = np.nan
            results[f"{method_name}_fallback_triggered"] = False

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    print("--- Starting Ablation Study (Parallel) ---")
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)

    # Pre-spawn RNG seeds
    master_seed_seq = SeedSequence(MASTER_SEED)
    child_seeds = master_seed_seq.spawn(N_REPLICATIONS)

    jobs_to_run = []
    for dist_name in DISTRIBUTIONS.keys():
        for n in SAMPLE_SIZES:
            for trial in range(1, N_REPLICATIONS + 1):
                jobs_to_run.append((dist_name, n, trial, child_seeds[trial - 1]))

    print(f"Total jobs to run: {len(jobs_to_run)}")

    # Execute in parallel
    with open(OUTPUT_CSV_FILE, "w", newline="") as f_out:
        write_header = True
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            # Note: requires pip install tqdm
            from tqdm.auto import tqdm

            for trial_results in tqdm(
                executor.map(run_ablation_trial, jobs_to_run), total=len(jobs_to_run)
            ):
                if trial_results:
                    df_row = pd.DataFrame([trial_results])
                    df_row.to_csv(f_out, header=write_header, index=False)
                    write_header = False

    print(f"Ablation study complete. Results saved to {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
