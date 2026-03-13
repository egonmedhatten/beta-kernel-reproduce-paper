#!/usr/bin/env python3
"""
Main script for running the "Bulletproof" Experiment 1. (PARALLEL VERSION)

This script executes the 7-distribution, 6-sample-size, 10-competitor
simulation, running 1000 trials for each configuration.

It has been modified to use `concurrent.futures.ProcessPoolExecutor`
to run trials in parallel, dramatically speeding up execution on
multi-core systems.

Random number generation is handled by spawning a unique child seed
from a master seed for each trial, ensuring reproducible and
independent random streams for all parallel jobs.

It records all specified metrics:
- LSCV Score (Primary, universal metric)
- ISE Score (Validation metric, for "nice" dists only)
- Computation Time (for the .fit() call)
- Bandwidth (h)
- is_fallback (for BETA_ROT)

Results are saved incrementally to 'simulation_results_full.csv'.
If the script is stopped, it can be resumed and will not re-run
completed trials.
"""

import concurrent.futures  # <-- NEW: For parallel processing
import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.special as sp
from numpy.random import SeedSequence  # <-- NEW: For robust RNG seeding
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import beta, truncnorm
from tqdm.auto import tqdm

# Import your custom KDE classes
# We assume KDE.py and KDE_Gauss.py are in the same directory
try:
    from KDE import BetaKernelKDE
    from KDE_Gauss import GaussianKDE
except ImportError:
    print(
        "Error: Could not import KDE and KDE_Gauss.py. Make sure they are in the same directory."
    )
    exit()


# --- Global Settings ---
N_REPLICATIONS = 1000  # Full run
SAMPLE_SIZES = [50, 100, 250, 500, 1000, 2000]  # Full run
OUTPUT_CSV_FILE = "data/experiment1/simulation_results_full.csv"
MASTER_SEED = 2025  # <-- NEW: Master seed for reproducibility
BETA_BOUNDS = (1e-4, 0.5)
GAUSS_REFLECT_BOUNDS = (1e-4, 0.5)
GAUSS_LOGIT_BOUNDS = (0.01, 2.0)

# --- NEW: Parallel Execution Settings ---
# Set to None to use all available CPU cores (default for ProcessPoolExecutor)
# Set to an integer to limit the number of worker processes, e.g., 4 or 8
MAX_WORKERS = 32


# --- Distribution Definitions ---
# *** MODIFIED ***
# All "generator" functions now accept an `rng` argument
# to ensure independent random streams per trial.
#
def trunc_norm_05_pdf(x):
    a, b = (0 - 0.5) / 0.15, (1 - 0.5) / 0.15
    return np.maximum(truncnorm.pdf(x, a=a, b=b, loc=0.5, scale=0.15), EPS)


def trunc_norm_05_pdf_pp(x):
    m, s = 0.5, 0.15
    v = s**2
    return (((x - m) ** 2) / (v**2) - 1 / v) * trunc_norm_05_pdf(x)


def trunc_norm_07_pdf(x):
    a, b = (0 - 0.7) / 0.15, (1 - 0.7) / 0.15
    return np.maximum(truncnorm.pdf(x, a=a, b=b, loc=0.7, scale=0.15), EPS)


def trunc_norm_07_pdf_pp(x):
    m, s = 0.7, 0.15
    v = s**2
    return (((x - m) ** 2) / (v**2) - 1 / v) * trunc_norm_07_pdf(x)


def beta_pdf_pp_factory(a, b):
    pdf_func = lambda x: np.maximum(beta.pdf(x, a, b), EPS)

    def pdf_pp(x_in):
        x = np.clip(np.atleast_1d(x_in), EPS, 1 - EPS)
        f = pdf_func(x)
        log_deriv = (a - 1) / x - (b - 1) / (1 - x)
        log_deriv_prime = -(a - 1) / (x**2) - (b - 1) / ((1 - x) ** 2)
        f_pp = f * (log_deriv**2 + log_deriv_prime)
        f_pp[f <= EPS] = 0.0
        return f_pp.item() if np.isscalar(x_in) else f_pp

    return pdf_pp


DISTRIBUTIONS = {
    "B(5, 5)": {
        "generator": lambda n, rng: rng.beta(5, 5, size=n),  # MODIFIED
        "true_pdf": lambda x: beta.pdf(x, 5, 5),
        "type": "nice",
        "oracle_params": (5, 5),
    },
    "B(2, 12)": {
        "generator": lambda n, rng: rng.beta(2, 12, size=n),  # MODIFIED
        "true_pdf": lambda x: beta.pdf(x, 2, 12),
        "type": "nice",
        "oracle_params": (2, 12),
    },
    "B(0.5, 0.5)": {
        "generator": lambda n, rng: rng.beta(0.5, 0.5, size=n),  # MODIFIED
        "true_pdf": lambda x: beta.pdf(x, 0.5, 0.5),
        "type": "hard",
        "oracle_params": (0.5, 0.5),
    },
    "B(0.8, 2.5)": {
        "generator": lambda n, rng: rng.beta(0.8, 2.5, size=n),  # MODIFIED
        "true_pdf": lambda x: beta.pdf(x, 0.8, 2.5),
        "type": "hard",
        "oracle_params": (0.8, 2.5),
    },
    "B(1.5, 1.5)": {
        "generator": lambda n, rng: rng.beta(1.5, 1.5, size=n),  # MODIFIED
        "true_pdf": lambda x: beta.pdf(x, 1.5, 1.5),
        "type": "hard",
        "oracle_params": (1.5, 1.5),
    },
    "NT(0.5, 0.15)": {
        "generator": lambda n, rng: truncnorm.rvs(  # MODIFIED
            a=(0 - 0.5) / 0.15,
            b=(1 - 0.5) / 0.15,
            loc=0.5,
            scale=0.15,
            size=n,
            random_state=rng,  # MODIFIED
        ),
        "true_pdf": trunc_norm_05_pdf,
        "true_pdf_pp": trunc_norm_05_pdf_pp,
        "type": "nice",
        "oracle_params": None,
    },
    "NT(0.7, 0.15)": {
        "generator": lambda n, rng: truncnorm.rvs(  # MODIFIED
            a=(0 - 0.7) / 0.15,
            b=(1 - 0.7) / 0.15,
            loc=0.7,
            scale=0.15,
            size=n,
            random_state=rng,  # MODIFIED
        ),
        "true_pdf": trunc_norm_07_pdf,
        "true_pdf_pp": trunc_norm_07_pdf_pp,
        "type": "nice",
        "oracle_params": None,
    },
    "BIMODAL": {
        "generator": lambda n, rng: np.concatenate(  # MODIFIED
            [
                rng.beta(10, 30, size=n // 2),  # MODIFIED
                rng.beta(30, 10, size=n - (n // 2)),  # MODIFIED
            ]
        ),
        "true_pdf": lambda x: 0.5 * beta.pdf(x, 10, 30) + 0.5 * beta.pdf(x, 30, 10),
        "true_pdf_pp": lambda x: 0.5 * beta_pdf_pp_factory(10, 30)(x)
        + 0.5 * beta_pdf_pp_factory(30, 10)(x),
        "type": "nice",
        "oracle_params": None,
    },
}

# --- Competitor Method Definitions ---
METHODS_TO_RUN = [
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


# %%
# =============================================================================
# HELPER FUNCTIONS (Copied from zz_test.py and our discussion)
# =============================================================================
# (All helper functions... lscv_score, ise, calculate_oracle_integrals...
# are unchanged, so they are omitted here for brevity.
# Just paste them in from the original script.)
# ...
def calculate_oracle_integrals(a=None, b=None, pdf_func=None, pdf_pp_func=None):
    if a is not None and b is not None:
        return calculate_oracle_integrals_beta(a, b)
    elif pdf_func is not None:
        return calculate_oracle_integrals_numerical(pdf_func, pdf_pp_func)
    else:
        raise ValueError(
            "Either (a,b) or pdf_func must be provided to calculate oracle integrals."
        )


def calculate_oracle_integrals_beta(a: float, b: float):
    if a <= 1.5 or b <= 1.5:
        return None, None
    try:
        log_I1 = (
            sp.gammaln(a - 0.5)
            + sp.gammaln(b - 0.5)
            - sp.gammaln(a + b - 1)
            - (sp.gammaln(a) + sp.gammaln(b) - sp.gammaln(a + b))
        )
        I1 = np.exp(log_I1)
        log_I2_num = (
            np.log(a - 1)
            + np.log(b - 1)
            + np.log(a * (3 * b - 4) - 4 * b + 6)
            + sp.gammaln(2 * a - 3)
            + sp.gammaln(2 * b - 3)
            + 2 * sp.gammaln(a + b)
        )
        log_I2_den = (
            np.log(2 * a + 2 * b - 5)
            + np.log(2 * a + 2 * b - 3)
            + 2 * sp.gammaln(a)
            + 2 * sp.gammaln(b)
            + sp.gammaln(2 * a + 2 * b - 6)
        )
        I2 = np.exp(log_I2_num - log_I2_den)
        if not np.isfinite(I1) or not np.isfinite(I2) or I2 == 0:
            return None, None
        return I1, I2
    except Exception:
        return None, None


EPS = 1e-10
H_STEP = 1e-6


def calculate_oracle_integrals_numerical(pdf_func, pdf_pp_func=None):
    try:
        if pdf_pp_func:
            f_pp = pdf_pp_func
        else:
            warnings.warn(
                "[Info] Using numerical derivative for f''(x).", RuntimeWarning
            )

            def f_pp_numerical(x):
                h = H_STEP
                f_plus_h = pdf_func(x + h)
                f_minus_h = pdf_func(x - h)
                f_x = pdf_func(x)
                if x < h or x > 1 - h:
                    return (pdf_func(x + 2 * h) - 2 * pdf_func(x + h) + f_x) / (h**2)
                return (f_plus_h - 2 * f_x + f_minus_h) / (h**2)

            f_pp = f_pp_numerical

        integrand_I1 = lambda x: pdf_func(x) / np.sqrt(x * (1 - x) + EPS)
        I1, i1_err = quad(integrand_I1, EPS, 1 - EPS, limit=100)
        if not np.isfinite(I1):
            return None, None

        integrand_I2 = lambda x: (x * (1 - x) * f_pp(x)) ** 2
        I2, i2_err = quad(integrand_I2, EPS, 1 - EPS, limit=50)
        if not np.isfinite(I2) or I2 == 0:
            return None, None

        return I1, I2

    except Exception as e:
        print(f"Error during numerical integration for Oracle: {e}")
        return None, None


def ise(kde_pdf, true_pdf_func):
    """Calculates the Integrated Squared Error (ISE)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            integrand = lambda x: (kde_pdf(x) - true_pdf_func(x)) ** 2
            result, _ = quad(integrand, 0, 1, limit=50, epsabs=1e-4)
            return result
        except Exception:
            return np.nan


def calculate_ISE_for_bandwidth(h, true_pdf, kde):
    kde.bandwidth = h
    ISE, err = quad(lambda x: (kde.pdf(x) - true_pdf(x)) ** 2, 0, 1)
    return ISE


from sklearn.model_selection import KFold


def lscv_score(kde, n_folds=10):
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

    try:
        integrand_sq = lambda x: kde.pdf(x) ** 2
        term1, _ = quad(integrand_sq, 0, 1, limit=50, epsabs=1e-4)
        if not np.isfinite(term1):
            term1 = np.nan
    except Exception:
        term1 = np.nan

    term2_sum = 0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    try:
        for train_idx, test_idx in kf.split(data):
            X_train, X_test = data[train_idx], data[test_idx]
            if len(X_train) == 0 or len(X_test) == 0:
                continue

            kde_fold = kde_class(bandwidth=bandwidth, **kde_init_kwargs)
            kde_fold.fit(X_train)

            pdf_values = kde_fold.pdf(X_test)
            term2_sum += np.sum(pdf_values[np.isfinite(pdf_values)])

        term2 = 2 * (term2_sum / n)
        if not np.isfinite(term2):
            term2 = np.nan
    except Exception:
        term2 = np.nan

    if np.isnan(term1) or np.isnan(term2):
        return np.nan

    return term1 - term2


# ...
# End of unchanged helper functions
# %%
# =============================================================================
# EXPERIMENT 1 CORE LOGIC
# =============================================================================


def _evaluate_method(kde, dist_type, true_pdf_func):
    """
    Helper function to evaluate a fitted KDE instance.
    Returns the key metrics.
    """
    h = (
        kde.bandwidth
        if (kde.bandwidth is not None and np.isfinite(kde.bandwidth))
        else np.nan
    )
    # Use getattr for is_fallback, as it's specific to BetaKernelKDE
    is_fallback = getattr(kde, "is_fallback", False)

    # A. LSCV Score (Primary, Universal Metric)
    score_lscv = lscv_score(kde)

    # B. ISE Score (Validation Metric for "nice" dists)
    score_ise = np.nan
    if dist_type == "nice":
        # Only compute ISE if h is valid
        if not np.isnan(h):
            score_ise = ise(kde.pdf, true_pdf_func)

    # C. Absolute error of integral (deviation from 1.0)
    try:
        func = lambda x: kde.pdf(x)
        integral, _ = quad(func, 0, 1, limit=100)
        error = abs(integral - 1.0)
    except:
        error = np.nan  # Skip failed integrations

    return h, score_lscv, score_ise, is_fallback, error


def run_single_trial(dist_name, dist_config, n, I1, I2, rng):
    """
    Runs one full trial (one row of the CSV) for all 10 methods.

    *** MODIFIED ***
    This function now accepts an `rng` object to generate data.
    """
    # 1. Generate data and get helpers
    data = dist_config["generator"](n, rng)  # <-- MODIFIED
    true_pdf_func = dist_config["true_pdf"]
    dist_type = dist_config["type"]
    oracle_params_tuple = dist_config.get("oracle_params")

    trial_results_dict = {
        "distribution": dist_name,
        "n": n,
    }

    fitted_kdes = {}

    # --- 2. FIT ALL METHODS (using the correct .fit() calls) ---
    # ...
    # (The fitting logic here is complex but *unchanged*.
    # It is omitted for brevity. Just paste the entire
    # "FIT ALL METHODS" block from your original script.)
    # ...
    # --- BETA Methods ---
    try:
        kde = BetaKernelKDE(bandwidth="MISE_rule", verbose=0)
        start_time = time.perf_counter()
        kde.fit(data)
        fit_time = time.perf_counter() - start_time
        fitted_kdes["BETA_ROT"] = (kde, fit_time)
    except Exception:
        fitted_kdes["BETA_ROT"] = (None, np.nan)

    try:
        kde = BetaKernelKDE(bandwidth="LSCV", verbose=0)
        start_time = time.perf_counter()
        kde.fit(data, bandwidth_bounds=BETA_BOUNDS)
        fit_time = time.perf_counter() - start_time
        fitted_kdes["BETA_LSCV"] = (kde, fit_time)
    except Exception:
        fitted_kdes["BETA_LSCV"] = (None, np.nan)

    try:
        if dist_type != "nice":
            raise ValueError("ISE method skipped for 'hard' distribution")
        kde = BetaKernelKDE(bandwidth=0.1, verbose=0)
        start_time = time.perf_counter()
        kde.fit(data)
        to_minimise = lambda h: calculate_ISE_for_bandwidth(
            h, dist_config["true_pdf"], kde
        )
        res = minimize_scalar(to_minimise, bounds=BETA_BOUNDS, method="bounded")
        h = res.x
        kde.bandwidth = h
        fit_time = time.perf_counter() - start_time
        fitted_kdes["BETA_ISE"] = (kde, fit_time)
    except Exception:
        fitted_kdes["BETA_ISE"] = (None, np.nan)

    try:
        term = I1 / (2 * data.size * I2 * np.sqrt(np.pi))
        h_oracle = term**0.4
        kde = BetaKernelKDE(bandwidth=h_oracle, verbose=0)
        start_time = time.perf_counter()
        kde.fit(data)
        fit_time = time.perf_counter() - start_time
        fitted_kdes["BETA_ORACLE"] = (kde, fit_time)
    except Exception:
        fitted_kdes["BETA_ORACLE"] = (None, np.nan)

    # --- LOGIT Methods ---
    try:
        kde = GaussianKDE(bandwidth="silverman", method="logit", verbose=0)
        start_time = time.perf_counter()
        kde.fit(data)
        fit_time = time.perf_counter() - start_time
        fitted_kdes["LOGIT_SILV"] = (kde, fit_time)
    except Exception:
        fitted_kdes["LOGIT_SILV"] = (None, np.nan)

    try:
        kde = GaussianKDE(bandwidth="LSCV", method="logit", verbose=0)
        start_time = time.perf_counter()
        kde.fit(data)
        fit_time = time.perf_counter() - start_time
        fitted_kdes["LOGIT_LSCV"] = (kde, fit_time)
    except Exception:
        fitted_kdes["LOGIT_LSCV"] = (None, np.nan)

    try:
        if dist_type != "nice":
            raise ValueError("ISE method skipped for 'hard' distribution")
        kde = GaussianKDE(bandwidth=0.1, method="logit", verbose=0)
        start_time = time.perf_counter()
        kde.fit(data)
        to_minimise = lambda h: calculate_ISE_for_bandwidth(
            h, dist_config["true_pdf"], kde
        )
        res = minimize_scalar(to_minimise, bounds=GAUSS_LOGIT_BOUNDS, method="bounded")
        h = res.x
        kde.bandwidth = h
        fit_time = time.perf_counter() - start_time
        fitted_kdes["LOGIT_ISE"] = (kde, fit_time)
    except Exception:
        fitted_kdes["LOGIT_ISE"] = (None, np.nan)

    # --- REFLECT Methods ---
    try:
        kde = GaussianKDE(bandwidth="silverman", method="reflect", verbose=0)
        start_time = time.perf_counter()
        kde.fit(data)
        fit_time = time.perf_counter() - start_time
        fitted_kdes["REFLECT_SILV"] = (kde, fit_time)
    except Exception:
        fitted_kdes["REFLECT_SILV"] = (None, np.nan)

    try:
        kde = GaussianKDE(bandwidth="LSCV", method="reflect", verbose=0)
        start_time = time.perf_counter()
        kde.fit(data, bandwidth_bounds=GAUSS_REFLECT_BOUNDS)
        fit_time = time.perf_counter() - start_time
        fitted_kdes["REFLECT_LSCV"] = (kde, fit_time)
    except Exception:
        fitted_kdes["REFLECT_LSCV"] = (None, np.nan)

    try:
        if dist_type != "nice":
            raise ValueError("ISE method skipped for 'hard' distribution")
        kde = GaussianKDE(bandwidth=0.1, method="reflect", verbose=0)
        start_time = time.perf_counter()
        kde.fit(data)
        to_minimise = lambda h: calculate_ISE_for_bandwidth(
            h, dist_config["true_pdf"], kde
        )
        res = minimize_scalar(
            to_minimise, bounds=GAUSS_REFLECT_BOUNDS, method="bounded"
        )
        h = res.x
        kde.bandwidth = h
        fit_time = time.perf_counter() - start_time
        fitted_kdes["REFLECT_ISE"] = (kde, fit_time)
    except Exception:
        fitted_kdes["REFLECT_ISE"] = (None, np.nan)

    # ...
    # End of unchanged fitting block
    # ...

    # --- 3. Evaluate all fitted methods ---
    for method_handle in METHODS_TO_RUN:
        h, score_lscv, score_ise, is_fallback = np.nan, np.nan, np.nan, False
        fit_time = np.nan

        try:
            if method_handle in fitted_kdes:
                fitted_kde, fit_time = fitted_kdes[method_handle]
                if (
                    fitted_kde is not None
                    and fitted_kde.bandwidth is not None
                    and np.isfinite(fitted_kde.bandwidth)
                ):
                    h, score_lscv, score_ise, is_fallback, error = _evaluate_method(
                        fitted_kde, dist_type, true_pdf_func
                    )
        except Exception:
            pass

        # 4. Store results for this method
        trial_results_dict[f"{method_handle}_h"] = h
        trial_results_dict[f"{method_handle}_lscv_score"] = score_lscv
        trial_results_dict[f"{method_handle}_ise_score"] = score_ise
        trial_results_dict[f"{method_handle}_comp_time"] = fit_time
        if method_handle == "BETA_ROT":
            trial_results_dict[f"{method_handle}_is_fallback"] = is_fallback
        trial_results_dict[f"{method_handle}_integral_error"] = error

    return trial_results_dict


# %%
# =============================================================================
# NEW: PARALLEL WORKER FUNCTION
# =============================================================================


def run_parallel_trial(job_args):
    """
    Wrapper function for parallel execution.
    - Unpacks job arguments.
    - Creates a trial-specific RNG.
    - Calls the original `run_single_trial` function.
    - Handles exceptions within the worker.
    """
    dist_name, n, trial, I1, I2, child_seed_seq = job_args

    try:
        # 1. Get the distribution config (read-only, safe for parallel)
        dist_config = DISTRIBUTIONS[dist_name]

        # 2. Create the unique, reproducible RNG for this trial
        rng = np.random.default_rng(child_seed_seq)

        # 3. Run the simulation
        trial_results = run_single_trial(dist_name, dist_config, n, I1, I2, rng)

        # 4. Add trial number
        trial_results["trial"] = trial

        return trial_results

    except Exception as e:
        # Log errors from the worker
        print(f"--- ERROR in worker for job {(dist_name, n, trial)}: {e} ---")
        return None


# %%
# =============================================================================
# MAIN FUNCTION (MODIFIED FOR PARALLELISM)
# =============================================================================


def main():
    """
    Main function to run the full simulation experiment.

    *** MODIFIED ***
    - Pre-calculates integrals and RNG seeds.
    - Builds a list of all jobs to run.
    - Uses ProcessPoolExecutor to run jobs in parallel.
    - Writes results as they are completed.
    """
    print(f"--- Starting Experiment 1 (Parallel Version) ---")
    print(f"Distributions: {list(DISTRIBUTIONS.keys())}")
    print(f"Sample Sizes: {SAMPLE_SIZES}")
    print(f"Replications: {N_REPLICATIONS}")
    print(f"Output File: {OUTPUT_CSV_FILE}\n")

    # --- Define Column Order *Once* ---
    COLUMN_ORDER = ["distribution", "n", "trial"]
    for method in METHODS_TO_RUN:
        COLUMN_ORDER.append(f"{method}_h")
        COLUMN_ORDER.append(f"{method}_lscv_score")
        COLUMN_ORDER.append(f"{method}_ise_score")
        COLUMN_ORDER.append(f"{method}_comp_time")
        if method == "BETA_ROT":
            COLUMN_ORDER.append(f"{method}_is_fallback")
        COLUMN_ORDER.append(f"{method}_integral_error")

    # --- Setup for Resuming ---
    existing_results = set()
    output_dir = os.path.dirname(OUTPUT_CSV_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(OUTPUT_CSV_FILE):
        print("Found existing results file. Resuming experiment...")
        try:
            df_existing = pd.read_csv(OUTPUT_CSV_FILE)
            for _, row in df_existing.iterrows():
                existing_results.add((row["distribution"], row["n"], int(row["trial"])))
            print(f"Loaded {len(existing_results)} completed trials.")
            f_out = open(OUTPUT_CSV_FILE, "a", newline="")
            write_header = False
        except Exception as e:
            print(f"Warning: Could not read existing file. Starting new. Error: {e}")
            f_out = open(OUTPUT_CSV_FILE, "w", newline="")
            write_header = True
    else:
        print("No existing file found. Starting new experiment...")
        f_out = open(OUTPUT_CSV_FILE, "w", newline="")
        write_header = True

    # --- NEW: Pre-computation Step ---
    print("\nPre-computing Oracle integrals...")
    oracle_integrals_map = {}
    for dist_name, dist_config in DISTRIBUTIONS.items():
        a_b_tuple = dist_config.get("oracle_params")
        if a_b_tuple is not None:
            I1, I2 = calculate_oracle_integrals(a=a_b_tuple[0], b=a_b_tuple[1])
        else:
            I1, I2 = calculate_oracle_integrals(
                pdf_func=dist_config["true_pdf"],
                pdf_pp_func=dist_config.get("true_pdf_pp", None),
            )
        oracle_integrals_map[dist_name] = (I1, I2)

    print("Spawning child RNG seeds...")
    master_seed_seq = SeedSequence(MASTER_SEED)
    child_seeds = master_seed_seq.spawn(N_REPLICATIONS)

    # --- NEW: Build Job List ---
    total_runs = len(DISTRIBUTIONS) * len(SAMPLE_SIZES) * N_REPLICATIONS
    jobs_to_run = []

    print("Building job list...")
    for dist_name, dist_config in DISTRIBUTIONS.items():
        for n in SAMPLE_SIZES:
            for trial in range(1, N_REPLICATIONS + 1):
                # Check if we should skip
                if (dist_name, n, trial) in existing_results:
                    continue

                # Get precomputed values
                I1, I2 = oracle_integrals_map[dist_name]
                child_seed_seq = child_seeds[trial - 1]  # Use 0-indexed seed

                # Add job arguments to the list
                job_args = (dist_name, n, trial, I1, I2, child_seed_seq)
                jobs_to_run.append(job_args)

    n_jobs = len(jobs_to_run)
    n_skipped = total_runs - n_jobs
    print(f"Total jobs: {total_runs}, Skipped: {n_skipped}, To run: {n_jobs}")

    if n_jobs == 0:
        print("All trials already completed.")
        f_out.close()
        return

    # --- NEW: Parallel Execution ---
    try:
        # You can set max_workers to a specific number, e.g., max_workers=16
        # By default, it uses all available cores.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            print(f"Starting parallel execution on {executor._max_workers} workers...")

            # Use executor.map to run jobs and get results as they complete
            # Wrap in tqdm for a progress bar
            results_iterator = executor.map(run_parallel_trial, jobs_to_run)

            for trial_results in tqdm(
                results_iterator, total=n_jobs, desc="Running Simulation"
            ):
                # Handle a failed worker
                if trial_results is None:
                    continue

                # --- Write to CSV ---
                df_row = pd.DataFrame([trial_results])
                df_row = df_row.reindex(columns=COLUMN_ORDER)

                df_row.to_csv(f_out, header=write_header, index=False)
                write_header = False  # Only write header once

    except KeyboardInterrupt:
        print("\n--- Experiment Interrupted. Saving progress... ---")
    finally:
        f_out.close()
        print(f"\n--- Experiment Complete (or paused) ---")
        print(f"All results saved to '{OUTPUT_CSV_FILE}'")


if __name__ == "__main__":
    # This check is CRITICAL for multiprocessing to work
    # on Windows and macOS.

    # Suppress warnings from scipy.integrate and optimizations
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    main()
