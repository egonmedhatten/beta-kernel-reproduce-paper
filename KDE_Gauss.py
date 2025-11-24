import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.special as sp
from scipy.stats import norm


class GaussianKDE:
    """
    A class for performing Kernel Density Estimation (KDE) for data
    on the [0, 1] interval using a Gaussian kernel with boundary correction.

    Supports two correction methods:
    1. 'reflect': Reflection method for boundaries at 0 and 1.
    2. 'logit': Logit-transformation of the data.

    Supports bandwidth selection via Silverman's (Scott's) rule or LSCV.
    """

    VALID_METHODS: List[str] = ["reflect", "logit"]
    VALID_SELECTION_METHODS: List[str] = ["silverman", "LSCV"]

    # --- Attributes set during/after fit ---
    data_: Optional[np.ndarray] = None  # Original data [0, 1]
    transformed_data_: Optional[np.ndarray] = (
        None  # Transformed data (reflected or logit)
    )
    n_samples_: int = 0  # Number of *original* samples
    bandwidth: Optional[float] = None

    # --- Attributes set during init ---
    method: str
    verbose: int
    _bandwidth_param: Optional[Union[float, str]]
    _epsilon: float = 1e-10  # For clipping in logit transform

    def __init__(
        self,
        method: str,
        bandwidth: Optional[Union[float, str]] = None,
        verbose: int = 1,
    ):
        """
        Initializes the GaussianKDE estimator.

        Parameters
        ----------
        method : str
            The boundary correction method to use. Must be one of ['reflect', 'logit'].
        bandwidth : float, str, or None, optional
            - If float: The bandwidth parameter h. Must be positive.
            - If str: The name of the selection method to use during fit().
              Must be one of ['silverman', 'LSCV'].
            - If None: A selection method must be provided to fit().
        verbose : int, default=1
            Controls the verbosity for messages.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method: '{method}'. Choose from {self.VALID_METHODS}."
            )
        self.method = method
        self.verbose = verbose
        self._bandwidth_param = bandwidth

        if isinstance(bandwidth, (int, float)):
            if bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            self.bandwidth = float(bandwidth)
        elif isinstance(bandwidth, str):
            if bandwidth not in self.VALID_SELECTION_METHODS:
                raise ValueError(
                    f"Unknown bandwidth selection method: '{bandwidth}'. "
                    f"Choose from {self.VALID_SELECTION_METHODS}."
                )
        elif bandwidth is not None:
            raise TypeError("Bandwidth must be a float, string, or None.")

    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """Validates the input data array is 1D and on [0, 1]."""
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data must be a numpy array.")
        if X.ndim > 2 or (X.ndim == 2 and X.shape[1] != 1):
            raise ValueError(
                "Data must be a 1D array or a 2D array with a single column."
            )
        if X.ndim == 2:
            X = X.flatten()
        if not np.all((X >= 0) & (X <= 1)):
            raise ValueError("All data points must be within the [0, 1] interval.")
        return X

    def fit(
        self,
        X: np.ndarray,
        bandwidth_selection_method: Optional[str] = None,
        bandwidth_bounds: Optional[Tuple[float, float]] = None,  # <-- Default is None
        **selection_kwargs: Any,
    ) -> "GaussianKDE":
        """
        Fits the KDE model to the data, including data transformation
        and bandwidth selection (if required).

        Parameters
        ----------
        X : np.ndarray
            The 1D array of data, with values in [0, 1].
        bandwidth_selection_method : str, optional
            Overrides the method specified in __init__.
        bandwidth_bounds : tuple, optional
            The bounds to use for LSCV optimization. If None,
            a default is chosen based on the 'method':
            - 'reflect': (1e-4, 0.5)
            - 'logit': (0.01, 2.0)
        **selection_kwargs :
            Additional arguments passed to LSCV, such as 'grid_points'
            or 'heuristic_factor'.
        """
        X_valid = self._validate_data(X)
        self.data_ = X_valid
        self.n_samples_ = len(X_valid)

        # 1. Transform data based on the chosen method
        if self.method == "reflect":
            # X, -X, 2-X
            self.transformed_data_ = np.concatenate([X_valid, -X_valid, 2 - X_valid])
            # Set default bounds if user provided None
            if bandwidth_bounds is None:
                bandwidth_bounds = (1e-4, 0.5)

        elif self.method == "logit":
            # Clip data slightly to avoid -inf/inf at 0/1
            X_clipped = np.clip(X_valid, self._epsilon, 1 - self._epsilon)
            self.transformed_data_ = sp.logit(X_clipped)
            # Set default bounds if user provided None
            if bandwidth_bounds is None:
                bandwidth_bounds = (0.01, 2.0)

        # 2. Select bandwidth if not already provided
        effective_selection_method = bandwidth_selection_method
        if effective_selection_method is None and isinstance(
            self._bandwidth_param, str
        ):
            effective_selection_method = self._bandwidth_param

        if self.bandwidth is not None:
            if effective_selection_method is not None and self.verbose > 0:
                print(
                    f"Warning: A numeric bandwidth (h={self.bandwidth}) was provided. "
                    f"Ignoring selection method '{effective_selection_method}'."
                )
            return self

        if self.bandwidth is None:
            if effective_selection_method == "silverman":
                if self.verbose > 0:
                    print(
                        f"Applying Silverman's (Scott's) rule of thumb on '{self.method}' transformed data..."
                    )
                self.bandwidth = self.select_bandwidth_silverman()
                if self.verbose > 0:
                    print(f"Bandwidth selected: h = {self.bandwidth:.4f}")

            elif effective_selection_method == "LSCV":
                if self.verbose > 0:
                    print(
                        f"Starting LSCV bandwidth selection on '{self.method}' transformed data "
                        f"within hard bounds {bandwidth_bounds}..."
                    )
                # bandwidth_bounds is now guaranteed to be a tuple
                self.bandwidth = self.select_bandwidth_lscv(
                    bounds=bandwidth_bounds, **selection_kwargs
                )
                if self.verbose > 0:
                    print(
                        f"Optimal bandwidth selected by LSCV: h = {self.bandwidth:.4f}"
                    )

            elif effective_selection_method is not None:
                raise ValueError(
                    f"Unknown bandwidth selection method: '{effective_selection_method}'. "
                    f"Choose from {self.VALID_SELECTION_METHODS}."
                )

        if self.bandwidth is None:
            raise RuntimeError(
                "No bandwidth was provided and no selection method was specified. "
                "Please provide a bandwidth or specify a valid selection method."
            )

        return self

    def pdf(self, X_new: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Estimates the Probability Density Function (PDF) for new data points.

        Parameters
        ----------
        X_new : float or np.ndarray
            A single point or a 1D array of points in [0, 1] at which to evaluate the density.

        Returns
        -------
        float or np.ndarray
            The estimated density value(s).
        """
        if self.transformed_data_ is None or self.bandwidth is None:
            raise RuntimeError(
                "The KDE model has not been fitted yet. Call .fit() first."
            )

        is_scalar_input = False
        if isinstance(X_new, (int, float)):
            is_scalar_input = True
            X_new_array = np.array([X_new])
        else:
            X_new_array = X_new

        X_new_valid = self._validate_data(X_new_array)

        if self.method == "reflect":
            # Standard KDE on transformed data, but normalized by *original* N
            k_mat = self._gaussian_kernel_matrix(
                X_new_valid, self.transformed_data_, self.bandwidth
            )
            # Sum contributions from all 3N reflected points, divide by N
            pdf_result = np.sum(k_mat, axis=1) / self.n_samples_

        elif self.method == "logit":
            # Clip evaluation points
            X_new_clipped = np.clip(X_new_valid, self._epsilon, 1 - self._epsilon)

            # Transform evaluation points to logit space
            Y_new = sp.logit(X_new_clipped)

            # Calculate KDE in logit space
            # g(y) = 1/N * sum(K_h(y - Y_i))
            k_mat = self._gaussian_kernel_matrix(
                Y_new, self.transformed_data_, self.bandwidth
            )
            g_y = np.sum(k_mat, axis=1) / self.n_samples_

            # Apply change of variables formula: f(x) = g(logit(x)) * |d/dx logit(x)|
            # |d/dx logit(x)| = 1 / (x * (1 - x))
            jacobian = 1.0 / (X_new_clipped * (1 - X_new_clipped))

            f_x = g_y * jacobian

            # Ensure density is 0 outside the [0, 1] bounds (for points < epsilon or > 1-epsilon)
            pdf_result = np.where((X_new_valid <= 0) | (X_new_valid >= 1), 0.0, f_x)

        else:
            # This should not be reachable due to __init__ check
            raise RuntimeError(f"Invalid method '{self.method}' encountered in pdf.")

        if is_scalar_input:
            return pdf_result.item()
        else:
            return pdf_result

    @staticmethod
    def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
        """Standard Gaussian kernel. (Static method for utility)."""
        return norm.pdf(u)

    def _gaussian_kernel_matrix(
        self, x_eval: np.ndarray, data_pts: np.ndarray, h: float
    ) -> np.ndarray:
        """
        Calculates the (N_eval, N_data) matrix of Gaussian kernels.
        K_mat[i, j] = K_h(x_eval[i] - data_pts[j])
        """
        x_col = x_eval[:, np.newaxis]  # (N_eval, 1)
        data_row = data_pts[np.newaxis, :]  # (1, N_data)

        u = (x_col - data_row) / h

        # K_h(u) = K(u/h) / h
        k_mat = self._gaussian_kernel(u) / h
        return k_mat

    def select_bandwidth_silverman(self) -> float:
        """
        Selects bandwidth using Silverman's rule of thumb.
        h = 1.06 * std(data) * n_points^(-1/5)

        For 'reflect':
            - std(data) is from the *original* data (N points).
            - n_points is from the *transformed* data (3N points).
        For 'logit':
            - std(data) is from the *transformed* data (N points).
            - n_points is from the *transformed* data (N points).
        """
        if self.data_ is None or self.transformed_data_ is None:
            raise RuntimeError("Run .fit() before selecting bandwidth.")

        std_dev_data: np.ndarray
        n_points: int

        if self.method == "reflect":
            std_dev_data = self.data_
            n_points = len(self.transformed_data_)  # Use 3N
        elif self.method == "logit":
            std_dev_data = self.transformed_data_
            n_points = len(self.transformed_data_)  # Use N
        else:
            # Should be unreachable
            raise RuntimeError(f"Invalid method {self.method}")

        if len(std_dev_data) < 2:
            raise RuntimeError("Not enough data to estimate bandwidth (n < 2).")

        std_dev = np.std(std_dev_data, ddof=1)

        # Handle zero variance
        if std_dev == 0:
            if self.verbose > 0:
                warnings.warn(
                    "Data has zero variance. Using a small fallback bandwidth."
                )
            return 1e-5

        # Silverman's rule of thumb
        h = ((4 * std_dev**5) / (3 * n_points)) ** (
            1 / 5
        )  # 1.06 * std_dev * (n_points ** (-1.0 / 5.0))
        return h

    def _lscv_objective_gaussian(self, h: float) -> float:
        """
        Vectorized LSCV objective function for a Gaussian kernel.
        LSCV(h) = (1/N^2) * sum(K_h(sqrt(2)) * (Yi - Yj))
                  - (2/(N(N-1))) * sum(i!=j, K_h(Yi - Yj))
        """
        if h <= 0:
            return np.inf
        if self.transformed_data_ is None:
            raise RuntimeError("LSCV objective called before fitting data.")

        data = self.transformed_data_
        n = len(data)
        if n < 2:
            return np.inf

        data_col = data[:, np.newaxis]
        data_row = data[np.newaxis, :]
        diffs = data_col - data_row  # (N, N) matrix of (Yi - Yj)

        # Term 1: Integral term (convolution)
        # K_h(sqrt(2)) is Gaussian kernel with bandwidth h*sqrt(2)
        h_conv = h * np.sqrt(2.0)
        u_term1 = diffs / h_conv
        k_mat_term1 = self._gaussian_kernel(u_term1) / h_conv
        term1 = np.sum(k_mat_term1) / (n**2)

        # Term 2: Sum term
        u_term2 = diffs / h
        k_mat_term2 = self._gaussian_kernel(u_term2) / h
        # Subtract diagonal (i==j)
        sum_off_diag = np.sum(k_mat_term2) - np.sum(np.diag(k_mat_term2))

        term2 = (-2.0 / (n * (n - 1))) * sum_off_diag

        return term1 + term2

    def select_bandwidth_lscv(
        self,
        bounds: Tuple[float, float],
        grid_points: int = 30,
        heuristic_factor: float = 4.0,
        **kwargs: Any,
    ) -> float:
        """
        Selects the optimal bandwidth using a robust 3-stage (Gaussian) LSCV.

        1. Calculates Silverman's rule of thumb (h_rule)
           to create a "dynamic window" for the search.
        2. Intersects this window with the user's `bounds` (which act as
           hard "safety" bounds) to get the final search range.
        3. Performs a grid search over `grid_points` within this range.
        4. Runs a local refinement (`minimize_scalar`) around the grid
           search optimum for precision.

        Parameters
        ----------
        bounds : tuple
            The "hard" bounds (min_h, max_h) from `fit()`.
            The search will *never* go outside this range.
        grid_points : int, default=30
            The number of points to use in the initial coarse grid search.
        heuristic_factor : float, default=4.0
            Controls the size of the dynamic window around the heuristic.
            Window will be (h_rule / factor, h_rule * factor).
        **kwargs :
            Additional arguments passed to `scipy.optimize.minimize_scalar`.
        """
        if self.transformed_data_ is None:
            raise RuntimeError("The KDE model has not been fitted yet.")
        if grid_points < 2:
            raise ValueError("grid_points must be 2 or more.")
        if heuristic_factor <= 0:
            raise ValueError("heuristic_factor must be positive.")

        objective_func: Callable[[float], float] = (
            lambda h: self._lscv_objective_gaussian(h)
        )

        # --- Stage 1: Get Heuristic (Silverman's Rule) ---
        try:
            # We call the class's own silverman rule, which correctly
            # handles the 'reflect' or 'logit' transformed data.
            h_rule = self.select_bandwidth_silverman()
        except RuntimeError as e:
            if self.verbose > 0:
                warnings.warn(
                    f"LSCV: Silverman's rule failed ({e}), "
                    f"using midpoint of hard bounds {bounds} as heuristic."
                )
            h_rule = (bounds[0] + bounds[1]) / 2.0

        if self.verbose > 0:
            print(f"LSCV: Heuristic h_rule (Silverman)={h_rule:.5f}")

        # --- Stage 2: Define Search Bounds ---
        dynamic_min = h_rule / heuristic_factor
        dynamic_max = h_rule * heuristic_factor

        # Intersect with the user's hard bounds
        search_min = max(bounds[0], dynamic_min)
        search_max = min(bounds[1], dynamic_max)

        # Handle invalid intersection (e.g., dynamic range is
        # completely outside hard bounds)
        if search_min >= search_max:
            if self.verbose > 0:
                print(
                    f"LSCV: Dynamic window [{dynamic_min:.4f}, {dynamic_max:.4f}] "
                    f"has no overlap with hard bounds [{bounds[0]}, {bounds[1]}]. "
                    f"Using hard bounds."
                )
            final_search_bounds = bounds
        else:
            final_search_bounds = (search_min, search_max)

        if self.verbose > 0:
            print(f"LSCV: Final search bounds={final_search_bounds}")

        # --- Stage 3: Grid Search ---
        h_grid = np.linspace(
            final_search_bounds[0], final_search_bounds[1], grid_points
        )
        grid_step = h_grid[1] - h_grid[0] if grid_points > 1 else 0.0

        if self.verbose > 0:
            print(
                f"LSCV: Starting grid search with {grid_points} points in {final_search_bounds}..."
            )

        grid_scores = np.array([objective_func(h) for h in h_grid])

        if not np.any(np.isfinite(grid_scores)):
            raise RuntimeError(
                f"LSCV grid search failed: Objective function returned non-finite values {grid_scores}"
                " for all grid points. Try different bounds."
            )

        best_grid_idx = np.nanargmin(grid_scores)
        best_grid_h = h_grid[best_grid_idx]
        best_grid_score = grid_scores[best_grid_idx]

        if self.verbose > 0:
            print(
                f"LSCV: Grid search minimum found: h={best_grid_h:.5f} (score={best_grid_score:.5f})"
            )

        # --- Stage 4: Local Refinement ---
        refinement_lower = max(
            final_search_bounds[0], best_grid_h - grid_step
        )
        refinement_upper = min(
            final_search_bounds[1], best_grid_h + grid_step
        )
        refinement_bounds = (refinement_lower, refinement_upper)

        if self.verbose > 0:
            print(
                f"LSCV: Starting local refinement in bounds {refinement_bounds}..."
            )

        # Set default optimization options if not provided
        kwargs.setdefault("method", "bounded")

        result = scipy.optimize.minimize_scalar(
            objective_func,
            bounds=refinement_bounds,
            **kwargs,  # Passes 'method' and any other user args
        )

        # --- Final Selection ---
        # Trust the local optimizer only if it succeeded AND found a
        # better (or equal) score than the grid search.
        if result.success and result.fun <= best_grid_score:
            optimal_bandwidth = float(result.x)
            if self.verbose > 0:
                print(
                    f"LSCV: Local refinement successful. Final h={optimal_bandwidth:.5f}"
                )
        else:
            # Fallback to the best point from the coarse grid search
            optimal_bandwidth = best_grid_h
            if self.verbose > 0:
                if not result.success:
                    print(
                        f"LSCV: Local refinement failed ({result.message}). Using grid minimum."
                    )
                else:
                    print(
                        f"LSCV: Local refinement found worse score ({result.fun:.5f}). Using grid minimum."
                    )

        self.bandwidth = optimal_bandwidth
        return optimal_bandwidth
    
    def plot(
        self,
        eval_points: np.ndarray = None,
        show_histogram: bool = True,
        bins: int = 20,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the estimated Probability Density Function (PDF) and optionally
        an overlaid histogram of the original data.
        (This function is unchanged from your file)
        """
        if self.data_ is None or self.n_samples_ == 0:
            raise RuntimeError(
                "The KDE model has not been fitted yet. Call .fit() first."
            )
        if self.bandwidth is None:
            raise RuntimeError(
                "Bandwidth is not set. Please provide a bandwidth during initialization or implement a selection method in .fit()."
            )

        eval_points = (
            np.linspace(0, 1, 1000, endpoint=True)
            if eval_points is None
            else eval_points
        )

        # Compute PDF values
        if isinstance(eval_points, (int, float)):
            eval_points_arr = np.array([eval_points])
        else:
            eval_points_arr = eval_points

        pdf_values = self.pdf(eval_points_arr)

        created_ax = False
        fig: Optional[plt.Figure] = None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_ax = True
        else:
            fig = ax.figure

        plot_label = f"Gaussian KDE (h={self.bandwidth:.4f})" if label is None else label
        ax.plot(
            eval_points,
            pdf_values,
            label=plot_label,
            **kwargs,
        )

        if show_histogram:
            ax.hist(
                self.data_,
                bins=bins,
                density=True,
                alpha=0.5,
                label="Data Histogram",
                color="gray",
                edgecolor="black",
            )

        ax.set_title("Gaussian Kernel Density Estimation")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

        if created_ax and fig is not None:
            return fig, ax
        else:
            return ax


if __name__ == "__main__":
    # Example usage:
    np.random.seed(42)
    data_beta = np.random.beta(a=2, b=5, size=100)

    print("--- Testing 'reflect' method ---")
    kde_reflect = GaussianKDE(method="reflect", bandwidth="LSCV", verbose=1)
    kde_reflect.fit(data_beta, bandwidth_bounds=(0.01, 0.5))
    print(f"Reflect bandwidth (LSCV): {kde_reflect.bandwidth:.4f}")

    kde_reflect_silv = GaussianKDE(method="reflect", bandwidth="silverman", verbose=1)
    kde_reflect_silv.fit(data_beta)
    print(f"Reflect bandwidth (Silverman): {kde_reflect_silv.bandwidth:.4f}")

    eval_points = np.linspace(0.01, 0.99, 100)
    pdf_reflect = kde_reflect.pdf(eval_points)
    print(f"Reflect PDF shape: {pdf_reflect.shape}")

    print("\n--- Testing 'logit' method ---")
    kde_logit = GaussianKDE(method="logit", bandwidth="LSCV", verbose=1)
    # Logit-transformed data has a different scale, so bounds may need adjustment
    kde_logit.fit(data_beta, bandwidth_bounds=(0.1, 2.0))
    print(f"Logit bandwidth (LSCV): {kde_logit.bandwidth:.4f}")

    kde_logit_silv = GaussianKDE(method="logit", bandwidth="silverman", verbose=1)
    kde_logit_silv.fit(data_beta)
    print(f"Logit bandwidth (Silverman): {kde_logit_silv.bandwidth:.4f}")

    pdf_logit = kde_logit.pdf(eval_points)
    print(f"Logit PDF shape: {pdf_logit.shape}")

    # Example of plotting (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data_beta, bins=20, density=True, alpha=0.5, label="Data Histogram")

        ax.plot(
            eval_points,
            pdf_reflect,
            label=f"Reflect (LSCV h={kde_reflect.bandwidth:.3f})",
        )

        pdf_reflect_silv = kde_reflect_silv.pdf(eval_points)
        ax.plot(
            eval_points,
            pdf_reflect_silv,
            label=f"Reflect (Silv h={kde_reflect_silv.bandwidth:.3f})",
            linestyle="--",
        )

        ax.plot(
            eval_points, pdf_logit, label=f"Logit (LSCV h={kde_logit.bandwidth:.3f})"
        )

        pdf_logit_silv = kde_logit_silv.pdf(eval_points)
        ax.plot(
            eval_points,
            pdf_logit_silv,
            label=f"Logit (Silv h={kde_logit_silv.bandwidth:.3f})",
            linestyle="--",
        )

        ax.set_title("Gaussian KDE with Boundary Corrections")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True)
        print("\nPlotting example... close plot to exit.")
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping plot example.")
