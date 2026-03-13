"""Unified plot styling for all figures in the paper.

Provides a single ``setup_theme()`` call that every plot script should invoke
before creating figures, plus per-method style dictionaries that encode both
*colour* (for the online/digital version) and *linestyle + marker* (so every
series remains distinguishable when printed in black-and-white).

Colours are Paul Tol's colorblind-safe "bright" palette, specified as inline
hex codes so there is no external dependency on ``tol_colors``.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paul Tol "bright" colorblind-safe hex palette
# (https://personal.sron.nl/~pault/)
# ---------------------------------------------------------------------------
COLORS = {
    "blue":   "#4477AA",
    "cyan":   "#66CCEE",
    "green":  "#228833",
    "yellow": "#CCBB44",
    "red":    "#EE6677",
    "purple": "#AA3377",
    "grey":   "#BBBBBB",
    "indigo": "#332288",
    "teal":   "#44AA99",
    "olive":  "#999933",
}

# ---------------------------------------------------------------------------
# Theme setup — call once at the top of every plot script
# ---------------------------------------------------------------------------

def setup_theme():
    """Configure matplotlib + seaborn for publication-quality figures."""
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans",
        ],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.1,
        rc={
            "grid.linestyle": ":",
            "grid.color": "0.8",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        },
    )

# ---------------------------------------------------------------------------
# Experiment 1 & 2 — per-method styles (10 methods)
#
# Semantic grouping:
#   linestyle  "--"  (dashed)    = fast / rule-of-thumb selectors
#   linestyle  ":"   (dotted)    = LSCV selectors
#   linestyle  "-."  (dash-dot)  = oracle / benchmark (ISE-min, Oracle)
#   linestyle  "-"   (solid)     = proposed method (Beta Ref) — emphasised
#
# Each method also gets a unique marker so the series are distinguishable
# even without colour.
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    # --- Proposed ---
    "Beta (Ref)":          {"color": COLORS["red"],    "linestyle": "-",  "marker": "o", "dashes": (1, 0)},
    # --- Rule-of-thumb / fast ---
    "Logit (Silverman)":   {"color": COLORS["cyan"],   "linestyle": "--", "marker": "s", "dashes": (4, 1.5)},
    "Reflect (Silverman)": {"color": COLORS["green"],  "linestyle": "--", "marker": "^", "dashes": (4, 1.5)},
    # --- LSCV ---
    "Beta (LSCV)":         {"color": COLORS["blue"],   "linestyle": ":",  "marker": "D", "dashes": (1, 1)},
    "Logit (LSCV)":        {"color": COLORS["yellow"], "linestyle": ":",  "marker": "v", "dashes": (1, 1)},
    "Reflect (LSCV)":      {"color": COLORS["teal"],   "linestyle": ":",  "marker": "P", "dashes": (1, 1)},
    # --- Oracle / Benchmark ---
    "Beta (Oracle)":       {"color": COLORS["purple"], "linestyle": "-.", "marker": "X", "dashes": (5, 1, 1, 1)},
    "Beta (ISE)":          {"color": COLORS["olive"],  "linestyle": "-.", "marker": "*", "dashes": (5, 1, 1, 1)},
    "Logit (ISE-min)":     {"color": COLORS["indigo"], "linestyle": "-.", "marker": "h", "dashes": (5, 1, 1, 1)},
    "Reflect (ISE-min)":   {"color": COLORS["grey"],   "linestyle": "-.", "marker": "d", "dashes": (5, 1, 1, 1)},
}

# ---------------------------------------------------------------------------
# Convenience helpers for seaborn relplot keyword arguments
# ---------------------------------------------------------------------------

def get_color_map():
    """Return ``{Method label: color}`` for seaborn ``palette``."""
    return {k: v["color"] for k, v in METHOD_STYLES.items()}


def get_seaborn_dashes():
    """Return ``{Method label: dash_tuple}`` for seaborn ``dashes``."""
    return {k: v["dashes"] for k, v in METHOD_STYLES.items()}


def get_seaborn_markers():
    """Return ``{Method label: marker}`` for seaborn ``markers``."""
    return {k: v["marker"] for k, v in METHOD_STYLES.items()}


def get_method_order():
    """Ordered list of method labels (matches ``METHOD_STYLES`` insertion order)."""
    return list(METHOD_STYLES.keys())

# ---------------------------------------------------------------------------
# Ablation study styles (3 baselines)
# ---------------------------------------------------------------------------

ABLATION_STYLES = {
    "MODEL_A": {"label": "Baseline: Var Only",  "color": COLORS["red"],   "linestyle": "dotted",  "marker": "o"},
    "MODEL_B": {"label": "Baseline: Var + Skew","color": COLORS["green"], "linestyle": "dashed",  "marker": "s"},
    "MODEL_C": {"label": "Baseline: Var + Kurt","color": COLORS["blue"],  "linestyle": "dashdot", "marker": "^"},
}

# ---------------------------------------------------------------------------
# Kernel shape plot styles (6 evaluation points)
# ---------------------------------------------------------------------------

KERNEL_SHAPE_COLORS = [
    COLORS["red"],
    COLORS["blue"],
    COLORS["green"],
    COLORS["yellow"],
    COLORS["purple"],
    COLORS["teal"],
]

KERNEL_SHAPE_STYLES = [
    {"linestyle": "-",  "marker": "o"},
    {"linestyle": "--", "marker": "s"},
    {"linestyle": ":",  "marker": "^"},
    {"linestyle": "-.", "marker": "D"},
    {"linestyle": (0, (5, 1)),           "marker": "v"},
    {"linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "P"},
]
