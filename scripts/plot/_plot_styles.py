"""Unified plot styling for all figures in the paper.

Provides a single ``setup_theme()`` call that every plot script should invoke
before creating figures, plus per-method style dictionaries that encode both
*colour* (for the online/digital version) and *linestyle + marker* (so every
series remains distinguishable when printed in black-and-white).

Colours primarily use the Okabe-Ito colorblind-safe palette, supplemented 
by high-contrast Paul Tol colors, optimized for grayscale luminance.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Okabe-Ito & High-Contrast Colorblind-Safe Palette
# (Gold standard for scientific plotting and grayscale conversion)
# ---------------------------------------------------------------------------
COLORS = {
    # Okabe-Ito Base
    "black":     "#000000",
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",  # Dark, vibrant red/orange
    "purple":    "#CC79A7",
    "grey":      "#999999",
    # Supplementary high-contrast Paul Tol
    "indigo":    "#332288",
    "teal":      "#44AA99",
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
# Semantic grouping by LINELSTYLE (Selector):
#   linestyle  "-"   (solid)     = proposed method (Beta Ref) — emphasised
#   linestyle  "--"  (dashed)    = fast / rule-of-thumb selectors
#   linestyle  ":"   (dotted)    = LSCV selectors
#   linestyle  "-."  (dash-dot)  = oracle / benchmark (ISE-min, Oracle)
#
# Semantic grouping by COLOR (Estimator):
#   Reds/Warm = Beta, Blues = Logit, Greens = Reflect
#
# Unique markers guarantee distinguishability in B&W print.
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    # --- PROPOSED (High visibility Vermilion, Solid, Circle) ---
    "Beta (Ref)":          {"color": COLORS["vermilion"], "linestyle": "-",  "marker": "o", "dashes": ""},
    
    # --- BETA FAMILY (Warm/Red Hues) ---
    "Beta (LSCV)":         {"color": COLORS["orange"],    "linestyle": ":",  "marker": "D", "dashes": (1, 1.5)},
    "Beta (Oracle)":       {"color": COLORS["purple"],    "linestyle": "-.", "marker": "*", "dashes": (4, 1.5, 1, 1.5)},
    "Beta (ISE)":          {"color": COLORS["black"],     "linestyle": "-.", "marker": "X", "dashes": (4, 1.5, 1, 1.5)},

    # --- LOGIT FAMILY (Blue Hues) ---
    "Logit (Silverman)":   {"color": COLORS["blue"],      "linestyle": "--", "marker": "s", "dashes": (4, 1.5)},
    "Logit (LSCV)":        {"color": COLORS["skyblue"],   "linestyle": ":",  "marker": "v", "dashes": (1, 1.5)},
    "Logit (ISE-min)":     {"color": COLORS["indigo"],    "linestyle": "-.", "marker": "p", "dashes": (4, 1.5, 1, 1.5)},

    # --- REFLECT FAMILY (Green/Grey Hues) ---
    "Reflect (Silverman)": {"color": COLORS["green"],     "linestyle": "--", "marker": "^", "dashes": (4, 1.5)},
    "Reflect (LSCV)":      {"color": COLORS["teal"],      "linestyle": ":",  "marker": "P", "dashes": (1, 1.5)},
    "Reflect (ISE-min)":   {"color": COLORS["grey"],      "linestyle": "-.", "marker": "h", "dashes": (4, 1.5, 1, 1.5)},
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
    "MODEL_A": {"label": "Baseline: Var Only",  "color": COLORS["vermilion"], "linestyle": "dotted",  "marker": "o"},
    "MODEL_B": {"label": "Baseline: Var + Skew","color": COLORS["green"],     "linestyle": "dashed",  "marker": "s"},
    "MODEL_C": {"label": "Baseline: Var + Kurt","color": COLORS["blue"],      "linestyle": "dashdot", "marker": "^"},
}

# ---------------------------------------------------------------------------
# Kernel shape plot styles (6 evaluation points)
# ---------------------------------------------------------------------------

KERNEL_SHAPE_COLORS = [
    COLORS["vermilion"],
    COLORS["blue"],
    COLORS["green"],
    COLORS["orange"],
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