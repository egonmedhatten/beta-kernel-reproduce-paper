"""Unified plot styling for all figures in the paper.

Provides a single ``setup_theme()`` call that every plot script should invoke
before creating figures, plus per-method style dictionaries that encode both
*colour* (for the online/digital version) and *linestyle + marker* (so every
series remains distinguishable when printed in black-and-white).

Colours primarily use the Okabe-Ito colorblind-safe palette, supplemented
by high-contrast Paul Tol colors, optimized for grayscale luminance.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# ---------------------------------------------------------------------------
# Global configuration flags
# ---------------------------------------------------------------------------
SHOW_SUPTITLE = True          # Set False to omit figure suptitles (use LaTeX captions)
FIGURE_WIDTH_FULL = 6.5       # Single-column paper full width (inches)

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
    "grey":      "#666666",  # Darkened from #999999 for B&W print contrast
    # Supplementary high-contrast Paul Tol
    "indigo":    "#332288",
    "teal":      "#44AA99",
}

# ---------------------------------------------------------------------------
# Theme setup — call once at the top of every plot script
# ---------------------------------------------------------------------------

# Minimum font size (pt) — enforced for all text elements
_MIN_FONT_PT = 10

def setup_theme():
    """Configure matplotlib + seaborn for publication-quality figures.

    Important: ``plt.rcParams`` are set *after* ``sns.set_theme()`` so that
    our explicit font-size floors (>= 10 pt) are never overridden by seaborn's
    context-dependent scaling.
    """
    plt.switch_backend("Agg")

    # 1. Apply seaborn theme first (this resets rcParams)
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

    # 2. Override rcParams AFTER seaborn to guarantee >= 10pt everywhere
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans",
        ],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })

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

# ---------------------------------------------------------------------------
# Grouped legend helper — organises methods by kernel family
# ---------------------------------------------------------------------------

# Define the kernel-family groups and their ordering
METHOD_GROUPS = {
    "Beta":    ["Beta (Ref)", "Beta (LSCV)", "Beta (Oracle)", "Beta (ISE)"],
    "Logit":   ["Logit (Silverman)", "Logit (LSCV)", "Logit (ISE-min)"],
    "Reflect": ["Reflect (Silverman)", "Reflect (LSCV)", "Reflect (ISE-min)"],
}


def build_grouped_legend(ax_or_g, ncol=3, loc="lower center",
                         bbox_to_anchor=(0.5, -0.15), frameon=False,
                         **legend_kw):
    """Create a legend with methods grouped by kernel family.

    Builds handles directly from ``METHOD_STYLES`` so the legend is
    independent of what seaborn puts on the axes.  Each kernel family
    (Beta, Logit, Reflect) gets its own column with a bold header.

    Parameters
    ----------
    ax_or_g : matplotlib Axes, Figure, or seaborn FacetGrid
        Object to attach the legend to.
    ncol : int
        Number of legend columns (should equal len(METHOD_GROUPS)).
    **legend_kw : dict
        Extra keyword arguments forwarded to ``legend()``.

    Returns
    -------
    legend : matplotlib Legend
    """
    # Determine the figure object and remove any existing seaborn legend
    if hasattr(ax_or_g, "fig"):
        # seaborn FacetGrid
        fig = ax_or_g.fig
        if hasattr(ax_or_g, "_legend") and ax_or_g._legend is not None:
            ax_or_g._legend.remove()
            ax_or_g._legend = None
    elif hasattr(ax_or_g, "figure"):
        fig = ax_or_g.figure
    else:
        fig = ax_or_g

    # Build handles from METHOD_STYLES directly — reliable regardless of
    # seaborn's legend= parameter.
    max_group = max(len(v) for v in METHOD_GROUPS.values())

    ordered_handles = []
    ordered_labels = []
    for group_name, members in METHOD_GROUPS.items():
        # Group header — bold text, invisible handle
        blank = mlines.Line2D([], [], color="none", marker="None",
                              linestyle="None")
        ordered_handles.append(blank)
        ordered_labels.append(f"$\\bf{{{group_name}}}$")

        for m in members:
            style = METHOD_STYLES.get(m)
            if style is None:
                continue
            handle = mlines.Line2D(
                [], [],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=6,
                linewidth=1.8,
            )
            ordered_handles.append(handle)
            ordered_labels.append(m)

        # Pad shorter groups so columns stay aligned
        n_pad = max_group - len(members)
        for _ in range(n_pad):
            ordered_handles.append(
                mlines.Line2D([], [], color="none", marker="None",
                              linestyle="None")
            )
            ordered_labels.append("")

    leg = fig.legend(
        ordered_handles,
        ordered_labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=frameon,
        columnspacing=1.5,
        handletextpad=0.6,
        labelspacing=0.4,
        **legend_kw,
    )
    return leg