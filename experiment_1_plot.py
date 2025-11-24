# TODO: Name the rule appropriately in the plots
# TODO: Split LLSCV plot into two plots, one for "hard" and one for "nice" dists
# TODO: Plot BETA_ROT bandwidth together with oracle bandwidths in a separate plot
"""
Analyzes and plots the results from 'simulation_results_full.csv'.

This script is designed to read the "wide" output format from
'run_experiment_1.py' and "melt" it into a "long" (tidy) format
for visualization with Seaborn.

It generates three key multi-panel plots:
1. LSCV Score (Primary Metric) vs. Sample Size
2. ISE Score (Validation Metric) vs. Sample Size
3. Computation Time (Speed Metric) vs. Sample Size
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from tqdm.auto import tqdm

# --- Matplotlib Setup ---
plt.switch_backend('Agg')
sns.set_theme(
    context='paper',
    style='whitegrid',
    palette='deep',
    font_scale=1.1,
    rc={
        "grid.linestyle": ":",
        "grid.color": "0.8",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)
# --- End Setup ---

# --- MODIFICATION START: Define consistent Color and Style maps ---
palette = sns.color_palette('deep', 10)

# 1. Define a consistent COLOR_MAP for all 10 methods
COLOR_MAP = {
    "Beta (LSCV)": palette[0],
    "Reflect (Silverman)": palette[1],
    "Logit (Silverman)": palette[2],
    "Beta (Ref)": palette[3],
    "Logit (LSCV)": palette[4],
    "Reflect (LSCV)": palette[5],
    "Beta (Oracle)": palette[6],
    "Beta (ISE)": palette[7],
    "Logit (ISE-min)": palette[8],
    "Reflect (ISE-min)": palette[9],
}

# --- BUG FIX: Create TWO style maps ---
# 1. TUPLE map for seaborn's `dashes` argument
STYLE_MAP_TUPLES = {
    "Fast (Rule)": (4, 1.5),
    "Slow (LSCV)": (1, 1),
    "Benchmark (Oracle)": (5, 1, 1, 1),
}
# 2. STRING map for matplotlib's `linestyle` argument
STYLE_MAP_STRINGS = {
    "Fast (Rule)": "--",
    "Slow (LSCV)": ":",
    "Benchmark (Oracle)": "-.",
}
# --- END BUG FIX ---

# --- MODIFICATION START: Moved type to global scope ---
def type(method):
    """Helper function to get the selector type from a method name."""
    if "LSCV" in method: return "Slow (LSCV)"
    if "ISE" in method: return "Benchmark (Oracle)"
    if "ORACLE" in method: return "Benchmark (Oracle)"
    return "Fast (Rule)"
# --- MODIFICATION END ---


# --- MODIFICATION: Removed 'style' parameter from signature ---
def add_emphasis_to_plot(g, df_agg, y_col_name, x_col_name, label_to_emphasize, width):
    """
    Finds a specific line in a FacetGrid and re-plots it with emphasis.
    """
    # 1. Find the correct color/style from the legend
    emph_color = None
    if g.legend:
        for line in g.legend.get_lines():
            if line.get_label() == label_to_emphasize:
                # --- MODIFICATION: Get color from new COLOR_MAP ---
                emph_color = COLOR_MAP.get(label_to_emphasize)
                # Also update the legend line thickness
                line.set_linewidth(width)
                break # Found it
                
    # 2. Iterate over all axes and re-plot the emphasized line on top
    if emph_color:
        # Get the data for the emphasized method
        df_emph = df_agg[df_agg['Method'] == label_to_emphasize].copy() 
        
        # g.axes is a 1D array when col_wrap is used
        for ax in g.axes.flat:
            dist_name_on_plot = ax.get_title()
            if not dist_name_on_plot: # Skip empty facets
                continue

            # Filter data for this specific axis
            df_ax_emph = df_emph[df_emph['dist_name'] == dist_name_on_plot]
            
            if not df_ax_emph.empty:
                y_data = df_ax_emph[y_col_name]
                x_data = df_ax_emph[x_col_name]

                # --- BUG FIX: Get the string linestyle ---
                emph_style_string = STYLE_MAP_STRINGS.get(type("BETA_ROT"), "--")

                # Ensure 'linestyle' gets the string and 'linewidth' gets the float
                ax.plot(
                    x_data,
                    y_data,
                    color=emph_color,
                    linestyle=emph_style_string, # <-- Should be the string
                    linewidth=width,             # <-- Should be the float
                    marker='o',
                    label=label_to_emphasize,
                    zorder=10 
                )
                # --- END BUG FIX ---
# --- END NEW HELPER FUNCTION ---


def load_and_melt_data(csv_file):
    """
    Loads the "wide" CSV and melts it into a "long" (tidy) DataFrame.
    """
    print(f"Loading raw data from '{csv_file}'...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{csv_file}'")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Identify all base method names
    methods = [
        "BETA_ROT", "BETA_LSCV", "BETA_ISE", "BETA_ORACLE",
        "LOGIT_SILV", "LOGIT_LSCV", "LOGIT_ISE",
        "REFLECT_SILV", "REFLECT_LSCV", "REFLECT_ISE",
    ]
    
    all_melted_dfs = []
    id_vars = ['distribution', 'n', 'trial']

    # --- Melt all metrics one by one ---
    metrics_to_melt = ['h', 'lscv_score', 'ise_score', 'comp_time', 'integral_error']
    df_long = pd.DataFrame(columns=id_vars) # Start with an empty DF with the ID columns
    
    print("Melting data...")
    # This is a more robust way to melt
    for metric in metrics_to_melt:
        metric_cols = [f"{m}_{metric}" for m in methods if f"{m}_{metric}" in df.columns]
        if not metric_cols:
            continue
            
        df_metric = df.melt(
            id_vars=id_vars, 
            value_vars=metric_cols, 
            var_name='method_col', 
            value_name=metric
        )
        df_metric['method'] = df_metric['method_col'].str.replace(f'_{metric}', '')
        
        # Merge into the long dataframe
        if df_long.empty:
            df_long = df_metric
        else:
            df_long = df_long.merge(
                df_metric.drop(columns=['method_col']),
                on=id_vars + ['method'],
                how='outer'
            )

    # Get 'is_fallback' (only exists for BETA_ROT)
    fallback_col = 'BETA_ROT_is_fallback'
    if fallback_col in df.columns:
        df_fallback = df[id_vars + [fallback_col]].copy()
        df_fallback.rename(columns={fallback_col: 'is_fallback'}, inplace=True)
        df_fallback['method'] = 'BETA_ROT'
        
        df_long = df_long.merge(
            df_fallback,
            on=id_vars + ['method'],
            how='left'
        )
        # Ensure 'is_fallback' is False for all other methods
        df_long['is_fallback'].fillna(False, inplace=True)

    # --- Add Helper Columns for Plotting ---
    
    # 1. Format distribution names for panel titles
    def format_dist_name(name):
        match = re.match(r'B\((\d+\.?\d*), (\d+\.?\d*)\)', name)
        if match: return f"$B({match.group(1)}, {match.group(2)})$"
        match = re.match(r'NT\((\d+\.?\d*), (\d+\.?\d*)\)', name)
        if match:
            var = float(match.group(2))#**2
            return f"$\\mathcal{{NT}}({match.group(1)}, {var:.2f})$"
        if name == "BIMODAL": return "Bimodal"
        return name
        
    df_long['dist_name'] = df_long['distribution'].apply(format_dist_name)
    
    # Get logical sort order for panels
    try:
        def sort_key(dist_name):
            if '$B(0.5, 0.5)$' in dist_name: return 0
            if '$B(0.8, 2.5)$' in dist_name: return 1
            if '$B(1.5, 1.5)$' in dist_name: return 2
            if 'Bimodal' in dist_name: return 3
            if '$B(2, 12)$' in dist_name: return 4
            if 'mathcal{NT}' in dist_name: return 5
            if '$B(5, 5)$' in dist_name: return 6
            return 7
        dist_order = sorted(df_long['dist_name'].unique(), key=sort_key)
    except Exception:
        dist_order = sorted(df_long['dist_name'].unique())


    # 2. Add dist_type
    df_long['dist_type'] = 'nice'
    df_long.loc[df_long['distribution'].isin(['B(0.5, 0.5)', 'B(0.8, 2.5)', 'B(1.5, 1.5)']), 'dist_type'] = 'hard'

    # 3. Add Method type (for style)
    #    --- MODIFICATION: Now uses the global helper function ---
    df_long['type'] = df_long['method'].apply(type)

    # 4. Add Kernel type (for hue)
    def kernel_type(method):
        if "BETA" in method: return "Beta"
        if "LOGIT" in method: return "Logit-Gauss"
        if "REFLECT" in method: return "Reflect-Gauss"
        return "Other"
    df_long['kernel_type'] = df_long['method'].apply(kernel_type)
    
    # 5. Create a more readable method label
    method_rename_map = {
        "BETA_ROT": "Beta (Ref)",
        "BETA_LSCV": "Beta (LSCV)",
        "BETA_ISE": "Beta (ISE)",
        "BETA_ORACLE": "Beta (Oracle)",
        "LOGIT_SILV": "Logit (Silverman)",
        "LOGIT_LSCV": "Logit (LSCV)",
        "LOGIT_ISE": "Logit (ISE-min)",
        "REFLECT_SILV": "Reflect (Silverman)",
        "REFLECT_LSCV": "Reflect (LSCV)",
        "REFLECT_ISE": "Reflect (ISE-min)",
    }
    df_long['Method'] = df_long['method'].map(method_rename_map)

    print("Data processing complete.")
    return df_long, dist_order


def plot_lscv_vs_n(df_long, dist_order, output_file, emphasis_config):
    """
    Plots the primary metric: LSCV Score vs. Sample Size (N).
    This plot uses all 7 distributions.
    """
    print(f"Generating LSCV plot: {output_file}...")
    
    df_agg = df_long.groupby(
        ['dist_name', 'n', 'Method', 'type']
    ).mean(numeric_only=True).reset_index()

    g = sns.relplot(
        data=df_agg,
        kind='line',
        x='n',
        y='lscv_score',
        hue='Method',
        style='type',
        col='dist_name',
        col_wrap=3,
        col_order=dist_order,
        height=3.5,
        aspect=1.2,
        facet_kws={'sharey': False, 'despine': True},
        legend='full',
        lw=2.0,
        markers=True,
        # --- MODIFICATION: Use TUPLE map for dashes ---
        palette=COLOR_MAP,
        style_order=list(STYLE_MAP_TUPLES.keys()),
        dashes=STYLE_MAP_TUPLES
        # --- MODIFICATION END ---
    )
    
    # --- MODIFICATION: Definitive Y-Axis Fix ---
    g.set(xscale='log', 
          xlabel="Sample Size ($n$)", 
          ylabel="") # Remove buggy figure-level label
    for ax in g.axes.flat:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Mean LSCV Score (Lower is Better)")
    # --- MODIFICATION END ---
            
    g.set_titles(template="{col_name}")
    g.fig.suptitle("LSCV Score (Primary Metric) vs. Sample Size", fontsize=16)
    
    # --- MODIFICATION: Removed 'style' from call ---
    if emphasis_config['do_emphasis']:
        add_emphasis_to_plot(
            g=g,
            df_agg=df_agg,
            y_col_name='lscv_score',
            x_col_name='n',
            label_to_emphasize=emphasis_config['label'],
            width=emphasis_config['width']
        )
    # --- MODIFICATION END ---
    
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    g.fig.tight_layout(rect=[0, 0, 1, 0.95])

    g.savefig(output_file, bbox_inches='tight')
    plt.close(g.fig)


def plot_ise_vs_n(df_long, dist_order, output_file, emphasis_config):
    """
    Plots the validation metric: ISE Score vs. Sample Size (N).
    This plot *only* uses the "nice" distributions where ISE is stable.
    """
    print(f"Generating ISE plot: {output_file}...")
    
    df_nice = df_long[df_long['dist_type'] == 'nice'].copy()
    nice_dist_order = [d for d in dist_order if d in df_nice['dist_name'].unique()]
    df_agg = df_nice.groupby(
        ['dist_name', 'n', 'Method', 'type']
    ).mean(numeric_only=True).reset_index()
    
    col_wrap_val = 3
    if len(nice_dist_order) <= 3: col_wrap_val = len(nice_dist_order)
    elif len(nice_dist_order) == 4: col_wrap_val = 2

    g = sns.relplot(
        data=df_agg,
        kind='line',
        x='n',
        y='ise_score',
        hue='Method',
        style='type',
        col='dist_name',
        col_wrap=col_wrap_val, 
        col_order=nice_dist_order,
        height=3.5,
        aspect=1.2,
        facet_kws={'sharey': False, 'despine': True},
        legend='full',
        lw=2.0,
        markers=True,
        # --- MODIFICATION: Use TUPLE map for dashes ---
        palette=COLOR_MAP,
        style_order=list(STYLE_MAP_TUPLES.keys()),
        dashes=STYLE_MAP_TUPLES
        # --- MODIFICATION END ---
    )
    
    # --- MODIFICATION: Definitive Y-Axis Fix ---
    g.set(xscale='log', yscale='log',
          xlabel="Sample Size ($n$)", 
          ylabel="") # Remove buggy figure-level label
    for ax in g.axes.flat:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Mean ISE Score (Lower is Better)")
    # --- MODIFICATION END ---
            
    g.set_titles(template="{col_name}")
    g.fig.suptitle("ISE Score (Validation Metric) vs. Sample Size (on 'Nice' Dists)", fontsize=16)

    # --- MODIFICATION: Removed 'style' from call ---
    if emphasis_config['do_emphasis']:
        add_emphasis_to_plot(
            g=g,
            df_agg=df_agg,
            y_col_name='ise_score',
            x_col_name='n',
            label_to_emphasize=emphasis_config['label'],
            width=emphasis_config['width']
        )
    # --- MODIFICATION END ---
    
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    g.fig.tight_layout(rect=[0, 0, 1, 0.95])

    g.savefig(output_file, bbox_inches='tight')
    plt.close(g.fig)

def plot_time_vs_n(df_long, dist_order, output_file, emphasis_config):
    """
    Plots the speed metric: Computation Time vs. Sample Size (N).
    This plot shows all 7 distributions.
    """
    print(f"Generating Computation Time plot: {output_file}...")
    
    df_agg = df_long.groupby(
        ['dist_name', 'n', 'Method', 'type']
    ).mean(numeric_only=True).reset_index()
    
    g = sns.relplot(
        data=df_agg,
        kind='line',
        x='n',
        y='comp_time',
        hue='Method',
        style='type',
        col='dist_name',
        col_wrap=3,
        col_order=dist_order,
        height=3.5,
        aspect=1.2,
        facet_kws={'sharey': False, 'despine': True},
        legend='full',
        lw=2.0,
        markers=True,
        # --- MODIFICATION: Use TUPLE map for dashes ---
        palette=COLOR_MAP,
        style_order=list(STYLE_MAP_TUPLES.keys()),
        dashes=STYLE_MAP_TUPLES
        # --- MODIFICATION END ---
    )
    
    # --- MODIFICATION: DefinITIVE Y-Axis Fix ---
    g.set(xscale='log', yscale='log',
          xlabel="Sample Size ($n$)", 
          ylabel="") # Remove buggy figure-level label
    for ax in g.axes.flat:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Mean Computation Time (sec, Lower is Better)")
    # --- MODIFICATION END ---
            
    g.set_titles(template="{col_name}")
    g.fig.suptitle("Computation Time vs. Sample Size", fontsize=16)
    
    # --- MODIFICATION: Removed 'style' from call ---
    if emphasis_config['do_emphasis']:
        add_emphasis_to_plot(
            g=g,
            df_agg=df_agg,
            y_col_name='comp_time',
            x_col_name='n',
            label_to_emphasize=emphasis_config['label'],
            width=emphasis_config['width']
        )
    # --- MODIFICATION END ---
    
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    g.fig.tight_layout(rect=[0, 0, 1, 0.95])

    g.savefig(output_file, bbox_inches='tight')
    plt.close(g.fig)


def plot_bandwidth_vs_n(df_long, dist_order, output_file, emphasis_config):
    """
    Plots the selected bandwidth (h) vs. Sample Size (N) for
    the proposed rule and all oracle-based methods.
    This plot *only* uses the "nice" distributions.
    """
    print(f"Generating Bandwidth comparison plot (Nice Dists): {output_file}...")
    
    methods_to_plot = [
        'Beta (Ref)',
        'Beta (Oracle)',
        'Beta (ISE)',
    ]
    df_methods_filtered = df_long[df_long['Method'].isin(methods_to_plot)].copy()
    
    df_filtered = df_methods_filtered[df_methods_filtered['dist_type'] == 'nice'].copy()
    nice_dist_order = [d for d in dist_order if d in df_filtered['dist_name'].unique()]
    
    df_agg = df_filtered.groupby(
        ['dist_name', 'n', 'Method', 'type']
    ).mean(numeric_only=True).reset_index()
    
    col_wrap_val = 3
    if len(nice_dist_order) <= 3: col_wrap_val = len(nice_dist_order)
    elif len(nice_dist_order) == 4: col_wrap_val = 2
    
    g = sns.relplot(
        data=df_agg,
        kind='line',
        x='n',
        y='h',
        hue='Method',
        style='type',
        col='dist_name',
        col_wrap=col_wrap_val,
        col_order=nice_dist_order,
        height=3.5,
        aspect=1.2,
        facet_kws={'sharey': False, 'despine': True},
        legend='full',
        lw=2.0,
        markers=True,
        # --- MODIFICATION: Use TUPLE map for dashes ---
        palette=COLOR_MAP,
        style_order=list(STYLE_MAP_TUPLES.keys()),
        dashes=STYLE_MAP_TUPLES
        # --- MODIFICATION END ---
    )
    
    # --- MODIFICATION: Definitive Y-Axis Fix ---
    g.set(xscale='log', yscale='log',
          xlabel="Sample Size ($n$)", 
          ylabel="") # Remove buggy figure-level label
    for ax in g.axes.flat:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Mean Bandwidth ($h$)")
    # --- MODIFICATION END ---
            
    g.set_titles(template="{col_name}")
    g.fig.suptitle("Bandwidth ($h$) Comparison vs. Sample Size (on 'Nice' Dists)", fontsize=16)
    
    # --- MODIFICATION: Removed 'style' from call ---
    if emphasis_config['do_emphasis']:
        add_emphasis_to_plot(
            g=g,
            df_agg=df_agg,
            y_col_name='h',
            x_col_name='n',
            label_to_emphasize=emphasis_config['label'],
            width=emphasis_config['width']
        )
    # --- MODIFICATION END ---
    
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    g.fig.tight_layout(rect=[0, 0, 1, 0.95]) 

    g.savefig(output_file, bbox_inches='tight')
    plt.close(g.fig)


def plot_integral_error_vs_n(df_long, dist_order, output_file, emphasis_config):
    """
    Plots the selected bandwidth (h) vs. Sample Size (N) for
    the proposed rule and all oracle-based methods.
    This plot *only* uses the "nice" distributions.
    """
    print(f"Generating integral error comparison plot (Beta kernel): {output_file}...")
    
    methods_to_plot = [
        'Beta (Ref)',
        'Beta (Oracle)',
        'Beta (ISE)',
        'Beta (LSCV)',
    ]
    df_methods_filtered = df_long[df_long['Method'].isin(methods_to_plot)].copy()
    
    df_filtered = df_methods_filtered.copy() #[df_methods_filtered['dist_type'] == 'nice'].copy()
    nice_dist_order = [d for d in dist_order if d in df_filtered['dist_name'].unique()]
    
    df_agg = df_filtered.groupby(
        ['dist_name', 'n', 'Method', 'type']
    ).mean(numeric_only=True).reset_index()
    
    col_wrap_val = 3
    if len(nice_dist_order) <= 3: col_wrap_val = len(nice_dist_order)
    elif len(nice_dist_order) == 4: col_wrap_val = 2
    
    g = sns.relplot(
        data=df_agg,
        kind='line',
        x='n',
        y='integral_error',
        hue='Method',
        style='type',
        col='dist_name',
        col_wrap=col_wrap_val,
        col_order=nice_dist_order,
        height=3.5,
        aspect=1.2,
        facet_kws={'sharey': False, 'despine': True},
        legend='full',
        lw=2.0,
        markers=True,
        # --- MODIFICATION: Use TUPLE map for dashes ---
        palette=COLOR_MAP,
        style_order=list(STYLE_MAP_TUPLES.keys()),
        dashes=STYLE_MAP_TUPLES
        # --- MODIFICATION END ---
    )
    
    # --- MODIFICATION: Definitive Y-Axis Fix ---
    g.set(xscale='log', yscale='log',
          xlabel="Sample Size ($n$)", 
          ylabel="") # Remove buggy figure-level label
    for ax in g.axes.flat:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Mean error ($|\int\hat{f}(x)dx-1|$)")
    # --- MODIFICATION END ---
            
    g.set_titles(template="{col_name}")
    g.fig.suptitle("Integral error of Beta estimate", fontsize=16)
    
    # --- MODIFICATION: Removed 'style' from call ---
    if emphasis_config['do_emphasis']:
        add_emphasis_to_plot(
            g=g,
            df_agg=df_agg,
            y_col_name='integral_error',
            x_col_name='n',
            label_to_emphasize=emphasis_config['label'],
            width=emphasis_config['width']
        )
    # --- MODIFICATION END ---
    
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    g.fig.tight_layout(rect=[0, 0, 1, 0.95]) 

    g.savefig(output_file, bbox_inches='tight')
    plt.close(g.fig)

def plot_integral_error_vs_h(df_long, dist_order, output_file, emphasis_config):
    """
    Plots the selected bandwidth (h) vs. Sample Size (N) for
    the proposed rule and all oracle-based methods.
    This plot *only* uses the "nice" distributions.
    """
    print(f"Generating integral error comparison plot (Beta kernel vs h): {output_file}...")
    
    methods_to_plot = [
        'Beta (Ref)',
        'Beta (Oracle)',
        'Beta (ISE)',
        'Beta (LSCV)',
    ]
    df_methods_filtered = df_long[df_long['Method'].isin(methods_to_plot)].copy()
    
    df_filtered = df_methods_filtered.copy() #[df_methods_filtered['dist_type'] == 'nice'].copy()
    nice_dist_order = [d for d in dist_order if d in df_filtered['dist_name'].unique()]
    
    df_agg = df_filtered.groupby(
        ['dist_name', 'n', 'Method', 'type']
    ).mean(numeric_only=True).reset_index()
    
    col_wrap_val = 3
    if len(nice_dist_order) <= 3: col_wrap_val = len(nice_dist_order)
    elif len(nice_dist_order) == 4: col_wrap_val = 2
    
    g = sns.relplot(
        data=df_agg,
        kind='line',
        x='h',
        y='integral_error',
        hue='Method',
        style='type',
        col='dist_name',
        col_wrap=col_wrap_val,
        col_order=nice_dist_order,
        height=3.5,
        aspect=1.2,
        facet_kws={'sharey': False, 'despine': True},
        legend='full',
        lw=2.0,
        markers=True,
        # --- MODIFICATION: Use TUPLE map for dashes ---
        palette=COLOR_MAP,
        style_order=list(STYLE_MAP_TUPLES.keys()),
        dashes=STYLE_MAP_TUPLES
        # --- MODIFICATION END ---
    )
    
    # --- MODIFICATION: Definitive Y-Axis Fix ---
    g.set(xscale='log', yscale='log',
          xlabel="Sample Size ($n$)", 
          ylabel="") # Remove buggy figure-level label
    for ax in g.axes.flat:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Mean error ($|\int\hat{f}(x)dx-1|$)")
    # --- MODIFICATION END ---
            
    g.set_titles(template="{col_name}")
    g.fig.suptitle("Integral error of Beta estimate", fontsize=16)
    
    # --- MODIFICATION: Removed 'style' from call ---
    if emphasis_config['do_emphasis']:
        add_emphasis_to_plot(
            g=g,
            df_agg=df_agg,
            y_col_name='integral_error',
            x_col_name='h',
            label_to_emphasize=emphasis_config['label'],
            width=emphasis_config['width']
        )
    # --- MODIFICATION END ---
    
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    g.fig.tight_layout(rect=[0, 0, 1, 0.95]) 

    g.savefig(output_file, bbox_inches='tight')
    plt.close(g.fig)

def main():
    """
    Main function to run the analysis.
    """
    # --- Configuration ---
    INPUT_FILE = "data/experiment1/simulation_results_full.csv"
    OUTPUT_LSCV_PLOT = 'data/experiment1/plots/LSCV_Score_vs_N.pdf'
    OUTPUT_ISE_PLOT = 'data/experiment1/plots/ISE_Score_vs_N.pdf'
    OUTPUT_TIME_PLOT = 'data/experiment1/plots/Comp_Time_vs_N.pdf'
    OUTPUT_BW_PLOT = 'data/experiment1/plots/Bandwidth_vs_N_NiceDists.pdf'
    OUTPUT_INTEGRAL_PLOT = 'data/experiment1/plots/IntegralError_vs_N.pdf'
    OUTPUT_INTEGRAL_VS_H_PLOT = 'data/experiment1/plots/IntegralError_vs_H.pdf'

    
    # --- Emphasis Configuration ---
    EMPHASIZE_METHOD = True
    EMPHASIZED_LABEL = 'Beta (Ref)'
    EMPHASIZED_WIDTH = 3.0
    # --- MODIFICATION: This is no longer used, so it's commented out ---
    # EMPHASIZE_STYLE = '--' 
    # --- End Configuration ---
    
    df_long, dist_order = load_and_melt_data(INPUT_FILE)
    
    if df_long is not None:
        # --- MODIFICATION: Removed 'style' from config dict ---
        emphasis_config = {
            'do_emphasis': EMPHASIZE_METHOD,
            'label': EMPHASIZED_LABEL,
            'width': EMPHASIZED_WIDTH,
        }
        # --- MODIFICATION END ---
        
        plot_lscv_vs_n(df_long.copy(), dist_order, OUTPUT_LSCV_PLOT, emphasis_config)
        plot_ise_vs_n(df_long.copy(), dist_order, OUTPUT_ISE_PLOT, emphasis_config)
        plot_time_vs_n(df_long.copy(), dist_order, OUTPUT_TIME_PLOT, emphasis_config)
        plot_bandwidth_vs_n(df_long.copy(), dist_order, OUTPUT_BW_PLOT, emphasis_config)
        plot_integral_error_vs_n(df_long.copy(), dist_order, OUTPUT_INTEGRAL_PLOT, emphasis_config)
        plot_integral_error_vs_h(df_long.copy(), dist_order, OUTPUT_INTEGRAL_VS_H_PLOT, emphasis_config)

        
        print(f"\nAll plots generated successfully:")
        print(f"- {OUTPUT_LSCV_PLOT}")
        print(f"- {OUTPUT_ISE_PLOT}")
        print(f"- {OUTPUT_TIME_PLOT}")
        print(f"- {OUTPUT_BW_PLOT}")
        print(f"- {OUTPUT_INTEGRAL_PLOT}")
        print(f"- {OUTPUT_INTEGRAL_VS_H_PLOT}")
    else:
        print("Script failed because data could not be loaded.")

if __name__ == "__main__":
    # Suppress warnings from plotting with NaNs
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main()