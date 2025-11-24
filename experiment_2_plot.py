import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib_inline.backend_inline import set_matplotlib_formats
# Set the desired output format
set_matplotlib_formats('svg')
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import warnings
import os

# --- Imports Moved to Top ---
# This script MUST be in the same directory as your KDE.py and KDE_Gauss.py
# files, as it needs to import them to run the analysis.
try:
    from KDE import BetaKernelKDE
    from KDE_Gauss import GaussianKDE
except ImportError:
    print("Error: Could not import KDE and KDE_Gauss.py.")
    print("Please place this script in the same directory as your KDE class files.")
    exit()
# --- End Imports ---


# --- Consistent Plot Styling ---

# 1. Add the same seaborn theme from Experiment 1
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

# 2. Define the same method labels and types from Experiment 1
method_rename_map = {
    "BETA_ROT": "Beta (Ref)",
    "BETA_LSCV": "Beta (LSCV)",
    "LOGIT_SILV": "Logit (Silverman)",
    "LOGIT_LSCV": "Logit (LSCV)",
    "REFLECT_SILV": "Reflect (Silverman)",
    "REFLECT_LSCV": "Reflect (LSCV)",
}

def selector_type(method):
    if "LSCV" in method: return "Slow (LSCV)"
    return "Fast (Rule)"

# 3. Define the style and color mappings to match Experiment 1
palette = sns.color_palette('deep', 10)

# Map labels to the same colors used in Experiment 1's seaborn plots
COLOR_MAP = {
    "Beta (Ref)": palette[3], # Red
    "Beta (LSCV)": palette[0],          # Blue
    "Logit (Silverman)": palette[2],    # Green
    "Logit (LSCV)": palette[4],         # Purple
    "Reflect (Silverman)": palette[1],  # Orange
    "Reflect (LSCV)": palette[5],       # Brown
}

# Map selector types to the same linestyles from Experiment 1
# (This plot only uses ax.plot, so the simple STRING map is correct)
STYLE_MAP = {
    "Fast (Rule)": "--",
    "Slow (LSCV)": ":",
}

# 4. Define the (simpler) config for KDE classes
METHOD_CONFIG = {
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
# --- End Consistent Styling ---


def fetch_data():
    """
    Fetches the UCI Communities and Crime dataset.
    Returns a dictionary of the three data vectors.
    """
    try:
        communities_crime = fetch_ucirepo(id=183)
        X = communities_crime.data.features
        datasets = {
            "PctKids2Par": X["PctKids2Par"].dropna().values,
            "PctPopUnderPov": X["PctPopUnderPov"].dropna().values,
            "PctVacantBoarded": X["PctVacantBoarded"].dropna().values,
        }
        return datasets
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Please ensure 'ucimlrepo' is installed: pip install ucimlrepo")
        return None

def main():
    """
    Loads experiment 2 results and generates the 1x3
    qualitative fit plot for the paper.
    """
    print("Loading results from data/experiment2/experiment_2_summary.csv...")
    try:
        df_summary = pd.read_csv("data/experiment2/experiment_2_summary.csv")
    except FileNotFoundError:
        print("Error: 'data/experiment2/experiment_2_summary.csv' not found.")
        return

    print("Fetching raw data from UCI repository...")
    datasets = fetch_data()
    if datasets is None:
        return

    dataset_names = list(datasets.keys())
    
    # --- MODIFICATION: Change to a 3x1 vertical layout ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 12)) # Tall, narrower figure
    axes = axes.flatten()
    # --- END MODIFICATION ---

    print("Generating plots for each dataset...")
    for i, data_name in enumerate(dataset_names):
        ax = axes[i]
        data_vector = datasets[data_name]

        # 1. Plot histogram of the raw data
        ax.hist(
            data_vector,
            bins=30,
            density=True,
            alpha=0.4,
            label="Data Histogram",
            color="0.7", # Use a lighter gray
        )

        # 2. Get the results for this dataset
        df_data = df_summary[df_summary['dataset'] == data_name]
        
        # Plot points for the PDF
        x_plot = np.linspace(0, 1, 1000, endpoint=True)

        # 3. Fit and plot each method
        for _, row in df_data.iterrows():
            method_name = row['method']
            bandwidth = row['bandwidth']
            
            if method_name not in METHOD_CONFIG:
                continue

            # --- Apply consistent styling ---
            config = METHOD_CONFIG[method_name]
            
            # Get consistent labels, colors, and styles
            label = method_rename_map.get(method_name, method_name)
            sel_type = selector_type(method_name)
            color = COLOR_MAP.get(label, "black")
            style = STYLE_MAP.get(sel_type, "-")
            
            # Apply emphasis to the proposed rule
            if method_name == "BETA_ROT":
                lw = 3.0
                zorder = 10
                # Use the 'emphasis' style from Exp1
                style = "--"
            else:
                lw = 2.0
                zorder = 5
            
            plot_label = f"{label}"# (h={bandwidth:.3f})"
            # --- End styling ---
            
            try:
                # Initialize the correct KDE class with the
                # bandwidth found during the experiment
                kde = config['class'](
                    **config['init_args']
                )

                kde.bandwidth = bandwidth
                
                # Fit the model on the full raw data
                kde.fit(data_vector)
                
                # Get PDF values
                pdf_plot = kde.pdf(x_plot)

                # Plot with new styles
                ax.plot(x_plot, pdf_plot, 
                        label=plot_label, 
                        color=color,
                        linestyle=style,
                        linewidth=lw,
                        zorder=zorder)

            except Exception as e:
                print(f"Could not plot {method_name} for {data_name}: {e}")

        # --- MODIFICATION: Adjust titles and labels for 3x1 layout ---
        ax.set_title(f"{data_name}", fontsize=14)
        
        # Only add X-axis label to the bottom-most plot
        if i == len(dataset_names) - 1:
            ax.set_xlabel("Value")
        
        # Add Y-axis label to the middle plot for centering
        if i == 1:
            ax.set_ylabel("Density")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        # --- END MODIFICATION ---

    handles, labels = axes[0].get_legend_handles_labels()
    
    # --- MODIFICATION: Adjust legend for 3x1 layout ---
    # fig.suptitle("Qualitative PDF Fits on Real-World Data", fontsize=16)
    fig.legend(handles, labels, 
               loc='lower center', 
               ncol=3, # 3 columns fits well under this layout
               bbox_to_anchor=(0.5, -0.05), # Adjust anchor for new shape
               frameon=False) 
    
    # Adjust for legend and suptitle
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Give a bit more room at the bottom
    # --- END MODIFICATION ---
    
    output_filename = "data/experiment2/plots/experiment_2_visual_fits.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully saved figure to {output_filename}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Ensure the plot directory exists
    os.makedirs("data/experiment2/plots", exist_ok=True)
    
    main()