import pandas as pd
import os

def format_p_value(p):
    """Format p-value: <0.001 or specific value."""
    if pd.isna(p):
        return "-"
    if p < 0.001:
        return "$<0.001$"
    else:
        return f"{p:.3f}"

def format_value(v, best_v, is_min_best=True):
    """Format metric value, bolding the best."""
    if pd.isna(v):
        return "-"
    
    # Check if this is the best value (within tolerance)
    is_best = False
    if is_min_best:
        if v <= best_v + 1e-9:
            is_best = True
    else: # Max is best (for Log-Likelihood)
        if v >= best_v - 1e-9:
            is_best = True
            
    str_v = f"{v:.4f}"
    if is_best:
        return f"\\textbf{{{str_v}}}"
    return str_v

def get_method_macro(method_name):
    """Map CSV method names to user-defined LaTeX macros."""
    mapping = {
        'BETA_ROT': r'\rott',
        'BETA_LSCV': r'\blscvt',
        'LOGIT_SILV': r'\lsilvt',
        'LOGIT_LSCV': r'\llscvt',
        'REFLECT_SILV': r'\rsilvt',
        'REFLECT_LSCV': r'\rlscvt'
    }
    return mapping.get(method_name, method_name.replace('_', r'\_'))

def generate_appendix_tables(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    # Define method order matching your main text
    target_methods = ['BETA_ROT', 'BETA_LSCV', 'LOGIT_SILV', 'LOGIT_LSCV', 'REFLECT_SILV', 'REFLECT_LSCV']
    df = df[df['method'].isin(target_methods)]
    
    datasets = df['dataset'].unique()
    
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Table D.1: Wilcoxon P-values for LSCV
    # ---------------------------------------------------------
    d1_path = os.path.join(output_dir, 'table_d1_lscv_pvalues.tex')
    with open(d1_path, 'w') as f:
        f.write(r"\begin{tabular}{llc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Dataset} & \textbf{Method} & \textbf{p-value (vs \rott)} \\ \hline" + "\n")
        
        for dataset in datasets:
            subset = df[df['dataset'] == dataset].copy()
            
            # Sort by method order
            subset['method'] = pd.Categorical(subset['method'], categories=target_methods, ordered=True)
            subset = subset.sort_values('method')
            
            # Multirow for dataset name
            f.write(f"\\multirow{{6}}{{*}}{{\\textit{{{dataset}}}}} \n")
            
            for _, row in subset.iterrows():
                method_macro = get_method_macro(row['method'])
                
                # P-value for LSCV (Density)
                # The column 'density_p_value_wilcoxon (BETA_ROT >)' tests if BETA_ROT is significantly better
                p_val = row['density_p_value_wilcoxon (BETA_ROT >)']
                p_str = format_p_value(p_val)
                
                f.write(f" & {method_macro} & {p_str} \\\\\n")
            
            f.write(r"\hline" + "\n")
        
        f.write(r"\end{tabular}" + "\n")
    print(f"Generated {d1_path}")

    # ---------------------------------------------------------
    # Table D.2: Log-Likelihood Results
    # ---------------------------------------------------------
    d2_path = os.path.join(output_dir, 'table_d2_loglikelihood.tex')
    with open(d2_path, 'w') as f:
        f.write(r"\begin{tabular}{llcc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Dataset} & \textbf{Method} & \textbf{Log-Likelihood} & \textbf{p-value (vs \rott)} \\ \hline" + "\n")
        
        for dataset in datasets:
            subset = df[df['dataset'] == dataset].copy()
            
            # Best Log-Likelihood (Max is best)
            best_ll = subset['log_likelihood'].max()
            
            # Sort
            subset['method'] = pd.Categorical(subset['method'], categories=target_methods, ordered=True)
            subset = subset.sort_values('method')
            
            # Multirow
            f.write(f"\\multirow{{6}}{{*}}{{\\textit{{{dataset}}}}} \n")
            
            for _, row in subset.iterrows():
                method_macro = get_method_macro(row['method'])
                
                # Log Likelihood
                ll_val = row['log_likelihood']
                ll_str = format_value(ll_val, best_ll, is_min_best=False)
                
                # P-value
                p_val = row['loglik_p_value_wilcoxon (BETA_ROT >)']
                p_str = format_p_value(p_val)
                
                f.write(f" & {method_macro} & {ll_str} & {p_str} \\\\\n")
                
            f.write(r"\hline" + "\n")

        f.write(r"\end{tabular}" + "\n")
    print(f"Generated {d2_path}")

if __name__ == "__main__":
    # Adjust paths as needed
    csv_file = 'data/experiment2/experiment_2_summary.csv' # or 'data/experiment2/experiment_2_summary.csv'
    output_folder = 'data/experiment2/tables'
    
    generate_appendix_tables(csv_file, output_folder)