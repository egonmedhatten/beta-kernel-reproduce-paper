import pandas as pd

def format_time(t):
    """Format time: <0.0001 as <0.0001, small as 0.xxxx, large as 12.3"""
    if t < 0.0001:
        return "$<0.0001$"
    elif t < 1.0:
        return f"{t:.4f}"
    else:
        return f"{t:.1f}"

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

def generate_latex_file(csv_path, output_path):
    df = pd.read_csv(csv_path)

    # MODIFICATION: Added LOGIT_LSCV and REFLECT_LSCV back into the list
    target_methods = ['BETA_ROT', 'BETA_LSCV', 'LOGIT_SILV', 'LOGIT_LSCV', 'REFLECT_SILV', 'REFLECT_LSCV']
    df = df[df['method'].isin(target_methods)]

    datasets = df['dataset'].unique()
    
    with open(output_path, 'w') as f:
        # Write the table content (no begin/end table, no caption)
        f.write(r"\begin{tabular}{lccccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Dataset} & \textbf{Method} & \textbf{LSCV Score} & \textbf{Time (s)} & \textbf{Fallback Rate} \\ \hline" + "\n")

        for dataset in datasets:
            subset = df[df['dataset'] == dataset].copy()
            
            # Find best LSCV score (min) and Time (min) for bolding
            best_lscv = subset['lscv_score'].min()
            best_time = subset['comp_time_sec'].min()

            # Pretty print dataset name with multirow - Adjusted row count to 6
            f.write(f"\\multirow{{6}}{{*}}{{\\textit{{{dataset}}}}} \n")

            # Order methods: ROT first, then Benchmarks
            method_order = target_methods
            subset['method'] = pd.Categorical(subset['method'], categories=method_order, ordered=True)
            subset = subset.sort_values('method')

            for _, row in subset.iterrows():
                method_macro = get_method_macro(row['method'])
                
                # Bandwidth
                h_str = f"{row['bandwidth']:.4f}"
                
                # LSCV (Bold if best, within tolerance)
                lscv_val = row['lscv_score']
                if abs(lscv_val - best_lscv) < 1e-6:
                    lscv_str = f"\\textbf{{{lscv_val:.4f}}}"
                else:
                    lscv_str = f"{lscv_val:.4f}"

                # Time (Bold if best)
                time_val = row['comp_time_sec']
                time_str = format_time(time_val)
                if abs(time_val - best_time) < 1e-6:
                    time_str = f"\\textbf{{{time_str}}}"

                # Fallback Rate Logic
                if row['method'] == 'BETA_ROT':
                    rate = row['is_fallback_cv_mean'] * 100
                    fallback_str = f"{rate:.0f}\\%"
                else:
                    fallback_str = "-"

                f.write(f" & {method_macro} & {lscv_str} & {time_str} & {fallback_str} \\\\\n")
            
            f.write(r"\hline" + "\n")

        f.write(r"\end{tabular}" + "\n")

    print(f"Successfully generated table code in: {output_path}")


if __name__ == "__main__":
    # Run with your specific filename
    generate_latex_file('data/experiment2/experiment_2_summary.csv', 'data/experiment2/tables/experiment_2_table.tex')