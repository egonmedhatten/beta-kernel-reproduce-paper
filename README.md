# Supplementary Code: Beta Kernel Density Estimation with Boundary Correction

This repository contains the code and data to reproduce all experiments, tables, and figures in the accompanying paper submitted to the *Journal of Computational and Graphical Statistics*.

## Overview

The repository implements boundary-corrected kernel density estimation on the [0, 1] interval using the modified Beta kernel of Chen (1999), together with a novel MISE-optimal reference rule for bandwidth selection. It also provides Gaussian KDE baselines with reflection and logit-transformation boundary corrections.

### Repository Structure

```
├── KDE.py                      # Beta kernel density estimator (library)
├── KDE_Gauss.py                # Gaussian KDE with boundary correction (library)
├── scripts/
│   ├── _paths.py               # Root-path resolution helper
│   ├── run/                    # Long-running experiment scripts
│   │   ├── run_experiment_1.py
│   │   ├── run_experiment_2.py
│   │   └── run_ablation_study.py
│   ├── plot/                   # Figure generation
│   │   ├── plot_experiment_1.py
│   │   ├── plot_experiment_2.py
│   │   └── plot_kernel_shapes.py
│   ├── tables/                 # LaTeX table generation
│   │   ├── tables_experiment_1.py
│   │   ├── table_experiment_2.py
│   │   ├── tables_experiment_2_appendix.py
│   │   └── analysis_ablation.py
│   └── export/                 # Supplementary CSV export
│       ├── export_experiment_1.py
│       └── export_experiment_2.py
├── data/                       # Pre-computed experiment data (committed)
│   ├── experiment1/
│   ├── experiment2/
│   └── ablation_study/
└── output/                     # Generated artifacts (gitignored)
    ├── plots/
    ├── tables/
    └── supplementary/
```

### Data

Pre-computed results are provided in `data/` so that tables and figures can be reproduced without re-running the full experiments:

- `data/experiment1/simulation_results_full.csv` — Raw Experiment 1 results
- `data/experiment2/experiment_2_summary.csv` — Experiment 2 summary
- `data/experiment2/per_fold/experiment_2_per_fold_results.csv` — Per-fold CV results
- `data/ablation_study/ablation_results.csv` — Ablation study results

## Installation

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt
```

## Reproducing Tables and Figures

All tables and figures can be regenerated from the pre-computed data files.
Generated output is written to `output/tables/`, `output/plots/`, and
`output/supplementary/`.

```bash
# Experiment 1 tables
python scripts/tables/tables_experiment_1.py

# Experiment 1 figures
python scripts/plot/plot_experiment_1.py

# Experiment 2 table
python scripts/tables/table_experiment_2.py

# Experiment 2 appendix tables
python scripts/tables/tables_experiment_2_appendix.py

# Experiment 2 figures
python scripts/plot/plot_experiment_2.py

# Ablation study analysis
python scripts/tables/analysis_ablation.py

# Kernel shape illustration
python scripts/plot/plot_kernel_shapes.py

# Supplementary CSV exports
python scripts/export/export_experiment_1.py
python scripts/export/export_experiment_2.py
```

## Re-running the Experiments

> **Note:** The full experiments are computationally intensive. Experiment 1 runs approximately 32 hours on a 32-core workstation. Experiment 2 uses true leave-one-out cross-validation and is also time-consuming.

```bash
# Experiment 1 (parallelized; adjust MAX_WORKERS in the script)
python scripts/run/run_experiment_1.py

# Experiment 2 (fetches UCI data automatically)
python scripts/run/run_experiment_2.py

# Ablation study (parallelized)
python scripts/run/run_ablation_study.py
```

Experiment 1 supports resumption: if interrupted, re-running the script will skip already completed trials.

## Reference

Chen, S. X. (1999). Beta kernel estimators for density functions. *Computational Statistics & Data Analysis*, 31(2), 131–145.