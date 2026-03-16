# Supplementary Code: A Fast, Closed-Form Bandwidth Selector for the Beta Kernel Density Estimator

This repository contains the code and data to reproduce all experiments, tables,
and figures in the paper *"A Fast, Closed-Form Bandwidth Selector for the Beta
Kernel Density Estimator"*, submitted to the *Journal of Computational and
Graphical Statistics*.

## Quick Start

Reproducing every table, figure, and supplementary CSV from the pre-computed
data takes under a minute:

```bash
# 1. Clone the repository
git clone <repository-url>
cd beta-kernel-reproduce-paper

# 2. Install dependencies (pick one)
uv sync                          # recommended — uses the exact lockfile
# or: pip install -r requirements.txt

# 3. Reproduce all outputs
uv run python reproduce_all.py   # if using uv
# or: python reproduce_all.py
```

All generated files are written to `output/` (tables, plots, and supplementary
CSVs). On a typical laptop the script finishes in under a minute.

> **Note:** The Experiment 2 plotting script (`scripts/plot/plot_experiment_2.py`)
> downloads the UCI Communities and Crime dataset at runtime, so an internet
> connection is required when that script is first executed.

## Installation

Requires **Python >= 3.11**.

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. The
repository ships a `uv.lock` lockfile that pins every dependency to the exact
versions used during development.

```bash
uv sync
```

### Using pip

A traditional `requirements.txt` is also provided:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── KDE.py                          # Beta kernel density estimator
├── KDE_Gauss.py                    # Gaussian KDE with boundary corrections
├── reproduce_all.py                # One-command reproduction of all outputs
├── pyproject.toml                  # Project metadata & dependencies (uv / pip)
├── requirements.txt                # Pinned dependencies (pip fallback)
├── uv.lock                        # Exact dependency lockfile (uv)
├── scripts/
│   ├── _paths.py                   # Root-path resolution & output-dir creation
│   ├── run/                        # Experiment scripts (computationally intensive)
│   │   ├── run_experiment_1.py     #   Monte Carlo simulation (Experiment 1)
│   │   ├── run_experiment_2.py     #   Real-data cross-validation (Experiment 2)
│   │   └── run_ablation_study.py   #   Ablation study
│   ├── tables/                     # LaTeX table generation
│   │   ├── tables_experiment_1.py
│   │   ├── table_experiment_2.py
│   │   ├── tables_experiment_2_appendix.py
│   │   └── table_ablation.py
│   ├── plot/                       # Figure generation
│   │   ├── _plot_styles.py         #   Shared colour palette & theme
│   │   ├── plot_experiment_1.py
│   │   ├── plot_experiment_2.py
│   │   ├── plot_ablation.py
│   │   └── plot_kernel_shapes.py
│   └── export/                     # Supplementary CSV export
│       ├── export_experiment_1.py
│       └── export_experiment_2.py
├── data/                           # Pre-computed experiment results (committed)
│   ├── experiment1/
│   ├── experiment2/
│   └── ablation_study/
└── output/                         # Generated artifacts (created automatically)
    ├── tables/
    ├── plots/
    └── supplementary/
```

## Pre-computed Data

The `data/` directory contains pre-computed results so that all tables and
figures can be reproduced without re-running the experiments:

| File | Description |
|------|-------------|
| `data/experiment1/simulation_results_full.csv` | Raw Experiment 1 simulation results |
| `data/experiment2/experiment_2_summary.csv` | Experiment 2 summary statistics |
| `data/experiment2/per_fold/experiment_2_per_fold_results.csv` | Experiment 2 per-fold CV results |
| `data/ablation_study/ablation_results.csv` | Ablation study results |

## Reproducing Tables and Figures

### All at once

```bash
python reproduce_all.py
```

This runs every table, plot, and export script and writes output to
`output/tables/`, `output/plots/`, and `output/supplementary/`.
The `output/` directory is created automatically.

### Individual scripts

```bash
# --- Tables ---
python scripts/tables/tables_experiment_1.py          # Experiment 1 (main + appendix)
python scripts/tables/table_experiment_2.py            # Experiment 2
python scripts/tables/tables_experiment_2_appendix.py  # Experiment 2 appendix
python scripts/tables/table_ablation.py                # Ablation study

# --- Figures ---
python scripts/plot/plot_experiment_1.py               # Experiment 1 figures
python scripts/plot/plot_experiment_2.py               # Experiment 2 density fits
python scripts/plot/plot_ablation.py                   # Ablation Δ-LSCV plot
python scripts/plot/plot_kernel_shapes.py              # Kernel shape illustration

# --- Supplementary CSVs ---
python scripts/export/export_experiment_1.py
python scripts/export/export_experiment_2.py
```

## Re-running the Experiments (Optional)

> **Warning:** The full experiment pipeline is computationally intensive.
> Experiment 1 takes approximately 28 hours on a 32-core workstation.
> Pre-computed results are already provided in `data/`, so re-running is
> not required to reproduce the paper's tables and figures.

```bash
# Experiment 1 — Monte Carlo simulation (parallelised; set MAX_WORKERS in script)
python scripts/run/run_experiment_1.py

# Experiment 2 — real-data cross-validation (downloads UCI data automatically)
python scripts/run/run_experiment_2.py

# Ablation study (parallelised)
python scripts/run/run_ablation_study.py
```

Experiment 1 supports **resumption**: if interrupted, re-running the script
skips already completed trials.
