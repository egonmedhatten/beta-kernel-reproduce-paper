#!/usr/bin/env python3
"""Reproduce all tables, figures, and supplementary CSVs from pre-computed data.

Usage:
    python reproduce_all.py

This script runs every post-processing script (tables, plots, exports) and
writes output to ``output/tables/``, ``output/plots/``, and
``output/supplementary/``.  It does **not** re-run the experiments themselves;
see ``scripts/run/`` for that.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    # --- Tables ---
    ("Experiment 1 tables", "scripts/tables/tables_experiment_1.py"),
    ("Experiment 2 table", "scripts/tables/table_experiment_2.py"),
    ("Experiment 2 appendix tables", "scripts/tables/tables_experiment_2_appendix.py"),
    ("Ablation study table", "scripts/tables/table_ablation.py"),
    # --- Figures ---
    ("Experiment 1 figures", "scripts/plot/plot_experiment_1.py"),
    ("Experiment 2 figures", "scripts/plot/plot_experiment_2.py"),
    ("Ablation study figure", "scripts/plot/plot_ablation.py"),
    ("Kernel shape figure", "scripts/plot/plot_kernel_shapes.py"),
    # --- Supplementary CSVs ---
    ("Experiment 1 supplementary CSV", "scripts/export/export_experiment_1.py"),
    ("Experiment 2 supplementary CSV", "scripts/export/export_experiment_2.py"),
]


def main() -> int:
    failed = []

    for description, script in SCRIPTS:
        script_path = REPO_ROOT / script
        print(f"\n{'=' * 60}")
        print(f"  {description}")
        print(f"  {script}")
        print(f"{'=' * 60}")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO_ROOT),
        )

        if result.returncode != 0:
            print(f"  *** FAILED (exit code {result.returncode}) ***")
            failed.append(script)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if failed:
        print(f"  {len(SCRIPTS) - len(failed)}/{len(SCRIPTS)} scripts succeeded.")
        print(f"  Failed scripts:")
        for s in failed:
            print(f"    - {s}")
        print(f"{'=' * 60}")
        return 1

    print(f"  All {len(SCRIPTS)} scripts completed successfully.")
    print(f"  Output written to: {REPO_ROOT / 'output'}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
