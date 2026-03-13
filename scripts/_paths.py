"""Resolve repository root and configure import paths.

Every script under ``scripts/`` should begin with::

    from _paths import REPO_ROOT

This adds the repository root to ``sys.path`` so that ``KDE.py`` and
``KDE_Gauss.py`` can be imported directly, and provides ``REPO_ROOT`` as
a ``pathlib.Path`` for constructing absolute data / output paths.
"""

import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Ensure the repo root is on the import path so that
# ``from KDE import BetaKernelKDE`` works from any subdirectory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR: Path = REPO_ROOT / "data"
OUTPUT_DIR: Path = REPO_ROOT / "output"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
TABLES_DIR: Path = OUTPUT_DIR / "tables"
SUPPLEMENTARY_DIR: Path = OUTPUT_DIR / "supplementary"

for directory in [DATA_DIR, OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, SUPPLEMENTARY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)