"""Test configuration.

PyTorch and XGBoost each bundle their own OpenMP runtime; on macOS, loading both in one
Python process can crash it (segmentation fault). The PyTorch tests therefore run in a
separate process:

    uv run pytest                                      # all tests except PyTorch ones
    SWISSDELAY_TORCH_TESTS=1 uv run pytest             # only the PyTorch tests

(``make test`` runs both, one after the other.)
"""

import os
from pathlib import Path

TORCH_TESTS = ["test_graph_transformer.py"]
_HERE = Path(__file__).parent

if os.environ.get("SWISSDELAY_TORCH_TESTS"):
    collect_ignore = [p.name for p in _HERE.glob("test_*.py") if p.name not in TORCH_TESTS]
else:
    collect_ignore = list(TORCH_TESTS)
