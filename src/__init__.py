"""Person A — sensor-fusion core package.

Exposed top-level API (used by scripts and Person B):

    load
    swing.Swing
    fusion.reconstruct
    metrics.compute_metrics
"""

from . import config
from .load import load_csv, to_signal_arrays
from .swing import Swing
from .fusion import reconstruct
from .metrics import compute_metrics

__all__ = [
    "config",
    "load_csv",
    "to_signal_arrays",
    "Swing",
    "reconstruct",
    "compute_metrics",
]
