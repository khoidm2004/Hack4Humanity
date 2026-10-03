"""Person A + Person B — complete reconstruction & overlay package.

Public API:

  * Sensor core (Person A): load_csv, to_signal_arrays, reconstruct,
    compute_metrics, Swing
  * Video & overlay (Person B): video.probe/frame/map_imu_to_video,
    camera.calibrate/project_path/draw_overlay, plot.figure_3d/figure_signals
"""

from . import config
from .load import load_csv, to_signal_arrays
from .swing import Swing
from .fusion import reconstruct, estimate_gravity, integrate_orientation
from .metrics import compute_metrics

__all__ = [
    "config",
    "load_csv",
    "to_signal_arrays",
    "Swing",
    "reconstruct",
    "compute_metrics",
]

