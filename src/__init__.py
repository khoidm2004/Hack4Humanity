"""Person A + Person B — complete reconstruction & overlay package.

Public API:

  * Sensor core (Person A): load_csv, to_signal_arrays, reconstruct,
    compute_metrics, Swing
  * Video & overlay (Person B): video.probe/frame/map_imu_to_video,
    camera.calibrate/solve_pose/project_path/draw_overlay,
    plot.figure_3d/figure_signals
  * Overlay calibration: ``src.overlay_calib`` holds the committed
    racket-head annotations for ``swing_angle_1.mp4`` and solves the camera
    pose from them.  Import it directly (``from src import overlay_calib``);
    it is deliberately not pulled in here, so importing ``src`` does not read
    ``swing.npz`` or the sync tables.
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

