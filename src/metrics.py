"""Person A — metrics (PLAN.md §7 Person A: `metrics.py`).

Computes the headline numbers for the demo panel:

* peak racket-head speed from angular velocity (v = |ω| × r),
* peak g-force experienced,
* peak rotation speed in RPM,
* swing duration relative to the impact moment,
* impact time / index for anchoring.
"""

from __future__ import annotations

import numpy as np

from . import config
from .swing import Swing


def swing_window_mask(gyro_deg: np.ndarray,
                      impact_idx: int,
                      frac: float = 0.15) -> np.ndarray:
    """Boolean mask of the "active swing" around the impact.

    Samples are included from the last crossing *above a fraction of the peak*
    gyro magnitude before impact, up to ``impact_idx`` itself.  This rejects
    the slow setup phase at the start of the recording.
    """
    gmag = np.linalg.norm(gyro_deg, axis=1)
    threshold = float(gmag.max()) * frac if gmag.max() > 0 else 1.0
    start = 0
    for i in range(impact_idx, -1, -1):
        if gmag[i] < threshold:
            break
        start = i
    start = max(0, start - 1)
    mask = np.zeros(len(gmag), dtype=bool)
    mask[start:impact_idx + 1] = True
    return mask


def compute_metrics(swing: Swing) -> dict:
    """Compute the headline telemetry for the dashboard.

    Returns
    -------
    dict
        Keys: ``peak_head_speed_mps``, ``peak_head_speed_kmh``,
        ``peak_gforce``, ``peak_rpm``, ``swing_duration_s``,
        ``impact_time_s``, ``impact_idx``.
    """
    swing.validate()

    gyro_deg = swing.gyro_deg
    impact_idx = swing.impact_idx
    mask = swing_window_mask(gyro_deg, impact_idx)
    if not mask.any():
        mask[impact_idx] = True

    # --- Angular speed around the swing window ---------------------------- #
    omega = np.deg2rad(gyro_deg[mask])               # rad/s
    ang_speed = np.linalg.norm(omega, axis=1)        # rad/s

    # Peak racket-head speed: v = |ω| × r  (r = tip length from pivot/wrist).
    peak_ang = float(ang_speed.max())
    head_speed = peak_ang * config.RACKET_TIP_LEN    # m/s

    # --- g-force peak ------------------------------------------------------ #
    # Unitless specific force = |a| / g, peak over the window.
    accel_mag = np.linalg.norm(swing.accel[mask], axis=1)
    peak_g = float(accel_mag.max() / config.G)

    # --- RPM --------------------------------------------------------------- #
    peak_rpm = peak_ang * 60.0 / (2.0 * np.pi)

    # --- Swing duration & impact time ------------------------------------- #
    swing_t = swing.t[mask]
    duration = float(swing_t.max() - swing_t.min()) if len(swing_t) > 1 else 0.0
    impact_t = float(swing.t[impact_idx])

    return {
        "peak_head_speed_mps": round(head_speed, 2),
        "peak_head_speed_kmh": round(head_speed * 3.6, 2),
        "peak_gforce": round(peak_g, 2),
        "peak_rpm": round(peak_rpm, 0),
        "swing_duration_s": round(duration, 3),
        "impact_time_s": round(impact_t, 3),
        "impact_idx": int(impact_idx),
    }
