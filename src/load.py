"""Person A — data loading (PLAN.md §7 Person A: `load.py`).

Reads the raw IMU CSV into a :class:`pandas.DataFrame`, adds a proper time
axis in seconds (``t``), validates that no NaN / ±Inf values are present and
exposes the accelerometer / gyroscope row arrays used by the fusion stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Expected column names coming out of the raw sensor dump.
ACCEL_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]

# These are always present in a valid IMU dump.
REQUIRED_COLS = ACCEL_COLS + GYRO_COLS


def load_csv(path: str | None = None) -> pd.DataFrame:
    """Read the raw IMU CSV, add the ``t`` (seconds) axis and validate it.

    Parameters
    ----------
    path : str, optional
        CSV location.  Defaults to :data:`src.config.CSV_PATH`.

    Returns
    -------
    pandas.DataFrame
        Frame with an integer ``index`` column, IMU columns
        (``ax, ay, az, gx, gy, gz``) and an additional ``t`` column in seconds.

    Raises
    ------
    ValueError
        If required columns are missing or the frame contains NaN / ±Inf.
    """
    path = config.CSV_PATH if path is None else str(path)

    df = pd.read_csv(path)

    # Drop a stray "index" column if the device already wrote one, so we do
    # not shadow the signal columns.
    frame: pd.DataFrame
    if "index" in df.columns and "index" not in REQUIRED_COLS:
        frame = df.drop(columns=["index"])
    else:
        frame = df

    missing = [c for c in REQUIRED_COLS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"CSV '{path}' is missing required IMU columns: {missing}"
        )

    # Time axis in seconds: sample n happens at n * dt.
    n = len(frame)
    frame["t"] = np.arange(n) / config.FS

    # ---- Validation: no missing or infinite values ------------------------ #
    signal = frame[REQUIRED_COLS].to_numpy(dtype=float)
    if np.isnan(signal).any():
        raise ValueError(f"CSV '{path}' contains NaN values (not allowed).")
    if np.isinf(signal).any():
        raise ValueError(f"CSV '{path}' contains infinite values (not allowed).")

    return frame


def to_signal_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract the accelerometer and gyroscope arrays from a loaded frame.

    The CSV's accelerometer columns are in **g** (`config.ACCEL_UNITS`): `ax`
    hard-clips at 15.961504 for samples 181-199, the +/-16 g full scale of the
    part.  Every consumer's docstring says m/s2, so the conversion happens
    here, once, at the boundary.  Nothing used to convert, which made
    `fusion.rest_to_rest_cog` double-integrate numbers 9.80665x too small
    (`cog` span 0.161 m instead of 1.63 m) and made `metrics.peak_gforce`
    divide g by g.  `data/raw_data.csv` is NOT modified.

    This does not change `tip` or `quat`: `fusion.estimate_gravity` normalises
    its result to `config.G` regardless of the input scale, and the
    complementary-filter branch that would see the magnitude is off
    (`COMP_FILTER_ALPHA = 0.0`).  Verified: max |tip| difference 4.4e-16 m.

    Returns
    -------
    accel : (N, 3) float array, accelerometer in m/s2
    gyro  : (N, 3) float array, gyroscope in deg/s
    """
    accel = df[ACCEL_COLS].to_numpy(dtype=float)
    if config.ACCEL_UNITS == "g":
        accel = accel * config.G
    gyro = df[GYRO_COLS].to_numpy(dtype=float)
    return accel, gyro
