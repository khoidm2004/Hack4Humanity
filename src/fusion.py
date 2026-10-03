"""Person A — sensor fusion core (PLAN.md §7 Person A: `fusion.py`).

Implements the full IMU → 3D reconstruction chain:

1. Gravity reference estimation (accelerometer, low-motion window).
2. Orientation from gyroscope integration (quaternion) with a
   complementary filter that nudges roll/pitch toward the accel-measured
   gravity (SciPy ``Rotation``).
3. Pivot model translation: ``tip = pivot + R(t) @ [0, 0, L]``  (MAIN overlay).
4. Rest-to-rest translation: double integration of gravity-free accel with
   drift correction (high-pass + detrend) — the "from data" proof.
5. Impact (ball contact) detection via |gyro| peak + jerk spike.

Every function is unit-tested by ``scripts/verify_fusion.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.spatial.transform import Rotation

from . import config
from .swing import Swing


# --------------------------------------------------------------------------- #
# 1. Gravity reference
# --------------------------------------------------------------------------- #
def estimate_gravity(accel: np.ndarray) -> np.ndarray:
    """Estimate the sensor-frame gravity vector.

    With no guaranteed static segment we take the mean of the early, most
    static samples and normalise it to the expected magnitude ``g``.

    Parameters
    ----------
    accel : (N, 3) float, m/s2

    Returns
    -------
    (3,) float array — normalised gravity in sensor axes.
    """
    ref = accel[: config.GRAVITY_REF_SAMPLES].mean(axis=0)
    norm = np.linalg.norm(ref)
    if norm < 1e-6:  # degenerate — fall back to +z
        return np.array([0.0, 0.0, config.G])
    return ref * (config.G / norm)


# --------------------------------------------------------------------------- #
# 2. Orientation
# --------------------------------------------------------------------------- #
def _complementary_gravity(dir_world: np.ndarray, expected: np.ndarray,
                           alpha: float) -> np.ndarray:
    """Blend the world-space measured gravity direction toward the ideal
    ``[0, 0, -1]`` by rotating by ``alpha`` fraction of the angular error."""
    dir_world = dir_world / (np.linalg.norm(dir_world) + 1e-9)
    expected = expected / (np.linalg.norm(expected) + 1e-9)
    # --- rotation that takes dir_world -> expected ------------------------- #
    cross = np.cross(dir_world, expected)
    dot = np.clip(np.dot(dir_world, expected), -1.0, 1.0)
    axis = cross / (np.linalg.norm(cross) + 1e-9)
    angle = np.arccos(dot)
    corr = Rotation.from_rotvec(axis * (angle * alpha))
    return corr.apply(dir_world)


def integrate_orientation(accel: np.ndarray,
                          gyro_deg: np.ndarray,
                          gravity: np.ndarray,
                          alpha: float = config.COMP_FILTER_ALPHA,
                          ) -> np.ndarray:
    """Integrate the gyroscope into world quaternions (w,x,y,z).

    Parameters
    ----------
    accel : (N, 3) float, m/s2 — a low-pass filtered copy should be used to
        drive the complementary correction.
    gyro_deg : (N, 3) float, deg/s
    gravity : (3,) float, sensor-frame gravity estimate
    alpha : float
        Complementary weight applied to the accel-based roll/pitch correction.
        ``0.0`` == pure gyro integration.

    Returns
    -------
    (N, 4) float — world orientation as w,x,y,z quaternions.
    """
    # Initialize world orientation so sensor gravity maps to world "down".
    g_sensor = gravity / (np.linalg.norm(gravity) + 1e-9)
    q0 = _align_to_gravity(g_sensor)

    quats = np.empty((len(gyro_deg), 4))
    omega = np.deg2rad(gyro_deg) * config.DT  # incremental rotation vectors
    # Smooth accel for the tilt correction (removes linear-accel spikes).
    try:
        accel_smooth = savgol_filter(accel, config.R2R_SAVGOL_WINDOW, polyorder=2,
                                     axis=0, mode="interp")
    except Exception:
        accel_smooth = accel

    rot_accum = Rotation.from_quat(q0)
    for i in range(len(gyro_deg)):
        dq = Rotation.from_rotvec(omega[i])
        rot_accum = rot_accum * dq  # increments in world frame

        if alpha > 0.0:
            # Measured gravity direction in world frame = R @ g_sensor.
            g_world = rot_accum.apply(gravity)
            g_world = _complementary_gravity(g_world, np.array([0.0, 0.0, -config.G]),
                                             alpha)
            # Recover the rotation that aligns sensor gravity with this
            # corrected world gravity.
            rot_accum = _rotation_aligning(g_sensor, g_world)

        quats[i] = rot_accum.as_quat()  # (x,y,z,w) -> convert below

    # scipy as_quat order is (x,y,z,w); contract stores (w,x,y,z).
    return quats[:, [3, 0, 1, 2]]


def _align_to_gravity(g_sensor: np.ndarray) -> np.ndarray:
    """Initial orientation (w,x,y,z) mapping sensor gravity onto [0,0,-1]."""
    target = np.array([0.0, 0.0, -1.0])
    return _rotation_aligning(g_sensor, target).as_quat()[np.array([3, 0, 1, 2])]


def _rotation_aligning(vec_a: np.ndarray, vec_b: np.ndarray) -> Rotation:
    """Smallest rotation taking ``vec_a`` onto ``vec_b``."""
    a = vec_a / (np.linalg.norm(vec_a) + 1e-9)
    b = vec_b / (np.linalg.norm(vec_b) + 1e-9)
    axis = np.cross(a, b)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return Rotation.identity()
    axis = axis / n
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle)


# --------------------------------------------------------------------------- #
# 3. Pivot model — racket head tip
# --------------------------------------------------------------------------- #
def pivot_tip(quat: np.ndarray) -> np.ndarray:
    """Racket head-tip position in world space using the pivot (wrist) model.

    ``tip(t) = pivot + R(t) @ [0, 0, L]``.  The wrist/pivot is held fixed at
    the origin (the camera calibration handles its world offset), so the path
    traces the fixed-|radius| spherical surface — this is the overlay that
    always keeps the correct shape.

    Parameters
    ----------
    quat : (N, 4) float, w,x,y,z

    Returns
    -------
    (N, 3) float, metres
    """
    rots_yzx = quat[:, [1, 2, 3, 0]]          # back to (x,y,z,w)
    R = Rotation.from_quat(rots_yzx)
    local = np.array([0.0, 0.0, config.RACKET_TIP_LEN])
    return R.apply(local)                     # (N, 3)


# --------------------------------------------------------------------------- #
# 5. Impact detection
# --------------------------------------------------------------------------- #
def detect_impact(accel: np.ndarray, gyro_deg: np.ndarray) -> int:
    """Detect ball-contact (impact) sample index.

    Strategy: the contact happens at the moment the racket head reaches its
    peak angular speed, which corresponds to the largest |gyro| spike.  We
    refine that gyro-based candidate by searching for the largest jerk
    (|d(accel)/dt|) spike in its neighbourhood.

    Parameters
    ----------
    accel : (N, 3) float, m/s2
    gyro_deg : (N, 3) float, deg/s

    Returns
    -------
    int
        Sample index of the impact.
    """
    n = len(gyro_deg)
    gmag = np.linalg.norm(gyro_deg, axis=1)
    peak = float(gmag.max())
    target = peak * config.IMPACT_GYRO_PEAK_FRACTION

    # Latest sample reaching a large fraction of the gyro peak.
    gyro_candidates = np.where(gmag >= target)[0]
    gyro_candidate = int(gyro_candidates[-1]) if len(gyro_candidates) else int(
        np.argmax(gmag)
    )

    # Jerk magnitude around the gyro candidate.
    jerk = np.linalg.norm(np.diff(accel, axis=0), axis=1) / config.DT
    lo = max(0, gyro_candidate - config.IMPACT_JERK_SEARCH_WIDTH)
    hi = min(max(n - 1, 0), gyro_candidate + config.IMPACT_JERK_SEARCH_WIDTH)
    window = jerk[lo:hi + 1]
    if len(window) == 0:
        return gyro_candidate
    return int(lo + int(np.argmax(window)))


# --------------------------------------------------------------------------- #
# Top-level reconstruction
# --------------------------------------------------------------------------- #
def reconstruct(accel: np.ndarray,
                gyro_deg: np.ndarray,
                alpha: float = config.COMP_FILTER_ALPHA) -> Swing:
    """Run the whole fusion chain and build a :class:`Swing` record.

    Parameters
    ----------
    accel : (N, 3) float, m/s2
    gyro_deg : (N, 3) float, deg/s
    alpha : float
        Complementary-filter weight (0.0 == pure gyro).

    Returns
    -------
    Swing
        Filled contract record (validated).
    """
    gravity = estimate_gravity(accel)
    quat = integrate_orientation(accel, gyro_deg, gravity, alpha=alpha)
    tip = pivot_tip(quat)
    cog = rest_to_rest_cog(accel, quat, gravity)
    impact_idx = detect_impact(accel, gyro_deg)

    t = np.arange(len(accel), dtype=float) * config.DT
    swing = Swing(t=t, quat=quat, tip=tip, cog=cog,
                  gyro_deg=gyro_deg, accel=accel, impact_idx=impact_idx)
    swing.validate()
    return swing

def _highpass(x: np.ndarray, cutoff_hz: float = config.R2R_HP_CUTOFF_HZ) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth high-pass filter along axis=0."""
    nyq = 0.5 * config.FS
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return x
    sos = butter(2, cutoff_hz / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _detrend_rest_to_rest(v: np.ndarray) -> np.ndarray:
    """Remove a linear trend so the integrated path is rest-to-rest (v0=vN=0).

    Returned array has the same mean but zero best-fit linear slope.
    """
    t = np.arange(len(v), dtype=float)
    # Least-squares linear fit per channel.
    A = np.vstack([t, np.ones_like(t)]).T
    (slope, _), *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - np.outer(t, slope)


def rest_to_rest_cog(accel: np.ndarray, quat: np.ndarray,
                     gravity: np.ndarray) -> np.ndarray:
    """Sensor position (COG path) via gravity-free double integration.

    Strategy (PLAN.md §8):
      1. rotate accel into world frame,
      2. high-pass to remove gravity + sensor bias,
      3. integrate -> velocity, detrend (rest-to-rest velocity so v0=vN=0),
      4. integrate -> position, detrend to anchor the start at the origin.
    """
    rots = Rotation.from_quat(quat[:, [1, 2, 3, 0]])

    # 1. Specific force in world frame (a_world = R @ a_sensor).
    accel_world = rots.apply(accel)

    # 2. Strip the (near-constant in world frame) gravity via high-pass.
    lin_accel = _highpass(accel_world)

    # 3. Integrate -> velocity, one linear detrend: rest-to-rest v0=vN≈0.
    vel = np.cumsum(lin_accel, axis=0) * config.DT
    vel = _detrend_rest_to_rest(vel)

    # Smooth velocity before the second integration to limit position noise.
    try:
        vel = savgol_filter(vel, config.R2R_SAVGOL_WINDOW, polyorder=2, axis=0,
                            mode="interp")
    except Exception:
        pass

    # 4. Integrate -> position, detrend to center the path around the origin.
    pos = np.cumsum(vel, axis=0) * config.DT
    return _detrend_rest_to_rest(pos)

