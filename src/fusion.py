"""Person A — sensor fusion core (PLAN.md §7 Person A: `fusion.py`).

Implements the full IMU → 3D reconstruction chain:

1. Gravity reference estimation (accelerometer, low-motion window).
2. Orientation from gyroscope integration (quaternion) with a
   complementary filter that nudges roll/pitch toward the accel-measured
   gravity (SciPy ``Rotation``).
3. Pivot model translation: ``tip = pivot + R(t) @ (L * RACKET_LEVER_BODY)``
   (MAIN overlay).
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

    There is no static segment anywhere in this record: samples 0-24 carry
    |gyro| = 113 deg/s and |accel| = 7.48 m/s2 (0.76 g), and the quietest
    25-sample window in all 400 IS s0-24, at the same 113 deg/s
    (Artifacts/analysis.md §6).  So this is the least-moving window, not a
    static one, normalised to the expected magnitude `g`.  With
    `COMP_FILTER_ALPHA = 0.0` its only effect on `quat`/`tip` is the initial
    orientation `q0`, a pure LEFT multiplication, so it rotates the whole
    path rigidly.  Gyro bias removal was also tried and REJECTED: subtracting
    the s0-24 mean improves the fit-13 median to 17.0 px but degrades the
    held-out median to 329 px — that "bias" is the takeback, not a bias.

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
        Must be ``0.0`` — pure gyro integration.  The complementary-filter
        branch that used to accept a nonzero value here was DELETED, not
        fixed: see the `NotImplementedError` raised below and
        `config.COMP_FILTER_ALPHA`.

    Returns
    -------
    (N, 4) float — world orientation as w,x,y,z quaternions.
    """
    if alpha != 0.0:
        raise NotImplementedError(
            "COMP_FILTER_ALPHA must be 0.0. The complementary-filter branch "
            "that used to live here was DELETED, not fixed: it ended "
            "`rot_accum = _rotation_aligning(g_sensor, g_world)`, which "
            "replaced the integrated attitude with a gravity-only one and "
            "carried ZERO heading (measured: seed 137 deg of heading, run the "
            "branch, the component about gravity comes out 0.0000 deg). It "
            "was also worse end to end at every alpha tried. See "
            "config.COMP_FILTER_ALPHA and Artifacts/analysis.md §6."
        )
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

    # `_align_to_gravity` returns the contract's (w,x,y,z) order (it reorders
    # scipy's own output at :126), but `Rotation.from_quat` reads (x,y,z,w).
    # Feeding it the (w,x,y,z) array scrambled the initial orientation: the
    # intended R0 maps g_sensor onto [0,0,-1], the scrambled one mapped it onto
    # [-0.075, 0.179, 0.981] — 171.8 deg out.  scipy renormalises, so there was
    # no error to notice.  Reorder back to scipy's convention here.
    #
    # Because the loop below right-multiplies (`rot_accum * dq`), R(t) =
    # R0 . dR(t) and q0 is a pure LEFT multiplication: fixing it rotates the
    # whole path rigidly and changes `tip`'s arc length and tangent turning by
    # exactly zero (measured: both identical to 15 significant figures; max
    # |tip| difference 1.269 m, a constant 171.8 deg rotation).  It is a real
    # defect and is fixed on its own merits, not to move the arc — the arc is
    # a time-base problem, see FUSION_NOTES.md.
    rot_accum = Rotation.from_quat(q0[[1, 2, 3, 0]])
    for i in range(len(gyro_deg)):
        dq = Rotation.from_rotvec(omega[i])
        rot_accum = rot_accum * dq  # right-multiply: increments in the BODY
        # frame, which is correct for a body-mounted gyro.  The old comment
        # said "world frame", which is what left-multiplication would do;
        # composing that way measures 5.752 m / 871 deg against the body
        # frame's 5.113 m / 806 deg, i.e. the comment described the worse
        # option.  Code was right, comment was wrong.

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

    ``tip(t) = pivot + R(t) @ (RACKET_TIP_LEN * RACKET_LEVER_BODY)``.

    The lever direction is `config.RACKET_LEVER_BODY`, a MEASURED unit vector
    in sensor axes, currently -x.  Until 2026-10-05 this function hard-coded
    `[0, 0, L]` — the racket along the sensor's +z — which was an unverified
    assumption inherited from PLAN.md, not a measurement.  The accelerometer's
    own centripetal signal and a held-out video grid search independently put
    the racket ~90 deg away from that, along -x; see `config.RACKET_LEVER_BODY`
    and Artifacts/analysis.md §7.  The consequence of the wrong axis was a
    mean racket image-angle error of 116.7 deg across the follow-through
    (f146-220), against 9.1 deg once corrected.

    This returns the lever ONLY — the head relative to the pivot — so
    `|tip| == RACKET_TIP_LEN` for every sample and the path it traces is a
    sphere of that radius.  Until 2026-10-05 that WAS the whole model: the
    wrist sat at the world origin and the ~0.91 m it really travels across the
    annotated window was simply absent, which PnP could only absorb by pulling
    the camera to 55 % of the player's measured distance (|tvec| 2.400 m
    against the L-free 4.075 m anchor).  The translation now lives in
    `Swing.pivot` (`wrist_pivot` below) and the head is `Swing.head =
    pivot + tip`.  Keeping them separate is what preserves this radius
    invariant and the two lever-direction checks in
    `scripts/verify_fusion.py`.

    Parameters
    ----------
    quat : (N, 4) float, w,x,y,z

    Returns
    -------
    (N, 3) float, metres — |tip| == RACKET_TIP_LEN for every sample.
    """
    rots_yzx = quat[:, [1, 2, 3, 0]]          # back to (x,y,z,w)
    R = Rotation.from_quat(rots_yzx)
    local = np.asarray(config.RACKET_LEVER_BODY, float) * config.RACKET_TIP_LEN
    return R.apply(local)                     # (N, 3)


def wrist_pivot(cog: np.ndarray,
                scale: float = config.WRIST_PIVOT_COG_SCALE,
                removed_slope: np.ndarray | None = None,
                drift_gain: float = config.WRIST_PIVOT_DRIFT_GAIN,
                drift_pivot_sample: int = config.WRIST_PIVOT_DRIFT_PIVOT_SAMPLE,
                ) -> np.ndarray:
    """The wrist (pivot) path: ``scale * cog`` plus a restored linear drift.

    ``pivot(t) = k*cog(t) + b*(t - WRIST_PIVOT_DRIFT_PIVOT_SAMPLE)``, with
    ``b = k * drift_gain * removed_slope``.

    `pivot_tip` gives the racket head RELATIVE to the wrist; this gives where
    the wrist itself is, so the head in world space is `pivot + tip`
    (`Swing.head`).  `rest_to_rest_cog` is the only wrist-motion estimate the
    repo has from the data alone, and it correlates +0.914 (u) / +0.837 (v)
    with the annotated wrist pixels (Artifacts/analysis_followthrough.md §7.5).
    `scale` is a calibration, not a physical fraction — `cog` is detrended
    about the origin, so its amplitude is set by the detrending.  See
    `config.WRIST_PIVOT_COG_SCALE` for how 0.907 was measured and why the
    held-out reads were not used to pick it.

    `removed_slope` (from `rest_to_rest_cog(..., return_removed_slope=True)`
    or `wrist_drift_slope`) is the linear position trend
    `_detrend_rest_to_rest` subtracted from the double-integrated wrist
    path — found to be mostly REAL displacement, not integration error
    (findings/FUSION_NOTES.md §11). Passing `removed_slope=None` (the
    default) reproduces the pre-drift-term behaviour exactly — this is a
    drift *restoration*, not a new calibration.

    CRITICAL INVARIANT: the drift term goes in `pivot`, never in `tip`.
    Folding it into `tip` would break `|tip| == RACKET_TIP_LEN`, which both
    of `verify_fusion.py`'s lever-direction checks and
    `racket_pixel_scale`'s L-cancellation argument depend on (see
    `src/swing.py:48-55` and `findings/FUSION_NOTES.md` §10.3).

    Parameters
    ----------
    cog : (N, 3) float, metres — `rest_to_rest_cog` output.
    scale : float — `config.WRIST_PIVOT_COG_SCALE`.
    removed_slope : (3,) float or None — m/sample, from `rest_to_rest_cog`.
    drift_gain : float — `config.WRIST_PIVOT_DRIFT_GAIN`.
    drift_pivot_sample : int — `config.WRIST_PIVOT_DRIFT_PIVOT_SAMPLE`.

    Returns
    -------
    (N, 3) float, metres.
    """
    c = np.asarray(cog, float)
    pivot = float(scale) * c
    if removed_slope is None or float(drift_gain) == 0.0:
        return pivot
    b = float(scale) * float(drift_gain) * np.asarray(removed_slope, float)
    s = np.arange(len(c), dtype=float) - float(drift_pivot_sample)
    return pivot + np.outer(s, b)


def fit_centripetal_lever(accel: np.ndarray,
                          gyro_deg: np.ndarray,
                          lo: int = 205,
                          hi: int | None = None,
                          omega_min_dps: float = 0.0,
                          ) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Fit the sensor's body-frame lever from its own centripetal signal.

    A sensor at body-frame position ``r`` from the rotation centre reads a
    specific force ``f = (w w^T - |w|^2 I) r + b``, which is LINEAR in ``r``
    — so ``r`` comes straight out of a least-squares fit, with no camera, no
    annotations and no fusion output involved.  ``b`` absorbs gravity and
    bias.  This is the measurement behind `config.RACKET_LEVER_BODY`, and
    `scripts/verify_fusion.py` turns it into a regression check that the
    configured lever still agrees with the data (Artifacts/analysis.md §7.1).

    The angular-acceleration term ``alpha x r`` is deliberately NOT modelled:
    the fit is used as an AXIS estimate with a 40 deg tolerance, and three
    independent windows already agree to within 20 deg without it.

    Samples where any accel axis has hard-clipped at the part's +/-16 g full
    scale are excluded (`ax` sits on the rail for samples 182-199).

    Parameters
    ----------
    accel : (N, 3) float, m/s2 — RAW, as `load.to_signal_arrays` returns it.
    gyro_deg : (N, 3) float, deg/s — RAW.
    lo, hi : int — half-open sample window [lo, hi).  The default starts at
        205 because that window is entirely clear of the clip plateau and
        carries the highest |w|.
    omega_min_dps : float — optional |w| floor, deg/s.

    Returns
    -------
    (unit_dir, radius_m, bias, n_used)
        ``unit_dir`` is ``r / |r|``; ``radius_m`` is ``|r|``.
    """
    a = np.asarray(accel, float)
    w = np.deg2rad(np.asarray(gyro_deg, float))
    n = len(a)
    hi = n if hi is None else int(hi)
    rail = config.ACCEL_FULL_SCALE_G - 0.06          # 15.94 g
    unclipped = ~((np.abs(a) / config.G) >= rail).any(axis=1)
    idx = np.arange(n)
    sel = (unclipped & (idx >= int(lo)) & (idx < hi)
           & (np.linalg.norm(w, axis=1) >= np.deg2rad(omega_min_dps)))
    k = np.flatnonzero(sel)
    if len(k) < 20:
        raise ValueError(f"centripetal fit needs >=20 usable samples, got {len(k)}")
    eye = np.eye(3)
    rows = [np.hstack([np.outer(w[i], w[i]) - float(w[i] @ w[i]) * eye, eye])
            for i in k]
    A = np.vstack(rows)                  # (3m, 6)
    y = a[k].reshape(-1)                 # (3m,) — C-order matches the row blocks
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = x[:3]
    norm = float(np.linalg.norm(r))
    if norm < 1e-9:
        raise ValueError("centripetal fit returned a degenerate radius")
    return r / norm, norm, x[3:], int(len(k))


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
        Filled contract record (validated).  ``gyro_deg``/``accel`` inside the
        record are the RAW signals (for plotting/metrics); the gesture
        low-pass (racket-string ringing removal) is applied to the fusion
        inputs only.
    """
    # Gesture-band low-pass: the strings ring at ~165 Hz after impact and that
    # is not hand motion — keep only the actual swing shape (< ~25 Hz).
    accel_lp = _lowpass(accel)
    gyro_lp = _lowpass(gyro_deg)

    gravity = estimate_gravity(accel_lp)
    quat = integrate_orientation(accel_lp, gyro_lp, gravity, alpha=alpha)
    tip = pivot_tip(quat)
    cog, removed_slope = rest_to_rest_cog(accel_lp, quat, gravity,
                                          return_removed_slope=True)
    pivot = wrist_pivot(cog, removed_slope=removed_slope)
    impact_idx = detect_impact(accel_lp, gyro_lp)

    t = np.arange(len(accel), dtype=float) * config.DT
    swing = Swing(t=t, quat=quat, tip=tip, cog=cog,
                  gyro_deg=gyro_deg, accel=accel, impact_idx=impact_idx,
                  pivot=pivot)
    swing.validate()
    return swing

def _highpass(x: np.ndarray, cutoff_hz: float = config.R2R_HP_CUTOFF_HZ) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth high-pass filter along axis=0."""
    nyq = 0.5 * config.FS
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return x
    sos = butter(2, cutoff_hz / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _lowpass(x: np.ndarray, cutoff_hz: float = config.GESTURE_LP_HZ) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass (gesture band).

    Removes the racket-string ringing (~165 Hz after ball impact, measured in
    ``ay``/``az``) while keeping the swing shape (all gesture energy is below
    ~25 Hz).  Applied to the *fusion inputs* only — the raw signals stay
    untouched in the ``Swing`` record for plotting and metrics.
    """
    nyq = 0.5 * config.FS
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return x
    sos = butter(4, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _linear_slope(v: np.ndarray) -> np.ndarray:
    """Per-channel least-squares slope of ``v`` against sample index."""
    t = np.arange(len(v), dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    (slope, _), *_ = np.linalg.lstsq(A, v, rcond=None)
    return np.asarray(slope, float)


def _detrend_rest_to_rest(v: np.ndarray) -> np.ndarray:
    """Remove a linear trend so the integrated path is rest-to-rest (v0=vN=0).

    Returned array has the same mean but zero best-fit linear slope.  The
    slope it removes is NOT integration error alone: over this 400-sample
    record it carries 1.404 m of real position displacement against a
    surviving path of 0.782 m (findings/FUSION_NOTES.md §11).
    `rest_to_rest_cog(..., return_removed_slope=True)` hands it back so
    `wrist_pivot` can give the real part of it back to the pivot.
    """
    t = np.arange(len(v), dtype=float)
    return v - np.outer(t, _linear_slope(v))


def rest_to_rest_cog(accel: np.ndarray, quat: np.ndarray,
                     gravity: np.ndarray, *,
                     return_removed_slope: bool = False):
    """Sensor position (COG path) via gravity-free double integration.

    Strategy (PLAN.md §8):
      1. rotate accel into world frame,
      2. high-pass to remove gravity + sensor bias,
      3. integrate -> velocity, detrend (rest-to-rest velocity so v0=vN=0),
      4. integrate -> position, detrend to anchor the start at the origin.

    Parameters
    ----------
    return_removed_slope : bool
        When False (default), returns ``cog`` alone, bit-identical to this
        function's pre-drift-term behaviour. When True, returns
        ``(cog, slope)`` where ``slope`` is the per-channel linear slope
        (m/sample) that the final detrend step subtracted from the
        (otherwise identical) position path — the quantity
        `wrist_pivot`'s drift term is derived from.
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
    slope = _linear_slope(pos)
    cog = pos - np.outer(np.arange(len(pos), dtype=float), slope)
    if return_removed_slope:
        return cog, slope
    return cog


def wrist_drift_slope(accel: np.ndarray, gyro_deg: np.ndarray,
                      alpha: float = config.COMP_FILTER_ALPHA) -> np.ndarray:
    """The position slope `_detrend_rest_to_rest` removes, m/sample, from RAW inputs.

    One-call entry point for the scripts: it runs exactly the same preamble
    `reconstruct` does, so the slope it returns is the one the committed
    `Swing.pivot` was built with.  Measured on `data/raw_data.csv`:
    slope * 400 = [-1.2967, +0.1899, -0.5044] m, |.| = 1.4042 m.
    """
    accel_lp = _lowpass(accel)
    gyro_lp = _lowpass(gyro_deg)
    gravity = estimate_gravity(accel_lp)
    quat = integrate_orientation(accel_lp, gyro_lp, gravity, alpha=alpha)
    _cog, slope = rest_to_rest_cog(accel_lp, quat, gravity,
                                   return_removed_slope=True)
    return slope

