"""Central configuration for the racket-swing reconstruction pipeline.

Everything that might change between datasets / devices lives here so the rest
of the code can stay algorithm-focused.  See doc/PLAN.md (section 6.2) for the
mapping of each setting to its file.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "raw_data.csv"
OUTPUTS_DIR = DATA_DIR / "outputs"
SWING_NPZ = OUTPUTS_DIR / "swing.npz"
VIDEO_DIR = DATA_DIR / "video"

# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
FS = 416                     # Hz — IMU sampling rate (confirmed in PLAN.md §2)
DT = 1.0 / FS                # seconds per sample

# --------------------------------------------------------------------------- #
# Sensor units
# --------------------------------------------------------------------------- #
# `data/raw_data.csv` is in **g**, not m/s2: `ax` hard-clips at 15.961504 for
# samples 181-199, which is the +/-16 g full scale of the part.  Every
# docstring downstream says m/s2 and nothing used to convert, so `cog` came out
# 9.80665x too small (span 0.161 m instead of 1.63 m) and
# `metrics.peak_gforce` divided g by g.  `load.to_signal_arrays` now converts
# once, at the boundary; the CSV on disk is untouched.
ACCEL_UNITS = "g"            # units IN THE CSV
GYRO_UNITS = "deg/s"
G = 9.80665                  # standard gravity, m/s2
ACCEL_FULL_SCALE_G = 16.0    # the part's range; `ax` reaches 15.9615 g

# --------------------------------------------------------------------------- #
# Racket geometry (PLAN.md §7 Person A — config.py)
# --------------------------------------------------------------------------- #
# A standard tennis racket.
L_TOTAL = 0.686              # m — full racket length (butt → head tip)
L_HANDLE = 0.20              # m — distance from the wrist pivot to the grip end
# For the PIVOT model the racket tip is at this distance from the pivot (wrist).
RACKET_TIP_LEN = L_TOTAL

# Which BODY axis the racket points along, as a unit vector in sensor axes.
# MEASURED, not assumed.  The sensor is taped to the string bed (see
# GESTURE_LP_HZ below), so its axes are set by how it was taped, and the
# original `[0, 0, 1]` was an unverified assumption inherited from PLAN.md.
#
# Evidence (Artifacts/analysis.md §7.1): a sensor at body-frame radius `r`
# reads `f = (ww^T - |w|^2 I) r + b`, linear in `r`.  Least squares over the
# unclipped samples (`fusion.fit_centripetal_lever`) gives
#     post-impact 205-399 : |r| 0.213 m   [-0.997 +0.039 +0.061]
#     all unclipped       : |r| 0.285 m   [-0.970 -0.065 -0.234]
#     high-omega unclipped: |r| 0.185 m   [-0.949 -0.314 -0.010]
# — every window within ~15-20 deg of -x and ~90 deg off the modelled +z.
# Raw confirmation with no fitting at all: through the downswing `ax` climbs
# 79 -> 90 -> 114 -> 136 m/s2 then saturates at 156.5, while `ay`/`az` stay
# inside +/-55.
#
# CLEAN -x rather than an oblique axis, deliberately.  The three windows'
# off-axis components point in three DIFFERENT directions (+z-ish, -z-ish,
# -y-ish); a genuinely oblique mount would make them agree on which way it
# leans.  They do not, so the scatter is fit noise about a clean axis.
# A held-out video grid search (§7.2) optimises at [-0.981 +0.173 +0.087],
# 11 deg from here and 8 deg from the accelerometer answer — INDEPENDENT
# CONFIRMATION, from a channel the accelerometer fit never saw.  It is not
# adopted: committing the direction that scored best on the held-out set
# would be fitting to it and would destroy its value as evidence.
# For the record, pure -x scores 10.1 deg mean image-angle error against
# that held-out set and the grid optimum 5.3 deg; the pre-fix +z scores
# 114.3 deg.
#
# UNIT VECTOR ONLY — the length stays `RACKET_TIP_LEN`, so |tip| is
# unchanged and `overlay_calib.racket_pixel_scale`'s L-cancellation argument
# still holds.  Do not fold a length in here.
RACKET_LEVER_BODY = (-1.0, 0.0, 0.0)

# --------------------------------------------------------------------------- #
# Reconstruction mode
#    "pivot"        : tip = pivot + R(t) @ (L * RACKET_LEVER_BODY)  <-- main overlay
#    "rest_to_rest" : double-integrate gravity-free accel with drift correction
# --------------------------------------------------------------------------- #
MODE = "pivot"

# --------------------------------------------------------------------------- #
# Signal-processing knobs (PLAN.md §8)
# --------------------------------------------------------------------------- #
# High-pass cutoff to strip gravity / bias before double integration.
R2R_HP_CUTOFF_HZ = 0.8       # within the plan's recommended 0.5–1 Hz band
R2R_SAVGOL_WINDOW = 15       # Savitzky-Golay smoothing window (odd, samples)
# Number of leading samples used to seed an (approximate) gravity reference
# when no true static segment exists.
GRAVITY_REF_SAMPLES = 25

# --------------------------------------------------------------------------- #
# Gesture low-pass filter (racket-string ringing removal)
# --------------------------------------------------------------------------- #
# The sensor is mounted on the racket strings; the ball impact rings the
# strings at ~165 Hz (seen as strong peaks in `ay`/`az` after contact,
# SYNC_METHOD.md).  That ringing is NOT hand motion and must not leak into
# orientation / position.  Gesture content sits well below ~25 Hz, so a
# 4th-order Butterworth low-pass at this cutoff removes the ringing while
# keeping the swing shape intact.
GESTURE_LP_HZ = 25.0

# Complementary-filter weight.  MUST STAY 0.0: the only implementation this
# repo ever had was broken, and has been DELETED rather than fixed
# (`fusion.integrate_orientation` now raises on a nonzero alpha).
#
# What was wrong (Artifacts/analysis.md §6): the `alpha > 0` branch ended
# `rot_accum = _rotation_aligning(g_sensor, g_world)`, which REPLACES the
# integrated attitude instead of blending into it.  `_rotation_aligning`
# returns the minimal rotation between two vectors, so its axis is
# perpendicular to gravity by construction and it carries ZERO heading.
# Measured: seed 137 deg of pure heading, run the branch, and the component
# about gravity comes out 0.0000 deg.  It was a hard reset to a gravity-only
# attitude, every sample, at any alpha > 0.
#
# Not rewritten, deliberately: there is NO static window anywhere in this
# record to validate a gravity correction against (the quietest 25 samples
# carry 113 deg/s), and every alpha measured end to end was worse —
# 0.01 -> 33.4 px fit / 429 px held-out; 0.05 -> 28.9 / 270;
# 0.4 -> 24.7 / 277; 0.002 -> pose rejected outright.
COMP_FILTER_ALPHA = 0.0

# --------------------------------------------------------------------------- #
# Impact detection
# --------------------------------------------------------------------------- #
# Jerk is detected as |d(accel)/dt|.  Use the largest jerk spike near the gyro
# magnitude peak as the ball-contact instant.
IMPACT_GYRO_PEAK_FRACTION = 0.90   # candidate = index where |gyro| >= peak*fraction
IMPACT_JERK_SEARCH_WIDTH = 12      # +/- samples around gyro peak on each side

# --------------------------------------------------------------------------- #
# Video / overlay time base
# --------------------------------------------------------------------------- #
# `data/video/swing_angle_*.mp4` are SLOW-MOTION clips: captured at ~240 fps
# and written out at 30 fps.  Determined from evidence by
# `scripts/measure_capture_fps.py`, not assumed — see FUSION_NOTES.md §1:
#   * the containers carry NO capture-rate tag (no `udta`/`meta`/`ilst`, no
#     `com.apple.quicktime` key); their `stts`/`mdhd` give only the 30 fps
#     PLAYBACK rate, so container metadata cannot settle this;
#   * two clean ball free-fall windows, one in each clip, give 222 and 238 fps
#     (both lower bounds — the HSV mask erodes the ball, so the diameter and
#     hence the rate are underestimated);
#   * 240 fps is the only standard capture mode in that band (120 and 480 are
#     each off by ~2x);
#   * at 30 fps the ball's own parabola would make gravity 0.156 m/s2.
#
# SYNC_METHOD.md previously concluded "real time, slow motion decisively
# excluded".  That conclusion is WRONG and is corrected there: its scale scan
# only searched 0.5x-2.0x (`src/sync/align.py:181`), so an 8x factor was never
# in the search space and could not have been found.
CAPTURE_FPS = 240.0          # Hz — the rate the sensor actually sampled light
PLAYBACK_FPS = 30.0          # Hz — the rate the file declares (mdhd 600 / stts 20)
SLOWMO_SCALE = CAPTURE_FPS / PLAYBACK_FPS      # 8.0; video_t = imu_t*scale + offset

# Frame the IMU<->video mapping is anchored on, per sync label: the
# ball-racket contact instant `src/sync/align.py:276` fitted the offset to.
# `SyncedData` rescales `frame_exact` ABOUT this frame, so the one measured
# correspondence stays fixed and only the rate changes.  Matches
# `src/overlay_calib.CONTACT_FRAME_EXACT`; angle_2 has no strike, so no anchor.
SYNC_ANCHOR_FRAME: dict[str, float] = {"angle_1": 130.70344}

# --------------------------------------------------------------------------- #
# Drawn-path window (src/camera.py:draw_overlay)
# --------------------------------------------------------------------------- #
# `draw_overlay` used to project all 400 IMU samples as one static polyline.
# At 240 fps capture the whole record spans ~230 video frames while the swing
# occupies ~30, so the drawn path showed ~8x the motion belonging to the frame
# on screen - a double loop where the footage shows one sweep.  The polyline is
# now a trail around the displayed frame instead.
PATH_TRAIL_FRAMES = 12.0     # video frames of path drawn BEHIND the current frame
PATH_LEAD_FRAMES = 4.0       # video frames drawn AHEAD of it
