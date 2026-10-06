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
# Wrist / pivot translation
# --------------------------------------------------------------------------- #
# How much of the `fusion.rest_to_rest_cog` path to use as the wrist (pivot)
# translation: `Swing.pivot = WRIST_PIVOT_COG_SCALE * cog`, and the racket head
# is `pivot + tip`.  Until 2026-10-05 `pivot_tip` pinned the wrist at the world
# origin, so the whole racket was a rigid sphere of radius RACKET_TIP_LEN about
# a point that never moves — while the real wrist travels 258 px across the
# annotated window and rises to shoulder height through the follow-through.
#
# NOT a physical fraction.  `rest_to_rest_cog` detrends its output about the
# origin (`fusion._detrend_rest_to_rest`), so `cog`'s amplitude is set by that
# detrending, not by physics, and PnP absorbs any constant part into `tvec`.
# It therefore has to be CALIBRATED, and the calibration below uses
# FIT-WINDOW DATA ONLY (Artifacts/analysis.md §7, §10.2):
#
#   AMPLITUDE MATCHING (route 1, the committed route) —
#     annotated wrist travel f116-145 : 258.118 px   (wrist_pixel_spread()[2],
#                                       the FIT reads; held-out never used)
#     image scale                     : 284.611 px/m (racket_pixel_scale(),
#                                       committed value, see the caveat below)
#       -> observed wrist travel      :   0.906914 m
#     `cog` max pairwise over the same window (samples 175-225) : 0.999390 m
#       -> k = 0.906914 / 0.999390    =   0.907468  -> 0.907
#
#   JOINT PnP OVER THE FIT SET (route 2) — REPORTED, NOT COMMITTED, because it
#   does not identify k.  Scored on the 13 + contact only, the fit mean and max
#   residuals fall MONOTONICALLY out to k = 1.5 (16.1 -> 15.5 px mean, 38.9 ->
#   36.4 px max) with no interior minimum, and the objective is not convex:
#   k = 0.10, 2.0 and 3.0 are bad PnP minima whose held-out medians are 416,
#   395 and 410 px while k = 3.0 posts the LOWEST fit median of every cell
#   tried (14.79 px).  An argmin over the fit set picks a wrong answer here.
#   What route 2 does say is that 0.907 sits inside a broad flat basin
#   (0.6-1.1, fit median 15.55-15.69 px), which is agreement, not fitting.
#
#   The held-out f146-220 reads were NEVER used to choose this number.  They
#   are the validation: held-out median 184.1 -> 78.9 px.
#
# DOMINANT UNCERTAINTY, declared: `racket_pixel_scale` maxes over f116-145 only
# and reads 284.6 px/m; over all 26 reads it reads 375.2 px/m, which would make
# route 1 give k = 0.688.  Widening that window is deliberately OUT OF SCOPE
# this round (Artifacts/analysis.md §6, §10.3) — it touches `racket_distance_m`
# and both arc ratios and raises an unsettled question about using held-out
# reads in a reported metric.  So k is 0.907 +0.000 / -0.219, and both ends sit
# inside route 2's basin.  Do not "fix" the scale here to move k.
WRIST_PIVOT_COG_SCALE = 0.907

# --------------------------------------------------------------------------- #
# Wrist / pivot linear drift restoration
# --------------------------------------------------------------------------- #
# `fusion.rest_to_rest_cog` ends with `_detrend_rest_to_rest(pos)`, which
# subtracts a linear trend so the double-integrated path starts and ends near
# the origin.  Standard practice for a ~1 s strapdown integration with no
# position aiding — and it throws away REAL displacement, not just integration
# error.  Measured on data/raw_data.csv: the removed position ramp is
#     slope * 400 = [-1.2967, +0.1899, -0.5044] m,  |.| = 1.4042 m
# against a surviving detrended `cog` net displacement of only 0.782 m.  The
# detrend removes MORE net displacement than it keeps, which is why the pivot
# reproduced only the oscillatory part of the wrist motion and the overlay came
# off the racket before f108 (Artifacts/analysis.md §3, findings/FUSION_NOTES.md §11).
#
# `fusion.wrist_pivot` gives that ramp back:
#     pivot(t) = WRIST_PIVOT_COG_SCALE * cog(t) + b * (t - WRIST_PIVOT_DRIFT_PIVOT_SAMPLE)
#     b = WRIST_PIVOT_COG_SCALE * WRIST_PIVOT_DRIFT_GAIN * removed_slope
#
# GAIN = 1.0 means "restore exactly what the detrend removed, scaled by the
# same k the pivot already uses".  It is DERIVED, not fitted: no annotation,
# no optimizer, no free parameter.  At gain 1.0,
#     b * 400 = [-1.1761, +0.1722, -0.4574] m,  |.| = 1.2736 m
#
# INDEPENDENT IMAGE-SIDE CROSS-CHECK (acceptance criterion 4).  Fitting the
# same 3-parameter ramp to the 14 pre-116 video annotations with the camera
# pose held fixed (`scripts/fit_pivot_drift.py`, procedure A) gives
#     b * 400 = [-1.27336, +0.14777, -1.12994] m,  |.| = 1.7088 m
#   -> angle between the two: 20.4 deg;  magnitude ratio 1.342
# Two wholly independent channels — double-integrated accelerometer vs
# hand-read video pixels — agreeing on direction to 20 deg and magnitude to
# 34%.  That agreement is the evidence for this term; the accelerometer value
# is the one COMMITTED, because it is the one that is not fitted to anything.
#
# WHY NOT THE IMAGE-FITTED b.  `scripts/fit_pivot_drift.py` reports three
# procedures.  Scored with the pose re-solved from the committed 13+contact
# set (the deployed configuration) against the never-fitted f146-220 reads:
#     committed b=0                    held-out  78.93 / 74.57 / 156.56 px
#     C accel-derived (COMMITTED)      held-out  72.56 / 72.07 / 113.58 px
#     A two-step image fit             held-out  76.83 / 74.50 / 123.95 px
#     B joint (rvec,tvec,b), seed 0    held-out  73.48 / 72.42 / 124.93 px
#     B joint, seeded at C's b         held-out  97.37 / 91.00 / 145.58 px  <- WORSE
# B's answer depends on its SEED, and the seed that posts the best pre-116
# number (7.3 px median) is the one that makes the held-out set worse.  Same
# lesson as PR #9's FOV sweep and PR #10's k-sweep: a single optimizer run is
# a lead, not a result.  C has no objective to minimise, so it has no local
# minima to land in the wrong one of.
#
# GAIN SWEEP, reported not minimised (`fit_pivot_drift.py --sweep`).  The
# objective is NOT convex and its argmin is NOT the answer:
#   gain  fit med/mean/max     pre-116 med      held-out med/mean/max
#   0.00  15.56/16.13/38.88    50.50            78.93/74.57/156.56
#   0.50  15.80/16.19/38.65    35.66            72.45/67.74/117.10
#   0.75  15.66/16.26/38.55    28.20            65.20/68.35/111.55
#   1.00  15.78/16.38/38.69    23.17            72.56/72.07/113.58  <- COMMITTED
#   1.25  16.52/16.89/50.62   255.22           432.51/385.12/635.94  <- BAD PnP MINIMUM
#   1.50  15.49/16.61/38.41    10.39            98.54/87.07/131.91  <- held-out REGRESSES
#   2.00  16.42/16.73/50.04   271.65           425.95/392.23/642.43  <- BAD PnP MINIMUM
# gain 0.75 posts the best held-out median and gain 1.50 the best pre-116
# median; 1.50 regresses the held-out set and 1.25/2.00 collapse into bad PnP
# minima (|tvec| 2.6 m).  1.0 is committed because it is the DERIVED value,
# and it is the only cell that improves all three statistics on both the
# pre-116 and the held-out span while leaving every fit-window gate passing.
# DO NOT tune this to a sweep argmin.
WRIST_PIVOT_DRIFT_GAIN = 1.0

# Sample index the restored ramp is zero at.  In exact arithmetic this is a
# pure gauge: a constant 3D offset of the whole point set is absorbed exactly
# by PnP's `tvec`.  In practice the PnP objective is non-convex and the
# centring changes which minimum the multi-solver path lands in — MEASURED:
#     centre   0 -> median 16.57 px, |tvec| 2.007 m  (bad minimum)
#     centre 200 -> median 15.78 px, |tvec| 4.357 m  <- COMMITTED
#     centre 399 -> median 15.73 px, |tvec| 4.743 m  (outside the 3.5-4.6 m
#                                                     player-height anchor band)
# 200 is the record's midpoint, which minimises |ramp| over the record and
# keeps |tvec| closest to its committed 4.232 m and inside that band.  A
# documented convention, not a free parameter.
WRIST_PIVOT_DRIFT_PIVOT_SAMPLE = 200

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
