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
# Sensor units (confirmed in PLAN.md §2 / §6.2)
# --------------------------------------------------------------------------- #
ACCEL_UNITS = "m/s2"
GYRO_UNITS = "deg/s"
G = 9.80665                  # standard gravity, m/s2

# --------------------------------------------------------------------------- #
# Racket geometry (PLAN.md §7 Person A — config.py)
# --------------------------------------------------------------------------- #
# A standard tennis racket.
L_TOTAL = 0.686              # m — full racket length (butt → head tip)
L_HANDLE = 0.20              # m — distance from the wrist pivot to the grip end
# For the PIVOT model the racket tip is at this distance from the pivot (wrist).
RACKET_TIP_LEN = L_TOTAL

# --------------------------------------------------------------------------- #
# Reconstruction mode
#    "pivot"        : tip = pivot + R(t) @ [0, 0, L]   <-- main overlay
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

# Complementary-filter weight applied to the accel-measured roll/pitch.
# This sensor is mounted on the strings: impact and string vibration make its
# acceleration unsuitable as a gravity reference during the swing.  The
# previous 0.4 correction pulled the gyro orientation backwards and suppressed
# the follow-through.  Keep gyro integration as the default; setting a nonzero
# value remains an explicit experiment for data with a reliable gravity signal.
COMP_FILTER_ALPHA = 0.0

# --------------------------------------------------------------------------- #
# Impact detection
# --------------------------------------------------------------------------- #
# Jerk is detected as |d(accel)/dt|.  Use the largest jerk spike near the gyro
# magnitude peak as the ball-contact instant.
IMPACT_GYRO_PEAK_FRACTION = 0.90   # candidate = index where |gyro| >= peak*fraction
IMPACT_JERK_SEARCH_WIDTH = 12      # +/- samples around gyro peak on each side

# --------------------------------------------------------------------------- #
# Video / overlay mapping (used by Person B)
# --------------------------------------------------------------------------- #
# NOTE: SYNC_METHOD.md measured both videos as real-time 30 fps footage
# (best time-scale ≈ 1.0x, slow motion excluded).  When a per-video sync table
# exists (data/synced_data_<angle>.csv from scripts/run_sync.py) the app uses
# its exact frame_index mapping and this value is only a fallback.
SLOWMO_SCALE = 1.0           # real-time footage; was 8.33 under the old assumption
