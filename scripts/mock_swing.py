"""Person B — synthetic 3D trajectory (PLAN.md §7 Phase 2: `mock_swing.py`).

Generates a plausible swing.npz without needing the real fusion output, so the
overlay / dashboard work can be developed and demoed in parallel.  Run::

    python scripts/mock_swing.py

Writes ``data/outputs/swing.npz`` (same contract as the real reconstruction).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from src import config  # noqa: E402
from src.swing import Swing  # noqa: E402


def make_mock_swing(n: int = 400, fs: int = config.FS) -> Swing:
    """Build a smooth synthetic swing (pivot model) mirroring real timings.

    The trajectory is a tilted circle on a sphere of radius ``RACKET_TIP_LEN``.
    """
    dt = 1.0 / fs
    t = np.arange(n, dtype=float) * dt

    # Swing phase loosely matching the real recording (~0.4s before impact).
    phase = np.linspace(-0.6, -0.05, n)
    yaw = np.linspace(-0.7, 0.7, n)                       # horizontal sweep
    elev = 0.35 * np.sin(6.0 * (phase + 0.5)) + 0.1       # vertical arc

    # Quaternion from yaw/elev: R = Ry * Rx.
    q = []
    for y, e in zip(yaw, elev):
        r = Rotation.from_euler("yx", [y, e])
        q.append(r.as_quat())                              # (x,y,z,w)
    quat = np.array(q)[:, [3, 0, 1, 2]]                    # -> (w,x,y,z)

    # config.RACKET_LEVER_BODY (-x), not the old [0, 0, L] +z lever PR #9
    # retracted as a live copy of an unverified assumption.
    tip = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).apply(
        np.asarray(config.RACKET_LEVER_BODY, float) * config.RACKET_TIP_LEN)

    # Crude sensor path (slightly off the tip, bounded).
    cog = 0.92 * np.stack([
        tip[:, 0] + 0.02 * np.sin(8 * t),
        tip[:, 1] + 0.02 * np.cos(8 * t),
        tip[:, 2] * 0.9,
    ], axis=1)

    # Fake accel/gyro (only used for signal plotting).
    gyr = np.stack([np.gradient(yaw, dt), np.gradient(elev, dt),
                    np.zeros_like(yaw)], axis=1)
    acc = np.zeros_like(gyr)
    impact = int(np.argmin(tip[:, 2]))                      # bottom of sweep

    sw = Swing(t=t, quat=quat, tip=tip, cog=cog,
               gyro_deg=np.rad2deg(gyr), accel=acc, impact_idx=impact)
    sw.validate()
    return sw


def main() -> int:
    sw = make_mock_swing()
    sw.save()
    print(f"Mock swing written -> {config.SWING_NPZ}")
    print(f"samples={len(sw.t)} span={sw.t[-1]:.3f}s impact_idx={sw.impact_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
