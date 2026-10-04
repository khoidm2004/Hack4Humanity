#!/usr/bin/env python3
"""Deliverable 6 — deterministic before/after overlay renders.

For each requested video frame of `swing_angle_1.mp4`, builds the solved
camera pose, fetches the frame, draws the overlay with `src.camera.draw_overlay`
and writes a PNG to `Artifacts/overlay_evidence/`.

With `--whole-path`, forces `path_window_idx=(0, len(swing.tip) - 1)` — the
old whole-record polyline — so the *windowing* change (src/camera.py Step 9)
can be seen on the exact same (fixed, post-time-base-fix) reconstruction and
pose, isolating what the window itself changed in the picture.

No side effects beyond the PNGs: does not touch `data/outputs/swing.npz`.

Run::

    uv run python scripts/render_overlay_frames.py
    uv run python scripts/render_overlay_frames.py --whole-path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src import camera, config, overlay_calib as oc, video  # noqa: E402

DEFAULT_FRAMES = [116, 122, 126, 130, 137, 143]
OUT_DIR = ROOT / "Artifacts" / "overlay_evidence"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    ap.add_argument("--whole-path", action="store_true",
                    help="force path_window_idx=(0, len(swing.tip)-1), the "
                         "pre-windowing whole-record polyline, on the "
                         "SAME (already fixed) time base.")
    args = ap.parse_args()

    swing, sync = oc._load_inputs()
    video_path = config.VIDEO_DIR / oc.VIDEO_NAME
    meta = video.probe(str(video_path))
    sol = oc.solved_pose(meta.width, meta.height, oc.CALIB_FOV_DEG)
    if not sol.ok:
        print("FAIL: no usable pose; cannot render.")
        return 1
    cam = sol.camera(meta.width, meta.height)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "after_whole_path_f" if args.whole_path else "after_render_f"
    path_window_idx = ((0, len(swing.tip) - 1) if args.whole_path else None)

    written = []
    for f in args.frames:
        f = int(f)
        img = video.frame(str(video_path), f)
        idx = sync.imu_idx_for_frame_centered(f)
        out = camera.draw_overlay(
            img, cam, swing, f / meta.fps, meta.fps,
            dot_idx=idx, path_window_idx=path_window_idx,
        )
        dest = OUT_DIR / f"{prefix}{f:03d}.png"
        cv2.imwrite(str(dest), out)
        written.append(dest)
        print(f"  frame {f:3d} (imu idx {idx:3d}) -> {dest}")

    print(f"\nWrote {len(written)} PNG(s) to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
