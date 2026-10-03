#!/usr/bin/env python3
"""Phase G: write the before/during/after verification panels.

    uv run python scripts/verify_sync.py

For each chosen moment the panel puts the video frame on the left and the six
IMU traces on the right, with a vertical cursor and a dot on each trace marking
the sample that maps to that frame.

Interpretation note: the acceptance criterion asks for "a dot from the synced
data on a few sample frames". There is no 3D reconstruction yet, so there is no
physical point in the image at which to draw a dot - the reconstruction is the
*next* task and depends on this one. The dot is therefore placed on the IMU
traces beside the frame rather than on the frame itself, which is what is
actually checkable today: the cursor must sit at the start of the post-impact
ringdown exactly on the frame where the ball leaves the strings. This is flagged
as an interpretation in SYNC_METHOD.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sync import paths  # noqa: E402
from src.sync.video_probe import read_frames  # noqa: E402

TRACES = [("ax", "g"), ("ay", "g"), ("az", "g"),
          ("gx", "deg/s"), ("gy", "deg/s"), ("gz", "deg/s")]


def choose_moments(df: pd.DataFrame, impact_index: int) -> list[tuple[int, str]]:
    """Five CSV samples spanning the clip, named for what should be on screen."""
    n = len(df)
    return [
        (0, "start of window - racquet at rest, ball still approaching"),
        (impact_index // 2, "backswing - racquet taken back, gx/gz at their most negative"),
        (impact_index, "IMPACT - ball on the strings"),
        (min(n - 1, impact_index + 25), "just after impact - ball leaving, frame ringing"),
        (n - 1, "end of window - ringdown decayed, follow-through"),
    ]


def panel(frame_bgr, df, sample_index, title, subtitle, out_path):
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(6, 2, figure=fig, width_ratios=[1.35, 1], hspace=0.12, wspace=0.18)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(subtitle, fontsize=10)

    t = df["t_sample"].to_numpy()
    t_now = float(t[sample_index])
    for k, (col, unit) in enumerate(TRACES):
        a = fig.add_subplot(gs[k, 1])
        v = df[col].to_numpy()
        a.plot(t, v, lw=0.8)
        a.axvline(t_now, color="r", lw=1.0)
        a.plot([t_now], [v[sample_index]], "o", color="r", ms=5)
        a.set_ylabel(f"{col} ({unit})", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25)
        if k < len(TRACES) - 1:
            a.set_xticklabels([])
        else:
            a.set_xlabel("CSV time t_sample (s)", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not paths.SYNC_RESULT_JSON.exists():
        print(f"!! {paths.SYNC_RESULT_JSON} not found - run scripts/run_sync.py first")
        return 1
    result = json.loads(paths.SYNC_RESULT_JSON.read_text())
    impact_index = int(result["imu"]["impact_index"])

    paths.VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    for old in paths.VERIFY_DIR.glob("*.png"):
        old.unlink()

    any_written = False
    for label, video, csv_out in paths.ANGLES:
        info = result["videos"][label]
        print(f"\n=== {label} ===")
        if not info["syncable"]:
            print(f"  skipped: {info['reason']}")
            print("  no verification panels - there is nothing to verify against.")
            continue

        df = pd.read_csv(csv_out)
        moments = choose_moments(df, impact_index)
        wanted = []
        for idx, _why in moments:
            fi = int(df["frame_index"].iloc[idx])
            wanted.append(fi)
        frames = read_frames(video, [f for f in wanted if f >= 0])

        for (idx, why), fi in zip(moments, wanted):
            if fi < 0 or fi not in frames:
                print(f"  sample {idx}: maps outside the video - skipped")
                continue
            t_video = float(df["t_video"].iloc[idx])
            title = (f"{label}: CSV sample {idx} (t_sample={df['t_sample'].iloc[idx]:.4f} s)"
                     f"  ->  video frame {fi} (t_video={t_video:.4f} s)"
                     f"   [offset {info['offset_s']:+.4f} s]")
            out = paths.VERIFY_DIR / f"{label}_s{idx:03d}_f{fi:05d}.png"
            panel(frames[fi], df, idx, title, why, out)
            print(f"  sample {idx:3d} -> frame {fi:3d}  {why}")
            print(f"              {out}")
            any_written = True

    if any_written:
        print(f"\nverification panels -> {paths.VERIFY_DIR}")
        print("Check by eye: on the IMPACT panel the cursor must sit exactly at the")
        print("start of the ringdown, and the frame must show the ball on the strings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
