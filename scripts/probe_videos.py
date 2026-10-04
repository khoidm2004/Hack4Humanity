#!/usr/bin/env python3
"""Phase A CLI: decode and measure every video, print a report, write sample frames.

Run this (and read its output) before trusting any alignment number.

    uv run python scripts/probe_videos.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sync import paths  # noqa: E402
from src.sync.video_probe import (  # noqa: E402
    VideoProbe,
    build_contact_sheet,
    probe_video,
    read_frames,
    sample_frame_indices,
)


def write_sample_frames(probe: VideoProbe, out_dir: Path, n: int = 12) -> list[Path]:
    """Write ~n evenly spaced frames as PNG so a human can look at the footage."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = probe.path.stem
    # Re-runs must not leave stale frames behind (filenames embed the timestamp).
    for old in out_dir.glob(f"{stem}_f*.png"):
        old.unlink()
    idx = sample_frame_indices(probe, n)
    frames = read_frames(probe.path, list(idx))
    written = []
    for i in idx:
        frame = frames.get(int(i))
        if frame is None:
            continue
        t = probe.frame_times[i]
        dest = out_dir / f"{stem}_f{int(i):05d}_t{t:07.3f}s.png"
        cv2.imwrite(str(dest), frame)
        written.append(dest)
    return written


def write_contact_sheet(probe: VideoProbe, out_dir: Path, cols: int = 14,
                        thumb_w: int = 192) -> Path:
    """Build the contact sheet and write it to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = build_contact_sheet(probe, cols=cols, thumb_w=thumb_w)
    dest = out_dir / f"contactsheet_{probe.path.stem}.png"
    cv2.imwrite(str(dest), sheet)
    return dest


def main() -> int:
    paths.PROBE_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for label, video, _csv_out in paths.ANGLES:
        if not video.exists():
            print(f"!! missing video: {video}")
            return 1
        print(f"\n=== {label} ===")
        probe = probe_video(video)
        for line in probe.summary_lines():
            print(line)
        frames_written = write_sample_frames(probe, paths.PROBE_DIR)
        print(f"  sample frames written: {len(frames_written)}")
        sheet = write_contact_sheet(probe, paths.PROBE_DIR)
        print(f"  contact sheet        : {sheet}")
        report[label] = {
            "path": str(probe.path),
            "width": probe.width,
            "height": probe.height,
            "reported_fps": probe.reported_fps,
            "reported_frame_count": probe.reported_frame_count,
            "decoded_frame_count": probe.decoded_frame_count,
            "measured_fps": probe.measured_fps,
            "fps_source": probe.fps_source,
            "duration_s": probe.duration_s,
            "timestamps_usable": probe.timestamps_usable,
            "is_vfr": probe.is_vfr,
            "interval_stats": probe.interval_stats,
        }

    out = paths.EVIDENCE_DIR / "probe_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nprobe frames -> {paths.PROBE_DIR}")
    print(f"probe report -> {out}")
    print("\nSTOP: read the frames and this report before writing alignment code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
