#!/usr/bin/env python3
"""Phase A CLI: decode and measure every video, print a report, write sample frames.

Run this (and read its output) before trusting any alignment number.

    uv run python scripts/probe_videos.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sync import paths  # noqa: E402
from src.sync.video_probe import probe_video, write_contact_sheet  # noqa: E402


def main() -> int:
    paths.PROBE_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for label, video, _csv_out in paths.ANGLES:
        if not video.exists():
            print(f"!! missing video: {video}")
            return 1
        print(f"\n=== {label} ===")
        probe = probe_video(video, sample_frame_dir=paths.PROBE_DIR)
        for line in probe.summary_lines():
            print(line)
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
