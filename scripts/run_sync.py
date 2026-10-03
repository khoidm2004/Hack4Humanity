#!/usr/bin/env python3
"""Phases C-F: detect the events, align each video to the CSV, write the outputs.

    uv run python scripts/run_sync.py

Writes `data/synced_data_angle_{1,2}.csv`, the machine-readable numbers in
`Artifacts/sync_evidence/sync_result.json`, and the supporting plots and
annotated frames under `Artifacts/sync_evidence/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sync import paths  # noqa: E402
from src.sync.align import align_video, build_synced_frame  # noqa: E402
from src.sync.imu_events import band_limited_gyro, load_imu_events  # noqa: E402
from src.sync.video_motion import ball_contact, motion_energy  # noqa: E402
from src.sync.video_probe import probe_video, read_frames  # noqa: E402

# How far either side of the motion-energy peak to look for the ball.
BALL_SEARCH_HALF_WIDTH = 25


def plot_signals(label, probe, me, events, al, out_path):
    gyro_lp = band_limited_gyro(events, probe.measured_fps)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7))

    ax[0].plot(me.frame_times, me.energy, "o-", ms=2.5, lw=1,
               label="video motion energy (frac. pixels |d| > 40)")
    ax[0].set_title(f"{label}: video motion energy over the whole clip")
    ax[0].set_xlabel("video time (s)")
    if al.syncable:
        ax[0].axvline(al.t_contact_video, color="r", ls="--",
                      label=f"ball contact, frame {al.contact_frame_exact:.2f}")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    if al.syncable:
        t = events.t_sample + al.offset_s
        ax[1].plot(me.frame_times, me.energy / me.energy.max(), "o-", ms=2.5, lw=1,
                   label="video motion energy (normalised)")
        ax[1].plot(t, gyro_lp / np.abs(gyro_lp).max(),
                   label="IMU |gyro|, low-passed to video Nyquist")
        ax[1].plot(t, events.gyro_mag / events.gyro_mag.max(), alpha=0.3,
                   label="IMU |gyro| raw")
        ax[1].axvline(events.t_impact + al.offset_s, color="r", ls="--",
                      label=f"IMU impact, sample {events.impact_index}")
        if al.xcorr is not None:
            ax[1].axvline(events.t_impact + al.xcorr.offset_s, color="m", ls="-.",
                          label="impact at cross-correlation offset")
        lo = al.t_contact_video - 1.2
        ax[1].set_xlim(lo, lo + 3.0)
        ax[1].set_title(f"{label}: CSV mapped onto the video clock "
                        f"(offset {al.offset_s:+.4f} s)")
    else:
        ax[1].text(0.5, 0.5, f"{label} not syncable:\n{al.reason}",
                   ha="center", va="center", transform=ax[1].transAxes, wrap=True)
    ax[1].set_xlabel("video time (s)")
    if ax[1].get_legend_handles_labels()[0]:
        ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_ball_track(label, contact, out_path):
    if not contact.found:
        # Leave no stale plot from a previous run claiming a contact exists.
        out_path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, v, name in ((ax[0], contact.xs, "x"), (ax[1], contact.ys, "y")):
        a.plot(contact.frames, v, "o-", ms=3)
        a.axvline(contact.contact_frame_exact, color="r", ls="--",
                  label=f"contact {contact.contact_frame_exact:.2f}")
        a.set_xlabel("frame")
        a.set_ylabel(f"ball {name} (px)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(f"{label}: ball image trajectory; contact is the kink")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def annotate_contact_frames(label, video, probe, contact, out_dir):
    """Write the frames bracketing contact, with the detected ball marked."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{label}_contact_*.png"):
        old.unlink()
    if not contact.found:
        return []
    lo, hi = contact.contact_frame_bracket
    want = list(range(max(0, lo - 2), min(probe.decoded_frame_count - 1, hi + 2) + 1))
    frames = read_frames(video, want)
    written = []
    for i in want:
        img = frames.get(i)
        if img is None:
            continue
        hit = np.flatnonzero(contact.frames == i)
        if hit.size:
            k = int(hit[0])
            cv2.circle(img, (int(contact.xs[k]), int(contact.ys[k])), 22,
                       (0, 0, 255), 2)
        cv2.putText(img, f"frame {i}  t={probe.frame_times[i]:.3f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if lo <= i <= hi:
            cv2.putText(img, "CONTACT BRACKET", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        dest = out_dir / f"{label}_contact_f{i:05d}.png"
        cv2.imwrite(str(dest), img)
        written.append(dest)
    return written


def main() -> int:
    paths.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    events = load_imu_events(paths.RAW_CSV)
    print("=== IMU (data/raw_data.csv) ===")
    for line in events.summary_lines():
        print(line)
    # Sanity assertion from the reference plot: the impact is visibly at ~200.
    assert abs(events.impact_index - 200) <= 5, (
        f"impact detector returned {events.impact_index}; the reference plot "
        "data/raw_data_visualized.png puts the impact at sample ~200")
    if events.detector_disagreement > 2:
        print(f"  NOTE: the two impact detectors differ by "
              f"{events.detector_disagreement} samples - see SYNC_METHOD.md")

    result = {
        "imu": {
            "sample_rate_hz": events.sample_rate_hz,
            "n_samples": int(len(events.df)),
            "impact_index": events.impact_index,
            "impact_index_gyro": events.impact_index_gyro,
            "impact_index_accel_hf": events.impact_index_accel,
            "accel_channels_used": events.accel_cols_used,
            "t_impact_s": events.t_impact,
            "motion_onset_index": events.motion_onset_index,
            "t_motion_onset_s": events.t_motion_onset,
            "saturation": {k: v for k, v in events.saturation.items()
                           if v["n_saturated"]},
        },
        "videos": {},
    }

    for label, video, csv_out in paths.ANGLES:
        print(f"\n=== {label} ({video.name}) ===")
        probe = probe_video(video)
        me = motion_energy(video, probe.frame_times)
        lo = max(0, me.peak_frame - BALL_SEARCH_HALF_WIDTH)
        hi = min(probe.decoded_frame_count - 1, me.peak_frame + BALL_SEARCH_HALF_WIDTH)
        contact = ball_contact(video, lo, hi)

        print(f"  decoded frames       : {probe.decoded_frame_count} "
              f"@ {probe.measured_fps:.4f} fps, {probe.duration_s:.3f} s, "
              f"{'VFR/near-CFR' if probe.is_vfr else 'CFR'}")
        print(f"  motion-energy peak   : frame {me.peak_frame} "
              f"(t={me.peak_time:.3f} s); ball searched in [{lo}, {hi}]")
        if contact.found:
            print(f"  ball track           : {contact.detail['n_detections']} "
                  f"detections, speed {contact.pre_speed:.1f} -> "
                  f"{contact.post_speed:.1f} px/frame")
            print(f"  trajectory kink      : frame "
                  f"{contact.contact_frame_exact:.2f} "
                  f"(x fit {contact.detail['crossing_from_x']:.2f}, "
                  f"y fit {contact.detail['crossing_from_y']:.2f}, "
                  f"spread {contact.detail['crossing_spread_frames']:.2f} frames)")
            print(f"  looks like a strike  : {contact.is_strike}")
        else:
            print(f"  ball track           : FAILED - {contact.reason}")

        al = align_video(label, probe, me, events, contact)

        if al.syncable:
            print(f"  CONTACT FRAME        : {al.contact_frame_exact:.2f} "
                  f"(bracket {al.contact_frame_bracket[0]}-"
                  f"{al.contact_frame_bracket[1]}), "
                  f"t_video = {al.t_contact_video:.4f} s")
            print(f"  OFFSET               : {al.offset_s:+.4f} s "
                  f"({al.offset_frames:+.2f} frames)")
            print(f"  error bar            : +/-1 frame = "
                  f"+/-{al.error_bar_s * 1000:.1f} ms = "
                  f"+/-{al.error_bar_samples:.0f} CSV samples")
            x = al.xcorr
            print(f"  cross-correlation    : offset {x.offset_s:+.4f} s, "
                  f"peak r = {x.peak_correlation:.3f}, "
                  f"lag vs event {x.lag_vs_event_frames:+.2f} frames -> "
                  f"{'AGREES' if x.agrees else 'DISAGREES'} "
                  f"(criterion +/-1 frame)")
            print(f"    feature bias       : video energy peak is "
                  f"{x.video_peak_minus_contact_frames:+.2f} frames from contact, "
                  f"IMU |gyro| peak is {x.imu_peak_minus_impact_frames:+.2f} frames "
                  f"from impact -> expected correlation bias "
                  f"{x.expected_bias_frames:+.2f} frames")
            d = al.drift
            print(f"  rate check (E3)      : best time scale {d.best_scale:.2f}x, "
                  f"indistinguishable over {d.scale_plateau[0]:.2f}-"
                  f"{d.scale_plateau[1]:.2f}x")
            print(f"                         {d.verdict}")
        else:
            print(f"  NOT SYNCABLE         : {al.reason}")

        synced, outside = build_synced_frame(events, probe,
                                             al.offset_s if al.syncable else None)
        al.rows_outside_video = outside
        synced.to_csv(csv_out, index=False)
        print(f"  wrote                : {csv_out} "
              f"({len(synced)} rows, {outside} outside the video)")

        plot_signals(label, probe, me, events, al,
                     paths.EVIDENCE_DIR / f"signals_{label}.png")
        plot_ball_track(label, contact,
                        paths.EVIDENCE_DIR / f"ball_track_{label}.png")
        annotate_contact_frames(label, video, probe, contact, paths.CONTACT_DIR)

        result["videos"][label] = {
            "file": str(video),
            "decoded_frame_count": probe.decoded_frame_count,
            "measured_fps": probe.measured_fps,
            "fps_source": probe.fps_source,
            "reported_fps": probe.reported_fps,
            "duration_s": probe.duration_s,
            "is_vfr": probe.is_vfr,
            "interval_stats": probe.interval_stats,
            "motion_energy_peak_frame": me.peak_frame,
            "ball_search_window": [lo, hi],
            "ball_found": bool(contact.found),
            "ball_is_strike": bool(contact.is_strike),
            "ball_pre_speed_px_per_frame": contact.pre_speed,
            "ball_post_speed_px_per_frame": contact.post_speed,
            "ball_detail": contact.detail,
            "syncable": al.syncable,
            "reason": al.reason,
            "contact_frame_exact": al.contact_frame_exact,
            "contact_frame_bracket": list(al.contact_frame_bracket),
            "t_contact_video_s": al.t_contact_video,
            "offset_s": al.offset_s,
            "offset_frames": al.offset_frames,
            "error_bar_s": al.error_bar_s if al.syncable else None,
            "rows_outside_video": outside,
            "xcorr": None if al.xcorr is None else vars(al.xcorr),
            "drift_check": None if al.drift is None else vars(al.drift),
            "synced_csv": str(csv_out),
        }

    paths.SYNC_RESULT_JSON.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nnumbers -> {paths.SYNC_RESULT_JSON}")
    print(f"plots   -> {paths.EVIDENCE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
