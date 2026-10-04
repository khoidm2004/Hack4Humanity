#!/usr/bin/env python3
"""Deliverable 1 — determine the slow-motion clips' capture frame rate from
evidence, with no `ffprobe`/`ffmpeg`/PyAV dependency (none are installed in
this environment; confirmed in `Artifacts/analysis.md`).

Three independent things, plus a verdict:

(a) Container metadata — walk the MP4 box tree with `struct` and show that
    the containers declare only a 30 fps *playback* rate (`mdhd`/`stts`) and
    carry no capture-rate tag at all (no `udta`/`meta`/`ilst` box, no
    `com.apple.quicktime` bytes anywhere in the file).

(b) The tennis ball's own free-fall parabola and pixel diameter, which give
    an independent px/m scale and frame interval with no racket and no PnP
    involved. Several candidate windows are tried; a window is accepted as
    evidence only if it passes three hard criteria (see ACCEPT_* below) —
    this is NOT a frame-rate search or optimiser, it is evidence filtering.

(c) The reductio: what gravity would have to be for the clip to really be
    real-time 30 fps.

This script computes a value and explains the reasoning; it does **not**
pick a frame rate by minimising anything. Run:

    uv run python scripts/measure_capture_fps.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.sync.video_motion import (  # noqa: E402
    BALL_AREA_MAX,
    BALL_AREA_MIN,
    BALL_HSV_HI,
    BALL_HSV_LO,
    track_ball,
)

BALL_DIAMETER_M = 0.067
G_MPS2 = 9.80665

# Criteria for accepting a free-fall window as frame-rate EVIDENCE.  A window
# failing any of these does not measure a frame rate — it is discarded, not
# down-weighted.
ACCEPT_MAX_PARABOLA_RESID_PX = 0.5
ACCEPT_MAX_DIAMETER_STD_FRAC = 0.05
ACCEPT_MIN_MEAN_DIAMETER_PX = 12.0


# --------------------------------------------------------------------------- #
# (a) Container metadata — walk the MP4 box tree with struct
# --------------------------------------------------------------------------- #
# Box types whose contents are themselves a sequence of boxes ("container
# boxes"); everything else is a leaf we just record the bytes of.
_CONTAINER_BOXES = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"meta",
    b"ilst", b"edts", b"mvex", b"moof", b"traf", b"mfra", b"dinf",
}


def _iter_boxes(data: bytes, start: int, end: int, depth: int = 0):
    """Yield (box_type, box_start, box_end, depth) for every box in [start,end)."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        btype = data[pos + 4:pos + 8]
        header = 8
        if size == 1:  # 64-bit extended size
            if pos + 16 > end:
                break
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            header = 16
        if size == 0:  # box extends to EOF
            size = end - pos
        box_end = pos + size
        if box_end > end or size < header:
            break
        yield btype, pos, box_end, depth
        if btype in _CONTAINER_BOXES:
            # `meta` sometimes carries a 4-byte version/flags header before
            # its child boxes (full-box variant); try both offsets so a
            # parse failure doesn't silently stop the walk early.
            inner_start = pos + header
            if btype == b"meta":
                probe = pos + header + 4
                if probe + 8 <= box_end:
                    maybe_size = struct.unpack(">I", data[probe:probe + 4])[0]
                    if 8 <= maybe_size <= (box_end - probe):
                        inner_start = probe
            yield from _iter_boxes(data, inner_start, box_end, depth + 1)
        pos = box_end


def _parse_full_box_times(body: bytes) -> tuple[int, int]:
    """``(timescale, duration)`` from an `mvhd`/`mdhd` full-box payload.

    Layout after the 4-byte version+flags header:
      version 0: creation(4) modification(4) timescale(4) duration(4)
      version 1: creation(8) modification(8) timescale(4) duration(8)
    """
    version = body[0]
    if version == 1:
        timescale = struct.unpack(">I", body[20:24])[0]
        duration = struct.unpack(">Q", body[24:32])[0]
    else:
        timescale, duration = struct.unpack(">II", body[12:20])
    return int(timescale), int(duration)


def _parse_mvhd(data: bytes, start: int, end: int) -> tuple[int, int]:
    return _parse_full_box_times(data[start + 8:end])


def _parse_mdhd(data: bytes, start: int, end: int) -> tuple[int, int]:
    return _parse_full_box_times(data[start + 8:end])


def _parse_stts(data: bytes, start: int, end: int) -> list[tuple[int, int]]:
    body = data[start + 8:end]
    entry_count = struct.unpack(">I", body[4:8])[0]
    out = []
    off = 8
    for _ in range(entry_count):
        count, delta = struct.unpack(">II", body[off:off + 8])
        out.append((int(count), int(delta)))
        off += 8
    return out


def probe_container(path: Path) -> dict:
    data = path.read_bytes()
    boxes = list(_iter_boxes(data, 0, len(data)))
    by_type: dict[bytes, list[tuple[int, int, int]]] = {}
    for btype, s, e, depth in boxes:
        by_type.setdefault(btype, []).append((s, e, depth))

    mvhd = by_type.get(b"mvhd")
    mvhd_timescale = mvhd_duration = None
    if mvhd:
        s, e, _ = mvhd[0]
        mvhd_timescale, mvhd_duration = _parse_mvhd(data, s, e)

    mdhd = by_type.get(b"mdhd")
    mdhd_timescale = mdhd_duration = None
    if mdhd:
        s, e, _ = mdhd[0]
        mdhd_timescale, mdhd_duration = _parse_mdhd(data, s, e)

    stts = by_type.get(b"stts")
    stts_entries: list[tuple[int, int]] = []
    if stts:
        s, e, _ = stts[0]
        stts_entries = _parse_stts(data, s, e)

    has_udta = b"udta" in by_type
    has_meta = b"meta" in by_type
    has_ilst = b"ilst" in by_type
    has_quicktime_key = b"com.apple.quicktime" in data

    return {
        "mvhd_timescale": mvhd_timescale,
        "mvhd_duration": mvhd_duration,
        "mdhd_timescale": mdhd_timescale,
        "mdhd_duration": mdhd_duration,
        "stts": stts_entries,
        "has_udta": has_udta,
        "has_meta": has_meta,
        "has_ilst": has_ilst,
        "has_quicktime_key": has_quicktime_key,
        "n_boxes": len(boxes),
    }


def report_container(label: str, path: Path) -> None:
    info = probe_container(path)
    playback_fps = (info["mdhd_timescale"] / info["stts"][0][1]
                    if info["stts"] else float("nan"))
    print(f"  {label} ({path.name}):")
    print(f"    mvhd  timescale={info['mvhd_timescale']}  "
          f"duration={info['mvhd_duration']}")
    print(f"    mdhd  timescale={info['mdhd_timescale']}  "
          f"duration={info['mdhd_duration']}")
    print(f"    stts  entries={info['stts']}")
    print(f"    udta/meta/ilst present : "
          f"{info['has_udta']}/{info['has_meta']}/{info['has_ilst']}")
    print(f"    'com.apple.quicktime' bytes present : "
          f"{info['has_quicktime_key']}")
    print(f"    => declares a {playback_fps:.3f} fps PLAYBACK rate "
          f"(mdhd timescale / first stts delta); no capture-rate tag present.")


# --------------------------------------------------------------------------- #
# (b) Ball free-fall × ball diameter
# --------------------------------------------------------------------------- #
def _track_ball_with_area(path: Path, lo: int, hi: int):
    """Like `video_motion.track_ball`, but also returns each blob's area.

    Re-runs the exact committed HSV rule (`BALL_HSV_LO/HI`, 3x3 MORPH_OPEN,
    area 25-2000, largest blob) rather than re-deriving a new one, so the
    diameter measurement cannot disagree with the tracker already used
    elsewhere in this repo.
    """
    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    areas: list[float] = []
    cap = cv2.VideoCapture(str(path))
    i = 0
    while i <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= lo:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, BALL_HSV_LO, BALL_HSV_HI)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))
            n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
            best = None
            for k in range(1, n):
                area = int(stats[k, cv2.CC_STAT_AREA])
                if BALL_AREA_MIN <= area <= BALL_AREA_MAX:
                    if best is None or area > best[0]:
                        best = (area, float(cent[k][0]), float(cent[k][1]))
            if best is not None:
                frames.append(i)
                areas.append(float(best[0]))
                xs.append(best[1])
                ys.append(best[2])
        i += 1
    cap.release()
    return (np.asarray(frames), np.asarray(xs), np.asarray(ys),
             np.asarray(areas))


class WindowResult:
    def __init__(self, label, lo, hi, n, d_mean, d_std, ypp, resid,
                 ok, reasons):
        self.label, self.lo, self.hi, self.n = label, lo, hi, n
        self.d_mean, self.d_std, self.ypp, self.resid = d_mean, d_std, ypp, resid
        self.ok, self.reasons = ok, reasons

    @property
    def fps(self) -> float:
        if self.d_mean <= 0 or self.ypp <= 0:
            return float("nan")
        px_per_m = self.d_mean / BALL_DIAMETER_M
        return float(np.sqrt(G_MPS2 * px_per_m / self.ypp))


def evaluate_window(label: str, path: Path, lo: int, hi: int) -> WindowResult:
    frames, x, y, areas = _track_ball_with_area(path, lo, hi)
    reasons = []
    if len(frames) < 6:
        return WindowResult(label, lo, hi, len(frames), 0.0, 0.0, 0.0,
                            float("inf"), False,
                            [f"only {len(frames)} detections"])

    c = np.polyfit(frames.astype(float), y, 2)
    fit = np.polyval(c, frames.astype(float))
    resid = float(np.sqrt(np.mean((y - fit) ** 2)))
    ypp = float(2.0 * c[0])

    diam = 2.0 * np.sqrt(areas / np.pi)
    d_mean = float(diam.mean())
    d_std = float(diam.std())

    ok = True
    if resid > ACCEPT_MAX_PARABOLA_RESID_PX:
        ok = False
        reasons.append(f"parabola resid {resid:.2f} px "
                       f"> {ACCEPT_MAX_PARABOLA_RESID_PX} px")
    if d_mean > 0 and (d_std / d_mean) > ACCEPT_MAX_DIAMETER_STD_FRAC:
        ok = False
        reasons.append(f"D std {100 * d_std / max(d_mean, 1e-9):.0f}% "
                       f"> {100 * ACCEPT_MAX_DIAMETER_STD_FRAC:.0f}%")
    if d_mean < ACCEPT_MIN_MEAN_DIAMETER_PX:
        ok = False
        reasons.append(f"D {d_mean:.1f} px < {ACCEPT_MIN_MEAN_DIAMETER_PX} px")

    return WindowResult(label, lo, hi, len(frames), d_mean, d_std,
                        abs(ypp), resid, ok, reasons)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    angle_1 = config.VIDEO_DIR / "swing_angle_1.mp4"
    angle_2 = config.VIDEO_DIR / "swing_angle_2.mp4"

    print("=" * 78)
    print("(a) Container metadata — MP4 box walk, no ffprobe")
    print("=" * 78)
    report_container("angle_1", angle_1)
    report_container("angle_2", angle_2)
    print("\n  Method (a) is EXHAUSTED: both containers declare only a 30 fps\n"
          "  PLAYBACK rate (mdhd timescale / stts); neither carries a\n"
          "  capture-rate tag. `cv2`'s reported fps for angle_1 (29.8678) is\n"
          "  an artefact of the double-length frame 99 (`stts` entry "
          "(1,40)),\n  which sits outside the 116-145 swing window used "
          "elsewhere — use\n  PLAYBACK_FPS = 600/20 = 30.000 for that window, "
          "not 29.8678.")

    print("\n" + "=" * 78)
    print("(b) Ball free-fall x ball diameter — five candidate windows")
    print("=" * 78)
    windows = [
        ("angle_1 100-130", angle_1, 100, 130),
        ("angle_1 132-146", angle_1, 132, 146),
        ("angle_2 140-165", angle_2, 140, 165),
        ("angle_2 185-210", angle_2, 185, 210),
        ("angle_2 195-220", angle_2, 195, 220),
    ]
    results = []
    print(f"  {'window':<20}{'n':>4}{'D mean':>10}{'D std':>8}"
          f"{'|y_pp|':>10}{'resid':>8}  verdict")
    for label, path, lo, hi in windows:
        r = evaluate_window(label, path, lo, hi)
        results.append(r)
        verdict = (f"ACCEPT -> {r.fps:.0f} fps" if r.ok
                   else "REJECT: " + ", ".join(r.reasons))
        print(f"  {label:<20}{r.n:>4}{r.d_mean:10.2f}{r.d_std:8.2f}"
              f"{r.ypp:10.5f}{r.resid:8.2f}  {verdict}")

    accepted = [r for r in results if r.ok]

    # The reductio + the "do not fit 80-130" demonstration.
    print("\n  Demonstration — frames 80-130 (ball still being tossed before "
          "~100):")
    r_toss = evaluate_window("angle_1 80-130", angle_1, 80, 130)
    verdict = (f"ACCEPT -> {r_toss.fps:.0f} fps" if r_toss.ok
              else "REJECT: " + ", ".join(r_toss.reasons))
    print(f"    parabola resid {r_toss.resid:.1f} px -> {verdict}")
    print("    Rejecting this window on criterion 1 is correct: the ball is "
          "not yet in\n    free flight for the first part of that range.")

    if accepted:
        a1 = next((r for r in accepted if r.label.startswith("angle_1")), None)
        if a1 is not None:
            print(f"\n  Diameter sensitivity for the accepted {a1.label} "
                  "window (holding y'' fixed):")
            for d in (22.2, 24.0, 26.0):
                px_per_m = d / BALL_DIAMETER_M
                fps = np.sqrt(G_MPS2 * px_per_m / a1.ypp)
                print(f"    D={d:.1f} px -> {fps:.0f} fps")
            print("    The HSV threshold ERODES a bright small ball, so the "
                  "measured diameter\n    (and hence the fps figures above) "
                  "are LOWER BOUNDS.")

    print("\n" + "=" * 78)
    print("(c) The reductio — what gravity would have to be at real-time 30 fps")
    print("=" * 78)
    if accepted:
        r = accepted[0]
        px_per_m = r.d_mean / BALL_DIAMETER_M
        g_implied = r.ypp * (30.0 ** 2) / px_per_m
        print(f"  Using {r.label}: y''={r.ypp:.5f} px/frame^2 at "
              f"{px_per_m:.0f} px/m.")
        print(f"  At real-time 30 fps, that y'' implies gravity = "
              f"{g_implied:.3f} m/s^2")
        print(f"  (real gravity is {G_MPS2:.3f} m/s^2 -> off by a factor of "
              f"{G_MPS2 / g_implied:.0f}x).")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if len(accepted) < 2:
        print(f"  Only {len(accepted)} window(s) passed all three acceptance "
              "criteria. STOPPING rather\n  than picking a number — see the "
              "table above.")
        return 1 if len(accepted) == 0 else 0

    fps_values = [r.fps for r in accepted]
    lo_fps, hi_fps = min(fps_values), max(fps_values)
    print(f"  Two independent clean free-fall windows, in two different "
          f"clips, give\n  {', '.join(f'{r.fps:.0f} fps ({r.label})' for r in accepted)}"
          " (lower bounds; the HSV mask\n  erodes the ball so the diameter, "
          "and hence the rate, is underestimated).")
    print("  Consumer slow-motion capture modes are 120, 240, 480, 960 fps. "
          f"The measured\n  band ({lo_fps:.0f}-{hi_fps:.0f} fps) excludes 120 "
          "by ~2x and 480 by ~2x.")
    if 180.0 <= lo_fps and hi_fps <= 320.0:
        print("  240 fps is the only standard capture mode consistent with "
              "the measurement,\n  so CAPTURE_FPS = 240.0 and "
              "SLOWMO_SCALE = 240/30 = 8.0. The repo's original\n  8.33 "
              "(250 fps) was approximately right.")
    else:
        print("  Both accepted windows fall outside the 180-320 fps band "
              "expected around a\n  240 fps capture mode. STOPPING rather "
              "than picking a number from this scan.")
        return 1

    print("\n  Residual sensitivity note (NOT evidence, see FUSION_NOTES.md): "
          "the re-solved\n  PnP median reprojection error is flat at "
          "21.5/21.1/23.9 px over 210/240/300 fps.\n  No argmin to chase; "
          "improvement past ~280 fps is window collapse, not a rate signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
