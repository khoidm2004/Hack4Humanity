"""Annotation aid for the overlay calibration (see ``src/overlay_calib.py``).

Writes, per requested video frame, images that make a *quantitative* by-eye
read of the racket-head pixel possible in a headless environment:

* ``f<NNN>_full.png``  — the full frame with a labelled coordinate grid
  (lines every 100 px, ticks every 50 px, numeric labels on both axes).
* ``f<NNN>_crop.png``  — a 2x upscaled crop around the frame-difference blob
  cluster, with its own grid **labelled in original frame coordinates**.
* ``f<NNN>_diff.png``  — the same crop region of the thresholded frame
  difference ``|I_t - I_{t-1}|``, side by side with the raw crop.  The
  motion streak is what identifies the fast-moving racket (the arm/torso
  blob is slower and rounder).

and, in ``--verify`` mode,

* ``verify_f<NNN>.png`` — the frame with the *proposed* annotation drawn on
  it (circle + crosshair + frame number + coordinates), so the annotations
  committed in ``src/overlay_calib.py`` can be re-read and confirmed rather
  than trusted.

Nothing here is imported by the app; it exists so the provenance of
``RACKET_HEAD_PX_ANGLE_1`` is reproducible.

Usage (from the repo root)::

    uv run python scripts/dump_swing_frames.py              # grids + crops
    uv run python scripts/dump_swing_frames.py --verify     # marker overlay
    uv run python scripts/dump_swing_frames.py --frames 117,120,124
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

VIDEO = ROOT / "data" / "video" / "swing_angle_1.mp4"
OUT_DIR = ROOT / "Artifacts" / "overlay_evidence" / "frames"

# Swing window from data/synced_data_angle_1.csv.
SWING_FRAMES = list(range(116, 146))

DIFF_THRESHOLD = 35
GRID_MAJOR = 100
GRID_MINOR = 50

# The racket in this clip is a dark green/teal frame with a translucent
# string bed.  Sampled HSV at hand-picked racket pixels: hue 77-88, sat
# 46-57.  The things it has to be told apart from sample as: blue court
# hue 106 / sat 116, grey trousers sat 4, beige wall hue 19 / sat 16.  The
# window below separates all three.  It is an *aid* for the by-eye read
# (drawn as a red overlay), never a substitute: it fails outright from
# frame 138 on, where the racket passes over the wooden goal board and the
# string bed shows the wood through it.
RACKET_HSV_LO = (55, 25, 70)
RACKET_HSV_HI = (100, 130, 255)


def read_frames(path: Path, lo: int, hi: int) -> dict[int, np.ndarray]:
    """Decode frames [lo, hi] sequentially (seeking per frame is unreliable)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    out: dict[int, np.ndarray] = {}
    i = 0
    while i <= hi:
        ok, img = cap.read()
        if not ok:
            break
        if i >= lo:
            out[i] = img
        i += 1
    cap.release()
    return out


def diff_mask(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    a = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    d = cv2.absdiff(b, a)
    m = (d > DIFF_THRESHOLD).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def racket_chroma_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of racket-coloured pixels (see ``RACKET_HSV_LO/HI``)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, RACKET_HSV_LO, RACKET_HSV_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def draw_grid(img: np.ndarray, *, x0: int = 0, y0: int = 0,
              zoom: float = 1.0) -> np.ndarray:
    """Draw a labelled grid in ORIGINAL frame coordinates.

    ``(x0, y0)`` is the original-frame pixel at the image's top-left corner
    and ``zoom`` the upscale factor, so a crop's labels still read as frame
    coordinates.
    """
    out = img.copy()
    h, w = out.shape[:2]
    ox_lo, ox_hi = x0, x0 + int(w / zoom)
    oy_lo, oy_hi = y0, y0 + int(h / zoom)

    def sx(ox: int) -> int:
        return int(round((ox - x0) * zoom))

    def sy(oy: int) -> int:
        return int(round((oy - y0) * zoom))

    for ox in range(ox_lo - ox_lo % GRID_MINOR, ox_hi + GRID_MINOR, GRID_MINOR):
        major = ox % GRID_MAJOR == 0
        col = (0, 200, 255) if major else (90, 90, 90)
        cv2.line(out, (sx(ox), 0), (sx(ox), h), col, 2 if major else 1)
        if major:
            cv2.putText(out, str(ox), (sx(ox) + 3, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                        cv2.LINE_AA)
    for oy in range(oy_lo - oy_lo % GRID_MINOR, oy_hi + GRID_MINOR, GRID_MINOR):
        major = oy % GRID_MAJOR == 0
        col = (0, 200, 255) if major else (90, 90, 90)
        cv2.line(out, (0, sy(oy)), (w, sy(oy)), col, 2 if major else 1)
        if major:
            cv2.putText(out, str(oy), (3, sy(oy) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                        cv2.LINE_AA)
    return out


def blob_cluster_bbox(mask: np.ndarray, frame_w: int, frame_h: int,
                      pad: int = 90) -> tuple[int, int, int, int]:
    """Bounding box of the moving pixels, padded; falls back to centre crop."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        cx, cy = frame_w // 2, frame_h // 2
        return cx - 200, cy - 150, 400, 300
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(frame_w, int(xs.max()) + pad)
    y1 = min(frame_h, int(ys.max()) + pad)
    return x0, y0, x1 - x0, y1 - y0


def dump(frames: list[int], region: tuple[int, int, int, int] | None = None,
         zoom: float = 2.0, tag: str = "", chroma: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lo, hi = min(frames) - 1, max(frames)
    imgs = read_frames(VIDEO, lo, hi)
    h, w = imgs[hi].shape[:2]

    for f in frames:
        if f not in imgs or (f - 1) not in imgs:
            print(f"  frame {f}: unavailable")
            continue
        cur, prev = imgs[f], imgs[f - 1]
        mask = diff_mask(prev, cur)

        cv2.imwrite(str(OUT_DIR / f"f{f:03d}_full.png"), draw_grid(cur))

        if chroma:
            cm = racket_chroma_mask(cur)
            cur = cur.copy()
            cur[cm > 0] = (0, 0, 255)

        if region is not None:
            bx, by, bw, bh = region
        else:
            bx, by, bw, bh = blob_cluster_bbox(mask, w, h)
        crop = cur[by:by + bh, bx:bx + bw]
        mcrop = cv2.cvtColor(mask[by:by + bh, bx:bx + bw], cv2.COLOR_GRAY2BGR)
        crop_z = cv2.resize(crop, None, fx=zoom, fy=zoom,
                            interpolation=cv2.INTER_CUBIC)
        mcrop_z = cv2.resize(mcrop, None, fx=zoom, fy=zoom,
                             interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT_DIR / f"f{f:03d}{tag}_crop.png"),
                    draw_grid(crop_z, x0=bx, y0=by, zoom=zoom))
        side = np.hstack([draw_grid(crop_z, x0=bx, y0=by, zoom=zoom),
                          draw_grid(mcrop_z, x0=bx, y0=by, zoom=zoom)])
        cv2.imwrite(str(OUT_DIR / f"f{f:03d}{tag}_diff.png"), side)
        print(f"  frame {f}: crop bbox=({bx},{by},{bw},{bh})")


def verify(frames: list[int] | None = None) -> None:
    """Re-draw the committed annotations so they can be read back by eye."""
    from src.overlay_calib import RACKET_HEAD_PX_ANGLE_1, WRIST_PX_ANGLE_1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    marks = RACKET_HEAD_PX_ANGLE_1
    want = sorted(marks) if frames is None else [f for f in frames if f in marks]
    if not want:
        print("  no annotations to verify")
        return
    imgs = read_frames(VIDEO, min(want), max(want))
    for f in want:
        if f not in imgs:
            continue
        out = draw_grid(imgs[f])
        u, v = marks[f]
        iu, iv = int(round(u)), int(round(v))
        cv2.circle(out, (iu, iv), 22, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(out, (iu - 34, iv), (iu + 34, iv), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(out, (iu, iv - 34), (iu, iv + 34), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"f{f} head ({u:.0f},{v:.0f})", (iu + 26, iv - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if f in WRIST_PX_ANGLE_1:
            wu, wv = WRIST_PX_ANGLE_1[f]
            cv2.circle(out, (int(round(wu)), int(round(wv))), 14,
                       (255, 120, 0), 2, cv2.LINE_AA)
            cv2.line(out, (iu, iv), (int(round(wu)), int(round(wv))),
                     (255, 120, 0), 1, cv2.LINE_AA)
            cv2.putText(out, "wrist", (int(wu) + 16, int(wv) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2,
                        cv2.LINE_AA)
        cv2.imwrite(str(OUT_DIR / f"verify_f{f:03d}.png"), out)
        print(f"  verify frame {f}: ({u:.1f}, {v:.1f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="",
                    help="comma-separated frame indices (default: 116-145)")
    ap.add_argument("--verify", action="store_true",
                    help="draw the committed annotations instead of grids")
    ap.add_argument("--region", default="",
                    help="fixed crop 'x,y,w,h' in frame pixels (default: the "
                         "frame-difference blob bounding box)")
    ap.add_argument("--zoom", type=float, default=2.0, help="crop upscale")
    ap.add_argument("--tag", default="", help="suffix for the crop filenames")
    ap.add_argument("--chroma", action="store_true",
                    help="paint racket-coloured pixels red in the crops")
    args = ap.parse_args()

    frames = ([int(s) for s in args.frames.split(",") if s.strip()]
              if args.frames else SWING_FRAMES)
    region = None
    if args.region:
        vals = [int(s) for s in args.region.split(",")]
        if len(vals) != 4:
            ap.error("--region needs exactly x,y,w,h")
        region = (vals[0], vals[1], vals[2], vals[3])

    print(f"writing to {OUT_DIR}")
    if args.verify:
        verify(frames if args.frames else None)
    else:
        dump(frames, region=region, zoom=args.zoom, tag=args.tag,
             chroma=args.chroma)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
