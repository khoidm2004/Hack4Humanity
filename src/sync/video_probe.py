"""Phase A: decode-and-measure a video file.

Nothing else in the sync pipeline may trust container metadata. This module
decodes every frame, counts them itself, collects per-frame presentation
timestamps and decides CFR vs VFR. Everything downstream consumes the
`VideoProbe` it produces.

OpenCV here is the *headless* build (`opencv-python-headless`), so there is no
`cv2.imshow`/`waitKey`. This module only measures and builds in-memory frame
data - writing anything to disk (as PNG) is a `scripts/` job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# An inter-frame interval that varies by more than this fraction of the median
# is treated as evidence of variable frame rate.
VFR_REL_TOLERANCE = 0.05


@dataclass
class VideoProbe:
    """Measured (not metadata-trusted) description of one video file."""

    path: Path
    width: int
    height: int
    reported_fps: float
    reported_frame_count: int
    decoded_frame_count: int
    # Per-frame presentation time in seconds, length == decoded_frame_count.
    frame_times: np.ndarray
    # True when CAP_PROP_POS_MSEC gave usable, strictly increasing timestamps.
    timestamps_usable: bool
    is_vfr: bool
    fps_source: str
    measured_fps: float
    interval_stats: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Duration using the frame-covers-[t_i, t_i+1) convention."""
        if self.decoded_frame_count == 0:
            return 0.0
        return float(self.frame_times[-1]) + 1.0 / self.measured_fps

    def frame_for_time(self, t: float | np.ndarray) -> np.ndarray:
        """Map video-clock time(s) to frame index using a nearest-frame rule.

        Returns -1 for times that fall outside the decoded video, where
        "outside" means before the first frame's presentation time minus half a
        frame, or after the last frame's time plus half a frame. Callers must
        mark those rows rather than clipping them.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        times = self.frame_times
        # Nearest frame: searchsorted gives the insertion point, then compare
        # the neighbours on either side. This is correct for VFR too, because
        # it never assumes a constant interval.
        idx = np.searchsorted(times, t_arr, side="left")
        idx = np.clip(idx, 1, len(times) - 1)
        left = times[idx - 1]
        right = times[idx]
        nearest = np.where(np.abs(t_arr - left) <= np.abs(right - t_arr), idx - 1, idx)
        if len(times) == 1:
            nearest = np.zeros_like(t_arr, dtype=int)

        half = 0.5 / self.measured_fps
        outside = (t_arr < times[0] - half) | (t_arr > times[-1] + half)
        nearest = np.where(outside, -1, nearest)
        return nearest.astype(int)

    def summary_lines(self) -> list[str]:
        s = self.interval_stats
        lines = [
            f"  file                 : {self.path}",
            f"  resolution           : {self.width} x {self.height}",
            f"  reported fps         : {self.reported_fps:.6g}",
            f"  reported frame count : {self.reported_frame_count}",
            f"  DECODED frame count  : {self.decoded_frame_count}",
            f"  fps used (source)    : {self.measured_fps:.6f}  ({self.fps_source})",
            f"  duration (decoded)   : {self.duration_s:.4f} s",
            f"  POS_MSEC usable      : {self.timestamps_usable}",
        ]
        if s:
            lines += [
                "  inter-frame interval (ms): "
                f"min={s['min_ms']:.3f} median={s['median_ms']:.3f} "
                f"max={s['max_ms']:.3f} std={s['std_ms']:.4f}",
                f"  max |dev| from median : {s['max_rel_dev'] * 100:.2f} %",
                f"  irregular intervals  : {s['n_irregular_intervals']} / {s['n_intervals']}",
            ]
        if not self.is_vfr:
            verdict = "CFR (every interval within tolerance of the median)"
        elif s and s["n_irregular_intervals"] <= 0.02 * s["n_intervals"]:
            verdict = (
                f"near-CFR: {s['n_irregular_intervals']} irregular interval(s) "
                "(dropped frame(s)); per-frame PTS array is used regardless"
            )
        else:
            verdict = "VFR (genuinely variable) - per-frame PTS array required"
        lines.append(f"  VERDICT              : {verdict}")
        return lines


def _fallback_fps(frame_times: np.ndarray, reported_fps: float) -> tuple[float, str]:
    """Pick a trustworthy fps: median PTS diff first, container metadata last."""
    if len(frame_times) > 1:
        diffs = np.diff(frame_times)
        median = float(np.median(diffs))
        if math.isfinite(median) and median > 1e-6:
            return 1.0 / median, "median inter-frame PTS diff (measured)"
    if math.isfinite(reported_fps) and 0.1 < reported_fps < 10000:
        return float(reported_fps), "container CAP_PROP_FPS (no usable PTS)"
    raise RuntimeError(
        "Cannot establish a frame rate: no usable timestamps and no sane "
        "CAP_PROP_FPS. Adding `av` for exact PTS would be justified here."
    )


def probe_video(path: str | Path) -> VideoProbe:
    """Decode `path` end to end and measure it.

    Every frame is decoded and counted; `CAP_PROP_FRAME_COUNT` is reported but
    never trusted. This function only measures - it performs no file I/O;
    callers that want sample frames or a contact sheet written to disk should
    use `sample_frame_indices`/`build_contact_sheet` together with
    `read_frames` from a `scripts/` entry point.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")

    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # First pass: decode everything, collect POS_MSEC for each frame.
    #
    # POS_MSEC must be read *after* `read()`: measured on both source files,
    # reading it before returns the previous frame's value (a stale duplicate at
    # index 1), while reading it after yields the strictly increasing PTS of the
    # frame just decoded, starting at exactly 0.
    pos_msec: list[float] = []
    n = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        pos_msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        n += 1
    cap.release()

    if n == 0:
        raise RuntimeError(f"Decoded zero frames from {path}")

    raw_times = np.asarray(pos_msec, dtype=float) / 1000.0
    # Usable means: finite, non-degenerate, and non-decreasing.
    finite = np.all(np.isfinite(raw_times))
    spread = float(raw_times.max() - raw_times.min()) if finite else 0.0
    monotonic = finite and bool(np.all(np.diff(raw_times) >= -1e-9))
    timestamps_usable = bool(finite and monotonic and spread > 1e-6 and n > 1)

    if timestamps_usable:
        frame_times = raw_times - raw_times[0]
        measured_fps, fps_source = _fallback_fps(frame_times, reported_fps)
    else:
        measured_fps, fps_source = _fallback_fps(np.empty(0), reported_fps)
        frame_times = np.arange(n, dtype=float) / measured_fps
        fps_source += " -> synthetic CFR time base"

    interval_stats: dict = {}
    is_vfr = False
    if timestamps_usable and n > 2:
        diffs = np.diff(frame_times)
        median = float(np.median(diffs))
        rel_dev = float(np.max(np.abs(diffs - median)) / median) if median > 0 else 0.0
        irregular = int(np.sum(np.abs(diffs - median) > VFR_REL_TOLERANCE * median))
        interval_stats = {
            "min_ms": float(diffs.min() * 1000),
            "median_ms": median * 1000,
            "max_ms": float(diffs.max() * 1000),
            "std_ms": float(diffs.std() * 1000),
            "max_rel_dev": rel_dev,
            "n_intervals": int(len(diffs)),
            "n_irregular_intervals": irregular,
        }
        is_vfr = irregular > 0

    probe = VideoProbe(
        path=path,
        width=width,
        height=height,
        reported_fps=reported_fps,
        reported_frame_count=reported_frame_count,
        decoded_frame_count=n,
        frame_times=frame_times,
        timestamps_usable=timestamps_usable,
        is_vfr=is_vfr,
        fps_source=fps_source,
        measured_fps=measured_fps,
        interval_stats=interval_stats,
    )

    return probe


def sample_frame_indices(probe: VideoProbe, n: int = 12) -> np.ndarray:
    """Pick ~n evenly spaced, de-duplicated frame indices across the video."""
    return np.unique(np.linspace(0, probe.decoded_frame_count - 1, n).astype(int))


def build_contact_sheet(probe: VideoProbe, cols: int = 14,
                        thumb_w: int = 192) -> np.ndarray:
    """Build one numbered grid image of every frame in the video, in memory.

    This is what answers "what actually happens in this clip, and is it the same
    take as the other one" in a single look, without scrubbing a player that the
    headless OpenCV build cannot open. Returns the image array; writing it to
    disk is the caller's job (see `scripts/probe_videos.py`).
    """
    thumb_h = int(round(thumb_w * probe.height / probe.width))

    cap = cv2.VideoCapture(str(probe.path))
    thumbs: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        thumbs.append(cv2.resize(frame, (thumb_w, thumb_h),
                                 interpolation=cv2.INTER_AREA))
    cap.release()

    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        cv2.putText(thumb, str(i), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)
        sheet[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

    return sheet


def read_frames(path: str | Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Return the requested frames, decoded sequentially (seeking is unreliable)."""
    wanted = set(int(i) for i in indices if i >= 0)
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(path))
    last = max(wanted)
    i = 0
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = frame
        i += 1
    cap.release()
    return out
