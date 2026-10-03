"""Person B — video I/O & rough time mapping (PLAN.md §7 Phase 2: `video.py`).

Thin OpenCV wrapper that reads frame metadata and individual frames, plus the
rough IMU↔video time mapping used by path-projection:
``video_time = imu_time × slowmo_scale + offset`` (PLAN.md §8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass
class VideoMeta:
    """Basic metadata extracted with :func:`probe`."""

    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoMeta:
    """Open a video and read its metadata (no frames decoded)."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        return VideoMeta(
            path=str(path),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def frame(path: str, index: int, gray: bool = False) -> np.ndarray:
    """Return frame ``index`` as a BGR uint8 array (or grayscale)."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, img = cap.read()
        if not ok or img is None:
            raise IndexError(f"frame {index} unavailable in {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img
    finally:
        cap.release()


def read_annotated_frame(
    path: str,
    draw: "callable",
    *,
    frame_idx: int,
) -> np.ndarray:
    """Fetch a frame and run an external ``draw`` callback for overlays."""
    img = frame(path, frame_idx)
    return draw(img)


# --------------------------------------------------------------------------- #
# Rough IMU -> video time mapping
# --------------------------------------------------------------------------- #
def map_imu_to_video(swing_t: np.ndarray, fps: float, scale: float,
                     offset: float) -> np.ndarray:
    """Map IMU time (s) onto video frame indices (rough, PLAN.md §8).

        video_time = t_imu × scale + offset
        frame_idx  = round(video_time × fps)

    The same linear relation drives the animated path-projection dot.

    Parameters
    ----------
    swing_t : (N,) float, seconds
    fps : float, video frames per second
    scale : float, slow-motion factor (default config.SLOWMO_SCALE ≈ 8.33)
    offset : float, seconds added to align the swing to the video.

    Returns
    -------
    (N,) int32 array of frame indices (clamped >= 0).
    """
    video_time = swing_t * float(scale) + float(offset)
    idx = np.round(video_time * float(fps)).astype(np.int64)
    return np.clip(idx, 0, None)
