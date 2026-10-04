"""Per-video IMU↔video sync tables (produced by ``scripts/run_sync.py``).

Every video in ``data/video/`` has a matching ``data/synced_data_<label>.csv``
whose rows carry:

* ``t_sample`` — CSV-side time (index / 416)
* ``t_video``   — time on that video's clock (t_sample + offset), NaN if outside
* ``frame_exact`` — fractional frame index (PTS-interpolated)
* ``frame_index`` — nearest integer frame, ``-1`` if outside the video

``swing_angle_2.mp4`` is a different take with no ball strike, so its table is
*not* syncable (all ``frame_index = -1``); the app must handle that explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .sync import paths as sync_paths


@dataclass
class SyncedData:
    """Lookup table mapping IMU samples ↔ video frames for one recording."""

    label: str
    video_path: Path
    csv_path: Path
    t_sample: np.ndarray      # (N,) — CSV-side seconds
    frame_index: np.ndarray   # (N,) int — nearest frame, -1 if outside video
    syncable: bool
    offset_s: float | None    # CSV -> video offset, when syncable
    frame_exact: np.ndarray = field(default_factory=lambda: np.array([]))
    # (N,) float — fractional (PTS-interpolated) frame index; NaN outside the
    # video.  Kept alongside ``frame_index`` because the rounded integer loses
    # the sub-frame information that ``imu_idx_for_frame_centered`` needs.

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def for_video(cls, label: str, video_path, csv_path) -> "SyncedData":
        """Load the sync table for one camera angle."""
        video_path = Path(video_path)
        csv_path = Path(csv_path)
        syncable = bool(csv_path.exists())
        offset_s: float | None = None

        t_sample = np.array([], dtype=float)
        frame_index = np.array([], dtype=int)
        frame_exact = np.array([], dtype=float)

        if syncable:
            df = pd.read_csv(csv_path)
            t_sample = df["t_sample"].to_numpy(dtype=float)
            frame_index = df["frame_index"].to_numpy(dtype=int)
            frame_exact = df["frame_exact"].to_numpy(dtype=float)
            # Any valid row means the video is syncable to the CSV.
            syncable = bool((frame_index >= 0).any())
            if syncable:
                valid = frame_index >= 0
                offset_s = float(np.median(df.loc[valid, "t_video"]
                                             - df.loc[valid, "t_sample"]))

        return cls(
            label=label,
            video_path=video_path,
            csv_path=csv_path,
            t_sample=t_sample,
            frame_index=frame_index,
            syncable=syncable,
            offset_s=offset_s,
            frame_exact=frame_exact,
        )

    @staticmethod
    def all_angles() -> dict[str, "SyncedData"]:
        """Load the sync tables for every angle defined in ``paths.ANGLES``."""
        out = {}
        for label, video, csv_out in sync_paths.ANGLES:
            out[Path(video).name] = SyncedData.for_video(label, video, csv_out)
        return out

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def frame_for_imu(self, imu_idx: int) -> int:
        """Video frame that the given IMU sample maps to (valid samples only)."""
        if not self.syncable:
            return -1
        return int(self.frame_index[int(imu_idx)])

    def imu_idx_for_frame(self, frame: int) -> int:
        """Nearest IMU sample whose mapped frame matches ``frame``.

        Falls back to the closest *valid* sample (frame_index >= 0); if the
        whole table is unsyncable, returns 0.
        """
        if not self.syncable or len(self.frame_index) == 0:
            return 0
        valid = np.flatnonzero(self.frame_index >= 0)
        if valid.size == 0:
            return 0
        return int(valid[np.argmin(np.abs(self.frame_index[valid] - int(frame)))])

    def imu_idx_for_frame_centered(self, frame: int) -> int:
        """IMU sample closest to the **centre** of video frame ``frame``.

        ``imu_idx_for_frame`` argmins over the *rounded* ``frame_index``, so
        for any frame with several samples in its bin it returns the **first**
        one — whose ``frame_exact`` sits near ``frame - 0.5``.  At ~13.9 IMU
        samples per video frame that is a systematic half-frame (~7 sample)
        bias, and near impact the tip moves ~0.45 m inside one frame bin, so
        the bias is worth ~0.2 m of 3D position.  Minimising
        ``|frame_exact - frame|`` instead lands on the sample whose video time
        actually matches the frame.

        Frames outside the synced window clamp to the nearest valid sample,
        the same contract as :meth:`imu_idx_for_frame` (the app's scrub slider
        spans the whole clip, not just the swing).
        """
        if not self.syncable or len(self.frame_index) == 0:
            return 0
        if len(self.frame_exact) != len(self.frame_index):
            return self.imu_idx_for_frame(frame)
        valid = np.flatnonzero((self.frame_index >= 0)
                               & np.isfinite(self.frame_exact))
        if valid.size == 0:
            return self.imu_idx_for_frame(frame)
        return int(valid[np.argmin(np.abs(self.frame_exact[valid]
                                          - float(frame)))])

    def imu_idx_for_frame_exact(self, frame_exact: float) -> int:
        """IMU sample closest to a **fractional** frame index.

        Used for the sub-frame ball-contact instant, which does not land on
        an integer frame.
        """
        if not self.syncable or len(self.frame_exact) == 0:
            return 0
        valid = np.flatnonzero((self.frame_index >= 0)
                               & np.isfinite(self.frame_exact))
        if valid.size == 0:
            return 0
        return int(valid[np.argmin(np.abs(self.frame_exact[valid]
                                          - float(frame_exact)))])

    def impact_frame(self, impact_idx: int) -> int:
        """Frame of the impact marker (or the nearest valid one)."""
        if not self.syncable:
            return -1
        f = self.frame_index[int(impact_idx)]
        if f >= 0:
            return int(f)
        return self.imu_idx_for_frame(f)
