"""Per-video IMU↔video sync tables (produced by ``scripts/run_sync.py``).

Every video in ``data/video/`` has a matching ``data/synced_data_<label>.csv``
whose rows carry:

* ``t_sample`` — CSV-side time (index / 416)
* ``t_video``   — time on that video's clock (t_sample + offset), NaN if outside
* ``frame_exact`` — fractional frame index (PTS-interpolated)
* ``frame_index`` — nearest integer frame, ``-1`` if outside the video

``swing_angle_2.mp4`` is a different take with no ball strike, so its table is
*not* syncable (all ``frame_index = -1``); the app must handle that explicitly.

``frame_exact``/``frame_index`` are **rescaled at load time** by
``config.SLOWMO_SCALE`` about ``config.SYNC_ANCHOR_FRAME`` (see
``_apply_slowmo`` below) — the CSV on disk is the unscaled 30 fps table built
by ``scripts/run_sync.py`` under the (wrong) real-time assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .sync import paths as sync_paths
from .video import probe as _probe_video


def _apply_slowmo(label, video_path, frame_exact, frame_index):
    """Rescale a sync table from playback time to capture time.

    `data/synced_data_<label>.csv` was built by `scripts/run_sync.py` under the
    assumption that the footage is real-time 30 fps, so its `frame_exact` runs
    at `PLAYBACK_FPS / FS` = 30/416 frames per IMU sample.  The clips are
    actually ~240 fps slow motion (see `config.CAPTURE_FPS`), so the true rate
    is `CAPTURE_FPS / FS` = 240/416 - `config.SLOWMO_SCALE` times steeper.

    The rescale is applied ABOUT the ball-contact anchor, which is the single
    event `src/sync/align.py` fitted the offset to, so that correspondence is
    preserved exactly and only the rate changes:

        frame_exact_new = anchor + (frame_exact_old - anchor) * SLOWMO_SCALE

    This is done here, at load time, rather than by regenerating the CSV: the
    committed sync tables stay byte-identical on disk and every caller
    (`src/overlay_calib.py`, `app.py`) picks the correction up for free.

    `scale == 1.0`, a label with no anchor, or an unsyncable table all return
    the inputs unchanged.  Samples whose rescaled frame falls outside the
    decoded clip get `frame_index = -1`, the same contract
    `src/sync/align.build_synced_frame` uses.
    """
    scale = float(config.SLOWMO_SCALE)
    anchor = config.SYNC_ANCHOR_FRAME.get(label)
    if anchor is None or abs(scale - 1.0) < 1e-9 or len(frame_exact) == 0:
        return frame_exact, frame_index

    fe = np.asarray(frame_exact, float).copy()
    fi = np.asarray(frame_index, int).copy()
    good = np.isfinite(fe) & (fi >= 0)
    if not good.any():
        return fe, fi

    fe[good] = float(anchor) + (fe[good] - float(anchor)) * scale
    fi[good] = np.rint(fe[good]).astype(int)

    # Mark samples the widened window pushes off the end of the clip.
    n_frames = None
    try:
        n_frames = int(_probe_video(str(video_path)).n_frames)
    except Exception:                                      # noqa: BLE001
        n_frames = None
    outside = good & ((fe < -0.5) | ((fe > n_frames - 0.5) if n_frames else False))
    fi[outside] = -1
    fe[outside] = np.nan
    return fe, fi


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
            frame_exact, frame_index = _apply_slowmo(
                label, video_path, frame_exact, frame_index)
            # Any valid row means the video is syncable to the CSV.
            syncable = bool((frame_index >= 0).any())
            if syncable:
                valid = frame_index >= 0
                # NOTE: `t_video`/`offset_s` describe the UNRESCALED table and
                # are informational only — nothing outside this class reads
                # them, and with SLOWMO_SCALE != 1 the relation is no longer a
                # pure offset.  `frame_exact`/`frame_index` are authoritative.
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

    def covers_frame(self, frame: int, tol: float = 1.0) -> bool:
        """True when some valid sample maps within ``tol`` frames of ``frame``.

        The lookups clamp to the nearest valid sample, which is right for a
        scrub slider spanning the whole clip but means a frame outside the IMU
        record silently returns the record's first or last sample.  Callers use
        this to say so instead of drawing a frozen overlay as if it were real.
        """
        if not self.syncable or len(self.frame_exact) == 0:
            return False
        valid = np.flatnonzero((self.frame_index >= 0)
                               & np.isfinite(self.frame_exact))
        if valid.size == 0:
            return False
        return bool(np.min(np.abs(self.frame_exact[valid] - float(frame))) <= tol)
