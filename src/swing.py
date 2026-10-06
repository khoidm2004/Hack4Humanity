"""Person A — the CONTRACT dataclass (PLAN.md §5).

``Swing`` is the single artifact produced by the sensor-fusion stage and
consumed by the video / overlay stage.  It is persisted to ``swing.npz``
so Person A and Person B can develop in parallel against a stable interface.

Field ordering and units are FROZEN — treat them as a public API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Swing:
    """Reconstructed racket swing.

    Attributes
    ----------
    t : (N,) float, seconds
        Time axis for every sample.
    quat : (N, 4) float, w,x,y,z
        World orientation of the racket (scipy Rotation.as_quat convention).
    tip : (N, 3) float, metres
        Racket head-tip position in world space (pivot model).
    cog : (N, 3) float, metres
        Sensor (centre-of-gravity-ish) position in world space
        (rest-to-rest model).
    pivot : (N, 3) float, metres
        Wrist (pivot) position in world space, built by
        ``fusion.wrist_pivot(cog, removed_slope=...)`` —
        ``WRIST_PIVOT_COG_SCALE * cog`` plus the linear drift term
        ``config.WRIST_PIVOT_DRIFT_GAIN`` restores (see ``src/config.py``).
        Defaults to zeros, which is the pre-2026-10-05 behaviour (the
        wrist pinned at the world origin) and is what an older ``swing.npz``
        without this array loads as.
    gyro_deg : (N, 3) float, deg/s
        Raw gyroscope signal carried through for plotting / calibration.
    accel : (N, 3) float, m/s2
        Raw accelerometer signal carried through for plotting / metrics.
    impact_idx : int
        Sample index of the ball-contact (impact) moment.

    Notes
    -----
    ``tip`` is the head position **relative to the pivot**, so
    ``|tip| == config.RACKET_TIP_LEN`` exactly, for every sample, always.  The
    head's WORLD position is ``pivot + tip`` — use the derived :attr:`head`.
    `pivot` is a separate field rather than being folded into `tip` on purpose:
    folding would break `scripts/verify_fusion.py`'s radius check, both of its
    lever-direction checks, and `overlay_calib.racket_pixel_scale`'s
    L-cancellation argument simultaneously (|tip + 0.907*cog| runs
    0.686 -> ~1.5 m).  See Artifacts/analysis.md §9.3.
    """

    t: np.ndarray
    quat: np.ndarray
    tip: np.ndarray
    cog: np.ndarray
    gyro_deg: np.ndarray
    accel: np.ndarray
    impact_idx: int
    pivot: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Zeros, not None, for every consumer downstream: `pivot + tip` then
        # works unconditionally and an older npz with no `pivot` array loads
        # as the pre-2026-10-05 pinned-wrist model.
        if self.pivot is None:
            self.pivot = np.zeros_like(np.asarray(self.tip, float))

    @property
    def head(self) -> np.ndarray:
        """(N, 3) float, metres — racket head tip in WORLD space.

        ``pivot + tip``.  Every consumer that wants the head's position on
        screen or in the world wants THIS, not ``tip``: `tip` is the lever
        relative to the pivot.  Exposed as a property so the sum is written
        once instead of at seven call sites.  Cheap — one (N, 3) add.
        """
        return self.pivot + self.tip

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Check internal consistency of the record.

        Raises
        ------
        ValueError
            If any array has a wrong shape or a length mismatch.
        """
        n = len(self.t)
        for name, arr in (
            ("quat", self.quat),
            ("tip", self.tip),
            ("cog", self.cog),
            ("pivot", self.pivot),
            ("gyro_deg", self.gyro_deg),
            ("accel", self.accel),
        ):
            expected = (n, 4) if name == "quat" else (n, 3)
            if arr.shape != expected:
                raise ValueError(
                    f"Swing.{name} expected shape {expected}, got {arr.shape}"
                )
        if not (0 <= self.impact_idx < n):
            raise ValueError(
                f"impact_idx {self.impact_idx} out of range [0, {n})"
            )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path=None) -> None:
        """Persist the swing to ``swing.npz`` (the Person A -> B contract)."""
        path = config.SWING_NPZ if path is None else str(path)
        self.validate()
        out_path = config.PROJECT_ROOT / path if not __import__("pathlib").Path(
            path
        ).is_absolute() else __import__("pathlib").Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            t=self.t,
            quat=self.quat,
            tip=self.tip,
            cog=self.cog,
            pivot=self.pivot,
            gyro_deg=self.gyro_deg,
            accel=self.accel,
            impact_idx=np.int64(self.impact_idx),
        )

    # ------------------------------------------------------------------ #
    # Alternative constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path=None) -> "Swing":
        """Load a swing from ``swing.npz``."""
        path = config.SWING_NPZ if path is None else str(path)
        with np.load(path) as z:
            return cls(
                t=z["t"],
                quat=z["quat"],
                tip=z["tip"],
                cog=z["cog"],
                gyro_deg=z["gyro_deg"],
                accel=z["accel"],
                impact_idx=int(z["impact_idx"]),
                # Back-compat: an npz written before `pivot` existed has no
                # such array.  `None` -> `__post_init__` fills zeros, i.e. the
                # old pinned-wrist model, so old files still load.
                pivot=(z["pivot"] if "pivot" in z else None),
            )
