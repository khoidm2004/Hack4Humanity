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
    gyro_deg : (N, 3) float, deg/s
        Raw gyroscope signal carried through for plotting / calibration.
    accel : (N, 3) float, m/s2
        Raw accelerometer signal carried through for plotting / metrics.
    impact_idx : int
        Sample index of the ball-contact (impact) moment.
    """

    t: np.ndarray
    quat: np.ndarray
    tip: np.ndarray
    cog: np.ndarray
    gyro_deg: np.ndarray
    accel: np.ndarray
    impact_idx: int

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
            )
