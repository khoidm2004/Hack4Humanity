"""Phase C: load the IMU CSV, derive its time base, and detect its events.

Every index here is *computed* from the signal. The impact is known by eye to
sit near sample 200 in `data/raw_data_visualized.png`; that knowledge is used
only as an assertion on the detector's output, never as an input to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .paths import IMU_SAMPLE_RATE_HZ

ACCEL_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]

# Onset refinement: walk back from the jerk peak to the first sample that is
# still above this fraction of it.
ONSET_FRACTION = 0.20
ONSET_LOOKBACK = 12

# High-pass corner for the confirming accelerometer detector. Racquet-frame
# ringing after ball contact lives well above this; the swing itself does not.
HIGHPASS_HZ = 50.0
ENVELOPE_SMOOTH = 3

# Motion onset: gyro magnitude leaving its resting baseline.
BASELINE_SAMPLES = 40
MOTION_ONSET_K = 8.0


@dataclass
class ImuEvents:
    df: pd.DataFrame
    sample_rate_hz: float
    t_sample: np.ndarray
    impact_index: int
    impact_index_gyro: int
    impact_index_accel: int
    accel_cols_used: list[str]
    motion_onset_index: int
    gyro_mag: np.ndarray
    gyro_jerk: np.ndarray
    hf_accel_envelope: np.ndarray
    saturation: dict = field(default_factory=dict)

    @property
    def t_impact(self) -> float:
        return float(self.t_sample[self.impact_index])

    @property
    def t_motion_onset(self) -> float:
        return float(self.t_sample[self.motion_onset_index])

    @property
    def detector_disagreement(self) -> int:
        return abs(self.impact_index_gyro - self.impact_index_accel)

    def summary_lines(self) -> list[str]:
        lines = [
            f"  samples              : {len(self.df)} @ {self.sample_rate_hz:g} Hz "
            f"(assumed rate, not measured)",
            f"  clip duration        : {len(self.df) / self.sample_rate_hz:.4f} s",
            f"  impact (gyro jerk)   : index {self.impact_index_gyro} "
            f"= {self.t_sample[self.impact_index_gyro]:.4f} s",
            f"  impact (HF accel)    : index {self.impact_index_accel} "
            f"= {self.t_sample[self.impact_index_accel]:.4f} s "
            f"(channels {'+'.join(self.accel_cols_used)})",
            f"  detector disagreement: {self.detector_disagreement} sample(s)",
            f"  IMPACT (primary)     : index {self.impact_index} "
            f"= {self.t_impact:.4f} s",
            f"  motion onset         : index {self.motion_onset_index} "
            f"= {self.t_motion_onset:.4f} s",
            f"  swing duration       : {self.t_impact - self.t_motion_onset:.4f} s "
            f"(onset -> impact)",
        ]
        for col, info in self.saturation.items():
            if info["n_saturated"]:
                lines.append(
                    f"  SATURATED {col:<3}        : {info['n_saturated']} samples at "
                    f"{info['value']:+.3f} (longest run {info['longest_run']}, "
                    f"index {info['first_index']}..{info['last_index']})"
                )
        if not any(i["n_saturated"] for i in self.saturation.values()):
            lines.append("  saturation           : none detected")
        return lines


def _detect_saturation(x: np.ndarray, min_run: int = 3) -> dict:
    """Find runs of identical values at the signal's extreme - i.e. clipping."""
    out = {"n_saturated": 0, "value": float("nan"), "longest_run": 0,
           "first_index": -1, "last_index": -1}
    for extreme in (float(np.max(x)), float(np.min(x))):
        at = np.isclose(x, extreme, rtol=0, atol=1e-9)
        if at.sum() < min_run:
            continue
        # Longest consecutive run of the extreme value.
        idx = np.flatnonzero(at)
        splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
        longest = max(splits, key=len)
        if len(longest) < min_run:
            continue
        if int(at.sum()) > out["n_saturated"]:
            out = {
                "n_saturated": int(at.sum()),
                "value": extreme,
                "longest_run": int(len(longest)),
                "first_index": int(idx[0]),
                "last_index": int(idx[-1]),
            }
    return out


def _refine_to_onset(x: np.ndarray, peak: int, fraction: float,
                     lookback: int) -> int:
    """Walk back from `peak` to the first sample still above `fraction` of it."""
    thr = fraction * float(x[peak])
    i = peak
    lo = max(0, peak - lookback)
    while i > lo and x[i - 1] >= thr:
        i -= 1
    return int(i)


def detect_impact_gyro(gyro: np.ndarray) -> tuple[int, np.ndarray]:
    """Primary detector: the gyro *jerk* (sample-to-sample change) spikes at ball
    contact, because the racquet is struck and reverses angular rate within one
    or two samples. Returns (onset-refined index, jerk signal)."""
    jerk = np.linalg.norm(np.diff(gyro, axis=0), axis=1)
    jerk = np.concatenate([[0.0], jerk])  # align to sample index
    peak = int(np.argmax(jerk))
    return _refine_to_onset(jerk, peak, ONSET_FRACTION, ONSET_LOOKBACK), jerk


def detect_impact_accel_hf(accel: np.ndarray, fs: float) -> tuple[int, np.ndarray]:
    """Confirming detector: onset of high-frequency accelerometer energy.

    The racquet frame rings at ~100 Hz after contact. High-passing the
    accelerometer isolates that ringing from the swing's low-frequency arc; the
    first large threshold crossing is ball contact.

    `accel` must contain only non-saturated channels: a clipped channel has a
    hard corner where it enters and leaves the rail, and the high-pass turns
    that corner into a false onset tens of samples early. `load_imu_events`
    drops saturated channels before calling this.
    """
    nyq = fs / 2.0
    wn = min(HIGHPASS_HZ / nyq, 0.95)
    sos = signal.butter(4, wn, btype="highpass", output="sos")
    # A *causal* filter. `sosfiltfilt` is zero-phase but non-causal, so it leaks
    # the impact's energy ~25 samples backwards and the onset lands early.
    hp = signal.sosfilt(sos, accel, axis=0)
    # The vector magnitude of a high-passed signal is already a rectified
    # amplitude. Running `hilbert` over it (as for an analytic envelope) smears
    # the onset by tens of samples and is not meaningful on a rectified signal;
    # a short moving average is all the smoothing this needs.
    mag = np.linalg.norm(hp, axis=1)
    envelope = np.convolve(mag, np.ones(ENVELOPE_SMOOTH) / ENVELOPE_SMOOTH,
                           mode="same")
    peak = int(np.argmax(envelope))
    thr = ONSET_FRACTION * float(envelope[peak])
    above = np.flatnonzero(envelope[: peak + 1] >= thr)
    onset = int(above[0]) if above.size else peak
    return onset, envelope


def detect_motion_onset(gyro_mag: np.ndarray) -> int:
    """First sample where gyro magnitude leaves its resting baseline."""
    base = gyro_mag[:BASELINE_SAMPLES]
    thr = float(base.mean() + MOTION_ONSET_K * (base.std() + 1e-9))
    thr = max(thr, 0.05 * float(gyro_mag.max()))
    above = np.flatnonzero(gyro_mag >= thr)
    return int(above[0]) if above.size else 0


def load_imu_events(csv_path: str | Path,
                    sample_rate_hz: float = IMU_SAMPLE_RATE_HZ) -> ImuEvents:
    df = pd.read_csv(csv_path)
    t_sample = df["index"].to_numpy(dtype=float) / sample_rate_hz

    gyro = df[GYRO_COLS].to_numpy(dtype=float)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    saturation = {c: _detect_saturation(df[c].to_numpy(dtype=float))
                  for c in ACCEL_COLS + GYRO_COLS}

    idx_gyro, jerk = detect_impact_gyro(gyro)
    clean_accel_cols = [c for c in ACCEL_COLS if not saturation[c]["n_saturated"]]
    if not clean_accel_cols:
        clean_accel_cols = ACCEL_COLS
    idx_accel, envelope = detect_impact_accel_hf(
        df[clean_accel_cols].to_numpy(dtype=float), sample_rate_hz)
    motion_onset = detect_motion_onset(gyro_mag)

    return ImuEvents(
        df=df,
        sample_rate_hz=sample_rate_hz,
        t_sample=t_sample,
        # The gyro-jerk detector is the primary: it is a step change in a
        # non-saturated channel, so it is sharper than the accel envelope.
        impact_index=int(idx_gyro),
        impact_index_gyro=int(idx_gyro),
        impact_index_accel=int(idx_accel),
        accel_cols_used=clean_accel_cols,
        motion_onset_index=int(motion_onset),
        gyro_mag=gyro_mag,
        gyro_jerk=jerk,
        hf_accel_envelope=envelope,
        saturation=saturation,
    )


def band_limited_gyro(events: ImuEvents, video_fps: float) -> np.ndarray:
    """Low-pass gyro magnitude below the video's Nyquist, for cross-correlation.

    Without this the IMU's 100+ Hz post-impact ringing - which has no counterpart
    in 30 fps video - biases the correlation.
    """
    nyq_imu = events.sample_rate_hz / 2.0
    cutoff = min(0.9 * (video_fps / 2.0), 0.45 * events.sample_rate_hz)
    sos = signal.butter(4, cutoff / nyq_imu, btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, events.gyro_mag)
