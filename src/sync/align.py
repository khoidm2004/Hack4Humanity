"""Phase E: combine the IMU and video sides into one offset per video.

The model is a pure time offset:

    t_video = index / 416 + offset

No drift or scale term is fitted. Over a 0.96 s window with a single event,
a two-parameter fit would absorb noise rather than clock error; Phase E3
instead *checks* that no scale term is needed, by scanning one and confirming
that 1.0 is the best value the data supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .imu_events import ImuEvents, band_limited_gyro
from .video_motion import (MotionEnergy, STRIKE_MIN_POST_SPEED_PX_PER_FRAME,
                           STRIKE_MIN_SPEED_RATIO)
from .video_probe import VideoProbe

# Phase E2 pass criterion from the plan.
XCORR_AGREEMENT_FRAMES = 1.0
# Phase E3 passes when 1.0 lies inside the scale plateau and the plateau is
# narrow enough to exclude time-scaled footage. The criterion is stated this way
# - rather than as a tolerance on the best-fit scale - because it keeps the
# test's own resolving power visible instead of hiding it in a threshold.
SCALE_SLOW_MOTION_FACTOR = 2.0


@dataclass
class XCorrResult:
    offset_s: float
    peak_correlation: float
    lag_vs_event_s: float
    lag_vs_event_frames: float
    agrees: bool
    # Measured bias: where each signal's dominant peak sits relative to its own
    # impact instant. Explains a disagreement without being used to correct one.
    video_peak_minus_contact_frames: float
    imu_peak_minus_impact_frames: float
    expected_bias_frames: float


@dataclass
class DriftCheck:
    """Phase E3 evidence that an offset-only model is enough.

    `best_scale` and `scale_plateau` come from a scan of a time-scale factor
    through the cross-correlation. Both are diagnostic: the scale is never
    applied to the mapping.
    """

    best_scale: float
    scale_plateau: tuple[float, float]
    unity_in_plateau: bool
    slow_motion_excluded: bool
    agrees: bool
    verdict: str


@dataclass
class Alignment:
    label: str
    probe: VideoProbe
    events: ImuEvents
    syncable: bool
    reason: str
    contact_frame_exact: float = float("nan")
    contact_frame_bracket: tuple[int, int] = (-1, -1)
    t_contact_video: float = float("nan")
    offset_s: float = float("nan")
    offset_frames: float = float("nan")
    xcorr: XCorrResult | None = None
    drift: DriftCheck | None = None
    rows_outside_video: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def error_bar_s(self) -> float:
        """One video frame: the honest bound on a 30 fps contact identification."""
        return 1.0 / self.probe.measured_fps

    @property
    def error_bar_samples(self) -> float:
        return self.error_bar_s * self.events.sample_rate_hz


def frame_time_at(probe: VideoProbe, fractional_frame: float) -> float:
    """Presentation time of a fractional frame index, interpolated on the PTS array.

    Required because `swing_angle_1.mp4` drops a frame: `i / fps` is wrong after
    the gap, the PTS array is not.
    """
    idx = np.arange(len(probe.frame_times), dtype=float)
    return float(np.interp(fractional_frame, idx, probe.frame_times))


def frame_index_at(probe: VideoProbe, t: float) -> float:
    """Inverse of `frame_time_at`: fractional frame index at a video time."""
    idx = np.arange(len(probe.frame_times), dtype=float)
    return float(np.interp(t, probe.frame_times, idx))


def cross_correlate(events: ImuEvents, probe: VideoProbe, me: MotionEnergy,
                    event_offset_s: float, contact_frame: float,
                    search_margin_s: float = 1.5) -> XCorrResult:
    """Phase E2: independent offset estimate by cross-correlation.

    Both signals are brought onto the video frame grid, band-limited to what the
    video can represent, and z-scored. The searched lag range is restricted to a
    physically plausible window around the event-based offset rather than the
    whole clip, because a 0.96 s window correlated against an 7.5 s clip has
    many spurious maxima.
    """
    gyro_lp = band_limited_gyro(events, probe.measured_fps)
    ft = me.frame_times
    energy = me.energy

    step = 1.0 / (4.0 * probe.measured_fps)  # quarter-frame lag resolution
    offsets = np.arange(event_offset_s - search_margin_s,
                        event_offset_s + search_margin_s + step, step)
    best_corr, best_off = -np.inf, float("nan")
    for off in offsets:
        tv = events.t_sample + off
        inside = (tv >= ft[0]) & (tv <= ft[-1])
        if inside.sum() < 0.5 * len(tv):
            continue
        v = np.interp(tv[inside], ft, energy)
        a = gyro_lp[inside]
        a = (a - a.mean()) / (a.std() + 1e-12)
        v = (v - v.mean()) / (v.std() + 1e-12)
        c = float(np.mean(a * v))
        if c > best_corr:
            best_corr, best_off = c, float(off)

    lag_s = best_off - event_offset_s
    fps = probe.measured_fps
    video_peak_bias = me.peak_frame - contact_frame
    imu_peak_bias = ((int(np.argmax(gyro_lp)) - events.impact_index)
                     / events.sample_rate_hz) * fps
    return XCorrResult(
        offset_s=best_off,
        peak_correlation=best_corr,
        lag_vs_event_s=lag_s,
        lag_vs_event_frames=lag_s * fps,
        agrees=abs(lag_s * fps) <= XCORR_AGREEMENT_FRAMES,
        video_peak_minus_contact_frames=float(video_peak_bias),
        imu_peak_minus_impact_frames=float(imu_peak_bias),
        expected_bias_frames=float(video_peak_bias - imu_peak_bias),
    )


def check_drift(events: ImuEvents, probe: VideoProbe, me: MotionEnergy,
                event_offset_s: float) -> DriftCheck:
    """Phase E3: rate sanity check, not a drift fit.

    A time-scale factor is scanned through the cross-correlation, re-optimising
    the offset at every scale, and the width of the resulting plateau is
    reported alongside the best value. The scale is computed only to be
    reported: over a 0.96 s window containing one event, a fitted scale would
    absorb noise rather than clock error, so the mapping stays offset-only.

    The test the data can actually settle is whether the footage is real time.
    Slow motion would put the best scale near 0.5 (2x) or 0.125 (8x) and would
    break the offset-only model completely. It cannot resolve a few percent of
    clock drift - and over 0.96 s a few percent is far under one frame anyway.

    A duration-based variant (comparing the width of the motion burst on each
    side) was tried and dropped: the IMU capture is an impact-centred window
    that ends while the racquet is still ringing, so its burst is truncated by
    the window edge and its width is not comparable to the video's.
    """
    # Scale scan: t_video = scale * t_sample + offset, re-optimising the offset
    # at every scale so the two parameters are not confounded.
    gyro_lp = band_limited_gyro(events, probe.measured_fps)
    ft, energy = me.frame_times, me.energy
    scales = np.arange(0.5, 2.01, 0.02)
    curve = []
    for scale in scales:
        best = -np.inf
        for off in np.arange(event_offset_s - 0.5, event_offset_s + 0.5,
                             1.0 / (2 * probe.measured_fps)):
            tv = scale * events.t_sample + off
            inside = (tv >= ft[0]) & (tv <= ft[-1])
            if inside.sum() < 0.5 * len(tv):
                continue
            a = gyro_lp[inside]
            v = np.interp(tv[inside], ft, energy)
            a = (a - a.mean()) / (a.std() + 1e-12)
            v = (v - v.mean()) / (v.std() + 1e-12)
            best = max(best, float(np.mean(a * v)))
        curve.append(best)
    curve = np.asarray(curve)
    best_scale = float(scales[int(np.argmax(curve))])
    # Range of scales within 2% correlation of the best: the resolving power of
    # this test, reported rather than hidden.
    near = np.flatnonzero(curve >= curve.max() - 0.02)
    plateau = (float(scales[near[0]]), float(scales[near[-1]]))

    unity_in = bool(plateau[0] <= 1.0 <= plateau[1])
    # Slow motion means the CSV spans *more* video seconds than IMU seconds, i.e.
    # a scale well above 1; fast motion / frame drop, well below.
    slow_excluded = bool(plateau[1] < SCALE_SLOW_MOTION_FACTOR
                         and plateau[0] > 1.0 / SCALE_SLOW_MOTION_FACTOR)

    agrees = unity_in and slow_excluded
    if agrees:
        verdict = (f"real-time footage: scale 1.0 lies inside the plateau "
                   f"{plateau[0]:.2f}-{plateau[1]:.2f}x and 2x-or-greater slow "
                   f"motion is excluded, so the offset-only model holds")
    elif not unity_in:
        verdict = (f"SCALE MISMATCH - 1.0 is outside the plateau "
                   f"{plateau[0]:.2f}-{plateau[1]:.2f}x; the video fps or the "
                   f"416 Hz assumption is wrong")
    else:
        verdict = (f"INCONCLUSIVE - the plateau {plateau[0]:.2f}-{plateau[1]:.2f}x "
                   f"is too wide to rule out time-scaled footage")
    return DriftCheck(
        best_scale=best_scale,
        scale_plateau=plateau,
        unity_in_plateau=unity_in,
        slow_motion_excluded=slow_excluded,
        agrees=agrees,
        verdict=verdict,
    )


def build_synced_frame(events: ImuEvents, probe: VideoProbe,
                       offset_s: float | None) -> tuple[pd.DataFrame, int]:
    """Attach the video time base to every CSV row.

    All 400 rows are kept. Rows whose mapped time falls outside the decoded video
    get `frame_index = -1` and empty `t_video`/`frame_exact`; they are counted and
    reported, never dropped or clipped.
    """
    out = events.df.copy()
    out["t_sample"] = events.t_sample

    if offset_s is None or not np.isfinite(offset_s):
        out["t_video"] = np.nan
        out["frame_exact"] = np.nan
        out["frame_index"] = -1
        return out, len(out)

    t_video = events.t_sample + offset_s
    frame_index = probe.frame_for_time(t_video)
    outside = frame_index < 0

    frame_exact = np.array([frame_index_at(probe, t) for t in t_video])
    frame_exact[outside] = np.nan
    t_video_col = t_video.astype(float).copy()
    t_video_col[outside] = np.nan

    out["t_video"] = t_video_col
    out["frame_exact"] = frame_exact
    out["frame_index"] = frame_index
    return out, int(outside.sum())


def align_video(label: str, probe: VideoProbe, me: MotionEnergy,
                events: ImuEvents, contact) -> Alignment:
    """Phase E1-E3 for one video. `contact` is a `BallContact`."""
    if not contact.found or not contact.is_strike:
        why = contact.reason or (
            f"ball speed changes from {contact.pre_speed:.1f} to "
            f"{contact.post_speed:.1f} px/frame - too small for a racquet strike "
            f"(needs >= {STRIKE_MIN_POST_SPEED_PX_PER_FRAME:.0f} px/frame and a "
            f">= {STRIKE_MIN_SPEED_RATIO:.0f}x jump)")
        return Alignment(label=label, probe=probe, events=events,
                         syncable=False, reason=why)

    t_contact = frame_time_at(probe, contact.contact_frame_exact)
    offset = t_contact - events.t_impact

    al = Alignment(
        label=label, probe=probe, events=events, syncable=True,
        reason="ball-racquet contact identified in both modalities",
        contact_frame_exact=contact.contact_frame_exact,
        contact_frame_bracket=contact.contact_frame_bracket,
        t_contact_video=t_contact,
        offset_s=offset,
        offset_frames=offset * probe.measured_fps,
    )
    al.xcorr = cross_correlate(events, probe, me, offset,
                               contact.contact_frame_exact)
    al.drift = check_drift(events, probe, me, offset)
    return al
