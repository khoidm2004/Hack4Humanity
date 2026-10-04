"""Camera calibration for the `swing_angle_1.mp4` overlay.

Holds the committed racket-head annotations, turns them into 2D<->3D
correspondences against ``data/outputs/swing.npz``, and solves the camera
pose.  Pure library code: importing it reads nothing and solves nothing.

Why annotations rather than a tracker: the racket in this clip is a dark
green frame with a translucent string bed that spends half the swing in
front of grey trousers and the other half in front of a beige goal board,
motion-blurred into a streak.  Every automatic blob rule tried here picks
the arm or the torso at least as often as the racket, and nothing in the
footage tells a detector which blob is right.  The reads below were made by
eye and then re-read off a marker-overlay pass (see "Provenance").

The pose this module solves supersedes the *guessed* ``rvec``/``tvec`` that
``app.py`` used to build; the Z-up world -> Y-down camera convention from
``OVERLAY_FRAME.md`` still holds and is now absorbed into the solved
rotation rather than written out as ``[+pi/2, 0, 0]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import camera as _camera
from . import config

VIDEO_NAME = "swing_angle_1.mp4"
LABEL = "angle_1"

# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
# Provenance
# ----------
# Video     : data/video/swing_angle_1.mp4 (1280x720, 30 fps).
# Frames    : the swing window from data/synced_data_angle_1.csv (116-145).
# Date read : 2026-10-04.
# Method    : `uv run python scripts/dump_swing_frames.py --region ... --zoom 3
#             --chroma` writes each frame as a 3x crop carrying a coordinate
#             grid *labelled in original frame pixels* (major lines every
#             100 px, minor every 50) with racket-coloured pixels painted red.
#             Each (u, v) below was read off those grids by eye, then
#             re-checked with `--verify`, which draws the committed marker
#             back onto the full frame; every marker was looked at again and
#             the set re-read until each sat on the racket head.
# Convention: **the far outer edge of the racket head** — the rim point
#             furthest from the hand along the racket axis, i.e. the point
#             `config.RACKET_TIP_LEN` = 0.686 m from the wrist that
#             `src/fusion.py:pivot_tip` models as `tip`.  NOT the string-bed
#             centre (~0.09 m / ~25 px nearer the hand), NOT the hand, NOT the
#             centroid of the motion-blur streak.  Where the head is smeared
#             into a streak the far edge is taken at mid-streak (the head's
#             position at mid-exposure).
# Uncertainty: +/-12 px where the head rim is resolvable (116, 117, 137, 143,
#             145), +/-20 px where it is motion-blurred or edge-on (120-130,
#             139), +/-25 px on 141 (head rim partly behind the goal post).
RACKET_HEAD_PX_ANGLE_1: dict[int, tuple[float, float]] = {
    116: (420.0, 595.0),
    117: (432.0, 593.0),
    120: (482.0, 632.0),
    122: (524.0, 640.0),
    124: (578.0, 658.0),
    126: (652.0, 640.0),
    128: (688.0, 650.0),
    130: (740.0, 637.0),
    137: (877.0, 529.0),
    139: (917.0, 484.0),
    141: (945.0, 425.0),
    143: (965.0, 388.0),
    145: (978.0, 353.0),
}

# Frames deliberately left out, with the reason.  Nothing here was dropped to
# improve the fit; see `scripts/check_overlay_match.py` for the leave-one-out
# residuals of the frames that ARE used.
EXCLUDED_FRAMES: dict[int, str] = {
    118: "redundant — between 117 and 120, adds no new geometry",
    119: "redundant",
    121: "redundant",
    123: "redundant",
    125: "redundant",
    127: "redundant",
    129: "redundant",
    131: "racket head overlaps the struck ball; head rim not separable",
    132: "racket head overlaps the struck ball; head rim not separable",
    133: "racket and forearm merge into one blur streak",
    134: "racket and forearm merge into one blur streak",
    135: "racket and forearm merge into one blur streak",
    136: "racket and forearm merge into one blur streak",
    138: "redundant",
    140: "redundant",
    142: "redundant",
    144: "redundant",
}

# The wrist/hand pixel in the same frames, read with the same method.
# **Never fed to solvePnP.**  `src/fusion.py:pivot_tip` pins the wrist at the
# world origin, so the spread of these points is a direct measurement of the
# body translation the pivot model omits — the irreducible error floor for
# any single rigid camera pose.  `check_overlay_match.py` prints it.
WRIST_PX_ANGLE_1: dict[int, tuple[float, float]] = {
    116: (580.0, 505.0),
    117: (600.0, 503.0),
    120: (624.0, 498.0),
    122: (640.0, 510.0),
    124: (655.0, 520.0),
    126: (688.0, 478.0),
    128: (708.0, 465.0),
    130: (725.0, 472.0),
    137: (772.0, 420.0),
    139: (770.0, 400.0),
    141: (770.0, 388.0),
    143: (795.0, 382.0),
    145: (800.0, 370.0),
}

# --------------------------------------------------------------------------- #
# The one automatic correspondence: ball-racket contact
# --------------------------------------------------------------------------- #
# `src/sync/video_motion.ball_contact(path, 106, 156)` fits the ball's
# pre- and post-contact image trajectories and intersects them, recovering
# contact to a fraction of a frame.  Extrapolating the *pre*-contact line
# (frames 118-130) to that instant gives the sub-frame contact pixel below.
# `check_overlay_match.py` re-derives both and asserts agreement within 2 px,
# so this constant cannot drift away from the detector.
#
# Caveat, stated rather than hidden: the ball sits on the **string bed**,
# roughly 0.60 m from the wrist, while `tip` is the head tip at 0.686 m.  This
# correspondence therefore carries a built-in ~0.09 m (~25 px at this scale)
# bias toward the hand.  It is kept because it is the only correspondence
# with sub-frame timing, and because acceptance criterion 4 (the impact
# marker landing on the ball) is then satisfied *by the fit* instead of by
# translating the whole path onto one point.
CONTACT_FRAME_EXACT = 130.70344
CONTACT_PIXEL = (760.21, 584.38)

# Read once off the gridded full frames: head top y~212, shoe sole y~688 at
# frame 145.  Used only as an independent distance cross-check.
PLAYER_PIXEL_HEIGHT = 476.0
PLAYER_HEIGHT_M = 1.75

# --------------------------------------------------------------------------- #
# Calibrated FOV
# --------------------------------------------------------------------------- #
# `camera.solve_pose_fov_sweep` was run over 40-100 deg in 2.5 deg steps on
# the correspondences above; `check_overlay_match.py` re-runs it and prints
# the whole table.  **The sweep does not identify the FOV.**  Across that
# entire 60 deg range the all-point median moves by only ~7 px — less than
# the +/-12-25 px uncertainty on the annotations themselves — so there is no
# real minimum to find.  (The deeper dips in the robust column come from
# RANSAC selecting a different subset of points at each FOV, and its argmin
# sits at the end of the scanned range: the signature of an unidentifiable
# parameter, not a measurement of the lens.)
#
# So the FOV keeps the 60 deg the app already assumed.  That choice costs
# accuracy rather than buying it: at 60 deg the median is ~56 px, where the
# sweep's own argmin would have reported ~26 px at a 100 deg lens 1.4 m from
# the player.  Nothing here was moved to flatter the error.
CALIB_FOV_DEG = 60.0

# The median reprojection error actually achieved at CALIB_FOV_DEG, recorded so
# the tester has a real regression gate (one-sided: it fails on worse, never on
# better).  Was 56.0 px, measured against the pre-time-base-fix reconstruction;
# the pose is re-solved from the annotations on every run, so after
# SLOWMO_SCALE went 1.0 -> 8.0 the same fit lands at the value below.  This is a
# MEASUREMENT tightened to match reality, not a target moved to pass a check.
RECORDED_BASELINE_PX = 21.5

# Two image-side measurements used as independent cross-checks on the pose.
# `max |head_px - wrist_px|` over the annotated frames is the racket seen
# closest to fronto-parallel, so it converts `config.RACKET_TIP_LEN` into a
# pixels-per-metre scale without any PnP involved.
RACKET_MAX_APPARENT_PX = 195.2


# --------------------------------------------------------------------------- #
# Correspondences
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Correspondences:
    """Matched 2D annotations and 3D ``tip`` samples for one video."""

    frames: np.ndarray        # (M,) float — video frame (fractional for contact)
    imu_idx: np.ndarray       # (M,) int — chosen IMU sample
    img_pts: np.ndarray       # (M, 2) float — observed pixels
    obj_pts: np.ndarray       # (M, 3) float — swing.tip[imu_idx], metres
    labels: tuple[str, ...]   # per-point description, for printing

    def __len__(self) -> int:
        return len(self.frames)


def build_correspondences(swing, sync, *, include_contact: bool = True,
                          obj_mode: str = "center") -> Correspondences:
    """Pair each annotated frame with the ``tip`` sample at that frame's centre.

    ``obj_mode``:
      * ``"center"`` (default) — ``tip[i]`` for the sample minimising
        ``|frame_exact - frame|``.  This is the sample whose video time matches
        the frame; ``SyncedData.imu_idx_for_frame`` would return the *first*
        sample of the bin instead, a systematic half-frame bias.
      * ``"binmean"`` — the mean of ``tip`` over every sample in the frame's
        bin, which is better matched to a motion-blur centroid.  Offered as a
        sensitivity check only; ``check_overlay_match.py`` reports both.
    """
    if obj_mode not in ("center", "binmean"):
        raise ValueError(f"unknown obj_mode {obj_mode!r}")

    frames: list[float] = []
    idxs: list[int] = []
    img: list[tuple[float, float]] = []
    obj: list[np.ndarray] = []
    labels: list[str] = []

    for f in sorted(RACKET_HEAD_PX_ANGLE_1):
        i = sync.imu_idx_for_frame_centered(int(f))
        # The centred lookup must land inside the requested frame's bin.
        if int(sync.frame_index[i]) != int(f):
            raise ValueError(
                f"frame {f}: centred IMU sample {i} maps to frame "
                f"{int(sync.frame_index[i])}, not {f}")
        if obj_mode == "binmean":
            bin_ = np.flatnonzero(sync.frame_index == int(f))
            pt = swing.tip[bin_].mean(axis=0)
        else:
            pt = swing.tip[i]
        frames.append(float(f))
        idxs.append(int(i))
        img.append(RACKET_HEAD_PX_ANGLE_1[f])
        obj.append(np.asarray(pt, float))
        labels.append(f"f{f:d} head")

    if include_contact:
        i = sync.imu_idx_for_frame_exact(CONTACT_FRAME_EXACT)
        frames.append(float(CONTACT_FRAME_EXACT))
        idxs.append(int(i))
        img.append(CONTACT_PIXEL)
        obj.append(np.asarray(swing.tip[i], float))
        labels.append(f"f{CONTACT_FRAME_EXACT:.3f} ball contact")

    return Correspondences(
        frames=np.asarray(frames, float),
        imu_idx=np.asarray(idxs, int),
        img_pts=np.asarray(img, float),
        obj_pts=np.asarray(obj, float),
        labels=tuple(labels),
    )


# --------------------------------------------------------------------------- #
# Solved pose
# --------------------------------------------------------------------------- #
def _load_inputs():
    from .swing import Swing
    from .synced_data import SyncedData
    from .sync import paths as sync_paths

    swing = Swing.load(config.SWING_NPZ)
    for label, video, csv_out in sync_paths.ANGLES:
        if label == LABEL:
            return swing, SyncedData.for_video(label, video, csv_out)
    raise LookupError(f"no sync table configured for {LABEL!r}")


@lru_cache(maxsize=8)
def solved_pose(width: int = 1280, height: int = 720,
                fov_deg: float = CALIB_FOV_DEG) -> _camera.PoseSolution:
    """Solve (and memoise) the angle-1 camera pose from the annotations.

    Computed at runtime from the committed table rather than stored as a baked
    ``rvec``/``tvec`` pair, so the pose can never drift away from the
    annotations it came from, and so ``app.py`` and
    ``scripts/check_overlay_match.py`` necessarily report the same numbers.
    PnP on 14 points takes microseconds; the cache is keyed on everything that
    changes the answer.
    """
    swing, sync = _load_inputs()
    corr = build_correspondences(swing, sync)
    return _camera.solve_pose(int(width), int(height),
                              corr.img_pts, corr.obj_pts, float(fov_deg))


def has_annotations(video_name: str) -> bool:
    """True when a committed annotation set exists for this clip."""
    return video_name == VIDEO_NAME and bool(RACKET_HEAD_PX_ANGLE_1)


def wrist_pixel_spread() -> tuple[float, float, float]:
    """``(span_u, span_v, max_pairwise)`` of the annotated wrist pixels.

    The pivot model holds the wrist at the world origin, so this spread is
    error no rigid pose can remove.
    """
    w = np.asarray(list(WRIST_PX_ANGLE_1.values()), float)
    span = w.max(axis=0) - w.min(axis=0)
    d = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=-1)
    return float(span[0]), float(span[1]), float(d.max())


def player_height_distance_m(fov_deg: float = CALIB_FOV_DEG,
                             width: int = 1280) -> float:
    """Camera distance implied by the player's pixel height — independent of PnP."""
    fx = (0.5 * width) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return float(fx * PLAYER_HEIGHT_M / PLAYER_PIXEL_HEIGHT)


def racket_pixel_scale() -> tuple[float, float]:
    """``(px_per_metre, max_apparent_px)`` from the racket's own apparent length.

    ``head_px - wrist_px`` is the racket imaged end-to-end.  Its longest value
    over the annotated frames is the racket seen closest to fronto-parallel,
    so dividing by ``config.RACKET_TIP_LEN`` gives an image scale that owes
    nothing to PnP, the fusion output, or the FOV assumption.

    Known 20 % low, reported not fixed: the tennis ball's own pixel diameter
    (24.0 px / 0.067 m = 359 px/m, `scripts/measure_capture_fps.py`) disagrees
    with this function's 285 px/m by exactly the ratio 0.686/0.544 — i.e. the
    open question of what the wrist->head-rim lever really is (analysis §2.5,
    §7 Q4).  It MUST keep dividing by `config.RACKET_TIP_LEN`, the same length
    `fusion.pivot_tip` uses: the `L` then cancels out of
    `model_vs_video_arc`'s `ratio` exactly (analysis §3.1), and changing only
    one of the two would silently break that cancellation.
    """
    frames = sorted(set(RACKET_HEAD_PX_ANGLE_1) & set(WRIST_PX_ANGLE_1))
    v = np.array([np.subtract(RACKET_HEAD_PX_ANGLE_1[f], WRIST_PX_ANGLE_1[f])
                  for f in frames], float)
    longest = float(np.linalg.norm(v, axis=1).max())
    return longest / config.RACKET_TIP_LEN, longest


def racket_distance_m(fov_deg: float = CALIB_FOV_DEG, width: int = 1280) -> float:
    """Camera distance implied by the racket's apparent length — independent of PnP."""
    fx = (0.5 * width) / np.tan(np.deg2rad(fov_deg) / 2.0)
    scale, _ = racket_pixel_scale()
    return float(fx / scale)


def tangent_turning_deg(path: np.ndarray) -> float:
    """Total turning along a polyline: the sum of angles between successive
    tangent directions, in degrees.

    This is what "total turning" means and what reversals and loops show up in.
    `model_vs_video_arc` used to report `arc / RACKET_TIP_LEN` under the name
    `turn3d_total_deg`, which is arc divided by radius — it moves in exact
    lockstep with the arc and carries no information beyond it.  Works for
    (N,2) pixel paths and (N,3) metre paths alike.
    """
    p = np.asarray(path, float)
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-12
    if keep.sum() < 2:
        return 0.0
    u = d[keep] / n[keep, None]
    c = np.clip(np.einsum("ij,ij->i", u[:-1], u[1:]), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(c)).sum())


def model_vs_video_arc(swing, sync) -> dict:
    """How far the reconstructed tip travels, against how far the racket does.

    This is the measurement that decides whether *any* rigid camera pose can
    work.  A rigid pose is a similarity on the image plane to first order, so
    it can move and rotate and scale the path — it cannot change how much arc
    the path contains.  If the two arc lengths disagree, no pose fits, and the
    reprojection error has a floor that is nothing to do with the camera.

    The window is now the one the RESCALED time base gives (see
    `config.SLOWMO_SCALE`), not the whole 400-sample record: with
    `SLOWMO_SCALE = 1.0` this used to span samples 0-398 of 400 — 0.96 s of
    IMU motion compared against 0.12 s of real video time.

    Returns the 3D arc in metres, the image arc it should produce at the
    independently measured pixel scale, the arc the annotated racket head
    actually traces, their ratio, real tangent turning three ways, and the
    window the comparison spans.
    """
    frames = sorted(RACKET_HEAD_PX_ANGLE_1)
    lo = sync.imu_idx_for_frame_centered(frames[0])
    hi = sync.imu_idx_for_frame_centered(frames[-1])
    tip = swing.tip[lo:hi + 1]
    arc3d = float(np.sum(np.linalg.norm(np.diff(tip, axis=0), axis=1)))

    px = np.array([RACKET_HEAD_PX_ANGLE_1[f] for f in frames], float)
    arc2d = float(np.sum(np.linalg.norm(np.diff(px, axis=0), axis=1)))
    scale, _ = racket_pixel_scale()

    # Net turn of the racket vector, 3D and as imaged.
    def _ang(a, b):
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))

    v = np.array([np.subtract(RACKET_HEAD_PX_ANGLE_1[f], WRIST_PX_ANGLE_1[f])
                  for f in frames], float)
    wrist_max_px = wrist_pixel_spread()[2]
    return {
        "arc3d_m": arc3d,
        "arc3d_expected_px": arc3d * scale,
        "arc2d_observed_px": arc2d,
        "ratio": arc3d * scale / arc2d,
        # What a CORRECT wrist-pinned pivot model should give: `pivot_tip` holds
        # the wrist at the origin, so it can only produce the rotational part of
        # the head's motion, and the repo has measured `wrist_max_px` of wrist
        # travel it cannot represent. 1.0 is unreachable by construction.
        "ratio_expected": (arc2d - wrist_max_px) / arc2d,
        # `arc / RACKET_TIP_LEN`, kept for continuity with OVERLAY_FRAME.md and
        # renamed because it is NOT a turning measurement: it is the arc in
        # radius units and moves in exact lockstep with the arc.
        "arc_over_L_deg": float(np.rad2deg(arc3d / config.RACKET_TIP_LEN)),
        # Real total turning, three ways, all directly comparable.
        "turn3d_tangent_deg": tangent_turning_deg(tip),
        "turn3d_tangent_at_frames_deg": tangent_turning_deg(
            np.array([swing.tip[sync.imu_idx_for_frame_centered(f)]
                      for f in frames], float)),
        "turn2d_tangent_observed_deg": tangent_turning_deg(px),
        "turn3d_net_deg": _ang(tip[0], tip[-1]),
        "turn2d_net_deg": _ang(v[0], v[-1]),
        "px_per_m": scale,
        # The window the comparison actually spans — the evidence the time base
        # is right.  With SLOWMO_SCALE = 1.0 this was samples 0-398 of 400:
        # 0.96 s of IMU motion against 0.12 s of real video time.
        "window_lo": int(lo),
        "window_hi": int(hi),
        "window_n": int(hi - lo + 1),
        "window_frames": (int(frames[0]), int(frames[-1])),
        "slowmo_scale": float(config.SLOWMO_SCALE),
    }
