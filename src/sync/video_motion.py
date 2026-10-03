"""Phase D: video-side motion energy and the ball-racquet contact estimate.

Two signals come out of here:

* `motion_energy` - a per-frame scalar used to locate the swing and (Phase E2)
  to cross-correlate against the IMU.
* `ball_contact` - the ball's image-plane track, from which the instant of
  ball-racquet contact is recovered to sub-frame precision as the kink in the
  trajectory.

Why the motion energy is a *high-amplitude difference fraction* and not the
usual mean |I_t - I_{t-1}|: both source clips are handheld and H.264-encoded.
Mean absolute difference is dominated by (a) whole-frame camera shake and
(b) a quantisation step at every GOP keyframe, which shows up as a spurious
peak every 12 frames in both files and swamps the swing. Counting only pixels
that changed by more than `DIFF_THRESHOLD` grey levels rejects both: shake and
compression noise are low-amplitude and spread out, while a fast racquet or ball
produces a small number of very large pixel changes. Global-translation
compensation by phase correlation was also tried and changed the result by less
than one frame, so it is not carried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WORK_WIDTH = 320
BLUR_KSIZE = (5, 5)
DIFF_THRESHOLD = 40.0

# Tennis-ball segmentation in HSV (OpenCV hue 0-179). The ball is the only
# strongly saturated yellow-green object over the blue court.
BALL_HSV_LO = (22, 70, 120)
BALL_HSV_HI = (45, 255, 255)
BALL_AREA_MIN = 25
BALL_AREA_MAX = 2000

# A struck ball leaves the strings far faster than a bounced or tapped one.
# Used to decide, from the footage alone, whether a video contains a real shot.
STRIKE_MIN_POST_SPEED_PX_PER_FRAME = 20.0
STRIKE_MIN_SPEED_RATIO = 4.0
# The x and y fits of the trajectory kink are independent; if they disagree by
# more than this the two-line model does not fit and the result is rejected.
MAX_FIT_SPREAD_FRAMES = 2.0


@dataclass
class MotionEnergy:
    """Per-frame motion energy for one video."""

    path: Path
    # energy[i] = fraction of pixels differing from frame i-1 by more than
    # DIFF_THRESHOLD grey levels. energy[0] is 0 by definition.
    energy: np.ndarray
    frame_times: np.ndarray

    @property
    def peak_frame(self) -> int:
        return int(np.argmax(self.energy))

    @property
    def peak_time(self) -> float:
        return float(self.frame_times[self.peak_frame])


@dataclass
class BallContact:
    """Sub-frame ball-racquet contact recovered from the ball's trajectory."""

    frames: np.ndarray            # frames on which the ball was found
    xs: np.ndarray
    ys: np.ndarray
    contact_frame_exact: float    # fractional frame index of the trajectory kink
    contact_frame_bracket: tuple[int, int]
    pre_speed: float              # px/frame before contact
    post_speed: float             # px/frame after contact
    found: bool
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def is_strike(self) -> bool:
        """True when the speed change looks like a racquet strike, not a bounce."""
        return (self.found
                and self.post_speed >= STRIKE_MIN_POST_SPEED_PX_PER_FRAME
                and self.post_speed >= STRIKE_MIN_SPEED_RATIO * max(self.pre_speed, 1e-6))


def motion_energy(path: str | Path, frame_times: np.ndarray) -> MotionEnergy:
    """Decode `path` once and compute the high-amplitude difference fraction."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")

    energies: list[float] = []
    prev: np.ndarray | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = WORK_WIDTH / float(w)
        small = cv2.resize(frame, (WORK_WIDTH, max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, BLUR_KSIZE, 0).astype(np.float32)
        if prev is None:
            energies.append(0.0)
        else:
            energies.append(float((np.abs(grey - prev) > DIFF_THRESHOLD).mean()))
        prev = grey
    cap.release()

    energy = np.asarray(energies, dtype=float)
    n = min(len(energy), len(frame_times))
    return MotionEnergy(path=path, energy=energy[:n], frame_times=frame_times[:n])


def track_ball(path: str | Path, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray,
                                                            np.ndarray]:
    """Return (frames, x, y) of the largest ball-coloured blob per frame in [lo, hi]."""
    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    cap = cv2.VideoCapture(str(path))
    i = 0
    while i <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= lo:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, BALL_HSV_LO, BALL_HSV_HI)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
            best = None
            for k in range(1, n):
                area = int(stats[k, cv2.CC_STAT_AREA])
                if BALL_AREA_MIN <= area <= BALL_AREA_MAX:
                    if best is None or area > best[0]:
                        best = (area, float(cent[k][0]), float(cent[k][1]))
            if best is not None:
                frames.append(i)
                xs.append(best[1])
                ys.append(best[2])
        i += 1
    cap.release()
    return np.asarray(frames), np.asarray(xs), np.asarray(ys)


def _fit_intersection(f: np.ndarray, v: np.ndarray, pre: np.ndarray,
                      post: np.ndarray) -> tuple[float, float, float] | None:
    """Fit a line to v over `pre` and over `post`; return (crossing, slope_pre,
    slope_post). None if the slopes are too close to intersect meaningfully."""
    if pre.sum() < 3 or post.sum() < 3:
        return None
    a1, b1 = np.polyfit(f[pre], v[pre], 1)
    a2, b2 = np.polyfit(f[post], v[post], 1)
    if abs(a1 - a2) < 1e-6:
        return None
    return float((b2 - b1) / (a1 - a2)), float(a1), float(a2)


def ball_contact(path: str | Path, search_lo: int, search_hi: int) -> BallContact:
    """Locate ball-racquet contact as the kink in the ball's image trajectory.

    The ball is slow and falling before contact and fast and rising after it, so
    x(frame) and y(frame) are each well modelled by two straight lines whose
    intersection is the contact instant - recovered to a fraction of a frame,
    which is better than the +/-1 frame a visual read can give at 30 fps.
    """
    frames, xs, ys = track_ball(path, search_lo, search_hi)
    empty = BallContact(frames, xs, ys, float("nan"), (-1, -1), 0.0, 0.0, False)
    if len(frames) < 10:
        empty.reason = f"only {len(frames)} ball detections in [{search_lo},{search_hi}]"
        return empty

    # Only use the longest run of consecutive detections, so a dropout does not
    # create a false kink.
    splits = np.split(np.arange(len(frames)),
                      np.flatnonzero(np.diff(frames) != 1) + 1)
    run = max(splits, key=len)
    if len(run) < 10:
        empty.reason = f"longest consecutive ball run is only {len(run)} frames"
        return empty
    f, x, y = frames[run].astype(float), xs[run], ys[run]

    speed = np.hypot(np.diff(x), np.diff(y))
    # Contact is where the speed jumps most; the jump index refers to the step
    # between f[j] and f[j+1].
    j = int(np.argmax(np.diff(speed))) + 1
    pre = np.zeros(len(f), bool)
    post = np.zeros(len(f), bool)
    pre[max(0, j - 8):j + 1] = True
    post[min(len(f) - 1, j + 2):min(len(f), j + 12)] = True

    sols = []
    for v in (x, y):
        got = _fit_intersection(f, v, pre, post)
        if got is not None:
            sols.append(got)
    if not sols:
        empty.reason = "could not fit pre/post trajectory lines"
        return empty

    crossings = [s[0] for s in sols]
    contact = float(np.mean(crossings))
    pre_speed = float(np.median(speed[max(0, j - 6):j])) if j > 0 else 0.0
    post_speed = float(np.median(speed[j + 1:j + 8])) if j + 1 < len(speed) else 0.0
    spread = float(max(crossings) - min(crossings))

    # The x and y fits are independent estimates of the same instant. If they
    # disagree the two-line model does not describe this track (no real kink),
    # and the "contact" it returns is meaningless - say so rather than report it.
    if len(crossings) > 1 and spread > MAX_FIT_SPREAD_FRAMES:
        empty.reason = (f"x and y trajectory fits disagree by {spread:.1f} frames "
                        f"(limit {MAX_FIT_SPREAD_FRAMES}) - no clean kink, so this "
                        "track has no racquet strike in it")
        empty.pre_speed, empty.post_speed = pre_speed, post_speed
        empty.detail = {"crossing_from_x": crossings[0],
                        "crossing_from_y": crossings[1],
                        "crossing_spread_frames": spread,
                        "n_detections": int(len(f))}
        return empty
    if not search_lo <= contact <= search_hi:
        empty.reason = (f"fitted crossing at frame {contact:.1f} lies outside the "
                        f"search window [{search_lo}, {search_hi}]")
        empty.pre_speed, empty.post_speed = pre_speed, post_speed
        return empty

    return BallContact(
        frames=f.astype(int), xs=x, ys=y,
        contact_frame_exact=contact,
        contact_frame_bracket=(int(np.floor(contact)), int(np.ceil(contact))),
        pre_speed=pre_speed, post_speed=post_speed, found=True,
        detail={
            "crossing_from_x": crossings[0],
            "crossing_from_y": crossings[1] if len(crossings) > 1 else float("nan"),
            "crossing_spread_frames": float(max(crossings) - min(crossings)),
            "speed_jump_frame": int(f[j]),
            "n_detections": int(len(f)),
        },
    )
