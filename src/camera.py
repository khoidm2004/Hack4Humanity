"""Person B — camera calibration & path projection (PLAN.md §7 Phase 2).

Implements the "path-projection" overlay pipeline:

1. ``calibrate`` — recover the camera pose from a single frame using
   ``cv2.solvePnP`` with ≥4 non-collinear clicked racket points and known
   racket geometry.
2. ``project_path`` — project the entire 3D racket-head trajectory onto the
   image as a 2D curve via ``cv2.projectPoints``.
3. ``draw_overlay`` — draw a windowed path trail around the current sample,
   an animated dot for a given video time, and the impact marker.

The intrinsics are estimated from an assumed horizontal FOV (~60°) as the
plan requires; sliders refine pose/zoom when solvePnP is degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Camera:
    """Estimated projection model for one camera view."""

    K: np.ndarray          # (3,3) intrinsic matrix
    dist: np.ndarray       # (5,) distortion coefficients (zeros)
    rvec: np.ndarray       # (3,) world->camera rotation (Rodrigues)
    tvec: np.ndarray       # (3,) world->camera translation
    width: int
    height: int

    @classmethod
    def from_pnps(cls, K, dist, rvec, tvec, width, height) -> "Camera":
        if dist is None:
            dist = np.zeros(5)
        return cls(np.asarray(K, float), np.asarray(dist, float),
                   np.asarray(rvec, float).reshape(3), np.asarray(tvec, float).reshape(3),
                   int(width), int(height))


# --------------------------------------------------------------------------- #
# Intrinsics estimation
# --------------------------------------------------------------------------- #
def estimate_intrinsics(width: int, height: int,
                        fov_deg: float = 60.0) -> np.ndarray:
    """Pinhole intrinsics from an assumed horizontal FOV (square pixels)."""
    fx = (0.5 * width) / np.tan(np.deg2rad(fov_deg) / 2.0)
    fy = fx
    cx = 0.5 * width
    cy = 0.5 * height
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


# --------------------------------------------------------------------------- #
# Click-driven solvePnP calibration from one frame
# --------------------------------------------------------------------------- #
def calibrate(
    width: int,
    height: int,
    img_pts: np.ndarray,
    obj_pts: np.ndarray,
    fov_deg: float = 60.0,
) -> Camera:
    """Estimate camera pose from ≥4 non-collinear 2D↔3D point matches.

    Parameters
    ----------
    width, height : int — frame size in pixels
    img_pts : (M, 2) float — clicked pixel coordinates
    obj_pts : (M, 3) float — matching racket/world coordinates (metres)
    fov_deg : horizontal FOV guess for the intrinsics.

    Returns
    -------
    Camera
        Pose that projects ``obj_pts`` onto ``img_pts`` (with slider nudge
        driven by a separate helper).

    Raises
    ------
    ValueError
        If fewer than 4 points are supplied.
    """
    img_pts = np.asarray(img_pts, float).reshape(-1, 2)
    obj_pts = np.asarray(obj_pts, float).reshape(-1, 3)
    if len(img_pts) < 4 or len(obj_pts) < 4:
        raise ValueError("calibrate needs at least 4 point matches (non-collinear).")

    K = estimate_intrinsics(width, height, fov_deg)
    dist = None  # OpenCV 5: pass None to use an empty distortion model.

    # reshape to the C-contiguous float32 layout OpenCV expects.
    obj_f = np.ascontiguousarray(obj_pts.astype(np.float32))
    img_f = np.ascontiguousarray(img_pts.astype(np.float32))

    # OpenCV 5.0 returns a tuple (retval, rvec, tvec).
    try:
        result = cv2.solvePnP(obj_f, img_f, K, dist)
    except cv2.error as exc:
        raise ValueError(
            "solvePnP failed (degenerate points?). Provide ≥4 non-collinear, "
            "non-coplanar-ish racket points and avoid a perfectly frontal "
            "camera.  Fall back to the slider nudge." +
            (f"  Details: {exc}" if False else "")
        ) from exc

    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("solvePnP returned an unexpected result type.")
    ret, rvec, tvec = result
    if not ret or rvec is None or tvec is None:
        raise ValueError("solvePnP failed; check the clicked points are valid.")
    return Camera.from_pnps(K, dist, rvec, tvec, width, height)


# --------------------------------------------------------------------------- #
# Robust pose recovery from many 2D<->3D correspondences
# --------------------------------------------------------------------------- #
@dataclass
class PoseSolution:
    """A solved camera pose plus the diagnostics needed to judge it.

    ``calibrate`` above answers "give me a pose"; this answers "give me a pose
    *and* tell me how badly it fits", which is what a many-point fit to noisy
    hand annotations actually needs.  Nothing here silently discards a point:
    ``inliers`` records what RANSAC kept, ``residuals`` is reported for every
    input point, and ``messages`` carries every validation failure.
    """

    ok: bool
    rvec: np.ndarray
    tvec: np.ndarray
    K: np.ndarray
    fov_deg: float
    solver: str
    residuals: np.ndarray        # (M,) per-point reprojection error, px
    inliers: np.ndarray          # (M,) bool — RANSAC inlier mask
    messages: tuple[str, ...] = ()

    @property
    def median_px(self) -> float:
        return float(np.median(self.residuals)) if len(self.residuals) else float("nan")

    @property
    def mean_px(self) -> float:
        return float(np.mean(self.residuals)) if len(self.residuals) else float("nan")

    @property
    def max_px(self) -> float:
        return float(np.max(self.residuals)) if len(self.residuals) else float("nan")

    @property
    def camera_distance_m(self) -> float:
        """Distance from the camera centre to the world origin (the wrist)."""
        return float(np.linalg.norm(self.tvec))

    def camera(self, width: int, height: int) -> Camera:
        return Camera.from_pnps(self.K, np.zeros(5), self.rvec, self.tvec,
                                width, height)


def _reproject_errors(obj: np.ndarray, img: np.ndarray, K: np.ndarray,
                      rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    proj, _ = cv2.projectPoints(obj.astype(np.float32), rvec, tvec, K, None)
    return np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)


def _in_front(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Camera-frame depth of each world point (must be > 0 for a valid pose)."""
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    return (R @ obj.T).T[:, 2] + float(np.asarray(tvec).reshape(3)[2])


def _order_correlations(obj: np.ndarray, img: np.ndarray, K: np.ndarray,
                        rvec: np.ndarray, tvec: np.ndarray) -> tuple[float, float]:
    """``(corr_u, corr_v)`` between projected and observed pixel coordinates.

    A pose can hit a low median residual while running the path *backwards* —
    typically when a robust fit has latched onto one half of the data and
    mirrored the rest.  Requiring the projection to preserve the observed
    left-right and up-down ordering rejects those, and unlike a fixed
    world-axis sign rule it assumes nothing about which way the world frame's
    axes point: it compares the projection with the annotations themselves.
    """
    proj, _ = cv2.projectPoints(obj.astype(np.float32), rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    out = []
    for k in (0, 1):
        a, b = proj[:, k], img[:, k]
        if not np.all(np.isfinite(a)) or a.std() < 1e-9 or b.std() < 1e-9:
            out.append(float("nan"))
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return out[0], out[1]


def solve_pose(
    width: int,
    height: int,
    img_pts: np.ndarray,
    obj_pts: np.ndarray,
    fov_deg: float = 60.0,
    *,
    ransac_reproj_px: float = 40.0,
    iterations: int = 5000,
    confidence: float = 0.999,
    require_order: bool = True,
    use_ransac: bool = True,
) -> PoseSolution:
    """Recover a camera pose from >=4 noisy 2D<->3D matches.

    Runs three solvers and keeps the one with the lowest median residual
    rather than trusting any single one:

    * ``solvePnPRansac`` (SQPNP) followed by ``solvePnPRefineLM`` on its
      inliers — robust to one bad annotation;
    * plain ``solvePnP`` with ``SOLVEPNP_SQPNP``;
    * plain ``solvePnP`` with ``SOLVEPNP_ITERATIVE``.

    The focal length is **not** solved jointly (hopelessly ill-conditioned on
    ~12 noisy points); ``fov_deg`` fixes the intrinsics and
    :func:`solve_pose_fov_sweep` scans it instead.

    Two validity tests reject a pose outright rather than reporting it:
    any correspondence behind the camera, and (when ``require_order``, and
    there are at least 5 points) a projection that reverses the observed
    left-right or up-down ordering of the annotations.
    """
    img = np.ascontiguousarray(np.asarray(img_pts, float).reshape(-1, 2))
    obj = np.ascontiguousarray(np.asarray(obj_pts, float).reshape(-1, 3))
    if len(img) != len(obj):
        raise ValueError(f"{len(img)} image points vs {len(obj)} object points")
    if len(img) < 4:
        raise ValueError("solve_pose needs at least 4 point matches")

    K = estimate_intrinsics(width, height, fov_deg)
    obj_f = obj.astype(np.float32)
    img_f = img.astype(np.float32)
    msgs: list[str] = []
    cands: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray]] = []

    # ---- RANSAC + LM refinement ---- #
    # `use_ransac=False` leaves only the all-point fits, which is how the FOV
    # sweep gets an objective that is continuous in the FOV: a robust fit can
    # change WHICH points it explains from one FOV to the next, so its median
    # jumps around and its argmin says nothing about the lens.
    inlier_mask = np.ones(len(img), bool)
    try:
        res = cv2.solvePnPRansac(
            obj_f, img_f, K, None,
            reprojectionError=float(ransac_reproj_px),
            iterationsCount=int(iterations), confidence=float(confidence),
            flags=cv2.SOLVEPNP_SQPNP,
        ) if use_ransac else None
    except cv2.error as exc:                                    # pragma: no cover
        res = None
        msgs.append(f"solvePnPRansac raised: {exc}")
    if res is not None:
        # OpenCV 5 returns (retval, rvec, tvec, inliers); `inliers` is None
        # when RANSAC never had to reject anything.
        retval, rvec, tvec = res[0], res[1], res[2]
        inl = res[3] if len(res) > 3 else None
        if retval and rvec is not None and tvec is not None:
            mask = np.ones(len(img), bool)
            if inl is not None and len(np.asarray(inl).ravel()) > 0:
                mask = np.zeros(len(img), bool)
                mask[np.asarray(inl).ravel().astype(int)] = True
            if mask.sum() < 4:
                msgs.append(f"RANSAC kept only {int(mask.sum())} inliers; "
                            "refining on all points instead")
                mask = np.ones(len(img), bool)
            inlier_mask = mask
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_f[mask], img_f[mask], K, None,
                    np.asarray(rvec, float), np.asarray(tvec, float))
            except cv2.error as exc:                            # pragma: no cover
                msgs.append(f"solvePnPRefineLM raised: {exc}")
            err = _reproject_errors(obj, img, K, rvec, tvec)
            cands.append((float(np.median(err)), "ransac+lm",
                          np.asarray(rvec, float).reshape(3),
                          np.asarray(tvec, float).reshape(3), err))
        else:
            msgs.append("solvePnPRansac returned retval=False")

    # ---- Plain solvers, for comparison ---- #
    for name, flag in (("sqpnp", cv2.SOLVEPNP_SQPNP),
                       ("iterative", cv2.SOLVEPNP_ITERATIVE)):
        try:
            out = cv2.solvePnP(obj_f, img_f, K, None, flags=flag)
        except cv2.error as exc:
            msgs.append(f"solvePnP[{name}] raised: {exc}")
            continue
        ret, rvec, tvec = out[0], out[1], out[2]
        if not ret or rvec is None or tvec is None:
            msgs.append(f"solvePnP[{name}] returned retval=False")
            continue
        err = _reproject_errors(obj, img, K, rvec, tvec)
        cands.append((float(np.median(err)), name,
                      np.asarray(rvec, float).reshape(3),
                      np.asarray(tvec, float).reshape(3), err))

    # ---- Reject poses that are geometrically invalid ---- #
    valid = []
    for med, name, rvec, tvec, err in cands:
        depth = _in_front(obj, rvec, tvec)
        if np.any(depth <= 0):
            msgs.append(f"{name}: rejected, {int((depth <= 0).sum())} of "
                        f"{len(obj)} points behind the camera "
                        f"(min depth {depth.min():.3f} m)")
            continue
        if require_order and len(obj) >= 5:
            cu, cv_ = _order_correlations(obj, img, K, rvec, tvec)
            if not (cu > 0) or not (cv_ > 0):
                msgs.append(
                    f"{name}: rejected (median would have been {med:.1f} px), "
                    f"projection reverses the observed order — "
                    f"corr(proj u, obs u)={cu:+.2f}, "
                    f"corr(proj v, obs v)={cv_:+.2f}")
                continue
        valid.append((med, name, rvec, tvec, err))

    if not valid:
        return PoseSolution(False, np.zeros(3), np.array([0.0, 0.0, 10.0]), K,
                            float(fov_deg), "none", np.array([]),
                            np.zeros(len(img), bool),
                            tuple(msgs) or ("no solver produced a usable pose",))

    valid.sort(key=lambda c: c[0])
    med, name, rvec, tvec, err = valid[0]
    msgs.append("solver medians: " + ", ".join(
        f"{n}={m:.1f}px" for m, n, _r, _t, _e in valid))
    return PoseSolution(True, rvec, tvec, K, float(fov_deg), name, err,
                        inlier_mask, tuple(msgs))


def solve_pose_fov_sweep(
    width: int,
    height: int,
    img_pts: np.ndarray,
    obj_pts: np.ndarray,
    fovs: "np.ndarray | list[float] | None" = None,
    **kwargs,
) -> tuple[PoseSolution, list[tuple[float, float, str]]]:
    """Re-solve at each candidate FOV and keep the lowest median residual.

    Returns ``(best, table)`` where ``table`` is one
    ``(fov_deg, median_px, solver)`` row per candidate — print it, it is the
    evidence the FOV was *solved* rather than tuned.
    """
    if fovs is None:
        fovs = np.arange(40.0, 100.0 + 1e-9, 2.5)
    table: list[tuple[float, float, str]] = []
    best: PoseSolution | None = None
    for fov in np.asarray(fovs, float):
        sol = solve_pose(width, height, img_pts, obj_pts, float(fov), **kwargs)
        table.append((float(fov),
                      sol.median_px if sol.ok else float("inf"),
                      sol.solver))
        if sol.ok and (best is None or sol.median_px < best.median_px):
            best = sol
    if best is None:
        best = solve_pose(width, height, img_pts, obj_pts,
                          float(np.asarray(fovs, float)[0]), **kwargs)
    return best, table


# --------------------------------------------------------------------------- #
# Projection helpers
# --------------------------------------------------------------------------- #
def project_points(cam: Camera, pts3d: np.ndarray) -> np.ndarray:
    """Projects 3D world points to image pixels.

    Parameters
    ----------
    cam : Camera
    pts3d : (N, 3) or (3,) float, world metres

    Returns
    -------
    (N, 2) float — pixel coordinates (u, v).
    """
    pts3d = np.asarray(pts3d, float).reshape(-1, 3)
    img, _ = cv2.projectPoints(
        pts3d.astype(np.float32),
        cam.rvec, cam.tvec, cam.K, cam.dist,
    )
    return img.reshape(-1, 2)


def project_path(cam: Camera, tip3d: np.ndarray) -> np.ndarray:
    """Project the whole 3D racket-head trajectory into a 2D curve (PLAN §5)."""
    return project_points(cam, tip3d)


# --------------------------------------------------------------------------- #
# Nudge (fallback alignment when solvePnP is degenerate / visually off)
# --------------------------------------------------------------------------- #
def nudge(cam: Camera, *, yaw: float = 0, elev: float = 0,
          zoom: float = 1.0, pan_x: float = 0, pan_y: float = 0) -> Camera:
    """Return a copy of ``cam`` with a small pose/zoom adjustment.

    ``yaw/elev`` rotate the camera (degrees), ``zoom`` scales the effective
    focal length, ``pan_x/pan_y`` shift the principal point (pixels).  Used by
    the sidebar sliders to fix imperfect path→racket alignment.
    """
    r = cv2.Rodrigues(cam.rvec)[0]
    ry = cv2.Rodrigues(np.array([0.0, np.deg2rad(yaw), 0.0]))[0]
    rx = cv2.Rodrigues(np.array([np.deg2rad(elev), 0.0, 0.0]))[0]
    r = rx @ ry @ r

    K = cam.K.copy()
    K[0, 0] *= zoom
    K[1, 1] *= zoom
    K[0, 2] += pan_x
    K[1, 2] += pan_y

    return Camera.from_pnps(K, cam.dist, cv2.Rodrigues(r)[0], cam.tvec,
                            cam.width, cam.height)


# --------------------------------------------------------------------------- #
# Overlay drawing
# --------------------------------------------------------------------------- #
def _index_for_video_time(swing_t: np.ndarray, t_video: float,
                          fps: float, scale: float, offset: float) -> int:
    """IMU index whose video-time is closest to ``t_video`` (seconds)."""
    imu_t = (t_video - offset) / scale
    return int(np.argmin(np.abs(swing_t - imu_t)))


_DRAW_LIMIT = 1e5


def path_window(n_samples: int, idx: int) -> tuple[int, int]:
    """Inclusive sample range of the polyline to draw around sample ``idx``.

    `draw_overlay` used to project ALL of `swing.tip` as one static polyline.
    At 240 fps capture the 400-sample record spans ~230 video frames while the
    swing occupies ~30, so the drawn path held ~8x the motion belonging to the
    frame on screen: a double loop where the footage shows one sweep, and the
    same picture at every frame regardless of reconstruction quality.

    The window is stated in VIDEO FRAMES (`config.PATH_TRAIL_FRAMES` /
    `PATH_LEAD_FRAMES`) and converted with `FS / CAPTURE_FPS` IMU samples per
    frame, so it stays the same amount of *visible* motion if either rate is
    ever revised.  At FS=416, CAPTURE_FPS=240 that is 1.733 samples per frame:
    21 samples behind, 7 ahead, a 29-sample trail of ~0.12 s of playback.
    """
    spf = config.FS / float(config.CAPTURE_FPS)
    back = int(round(config.PATH_TRAIL_FRAMES * spf))
    fwd = int(round(config.PATH_LEAD_FRAMES * spf))
    lo = max(0, int(idx) - back)
    hi = min(int(n_samples) - 1, int(idx) + fwd)
    return lo, hi


def _drawable(pts: np.ndarray) -> np.ndarray:
    """Clamp projected pixels into a range ``int32`` can hold.

    ``cv2.projectPoints`` happily returns +/-inf, NaN, or values in the
    billions for a point near the camera plane, and casting those to
    ``int32`` wraps around into garbage coordinates.  The solved pose sits
    ~3 m from the swing so this should not happen, but the yaw/elevation
    sliders orbit the camera about its own centre and can swing points behind
    it.  Clamping draws a line that leaves the frame instead of one that
    wraps back into it.
    """
    pts = np.asarray(pts, float)
    pts = np.nan_to_num(pts, nan=_DRAW_LIMIT, posinf=_DRAW_LIMIT,
                        neginf=-_DRAW_LIMIT)
    return np.clip(pts, -_DRAW_LIMIT, _DRAW_LIMIT)


def draw_overlay(
    frame: np.ndarray,
    cam: Camera,
    swing,
    t_video: float,
    fps: float,
    scale: float = config.SLOWMO_SCALE,
    offset: float = 0.0,
    *,
    dot_idx: int | None = None,
    show_path: bool = True,
    show_dot: bool = True,
    show_impact: bool = True,
    path_color=(0, 255, 255),
    dot_color=(0, 0, 255),
    impact_color=(255, 0, 0),
    path_window_idx: tuple[int, int] | None = None,
    path_fade: bool = True,
) -> np.ndarray:
    """Draw the projected path, animated dot and impact marker onto a frame.

    Parameters
    ----------
    frame : BGR uint8 image
    cam : Camera calibrated for the current view
    swing : Swing (has ``t, tip, impact_idx``)
    t_video : float — current video time in seconds
    fps : video frames per second
    scale : slow-motion factor (video_time = imu_time*scale + offset)
    offset : rough sync offset in seconds
    dot_idx : int, optional
        Exact IMU sample index for the animated dot.  When given, this wins
        over the ``t_video``-based ``scale/offset`` mapping (used when the
        per-video synced table from ``scripts/run_sync.py`` is available).
    path_window_idx : (int, int), optional
        Inclusive sample range to draw the path over.  Defaults to
        ``path_window(len(swing.tip), idx)`` — a trail BEHIND/AHEAD of the
        current sample (``config.PATH_TRAIL_FRAMES``/``PATH_LEAD_FRAMES``),
        not the whole record.  Pass ``(0, len(swing.tip) - 1)`` to force the
        old whole-record polyline (used for before/after comparisons).
    path_fade : bool
        Draw the trail dim/thin at its oldest end and bright/thick at the
        current sample, so it reads as a direction of travel.
    ...
    """
    out = frame.copy()

    # The sample the displayed frame belongs to.  Computed BEFORE the path is
    # drawn, because the path is now a window around it rather than the whole
    # record.
    if dot_idx is None:
        idx = _index_for_video_time(swing.t, t_video, fps, scale, offset)
    else:
        idx = int(dot_idx)
    idx = int(np.clip(idx, 0, len(swing.tip) - 1))

    # Projected path — a trail around the current sample, not the whole record.
    if show_path:
        lo, hi = (path_window_idx if path_window_idx is not None
                  else path_window(len(swing.tip), idx))
        lo = max(0, int(lo))
        hi = min(len(swing.tip) - 1, int(hi))
        if hi > lo:
            curve = _drawable(project_path(cam, swing.tip[lo:hi + 1]))
            pts = curve.astype(np.int32)
            nseg = len(pts) - 1
            for k in range(nseg):
                if path_fade:
                    # Oldest end dim and thin, newest end bright and thick, so
                    # the stroke reads as a direction of travel rather than a
                    # static shape.
                    w = (k + 1) / nseg
                    col = tuple(int(round(c * (0.30 + 0.70 * w)))
                                for c in path_color)
                    thick = 1 + int(round(2 * w))
                else:
                    col, thick = path_color, 2
                cv2.line(out, tuple(pts[k]), tuple(pts[k + 1]), col,
                         thick, lineType=cv2.LINE_AA)

    # Impact marker (always at its true projected location).
    if show_impact and 0 <= swing.impact_idx < len(swing.tip):
        imp = _drawable(project_points(cam, swing.tip[swing.impact_idx])).ravel()
        cv2.circle(out, tuple(int(round(v)) for v in imp), 12, impact_color, 2)
        cv2.putText(out, "IMPACT", (int(imp[0]) + 12, int(imp[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, impact_color, 2)

    # Animated dot at the current video time.
    if show_dot:
        dot = _drawable(project_points(cam, swing.tip[idx])).ravel()
        cv2.circle(out, tuple(int(round(v)) for v in dot), 7, dot_color, -1)
        cv2.circle(out, tuple(int(round(v)) for v in dot), 12, (255, 255, 255), 2)

    return out


def overlay_status_text(frame: np.ndarray, txt: str) -> np.ndarray:
    """Overlay small status text (e.g. frame/time) on the top-left corner."""
    out = frame.copy()
    cv2.putText(out, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return out
