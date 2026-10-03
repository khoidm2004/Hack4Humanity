"""Person B — camera calibration & path projection (PLAN.md §7 Phase 2).

Implements the "path-projection" overlay pipeline:

1. ``calibrate`` — recover the camera pose from a single frame using
   ``cv2.solvePnP`` with ≥4 non-collinear clicked racket points and known
   racket geometry.
2. ``project_path`` — project the entire 3D racket-head trajectory onto the
   image as a 2D curve via ``cv2.projectPoints``.
3. ``draw_overlay`` — draw the static path polyline, an animated dot for a
   given video time, and the impact marker.

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
    ...
    """
    out = frame.copy()

    # Static projected path.
    if show_path:
        curve = project_path(cam, swing.tip)
        pts = curve.astype(np.int32)[:, None, :]
        cv2.polylines(out, [pts], isClosed=False, color=path_color,
                      thickness=2, lineType=cv2.LINE_AA)

    # Impact marker (always at its true projected location).
    if show_impact and 0 <= swing.impact_idx < len(swing.tip):
        imp = project_points(cam, swing.tip[swing.impact_idx]).ravel()
        cv2.circle(out, tuple(int(round(v)) for v in imp), 12, impact_color, 2)
        cv2.putText(out, "IMPACT", (int(imp[0]) + 12, int(imp[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, impact_color, 2)

    # Animated dot at the current video time.
    if show_dot:
        if dot_idx is None:
            idx = _index_for_video_time(swing.t, t_video, fps, scale, offset)
        else:
            idx = int(dot_idx)
        dot = project_points(cam, swing.tip[idx]).ravel()
        cv2.circle(out, tuple(int(round(v)) for v in dot), 7, dot_color, -1)
        cv2.circle(out, tuple(int(round(v)) for v in dot), 12, (255, 255, 255), 2)

    return out


def overlay_status_text(frame: np.ndarray, txt: str) -> np.ndarray:
    """Overlay small status text (e.g. frame/time) on the top-left corner."""
    out = frame.copy()
    cv2.putText(out, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return out
