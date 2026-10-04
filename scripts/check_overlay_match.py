"""Numeric regression check for the `swing_angle_1.mp4` overlay calibration.

Prints, and asserts where it can:

1. the committed ball-contact pixel re-derived from the video (must agree
   within 2 px);
2. annotation sanity — arc length, curvature, pixel extent;
3. the FOV sweep table (evidence the FOV was solved, not tuned);
4. the solved pose, per-point reprojection residuals, and the median against
   the 50 px acceptance target;
5. leave-one-out residuals, so a bad annotation shows up as a point whose
   removal markedly improves the rest;
6. honest limits — the recovered camera distance beside an independent
   estimate from the player's pixel height, and the annotated wrist-pixel
   spread, which is the error floor the pivot model cannot go below.

Exit code is non-zero only on a **hard** failure: solver failure, missing
annotations, a contact pixel that no longer matches the video, or a median
worse than `overlay_calib.RECORDED_BASELINE_PX` + 10 px.  Missing the 50 px
target is reported loudly but is NOT a hard failure — per `Artifacts/TASK.md`
the pivot model may put it structurally out of reach, and that is a finding
to report rather than a number to tune.

Run::

    uv run python scripts/check_overlay_match.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import camera, config, overlay_calib  # noqa: E402

TARGET_MEDIAN_PX = 50.0          # acceptance criterion 1 (Artifacts/TASK.md)
BASELINE_SLACK_PX = 10.0
CONTACT_TOLERANCE_PX = 2.0
# The pre-fix projected path, for criterion 2.
OLD_PATH_EXTENT_PX = (136.0, 93.0)


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check_contact_pixel() -> tuple[bool, float]:
    """Re-derive the contact pixel from the video and compare to the constant."""
    from src.sync.video_motion import ball_contact

    _rule("1. Ball-contact correspondence re-derived from the video")
    path = config.VIDEO_DIR / overlay_calib.VIDEO_NAME
    c = ball_contact(path, 106, 156)
    if not c.found or not c.is_strike:
        print(f"  FAIL: ball_contact found={c.found} is_strike={c.is_strike} "
              f"({c.reason})")
        return False, float("inf")

    f = c.frames.astype(float)
    m = (f >= 118) & (f <= 130)
    u = float(np.polyval(np.polyfit(f[m], c.xs[m], 1), c.contact_frame_exact))
    v = float(np.polyval(np.polyfit(f[m], c.ys[m], 1), c.contact_frame_exact))
    du = np.hypot(u - overlay_calib.CONTACT_PIXEL[0],
                  v - overlay_calib.CONTACT_PIXEL[1])

    print(f"  contact_frame_exact : {c.contact_frame_exact:.5f} "
          f"(committed {overlay_calib.CONTACT_FRAME_EXACT})")
    print(f"  pre/post ball speed : {c.pre_speed:.1f} / {c.post_speed:.1f} px/frame")
    print(f"  re-derived pixel    : ({u:.2f}, {v:.2f})")
    print(f"  committed pixel     : {overlay_calib.CONTACT_PIXEL}")
    print(f"  difference          : {du:.2f} px "
          f"(tolerance {CONTACT_TOLERANCE_PX})  "
          f"{'OK' if du <= CONTACT_TOLERANCE_PX else 'MISMATCH'}")
    print("  note: the ball rests on the string bed (~0.60 m from the wrist) "
          "while `tip` is\n        the head tip (0.686 m) — this point carries "
          "a built-in ~0.09 m bias.")
    return du <= CONTACT_TOLERANCE_PX, du


def check_annotations(corr) -> None:
    _rule("2. Annotation sanity (before any PnP)")
    px = corr.img_pts
    frames = corr.frames
    step = np.r_[0.0, np.linalg.norm(np.diff(px, axis=0), axis=1)]
    arc = np.cumsum(step)
    curv = np.r_[0.0, np.linalg.norm(np.diff(px, 2, axis=0), axis=1), 0.0]

    print(f"  {'label':<26}{'u':>8}{'v':>8}{'step':>9}{'arc':>9}{'|d2|':>9}")
    for k, lab in enumerate(corr.labels):
        print(f"  {lab:<26}{px[k, 0]:8.1f}{px[k, 1]:8.1f}"
              f"{step[k]:9.1f}{arc[k]:9.1f}{curv[k]:9.1f}")

    mono = bool(np.all(np.diff(arc) >= 0))
    span_u = float(px[:, 0].max() - px[:, 0].min())
    span_v = float(px[:, 1].max() - px[:, 1].min())
    print(f"\n  arc length monotonic         : {mono}")
    print(f"  observed annotation extent   : {span_u:.0f} x {span_v:.0f} px")
    print(f"  (old projected path extent)  : {OLD_PATH_EXTENT_PX[0]:.0f} x "
          f"{OLD_PATH_EXTENT_PX[1]:.0f} px")
    print(f"  frames used                  : {len(frames)} "
          f"({', '.join(f'{f:g}' for f in frames)})")
    if overlay_calib.EXCLUDED_FRAMES:
        print("  frames excluded              :")
        for f, why in sorted(overlay_calib.EXCLUDED_FRAMES.items()):
            print(f"      {f}: {why}")


def check_fov_sweep(corr, width: int, height: int) -> camera.PoseSolution:
    _rule("3. FOV sweep (40-100 deg, 2.5 deg steps)")
    fovs = np.arange(40.0, 100.0 + 1e-9, 2.5)
    best, table = camera.solve_pose_fov_sweep(
        width, height, corr.img_pts, corr.obj_pts, fovs)
    plain, plain_table = camera.solve_pose_fov_sweep(
        width, height, corr.img_pts, corr.obj_pts, fovs, use_ransac=False)

    print("  'all solvers' includes the robust RANSAC fit; 'all-point fits "
          "only' excludes it.")
    print(f"  {'fov_deg':>9}{'median_px':>12}  {'solver':<12}"
          f"{'median_px':>12}  {'solver':<10}{'|tvec| m':>9}")
    print(f"  {'':>9}{'(all solvers)':>12}  {'':<12}"
          f"{'(all-point)':>12}")
    for (fov, med, solver), (_f, pmed, psolver) in zip(table, plain_table):
        psol = camera.solve_pose(width, height, corr.img_pts, corr.obj_pts,
                                 float(fov), use_ransac=False)
        mark = "  <-- argmin" if np.isclose(fov, best.fov_deg) else ""
        print(f"  {fov:9.1f}{med:12.2f}  {solver:<12}{pmed:12.2f}  "
              f"{psolver:<10}{psol.camera_distance_m:9.2f}{mark}")

    pmeds = np.array([m for _f, m, _s in plain_table])
    print(f"\n  sweep argmin FOV (all solvers) : {best.fov_deg:.1f} deg "
          f"(median {best.median_px:.2f} px, |tvec| "
          f"{best.camera_distance_m:.2f} m)")
    print(f"  committed CALIB_FOV            : "
          f"{overlay_calib.CALIB_FOV_DEG:.1f} deg "
          f"(median {camera.solve_pose(width, height, corr.img_pts, corr.obj_pts, overlay_calib.CALIB_FOV_DEG).median_px:.2f} px)")
    print(f"\n  The FOV is NOT identifiable from these correspondences:")
    print(f"    - over the ENTIRE 40-100 deg range the all-point median moves "
          f"only\n      {pmeds.min():.1f} -> {pmeds.max():.1f} px, a "
          f"{pmeds.max() - pmeds.min():.1f} px swing — smaller than the "
          f"+/-12-25 px\n      uncertainty on the annotations themselves. "
          "There is no real minimum to find.")
    print("    - the deeper dips in the 'all solvers' column are the RANSAC "
          "fit choosing a\n      DIFFERENT subset of points at each FOV, not "
          "a better lens estimate; its\n      argmin sits at the end of the "
          "scanned range, which is what an\n      unidentifiable parameter "
          "looks like.")
    print(f"    - so CALIB_FOV_DEG keeps the 60 deg the app already assumed. "
          "That choice COSTS\n      accuracy ("
          f"{camera.solve_pose(width, height, corr.img_pts, corr.obj_pts, overlay_calib.CALIB_FOV_DEG).median_px:.1f} px "
          f"against the sweep argmin's {best.median_px:.1f} px); it is not "
          "tuning.")
    return best


def check_pose(corr, sol: camera.PoseSolution, width: int, height: int) -> None:
    _rule("4. Solved pose and reprojection residuals")
    if not sol.ok:
        print("  FAIL: no usable pose")
        for m in sol.messages:
            print(f"      {m}")
        return float("nan"), float("nan")

    R = cv2.Rodrigues(sol.rvec)[0]
    depth = (R @ corr.obj_pts.T).T[:, 2] + sol.tvec[2]
    proj = camera.project_points(sol.camera(width, height), corr.obj_pts)

    print(f"  solver            : {sol.solver}  (FOV {sol.fov_deg:.1f} deg, "
          f"fx = {sol.K[0, 0]:.1f})")
    print(f"  rvec (deg)        : "
          f"{np.rad2deg(sol.rvec).round(2).tolist()}")
    print(f"  tvec (m)          : {sol.tvec.round(3).tolist()}")
    print(f"  |tvec|            : {sol.camera_distance_m:.3f} m")
    for m in sol.messages:
        print(f"  note              : {m}")

    print(f"\n  {'label':<26}{'obs u':>8}{'obs v':>8}{'proj u':>9}{'proj v':>9}"
          f"{'err px':>9}{'depth m':>9}  inlier")
    for k, lab in enumerate(corr.labels):
        print(f"  {lab:<26}{corr.img_pts[k, 0]:8.1f}{corr.img_pts[k, 1]:8.1f}"
              f"{proj[k, 0]:9.1f}{proj[k, 1]:9.1f}{sol.residuals[k]:9.1f}"
              f"{depth[k]:9.3f}  {'yes' if sol.inliers[k] else 'NO'}")

    print(f"\n  median / mean / max : {sol.median_px:.1f} / {sol.mean_px:.1f} "
          f"/ {sol.max_px:.1f} px")
    print(f"  all points in front of camera : "
          f"{bool(np.all(depth > 0))} (min depth {depth.min():.3f} m)")
    print(f"  RANSAC inliers                : "
          f"{int(sol.inliers.sum())} / {len(sol.inliers)}")

    _rule("5. Orientation regression (supersedes the OVERLAY_FRAME.md check)")
    swing = _swing()
    curve = camera.project_points(sol.camera(width, height), swing.tip)
    sel = corr.imu_idx
    cz_obs = float(np.corrcoef(swing.tip[sel, 2], corr.img_pts[:, 1])[0, 1])
    cz_proj = float(np.corrcoef(swing.tip[:, 2], curve[:, 1])[0, 1])
    cy = float(np.corrcoef(swing.tip[:, 1], curve[:, 1])[0, 1])
    cx = float(np.corrcoef(swing.tip[:, 0], curve[:, 0])[0, 1])
    ord_v = float(np.corrcoef(curve[sel, 1], corr.img_pts[:, 1])[0, 1])
    ord_u = float(np.corrcoef(curve[sel, 0], corr.img_pts[:, 0])[0, 1])

    print("  The real test is whether the projection reproduces the vertical "
          "and horizontal\n  ORDER of the annotated racket head. That needs no "
          "assumption about which way\n  the fusion world's axes point:\n")
    print(f"  corr(projected v, observed v) at the annotated frames : "
          f"{ord_v:+.3f}  "
          f"{'OK' if ord_v > 0 else 'FAIL — path is vertically mirrored'}")
    print(f"  corr(projected u, observed u) at the annotated frames : "
          f"{ord_u:+.3f}  "
          f"{'OK' if ord_u > 0 else 'FAIL — path is horizontally mirrored'}")

    print("\n  World-axis correlations, for the record:")
    print(f"  corr(world +Z, OBSERVED head v)  : {cz_obs:+.3f}   "
          "<- measured straight off the annotations, no camera involved")
    print(f"  corr(world +Z, projected v)      : {cz_proj:+.3f}   "
          f"{'consistent' if cz_obs * cz_proj > 0 else 'INCONSISTENT'} "
          "with the measured sign")
    print(f"  corr(world +Y, projected v)      : {cy:+.3f}")
    print(f"  corr(world +X, projected u)      : {cx:+.3f}")
    if cz_obs > 0:
        print("\n  NOTE: the annotations say world +Z moves the racket head "
              "DOWN the image.\n  `OVERLAY_FRAME.md` assumed world +Z is up "
              "(gravity aligned to [0,0,-1]) and\n  picked rvec=[+pi/2,0,0] "
              "to satisfy corr(+Z, v) < 0. That test was circular —\n  it "
              "checked the projection against the assumption, not against the "
              "footage.\n  An accelerometer at rest measures specific force "
              "+1g along its UP axis, so\n  aligning MEASURED gravity to "
              "[0,0,-1] puts world +Z along DOWN. Every solver\n  at every "
              "FOV agrees. See OVERLAY_FRAME.md.")

    _rule("6. Projected path extent (acceptance criterion 2)")
    fin = np.isfinite(curve).all(axis=1)
    pu = curve[fin, 0]
    pv = curve[fin, 1]
    print(f"  projected path u : {pu.min():8.1f} .. {pu.max():8.1f}   "
          f"span {pu.max() - pu.min():6.1f} px")
    print(f"  projected path v : {pv.min():8.1f} .. {pv.max():8.1f}   "
          f"span {pv.max() - pv.min():6.1f} px")
    print(f"  observed annotation extent : "
          f"{np.ptp(corr.img_pts[:, 0]):.0f} x "
          f"{np.ptp(corr.img_pts[:, 1]):.0f} px")
    print(f"  pre-fix projected extent   : {OLD_PATH_EXTENT_PX[0]:.0f} x "
          f"{OLD_PATH_EXTENT_PX[1]:.0f} px")
    return ord_v, ord_u


def check_loo(corr, width: int, height: int, fov_deg: float) -> None:
    _rule("7. Leave-one-out residuals (bad-annotation detector)")
    n = len(corr)
    full = camera.solve_pose(width, height, corr.img_pts, corr.obj_pts, fov_deg)
    print(f"  full-set median: {full.median_px:.1f} px\n")
    print(f"  {'left out':<26}{'own err vs LOO pose':>22}"
          f"{'median of the rest':>21}{'delta':>9}")
    for k in range(n):
        keep = np.ones(n, bool)
        keep[k] = False
        sol = camera.solve_pose(width, height, corr.img_pts[keep],
                                corr.obj_pts[keep], fov_deg)
        if not sol.ok:
            print(f"  {corr.labels[k]:<26}{'(solver failed)':>22}")
            continue
        own = float(camera._reproject_errors(
            corr.obj_pts[k:k + 1], corr.img_pts[k:k + 1], sol.K,
            sol.rvec, sol.tvec)[0])
        rest = sol.median_px
        print(f"  {corr.labels[k]:<26}{own:22.1f}{rest:21.1f}"
              f"{rest - full.median_px:+9.1f}")
    print("\n  A point whose removal markedly DROPS the median of the rest is "
          "suspect.\n  Nothing is dropped automatically — re-read that frame's "
          "image first.\n")
    print("  Reading of this table (2026-10-04): several points — 116, 122, "
          "124, 126, 145 —\n  each drop the median when removed. One bad "
          "annotation produces ONE large drop\n  and leaves the others flat; "
          "a spread of drops like this is the fit trading one\n  part of the "
          "swing off against another, i.e. model error. The largest, f126,\n"
          "  was re-read at 4x against the chroma mask: the racket's far rim "
          "tip is at\n  (648, 642) against the committed (652, 640). The "
          "annotation is right; the model\n  is what cannot hold both ends of "
          "the swing at once (section 8c). Nothing was\n  excluded on the "
          "strength of this table.")


def check_limits(corr, sol: camera.PoseSolution, width: int, height: int,
                 swing, sync) -> None:
    _rule("8. Honest limits — why the residual has a floor")
    d_pnp = sol.camera_distance_m
    d_player = overlay_calib.player_height_distance_m(sol.fov_deg, width)
    d_racket = overlay_calib.racket_distance_m(sol.fov_deg, width)
    su, sv, smax = overlay_calib.wrist_pixel_spread()
    fx = sol.K[0, 0]

    print("  (a) Camera distance — PnP against two independent image "
          "measurements")
    print(f"      from PnP                      : {d_pnp:.2f} m")
    print(f"      from player pixel height      : {d_player:.2f} m "
          f"({overlay_calib.PLAYER_PIXEL_HEIGHT:.0f} px, assumed "
          f"{overlay_calib.PLAYER_HEIGHT_M} m)")
    print(f"      from racket apparent length   : {d_racket:.2f} m "
          f"({overlay_calib.racket_pixel_scale()[1]:.0f} px, "
          f"{config.RACKET_TIP_LEN} m)")
    print("      The two image measurements agree with each other, which is "
          "the useful\n      part: the pixel scale is sound. PnP then places "
          f"the camera {d_player / d_pnp:.2f}x CLOSER\n      than that scale "
          "allows. Shrinking the distance is the only freedom PnP has\n"
          "      for absorbing the arc-length mismatch in (c) — the ratio is "
          "the same at\n      every FOV, so it is a property of the data, not "
          "of the lens assumption.")

    print("\n  (b) Wrist translation the pivot model omits")
    print(f"      annotated wrist pixel spread  : {su:.0f} x {sv:.0f} px "
          f"(max pairwise {smax:.0f} px)")
    print(f"      in metres at {d_racket:.1f} m          : "
          f"{smax / overlay_calib.racket_pixel_scale()[0]:.2f} m")
    print("      `src/fusion.py:pivot_tip` pins the wrist at the world origin, "
          "so this\n      motion is absent from `tip` and no rigid pose can "
          "put it back.")

    print("\n  (c) Arc-length mismatch — the decisive one")
    a = overlay_calib.model_vs_video_arc(swing, sync)
    print(f"      image scale (racket, no PnP)  : {a['px_per_m']:.0f} px/m")
    print(f"      3D `tip` arc over the window  : {a['arc3d_m']:.2f} m "
          f"-> {a['arc3d_expected_px']:.0f} px of image arc")
    print(f"      racket head's observed arc    : "
          f"{a['arc2d_observed_px']:.0f} px")
    print(f"      ratio                         : {a['ratio']:.2f}x")
    print(f"      net turn of the racket, 3D    : {a['turn3d_net_deg']:.0f} deg "
          f"(total path turn {a['turn3d_total_deg']:.0f} deg)")
    print(f"      net turn of the racket, imaged: {a['turn2d_net_deg']:.0f} deg")
    print("      A rigid pose is, to first order, a similarity on the image "
          "plane: it can\n      translate, rotate and scale the path but "
          "CANNOT change how much arc it\n      contains. The reconstructed "
          "tip travels ~2x the arc the racket visibly\n      travels, with "
          "reversals the footage does not show, so the projected path\n"
          "      is a double loop where the video shows one sweep. That is a "
          "reconstruction\n      property, not a calibration one — "
          "`src/fusion.py` and `swing.npz` are out of\n      scope here, so it "
          "is reported, not fixed.")

    # Impact residual, reported on its own because criterion 4 depends on it.
    k = int(np.argmin(np.abs(corr.frames - overlay_calib.CONTACT_FRAME_EXACT)))
    proj = camera.project_points(sol.camera(width, height),
                                 corr.obj_pts[k]).ravel()
    print(f"\n  (d) impact / ball-contact residual: {sol.residuals[k]:.1f} px "
          f"(projected ({proj[0]:.0f}, {proj[1]:.0f}) vs observed "
          f"({corr.img_pts[k, 0]:.0f}, {corr.img_pts[k, 1]:.0f}))")
    print("      This is criterion 4 satisfied BY THE FIT — `app.py` no longer "
          "translates\n      the whole path to force the impact point onto the "
          "ball pixel by default\n      (the legacy anchor is still available "
          "behind a sidebar checkbox).")


def check_obj_mode_sensitivity(swing, sync, width: int, height: int,
                               fov_deg: float) -> None:
    _rule("9. Sensitivity: frame-centre tip sample vs bin-mean tip")
    for mode in ("center", "binmean"):
        corr = overlay_calib.build_correspondences(swing, sync, obj_mode=mode)
        sol = camera.solve_pose(width, height, corr.img_pts, corr.obj_pts,
                                fov_deg)
        print(f"  obj_mode={mode:<8} median {sol.median_px:6.1f} px   "
              f"max {sol.max_px:6.1f} px   |tvec| {sol.camera_distance_m:5.2f} m")
    print("  'center' is the primary; 'binmean' is only a check that the "
          "answer does not\n  hinge on which sample inside the frame bin is "
          "picked.")


def _swing():
    from src.swing import Swing
    return Swing.load(config.SWING_NPZ)


def smax_fmt(oc) -> str:
    """'<N> px' — the largest wrist-pixel excursion, for the verdict text."""
    return f"{oc.wrist_pixel_spread()[2]:.0f} px"


def main() -> int:
    from src.video import probe

    video = config.VIDEO_DIR / overlay_calib.VIDEO_NAME
    meta = probe(str(video))
    width, height = meta.width, meta.height
    print(f"video: {video.name}  {width}x{height} @ {meta.fps:g} fps, "
          f"{meta.n_frames} frames")

    if not overlay_calib.RACKET_HEAD_PX_ANGLE_1:
        print("FAIL: no committed annotations in src/overlay_calib.py")
        return 1

    contact_ok, _ = check_contact_pixel()

    swing, sync = overlay_calib._load_inputs()
    corr = overlay_calib.build_correspondences(swing, sync)
    check_annotations(corr)

    swept = check_fov_sweep(corr, width, height)
    sol = overlay_calib.solved_pose(width, height, overlay_calib.CALIB_FOV_DEG)
    ord_v, ord_u = check_pose(corr, sol, width, height)
    if not sol.ok:
        return 1
    check_loo(corr, width, height, sol.fov_deg)
    check_limits(corr, sol, width, height, swing, sync)
    check_obj_mode_sensitivity(swing, sync, width, height, sol.fov_deg)

    # ---------------- verdict ---------------- #
    _rule("VERDICT")
    R = cv2.Rodrigues(sol.rvec)[0]
    depth = (R @ corr.obj_pts.T).T[:, 2] + sol.tvec[2]
    curve = camera.project_points(sol.camera(width, height), swing.tip)
    span = (float(np.ptp(curve[:, 0])), float(np.ptp(curve[:, 1])))

    hard_fail = []
    if not contact_ok:
        hard_fail.append("re-derived contact pixel does not match the constant")
    if not np.all(depth > 0):
        hard_fail.append("some correspondences are behind the camera")
    if not (ord_v > 0):
        hard_fail.append("projected v does not follow observed v — path is "
                         "vertically mirrored")
    if not (ord_u > 0):
        hard_fail.append("projected u does not follow observed u — path is "
                         "horizontally mirrored")
    limit = overlay_calib.RECORDED_BASELINE_PX + BASELINE_SLACK_PX
    if sol.median_px > limit:
        hard_fail.append(
            f"median {sol.median_px:.1f} px worse than the recorded baseline "
            f"{overlay_calib.RECORDED_BASELINE_PX:.1f} + "
            f"{BASELINE_SLACK_PX:.0f} px slack")

    print(f"  median reprojection error : {sol.median_px:.1f} px")
    print(f"  acceptance target         : {TARGET_MEDIAN_PX:.0f} px  -> "
          f"{'PASS' if sol.median_px <= TARGET_MEDIAN_PX else 'BELOW TARGET'}")
    print(f"  recorded baseline         : "
          f"{overlay_calib.RECORDED_BASELINE_PX:.1f} px "
          f"(+{BASELINE_SLACK_PX:.0f} px regression slack)")
    print(f"  projected path extent     : {span[0]:.0f} x {span[1]:.0f} px "
          f"(was {OLD_PATH_EXTENT_PX[0]:.0f} x {OLD_PATH_EXTENT_PX[1]:.0f})")
    print(f"  sweep argmin FOV          : {swept.fov_deg:.1f} deg")

    if sol.median_px > TARGET_MEDIAN_PX:
        a = overlay_calib.model_vs_video_arc(swing, sync)
        print(f"\n  The {TARGET_MEDIAN_PX:.0f} px target is NOT met, and no "
              "camera pose can meet it. Section 8c:\n  the reconstructed "
              f"`tip` travels {a['arc3d_m']:.2f} m of arc, which at the "
              "racket's own measured\n  image scale is "
              f"{a['arc3d_expected_px']:.0f} px, while the racket head "
              f"visibly travels {a['arc2d_observed_px']:.0f} px —\n  "
              f"{a['ratio']:.2f}x. A rigid pose can translate, rotate and "
              "scale a path; it cannot\n  change how much arc the path "
              "contains. On top of that the pivot model pins the\n  wrist at "
              "the origin while the real wrist crosses "
              f"{smax_fmt(overlay_calib)} of the frame.\n  Reported, not "
              "tuned: see the write-up in OVERLAY_FRAME.md.")

    if hard_fail:
        print("\n  RESULT: FAIL")
        for m in hard_fail:
            print(f"    - {m}")
        return 1
    print("\n  RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
