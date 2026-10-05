"""Deliverable 6 + 7 — per-frame projected-pixel table and the wrist-sweep report.

Reconstructs the swing **in memory** and never writes `data/outputs/swing.npz`
(no `swing.save()` call anywhere here), so `--lever` can score an alternative
racket body lever against both the before and after numbers from the same
code path, without touching the committed contract file.

Deliberately calls `src.camera.solve_pose` directly on an in-memory `tip`
rather than `src.overlay_calib.solved_pose`: that function is `lru_cache`d on
`(width, height, fov)` only and loads `swing.npz` *inside* the cached call, so
regenerating the npz and then re-solving in the same process would silently
return the stale pose (Artifacts/analysis.md §9.4). Calling `solve_pose`
directly sidesteps the cache entirely.

Run::

    uv run python scripts/report_followthrough.py            # after: -x lever
    uv run python scripts/report_followthrough.py --lever 0,0,1   # before
    uv run python scripts/report_followthrough.py --wrist-sweep   # deliverable 7

Writes nothing to disk. Exit code is always 0 — this is a reporting script,
not a gate.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from src import camera, config, fusion, load_csv, overlay_calib as oc  # noqa: E402
from src import to_signal_arrays  # noqa: E402

WIDTH, HEIGHT = 1280, 720
TABLE_FRAMES = range(112, 226)       # f112..f225 inclusive


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def tip_for_lever(quat: np.ndarray, lever) -> np.ndarray:
    u = np.asarray(lever, float)
    u = u / np.linalg.norm(u)
    R = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    return R.apply(u * config.RACKET_TIP_LEN)


def solve_for(swing_alt, sync):
    corr = oc.build_correspondences(swing_alt, sync)     # the 13 + contact, unchanged
    sol = camera.solve_pose(WIDTH, HEIGHT, corr.img_pts, corr.obj_pts,
                            oc.CALIB_FOV_DEG)
    return corr, sol


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _fmt_corr(c: float) -> str:
    return "n/a" if not np.isfinite(c) else f"{c:+.3f}"


def wrap180(deg: float) -> float:
    return float(((deg + 180.0) % 360.0) - 180.0)


def held_out_residuals(sol, swing_alt, sync, lo_sample_lookup):
    """Per held-out frame: projected head pixel vs the §3 reads, px residual."""
    rows = []
    cam = sol.camera(WIDTH, HEIGHT)
    for f in sorted(oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1):
        idx = sync.imu_idx_for_frame_centered(f)
        proj = camera.project_points(cam, swing_alt.tip[idx]).ravel()
        obs = np.asarray(oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1[f], float)
        resid = float(np.linalg.norm(proj - obs))
        rows.append((f, idx, proj[0], proj[1], obs[0], obs[1], resid))
    return rows


def held_out_image_angle(sol, swing_alt, sync):
    """Per held-out frame: signed racket image-angle error, model vs video."""
    cam = sol.camera(WIDTH, HEIGHT)
    origin_px = camera.project_points(cam, np.zeros(3)).ravel()
    rows = []
    for f in sorted(oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1):
        idx = sync.imu_idx_for_frame_centered(f)
        head_px = camera.project_points(cam, swing_alt.tip[idx]).ravel()
        m = head_px - origin_px
        o = (np.asarray(oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1[f], float)
             - np.asarray(oc.WRIST_PX_HELDOUT_ANGLE_1[f], float))
        err = wrap180(np.degrees(np.arctan2(m[1], m[0])
                                  - np.arctan2(o[1], o[0])))
        rows.append((f, err))
    return rows, origin_px


def print_lever_and_fit(lever, accel, gyro):
    _rule("1. Lever and the centripetal fit")
    u = np.asarray(lever, float)
    u = u / np.linalg.norm(u)
    print(f"  lever in use (unit)   : {np.round(u, 4).tolist()}")
    windows = [
        ("post-impact 205-399", 205, None, 0.0),
        ("all unclipped", 0, None, 0.0),
        ("high-omega unclipped", 0, None, 400.0),
    ]

    def _axis_angle(a, b):
        c = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        return float(np.rad2deg(np.arccos(min(c, 1.0))))

    for label, lo, hi, omega_min in windows:
        fit_u, r, _bias, n = fusion.fit_centripetal_lever(
            accel, gyro, lo=lo, hi=hi, omega_min_dps=omega_min)
        ang_lever = _axis_angle(fit_u, u)
        ang_pz = _axis_angle(fit_u, np.array([0.0, 0.0, 1.0]))
        print(f"  {label:<22} |r|={r:.3f} m  unit={np.round(fit_u, 3).tolist()} "
              f"n={n:4d}  -> lever {ang_lever:5.1f} deg, +z {ang_pz:5.1f} deg")


def print_pose(corr, sol):
    _rule("2. Pose")
    if not sol.ok:
        print("  FAIL: no usable pose")
        for m in sol.messages:
            print(f"    {m}")
        return
    print(f"  solver             : {sol.solver}")
    print(f"  median/mean/max px : {sol.median_px:.1f} / {sol.mean_px:.1f} / "
          f"{sol.max_px:.1f}")
    print(f"  inliers            : {int(sol.inliers.sum())} / {len(sol.inliers)}")
    print(f"  |tvec|             : {sol.camera_distance_m:.3f} m")
    print(f"\n  {'label':<26}{'obs u':>8}{'obs v':>8}{'proj u':>9}{'proj v':>9}"
          f"{'err px':>9}  inlier")
    cam = sol.camera(WIDTH, HEIGHT)
    proj = camera.project_points(cam, corr.obj_pts)
    for k, lab in enumerate(corr.labels):
        print(f"  {lab:<26}{corr.img_pts[k, 0]:8.1f}{corr.img_pts[k, 1]:8.1f}"
              f"{proj[k, 0]:9.1f}{proj[k, 1]:9.1f}{sol.residuals[k]:9.1f}  "
              f"{'yes' if sol.inliers[k] else 'NO'}")


def print_pixel_table(sol, swing_alt, sync):
    _rule("3. Per-frame projected-pixel table, f112-225")
    cam = sol.camera(WIDTH, HEIGHT)
    fit_frames = set(oc.RACKET_HEAD_PX_ANGLE_1)
    held_frames = set(oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1)
    print(f"  {'frm':>4}{'smp':>5}{'u':>9}{'v':>9}{'step':>8}{'in':>4}"
          f"{'cov':>4}{'obs_u':>8}{'obs_v':>8}{'resid':>8}")
    prev = None
    out_of_frame = []
    uvs = []
    steps_at = {}
    first_settle = None
    for f in TABLE_FRAMES:
        covered = sync.covers_frame(f)
        idx = sync.imu_idx_for_frame_centered(f)
        proj = camera.project_points(cam, swing_alt.tip[idx]).ravel()
        u, v = float(proj[0]), float(proj[1])
        step = float(np.linalg.norm(proj - prev)) if prev is not None else float("nan")
        prev = proj
        in_frame = 0 <= u < 1280 and 0 <= v < 720
        if not in_frame:
            out_of_frame.append(f)
        if covered:
            uvs.append((u, v))
        tag = ""
        obs_u = obs_v = resid = None
        if f in fit_frames:
            tag = "fit"
            obs_u, obs_v = oc.RACKET_HEAD_PX_ANGLE_1[f]
            resid = float(np.hypot(u - obs_u, v - obs_v))
        elif f in held_frames:
            tag = "HELD"
            obs_u, obs_v = oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1[f]
            resid = float(np.hypot(u - obs_u, v - obs_v))
        step_str = "" if np.isnan(step) else f"{step:8.1f}"
        obs_str = (f"{obs_u:8.1f}{obs_v:8.1f}{resid:8.1f}" if obs_u is not None
                  else f"{'':8}{'':8}{'':8}")
        print(f"  {f:4d}{idx:5d}{u:9.1f}{v:9.1f}{step_str}"
              f"{'Y' if in_frame else 'N':>4}{'Y' if covered else 'N':>4}"
              f"{obs_str}  {tag}")
        if f in (204, 208, 212, 216, 220, 224):
            steps_at[f] = step
        if f >= 180 and first_settle is None and not np.isnan(step) and step <= 15.0:
            # must stay <=15 for the rest of the table to count as "settled"
            first_settle = f

    uv = np.array(uvs, float)
    print(f"\n  out-of-frame count over f112-225 : {len(out_of_frame)}")
    if out_of_frame:
        print(f"  out-of-frame frames              : {out_of_frame}")
    print(f"  covered-frame u range : {uv[:, 0].min():.1f} .. {uv[:, 0].max():.1f} "
          f"(span {np.ptp(uv[:, 0]):.1f})")
    print(f"  covered-frame v range : {uv[:, 1].min():.1f} .. {uv[:, 1].max():.1f} "
          f"(span {np.ptp(uv[:, 1]):.1f})")
    print("  step (px/frame) at f204/208/212/216/220/224 : " +
          ", ".join(f"f{f}={steps_at[f]:.1f}" for f in sorted(steps_at)))
    print(f"  first frame >= f180 where step <= 15 px/frame : {first_settle}")


def print_held_out_residual(sol, swing_alt, sync):
    _rule("4. Held-out residual, f146-220 (criterion 4a)")
    rows = held_out_residuals(sol, swing_alt, sync, None)
    print(f"  {'frame':>6}{'smp':>6}{'proj u':>9}{'proj v':>9}{'obs u':>8}"
          f"{'obs v':>8}{'resid':>8}")
    for f, idx, pu, pv, ou, ov, resid in rows:
        print(f"  {f:6d}{idx:6d}{pu:9.1f}{pv:9.1f}{ou:8.1f}{ov:8.1f}{resid:8.1f}")
    resids = np.array([r[-1] for r in rows], float)
    print(f"\n  median / mean / max : {np.median(resids):.1f} / "
          f"{np.mean(resids):.1f} / {np.max(resids):.1f} px")


def print_held_out_angle(sol, swing_alt, sync):
    _rule("5. Held-out racket image-angle error, f146-220 (criterion 4b)")
    rows, origin_px = held_out_image_angle(sol, swing_alt, sync)
    print(f"  pivot projects to approx ({origin_px[0]:.1f}, {origin_px[1]:.1f}) px")
    print(f"  {'frame':>6}{'signed err deg':>16}")
    for f, err in rows:
        print(f"  {f:6d}{err:16.1f}")
    errs = np.array([r[1] for r in rows], float)
    print(f"\n  mean |error| : {np.mean(np.abs(errs)):.1f} deg")


def print_windowed_extent(sol, swing_alt, sync):
    _rule("6. Windowed extent, f116-145 (criterion 6)")
    cam = sol.camera(WIDTH, HEIGHT)
    frames = sorted(oc.RACKET_HEAD_PX_ANGLE_1)
    lo = sync.imu_idx_for_frame_centered(frames[0])
    hi = sync.imu_idx_for_frame_centered(frames[-1])
    curve = camera.project_points(cam, swing_alt.tip[lo:hi + 1])
    fin = np.isfinite(curve).all(axis=1)
    span_u = float(np.ptp(curve[fin, 0]))
    span_v = float(np.ptp(curve[fin, 1]))
    print(f"  projected extent f116-145 : {span_u:.0f} x {span_v:.0f} px")
    print(f"  observed annotation extent: 558 x 305 px")


def print_arc_ratio(swing_alt, sync):
    _rule("7. Arc ratio, both windows (criterion 7)")
    a = oc.model_vs_video_arc(swing_alt, sync)
    b = oc.fullspan_arc_bound(swing_alt, sync)
    print(f"  annotated window f{a['window_frames'][0]}-{a['window_frames'][1]} "
          f"(samples {a['window_lo']}-{a['window_hi']}):")
    print(f"    ratio          : {a['ratio']:.3f}")
    print(f"    ratio_expected : {a['ratio_expected']:.3f} "
          f"(wrist spread used: annotated-window only)")
    print(f"  full span f{b['frames'][0]}-{b['frames'][1]} "
          f"(samples {b['window_lo']}-{b['window_hi']}, {b['n_reads']} reads):")
    print(f"    ratio          : {b['ratio']:.3f}")
    print(f"    ratio_expected : {b['ratio_expected']:.3f} "
          f"(wrist spread used: full-span, wrist_max_px={b['wrist_max_px']:.1f})")
    print("  Neither number above is to be chased by scaling anything.")


def run_wrist_sweep(quat, lever, sync, accel, gyro, swing0):
    _rule("8. Wrist-sweep experiment (deliverable 7, --wrist-sweep only)")
    ks = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    print(f"  {'k':>6}{'fit13 med':>12}{'held-out med':>14}{'mean |ang|':>12}"
          f"{'max resid':>12}{'inliers':>10}")
    base_tip = tip_for_lever(quat, lever)
    results = {}
    for k in ks:
        tip_k = k * swing0.cog + base_tip
        swing_k = replace(swing0, tip=tip_k)
        corr, sol = solve_for(swing_k, sync)
        held_rows = held_out_residuals(sol, swing_k, sync, None)
        held_resids = np.array([r[-1] for r in held_rows], float)
        angle_rows, _ = held_out_image_angle(sol, swing_k, sync)
        angle_errs = np.array([r[1] for r in angle_rows], float)
        results[k] = dict(
            fit_med=sol.median_px, held_med=float(np.median(held_resids)),
            mean_abs_angle=float(np.mean(np.abs(angle_errs))),
            max_resid=sol.max_px, inliers=int(sol.inliers.sum()),
        )
        print(f"  {k:6.2f}{sol.median_px:12.1f}{float(np.median(held_resids)):14.1f}"
              f"{float(np.mean(np.abs(angle_errs))):12.1f}{sol.max_px:12.1f}"
              f"{int(sol.inliers.sum()):10d}")

    # Correlations at k=0.
    swing_k0 = replace(swing0, tip=base_tip)
    corr0, sol0 = solve_for(swing_k0, sync)
    cam0 = sol0.camera(WIDTH, HEIGHT)
    head_rows = held_out_residuals(sol0, swing_k0, sync, None)
    wrist_disp_v, wrist_disp_u, resid_v, resid_u = [], [], [], []
    wrist0 = np.asarray(oc.WRIST_PX_HELDOUT_ANGLE_1[sorted(oc.WRIST_PX_HELDOUT_ANGLE_1)[0]], float)
    cog_proj_u, cog_proj_v, obs_wrist_u, obs_wrist_v = [], [], [], []
    for f, idx, pu, pv, ou, ov, _resid in head_rows:
        resid_vec = np.array([pu - ou, pv - ov])
        wrist_px = np.asarray(oc.WRIST_PX_HELDOUT_ANGLE_1[f], float)
        disp = wrist_px - wrist0
        resid_u.append(resid_vec[0]); resid_v.append(resid_vec[1])
        wrist_disp_u.append(disp[0]); wrist_disp_v.append(disp[1])
        cog_px = camera.project_points(cam0, swing0.cog[idx]).ravel()
        cog_proj_u.append(cog_px[0]); cog_proj_v.append(cog_px[1])
        obs_wrist_u.append(wrist_px[0]); obs_wrist_v.append(wrist_px[1])

    print(f"\n  corr(residual v, wrist displacement v) @ k=0 : "
          f"{_fmt_corr(_corr(resid_v, wrist_disp_v))}")
    print(f"  corr(residual u, wrist displacement u) @ k=0 : "
          f"{_fmt_corr(_corr(resid_u, wrist_disp_u))}")
    print(f"  corr(projected cog u, observed wrist u)      : "
          f"{_fmt_corr(_corr(cog_proj_u, obs_wrist_u))}")
    print(f"  corr(projected cog v, observed wrist v)      : "
          f"{_fmt_corr(_corr(cog_proj_v, obs_wrist_v))}")

    # Fixed rule, decided in advance.
    base = results[0.0]
    helps = []
    for k in ks:
        if k == 0.0:
            continue
        r = results[k]
        held_improve = (base["held_med"] - r["held_med"]) / base["held_med"] >= 0.20
        fit_ok = (r["fit_med"] - base["fit_med"]) <= 2.0
        angle_ok = r["mean_abs_angle"] <= base["mean_abs_angle"]
        if held_improve and fit_ok and angle_ok:
            helps.append(k)
    if helps:
        print(f"\n  VERDICT: helps (k={helps}), but NOT landed in this PR — "
              "adding `cog` to `tip` breaks the |tip|=RACKET_TIP_LEN invariant "
              "that verify_fusion.py's radius check and racket_pixel_scale's "
              "L-cancellation both rest on. Report to the coordinator as a "
              "follow-up task.")
    else:
        print("\n  VERDICT: does not clear the fixed rule (held-out median "
              ">=20% better, fit-13 median <=2px worse, mean |angle error| "
              "not worse) at any sampled k. Not included.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lever", type=str, default=None,
                   help="x,y,z body lever to score; default config.RACKET_LEVER_BODY")
    p.add_argument("--wrist-sweep", action="store_true")
    args = p.parse_args()

    lever = (config.RACKET_LEVER_BODY if args.lever is None
              else tuple(float(x) for x in args.lever.split(",")))

    df = load_csv()
    accel, gyro = to_signal_arrays(df)
    swing0 = fusion.reconstruct(accel, gyro)   # quat is lever-independent

    from src.synced_data import SyncedData
    from src.sync import paths as sync_paths
    sync = None
    for label, video, csv_out in sync_paths.ANGLES:
        if label == oc.LABEL:
            sync = SyncedData.for_video(label, video, csv_out)
    if sync is None:
        raise LookupError(f"no sync table configured for {oc.LABEL!r}")

    tip_alt = tip_for_lever(swing0.quat, lever)
    swing_alt = replace(swing0, tip=tip_alt)

    corr, sol = solve_for(swing_alt, sync)

    print_lever_and_fit(lever, accel, gyro)
    print_pose(corr, sol)
    print_pixel_table(sol, swing_alt, sync)
    print_held_out_residual(sol, swing_alt, sync)
    print_held_out_angle(sol, swing_alt, sync)
    print_windowed_extent(sol, swing_alt, sync)
    print_arc_ratio(swing_alt, sync)

    if args.wrist_sweep:
        run_wrist_sweep(swing0.quat, lever, sync, accel, gyro, swing0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
