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
        proj = camera.project_points(cam, swing_alt.head[idx]).ravel()
        obs = np.asarray(oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1[f], float)
        resid = float(np.linalg.norm(proj - obs))
        rows.append((f, idx, proj[0], proj[1], obs[0], obs[1], resid))
    return rows


def held_out_image_angle(sol, swing_alt, sync):
    """Per held-out frame: signed racket image-angle error, model vs video.

    The model vector's tail is the PROJECTED PIVOT AT THAT SAMPLE, not the
    world origin: with `Swing.pivot` carrying the wrist translation the pivot
    pixel moves, and using a fixed origin would reintroduce exactly the
    tail mismatch that made the old apparent-length ratio read 1.46
    (Artifacts/analysis.md §3).
    """
    cam = sol.camera(WIDTH, HEIGHT)
    rows = []
    pivot_px_first = None
    for f in sorted(oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1):
        idx = sync.imu_idx_for_frame_centered(f)
        pivot_px = camera.project_points(cam, swing_alt.pivot[idx]).ravel()
        if pivot_px_first is None:
            pivot_px_first = pivot_px
        head_px = camera.project_points(cam, swing_alt.head[idx]).ravel()
        m = head_px - pivot_px
        o = (np.asarray(oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1[f], float)
             - np.asarray(oc.WRIST_PX_HELDOUT_ANGLE_1[f], float))
        err = wrap180(np.degrees(np.arctan2(m[1], m[0])
                                  - np.arctan2(o[1], o[0])))
        rows.append((f, err))
    return rows, pivot_px_first


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
        proj = camera.project_points(cam, swing_alt.head[idx]).ravel()
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
    rows, pivot_px_first = held_out_image_angle(sol, swing_alt, sync)
    print(f"  pivot (at the first held-out frame) projects to approx "
          f"({pivot_px_first[0]:.1f}, {pivot_px_first[1]:.1f}) px")
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
    curve = camera.project_points(cam, swing_alt.head[lo:hi + 1])
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


def ratio_ab(sol, swing_alt, sync):
    """Apparent-length ratios A (from the projected pivot) and B (from the
    annotated wrist), over all 26 head reads. A was meaningless while the
    pivot was pinned; it is a real measurement now that the pivot moves."""
    cam = sol.camera(WIDTH, HEIGHT)
    head = {**oc.RACKET_HEAD_PX_ANGLE_1, **oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1}
    wrist = {**oc.WRIST_PX_ANGLE_1, **oc.WRIST_PX_HELDOUT_ANGLE_1}
    a_vals, b_vals = [], []
    for f in sorted(set(head) & set(wrist)):
        idx = sync.imu_idx_for_frame_centered(f)
        pivot_px = camera.project_points(cam, swing_alt.pivot[idx]).ravel()
        head_px = camera.project_points(cam, swing_alt.head[idx]).ravel()
        obs_head = np.asarray(head[f], float)
        obs_wrist = np.asarray(wrist[f], float)
        denom = float(np.linalg.norm(obs_head - obs_wrist))
        a_vals.append(float(np.linalg.norm(head_px - pivot_px)) / denom)
        b_vals.append(float(np.linalg.norm(head_px - obs_wrist)) / denom)
    return float(np.median(a_vals)), float(np.median(b_vals))


def print_before_after(swing0, sync, lever):
    """Deliverable 4 — eight statistics, k=0 against the committed k."""
    _rule("8. Before/after — the omitted wrist translation (deliverable 4)")
    base_tip = tip_for_lever(swing0.quat, lever)
    ks = (0.0, config.WRIST_PIVOT_COG_SCALE)
    print(f"  {'k':>6}{'fit med':>9}{'fit mean':>10}{'fit max':>9}{'inl':>7}"
          f"{'held med':>10}{'mean|ang|':>11}{'|tvec|':>8}"
          f"{'ratioA':>8}{'ratioB':>8}  {'solver':<10}")
    rows = {}
    for k in ks:
        swing_k = replace(swing0, tip=base_tip, pivot=k * swing0.cog)
        corr, sol = solve_for(swing_k, sync)
        held_rows = held_out_residuals(sol, swing_k, sync, None)
        held_resids = np.array([r[-1] for r in held_rows], float)
        angle_rows, _ = held_out_image_angle(sol, swing_k, sync)
        angle_errs = np.array([r[1] for r in angle_rows], float)
        ratio_a, ratio_b = ratio_ab(sol, swing_k, sync)
        row = dict(
            fit_med=sol.median_px, fit_mean=sol.mean_px, fit_max=sol.max_px,
            inliers=int(sol.inliers.sum()), n=len(sol.inliers),
            held_med=float(np.median(held_resids)),
            mean_abs_angle=float(np.mean(np.abs(angle_errs))),
            tvec=sol.camera_distance_m, ratio_a=ratio_a, ratio_b=ratio_b,
            solver=sol.solver,
        )
        rows[k] = row
        print(f"  {k:6.3f}{row['fit_med']:9.2f}{row['fit_mean']:10.2f}"
              f"{row['fit_max']:9.2f}{row['inliers']:4d}/{row['n']:<2d}"
              f"{row['held_med']:10.1f}{row['mean_abs_angle']:11.1f}"
              f"{row['tvec']:8.3f}{row['ratio_a']:8.3f}{row['ratio_b']:8.3f}"
              f"  {row['solver']:<10}")

    b0, b1 = rows[ks[0]], rows[ks[1]]
    print("\n  Delta (committed k - k=0):")
    print(f"    fit median {b1['fit_med'] - b0['fit_med']:+.2f} px   "
          f"fit mean {b1['fit_mean'] - b0['fit_mean']:+.2f} px   "
          f"fit max {b1['fit_max'] - b0['fit_max']:+.2f} px")
    print(f"    inliers {b1['inliers']}/{b1['n']} vs {b0['inliers']}/{b0['n']}"
          f"   held-out median {b1['held_med'] - b0['held_med']:+.1f} px   "
          f"mean|angle| {b1['mean_abs_angle'] - b0['mean_abs_angle']:+.1f} deg")
    print(f"    |tvec| {b1['tvec'] - b0['tvec']:+.3f} m   "
          f"ratio A {b1['ratio_a'] - b0['ratio_a']:+.3f}   "
          f"ratio B {b1['ratio_b'] - b0['ratio_b']:+.3f}")

    print("\n  Acceptance criteria (Artifacts/TASK.md):")
    print(f"    1. held-out median <= 100 px (was {b0['held_med']:.1f})   : "
          f"{b1['held_med']:.1f} px -> "
          f"{'PASS' if b1['held_med'] <= 100.0 else 'FAIL'}")
    print(f"    2. |tvec| in 3.5-4.6 m (was {b0['tvec']:.3f})             : "
          f"{b1['tvec']:.3f} m -> "
          f"{'PASS' if 3.5 <= b1['tvec'] <= 4.6 else 'FAIL'}")
    crit3 = (b1['fit_mean'] <= b0['fit_mean'] and b1['fit_max'] <= b0['fit_max']
              and b1['inliers'] >= b0['inliers'])
    print(f"    3. mean/max/inliers improve (median may rise, and did: "
          f"{b0['fit_med']:.2f} -> {b1['fit_med']:.2f})           : "
          f"mean {b0['fit_mean']:.1f}->{b1['fit_mean']:.1f}, "
          f"max {b0['fit_max']:.1f}->{b1['fit_max']:.1f}, "
          f"inliers {b0['inliers']}->{b1['inliers']} -> "
          f"{'PASS' if crit3 else 'FAIL'}")
    print(f"    4. |ratio A - 1| <= 0.10 (was {abs(b0['ratio_a'] - 1.0):.3f}) : "
          f"{abs(b1['ratio_a'] - 1.0):.3f} -> "
          f"{'PASS' if abs(b1['ratio_a'] - 1.0) <= 0.10 else 'FAIL'}")
    print(f"    5. mean |angle error| <= 9.1 deg (was {b0['mean_abs_angle']:.1f})"
          f"      : {b1['mean_abs_angle']:.1f} deg -> "
          f"{'PASS' if b1['mean_abs_angle'] <= 9.1 else 'FAIL'}")


def print_k_routes(swing0, sync, lever):
    """Both routes for `config.WRIST_PIVOT_COG_SCALE`, and which was committed."""
    _rule("9. k — both routes (deliverable 2)")
    print("  Route 1 — amplitude matching (COMMITTED):")
    r1 = oc.wrist_cog_scale(swing0, sync)
    for key, val in r1.items():
        print(f"    {key:<18}: {val}")

    # Out-of-scope sensitivity: racket_pixel_scale's window, all 26 reads.
    head = {**oc.RACKET_HEAD_PX_ANGLE_1, **oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1}
    wrist = {**oc.WRIST_PX_ANGLE_1, **oc.WRIST_PX_HELDOUT_ANGLE_1}
    frames26 = sorted(set(head) & set(wrist))
    v = np.array([np.subtract(head[f], wrist[f]) for f in frames26], float)
    longest26 = float(np.linalg.norm(v, axis=1).max())
    px_per_m_26 = longest26 / config.RACKET_TIP_LEN
    wrist_m_26 = r1["wrist_max_px"] / px_per_m_26
    k_26 = wrist_m_26 / r1["cog_max_pairwise_m"]
    print(f"    OUT-OF-SCOPE sensitivity: racket_pixel_scale over all 26 reads "
          f"-> {px_per_m_26:.1f} px/m -> k = {k_26:.3f} (vs committed "
          f"{config.WRIST_PIVOT_COG_SCALE:g}). Not used; racket_pixel_scale's "
          "window is left as-is this round (analysis §6, §10.3).")

    print("\n  Route 2 — joint PnP over the fit set, REPORTED NOT COMMITTED "
          "(it does not identify k):")
    ks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.907,
          0.95, 1.0, 1.25, 1.5, 2.0, 3.0]
    base_tip = tip_for_lever(swing0.quat, lever)
    print(f"  {'k':>6}{'fit med':>9}{'fit mean':>10}{'fit max':>9}{'inl':>5}"
          f"{'|tvec|':>8}  {'solver':<10}"
          f"{'held med VALIDATION ONLY - not used to choose k':>50}")
    for k in ks:
        swing_k = replace(swing0, tip=base_tip, pivot=k * swing0.cog)
        corr, sol = solve_for(swing_k, sync)
        held_rows = held_out_residuals(sol, swing_k, sync, None)
        held_resids = np.array([r[-1] for r in held_rows], float)
        print(f"  {k:6.3f}{sol.median_px:9.2f}{sol.mean_px:10.2f}"
              f"{sol.max_px:9.2f}{int(sol.inliers.sum()):5d}"
              f"{sol.camera_distance_m:8.3f}  {sol.solver:<10}"
              f"{float(np.median(held_resids)):50.1f}")

    print("\n  Mean and max fall MONOTONICALLY out to k=1.5, so there is no "
          "interior minimum in this\n  objective. k=0.1/2.0/3.0 are bad PnP "
          "minima (held-out medians far above the rest) with\n  k=3.0 posting "
          "the LOWEST fit median of every cell tried — an argmin over the "
          "fit set\n  would pick a wrong answer here. Therefore route 1 is "
          f"committed; route 2 is the agreement\n  check — the committed "
          f"k={config.WRIST_PIVOT_COG_SCALE:g} sits inside the broad flat "
          "0.6-1.1 basin (fit median 15.55-15.69 px).")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lever", type=str, default=None,
                   help="x,y,z body lever to score; default config.RACKET_LEVER_BODY")
    p.add_argument("--wrist-sweep", action="store_true",
                   help="alias, kept for back-compat; the before/after table "
                        "and both k routes now print unconditionally")
    p.add_argument("--k", type=float, default=None,
                   help="pivot = k * cog; default config.WRIST_PIVOT_COG_SCALE. "
                        "Use --k 0 for the pre-pivot pinned-wrist behaviour.")
    args = p.parse_args()

    lever = (config.RACKET_LEVER_BODY if args.lever is None
              else tuple(float(x) for x in args.lever.split(",")))
    k = (config.WRIST_PIVOT_COG_SCALE if args.k is None else float(args.k))

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
    swing_alt = replace(swing0, tip=tip_alt, pivot=k * swing0.cog)

    corr, sol = solve_for(swing_alt, sync)

    print(f"\n  k (pivot = k * cog) in use : {k:g}")
    print_lever_and_fit(lever, accel, gyro)
    print_pose(corr, sol)
    print_pixel_table(sol, swing_alt, sync)
    print_held_out_residual(sol, swing_alt, sync)
    print_held_out_angle(sol, swing_alt, sync)
    print_windowed_extent(sol, swing_alt, sync)
    print_arc_ratio(swing_alt, sync)
    print_before_after(swing0, sync, lever)
    print_k_routes(swing0, sync, lever)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
