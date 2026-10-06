"""Procedure comparison for the pivot's restored linear drift term.

Reports three candidate procedures for `b` in
``pivot(t) = k*cog(t) + b*(t - WRIST_PIVOT_DRIFT_PIVOT_SAMPLE)``:

  * **C** — accel-derived, COMMITTED (`config.WRIST_PIVOT_DRIFT_GAIN`).
    ``b = k * removed_slope``, zero free parameters, zero annotations used.
  * **A** — two-step: camera pose held fixed at the committed (b=0) solution,
    `b` fitted by least-squares against the 14 pre-116 reads only.
  * **B** — joint `(rvec, tvec, b)` least-squares, fitting the committed
    13+contact set *plus* the 14 pre-116 reads together. Run twice, from two
    different seeds, to show the fit is seed-dependent (non-convex).

This script is reporting-only: it never calls `overlay_calib.solved_pose`
(its `lru_cache` would hand back a stale pose after any `swing.npz` write —
and this script does not write one anyway), never calls `swing.save()`, and
never scores any candidate's fit against the f146-220 held-out set while
*fitting* it — held-out is read only to print, exactly like `pre-116` is
pure validation under procedure C.

Run::

    uv run python scripts/fit_pivot_drift.py
    uv run python scripts/fit_pivot_drift.py --procedures C
    uv run python scripts/fit_pivot_drift.py --no-sweep

Writes nothing. Always exits 0 — a reporting script, not a gate.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from src import camera, config, fusion, load_csv, overlay_calib as oc  # noqa: E402
from src import to_signal_arrays  # noqa: E402

WIDTH, HEIGHT = 1280, 720
PIVOT_SAMPLE = config.WRIST_PIVOT_DRIFT_PIVOT_SAMPLE   # 200
K = config.WRIST_PIVOT_COG_SCALE

# Gates and recorded baselines, restated from `scripts/check_overlay_match.py`
# (the source of truth for these numbers) — never redefine them to anything
# else, and never loosen them to make a candidate pass.
GATE_MED = oc.RECORDED_BASELINE_PX + 10.0          # 23.7
GATE_MEAN = oc.RECORDED_BASELINE_MEAN_PX + 6.0      # 22.2
GATE_MAX = oc.RECORDED_BASELINE_MAX_PX + 15.0       # 54.0
GATE_INL = oc.RECORDED_BASELINE_INLIERS - 1         # 12
HELD_BAR = (78.93, 74.57, 156.56)
PRE_BASE = (50.50, 59.92, 145.40)


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def wrap180(deg: float) -> float:
    return float(((deg + 180.0) % 360.0) - 180.0)


def _load_sync():
    from src.synced_data import SyncedData
    from src.sync import paths as sync_paths
    for label, video, csv_out in sync_paths.ANGLES:
        if label == oc.LABEL:
            return SyncedData.for_video(label, video, csv_out)
    raise LookupError(f"no sync table configured for {oc.LABEL!r}")


def swing_with_b(swing0, b: np.ndarray):
    """``swing0`` with ``pivot = k*cog + b*(sample - 200)``. Never mutates swing0."""
    s = (np.arange(len(swing0.cog), dtype=float) - PIVOT_SAMPLE)[:, None]
    return replace(swing0, pivot=K * swing0.cog + s * np.asarray(b, float))


def _residual_stats(sw, sol, sync, frame_dict):
    cam = sol.camera(WIDTH, HEIGHT)
    resids = []
    for f in sorted(frame_dict):
        idx = sync.imu_idx_for_frame_centered(int(f))
        proj = camera.project_points(cam, sw.head[idx]).ravel()
        obs = np.asarray(frame_dict[f], float)
        resids.append(float(np.linalg.norm(proj - obs)))
    r = np.array(resids, float)
    return float(np.median(r)), float(np.mean(r)), float(np.max(r))


def _image_angle_stat(sw, sol, sync, head_dict, wrist_dict):
    """mean |signed image-angle error| over ``head_dict``'s frames, degrees.

    Model vector tail is the PROJECTED PIVOT at that sample (not a fixed
    world origin) — see `report_followthrough.py:93-117`, same formula.
    """
    cam = sol.camera(WIDTH, HEIGHT)
    errs = []
    for f in sorted(head_dict):
        idx = sync.imu_idx_for_frame_centered(int(f))
        pivot_px = camera.project_points(cam, sw.pivot[idx]).ravel()
        head_px = camera.project_points(cam, sw.head[idx]).ravel()
        m = head_px - pivot_px
        o = (np.asarray(head_dict[f], float)
             - np.asarray(wrist_dict[f], float))
        errs.append(wrap180(np.degrees(np.arctan2(m[1], m[0])
                                        - np.arctan2(o[1], o[0]))))
    return float(np.mean(np.abs(np.array(errs, float))))


def score(swing0, sync, b, *, use_ransac: bool = True) -> dict:
    """Re-solve the pose on the COMMITTED 13+contact set, then score 3 spans.

    The pose MUST be re-solved: that is what the deployed pipeline does
    (`solved_pose` re-solves from `build_correspondences`, whose obj_pts are
    `swing.head`, which carries the drift). Scoring a candidate against the
    b=0 pose instead changes the fit-window numbers by up to 8 px.
    """
    sw = swing_with_b(swing0, b)
    corr = oc.build_correspondences(sw, sync)       # include_pre116=False
    sol = camera.solve_pose(WIDTH, HEIGHT, corr.img_pts, corr.obj_pts,
                            oc.CALIB_FOV_DEG, use_ransac=use_ransac)
    fit = (sol.median_px, sol.mean_px, sol.max_px)
    pre = _residual_stats(sw, sol, sync, oc.RACKET_HEAD_PX_PRE116_ANGLE_1)
    ho = _residual_stats(sw, sol, sync, oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1)
    fit_angle = _image_angle_stat(sw, sol, sync, oc.RACKET_HEAD_PX_ANGLE_1,
                                  oc.WRIST_PX_ANGLE_1)
    pre_angle = _image_angle_stat(sw, sol, sync, oc.RACKET_HEAD_PX_PRE116_ANGLE_1,
                                  oc.WRIST_PX_PRE116_ANGLE_1)
    ho_angle = _image_angle_stat(sw, sol, sync, oc.RACKET_HEAD_PX_HELDOUT_ANGLE_1,
                                 oc.WRIST_PX_HELDOUT_ANGLE_1)
    return dict(b=np.asarray(b, float), sol=sol, fit=fit, pre=pre, ho=ho,
                fit_angle=fit_angle, pre_angle=pre_angle, ho_angle=ho_angle,
                inliers=int(sol.inliers.sum()), n=len(sol.inliers),
                tvec=sol.camera_distance_m, solver=sol.solver)


def _fmt_span(label: str, stats: tuple[float, float, float]) -> str:
    return f"{label:<10} med/mean/max = {stats[0]:6.2f} / {stats[1]:6.2f} / {stats[2]:7.2f} px"


def print_score_row(tag: str, row: dict) -> None:
    print(f"  [{tag}] |b|*400={float(np.linalg.norm(row['b'] * 400)):.4f} m  "
          f"solver={row['solver']:<10} inliers={row['inliers']}/{row['n']}  "
          f"|tvec|={row['tvec']:.3f} m")
    print(f"         {_fmt_span('fit', row['fit'])}")
    print(f"         {_fmt_span('pre-116', row['pre'])}")
    print(f"         {_fmt_span('held-out', row['ho'])}")
    print(f"         image-angle mean|err| : fit={row['fit_angle']:.2f} deg  "
          f"pre-116={row['pre_angle']:.2f} deg  held-out={row['ho_angle']:.2f} deg")


def gate_verdict(row: dict) -> str:
    ok = (row['fit'][0] <= GATE_MED and row['fit'][1] <= GATE_MEAN
          and row['fit'][2] <= GATE_MAX and row['inliers'] >= GATE_INL)
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------- #
# Procedure A — two-step, pose fixed, b fitted on the 14 pre-116 reads only.
# --------------------------------------------------------------------------- #
def procedure_A(swing0, sync, corr0, sol0):
    cam0 = sol0.camera(WIDTH, HEIGHT)
    frames = sorted(oc.RACKET_HEAD_PX_PRE116_ANGLE_1)
    idxs = np.array([sync.imu_idx_for_frame_centered(int(f)) for f in frames])
    obs = np.array([oc.RACKET_HEAD_PX_PRE116_ANGLE_1[f] for f in frames], float)
    base_pivot = K * swing0.cog

    def resid_A(bv):
        pts = (base_pivot[idxs] + (idxs - PIVOT_SAMPLE)[:, None] * bv[None, :]
               + swing0.tip[idxs])
        return (camera.project_points(cam0, pts) - obs).ravel()

    res = least_squares(resid_A, np.zeros(3), method="lm", xtol=1e-14, ftol=1e-14)
    return res.x


# --------------------------------------------------------------------------- #
# Procedure B — joint (rvec, tvec, b), two seeds.
# --------------------------------------------------------------------------- #
def procedure_B(swing0, sol0, corr_joint, seed_b: np.ndarray):
    base_pivot = K * swing0.cog
    jidx = corr_joint.imu_idx
    img = corr_joint.img_pts
    Kmat = sol0.K

    def resid(p):
        rvec, tvec, b = p[:3], p[3:6], p[6:9]
        pts = (base_pivot[jidx] + (jidx - PIVOT_SAMPLE)[:, None] * b[None, :]
               + swing0.tip[jidx])
        proj, _ = cv2.projectPoints(pts.astype(np.float32), rvec, tvec, Kmat, None)
        return (proj.reshape(-1, 2) - img).ravel()

    p0 = np.concatenate([sol0.rvec, sol0.tvec, np.asarray(seed_b, float)])
    res = least_squares(resid, p0, method="lm", xtol=1e-14, ftol=1e-14)
    rvec, tvec, b = res.x[:3], res.x[3:6], res.x[6:9]
    return b, rvec, tvec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procedures", default="A,B,C",
                    help="comma-separated subset of A,B,C to run")
    ap.add_argument("--sweep", dest="sweep", action="store_true", default=True,
                    help="run the gain/centring sweeps (default on)")
    ap.add_argument("--no-sweep", dest="sweep", action="store_false",
                    help="skip the gain/centring sweeps")
    args = ap.parse_args()
    procs = {p.strip().upper() for p in args.procedures.split(",") if p.strip()}

    df = load_csv()
    accel, gyro = to_signal_arrays(df)
    sync = _load_sync()

    # `swing_full` already carries the COMMITTED drift (gain=1.0) — fusion's
    # default behaviour. `swing0` is the b=0 baseline (PR #10's pivot = k*cog)
    # that every procedure here perturbs with its own candidate b.
    swing_full = fusion.reconstruct(accel, gyro)
    swing0 = replace(swing_full, pivot=K * swing_full.cog)

    # ------------------------------------------------------------------- #
    # Section 1 — accelerometer side (criterion 4)
    # ------------------------------------------------------------------- #
    _rule("1. Accelerometer side (acceptance criterion 4)")
    removed_slope = fusion.wrist_drift_slope(accel, gyro)
    slope400 = removed_slope * 400.0
    k_slope400 = K * slope400
    print(f"  removed ramp * 400 (raw detrend slope)   : "
          f"{np.round(slope400, 4).tolist()} m  |.|={np.linalg.norm(slope400):.4f} m")
    print(f"  * k = the pivot's lost ramp (gain=1)      : "
          f"{np.round(k_slope400, 4).tolist()} m  |.|={np.linalg.norm(k_slope400):.4f} m")
    cog_net = float(np.linalg.norm(swing0.cog[-1] - swing0.cog[0]))
    cog_max_pair = float(np.max(np.linalg.norm(
        swing0.cog[:, None, :] - swing0.cog[None, :, :], axis=-1)))
    print(f"  surviving detrended cog: net |cog[-1]-cog[0]| = {cog_net:.3f} m, "
          f"max pairwise = {cog_max_pair:.3f} m")
    expect_slope400 = np.array([-1.2967, 0.1899, -0.5044])
    expect_k_slope400 = np.array([-1.1761, 0.1722, -0.4574])
    agree = (np.allclose(slope400, expect_slope400, atol=2e-3)
             and np.allclose(k_slope400, expect_k_slope400, atol=2e-3))
    print(f"  agreement with Step 1's independent re-derivation : "
          f"{'YES' if agree else 'NO -- STOP AND REPORT'}")

    b_C = K * removed_slope   # procedure C's answer, m/sample

    # Pose on the committed 13+contact set (b=0), used as the "pose held
    # fixed" baseline for procedure A and as the seed pose for procedure B.
    corr0 = oc.build_correspondences(swing0, sync)
    sol0 = camera.solve_pose(WIDTH, HEIGHT, corr0.img_pts, corr0.obj_pts,
                             oc.CALIB_FOV_DEG)

    results: dict[str, dict] = {}

    # ------------------------------------------------------------------- #
    # Section 2 — Procedure C, COMMITTED
    # ------------------------------------------------------------------- #
    if "C" in procs:
        _rule("2. Procedure C -- accel-derived, COMMITTED (zero free parameters)")
        row_C = score(swing0, sync, b_C, use_ransac=True)
        print_score_row("C", row_C)
        row_C_norans = score(swing0, sync, b_C, use_ransac=False)
        print("\n  Same, without RANSAC (answer should not hinge on the robust solver):")
        print_score_row("C no-ransac", row_C_norans)
        results["C"] = row_C

    # ------------------------------------------------------------------- #
    # Section 3 — Procedure A
    # ------------------------------------------------------------------- #
    b_A = None
    if "A" in procs:
        _rule("3. Procedure A -- two-step, pose fixed, b fit on 14 pre-116 reads only")
        b_A = procedure_A(swing0, sync, corr0, sol0)
        print(f"  b_A * 400 = {np.round(b_A * 400, 5).tolist()} m  "
              f"|.|={np.linalg.norm(b_A * 400):.4f} m")
        row_A = score(swing0, sync, b_A, use_ransac=True)
        print_score_row("A", row_A)
        results["A"] = row_A

    # ------------------------------------------------------------------- #
    # Section 4 — Procedure B, two seeds (LESS STABLE)
    # ------------------------------------------------------------------- #
    if "B" in procs:
        _rule("4. Procedure B -- joint (rvec,tvec,b), 13+contact+14 pre-116 "
              "-- LESS STABLE, reported for comparison, NOT committed")
        corr_joint = oc.build_correspondences(swing0, sync, include_pre116=True)
        b_B0, rvec_B0, tvec_B0 = procedure_B(swing0, sol0, corr_joint, np.zeros(3))
        print(f"  seed=0        b_B*400 = {np.round(b_B0 * 400, 4).tolist()} m  "
              f"|.|={np.linalg.norm(b_B0 * 400):.4f} m")
        row_B0 = score(swing0, sync, b_B0, use_ransac=True)
        print_score_row("B seed=0", row_B0)

        b_BC, rvec_BC, tvec_BC = procedure_B(swing0, sol0, corr_joint, b_C)
        print(f"\n  seed=b_C      b_B*400 = {np.round(b_BC * 400, 4).tolist()} m  "
              f"|.|={np.linalg.norm(b_BC * 400):.4f} m")
        row_BC = score(swing0, sync, b_BC, use_ransac=True)
        print_score_row("B seed=b_C", row_BC)

        print(f"\n  CONCLUSION: the two seeds disagree -- seed=0 gives |b|*400="
              f"{np.linalg.norm(b_B0 * 400):.3f} m, seed=b_C gives "
              f"{np.linalg.norm(b_BC * 400):.3f} m. seed=b_C posts the better "
              f"pre-116 median ({row_BC['pre'][0]:.2f} vs {row_B0['pre'][0]:.2f} px) "
              f"but its held-out median/mean REGRESS "
              f"({row_BC['ho'][0]:.2f}/{row_BC['ho'][1]:.2f} vs "
              f"{row_B0['ho'][0]:.2f}/{row_B0['ho'][1]:.2f} px, committed b=0 "
              f"baseline is {HELD_BAR[0]:.2f}/{HELD_BAR[1]:.2f} px). "
              "This is the procedural-sensitivity finding: do not commit B.")
        results["B_seed0"] = row_B0
        results["B_seedC"] = row_BC

        # ------------------------------------------------------------- #
        # Section 5 — identifiability: B's optimizer on the fit window alone
        # ------------------------------------------------------------- #
        _rule("5. Identifiability -- procedure B's optimizer on the 13+contact "
              "fit window ALONE (no pre-116)")
        b_fitonly, _rv, _tv = procedure_B(swing0, sol0, corr0, np.zeros(3))
        print(f"  b (fit-window-only) * 400 = {np.round(b_fitonly * 400, 5).tolist()} m  "
              f"|.|={np.linalg.norm(b_fitonly * 400):.5f} m")
        print("  Expected: close to zero (well under 0.05 m) -- the fit window "
              "alone cannot see `b`.\n  This is why the drift must come from "
              "the accelerometer (or reads outside f116-145): the fit-window\n"
              "  annotations that already constrain the pose carry essentially "
              "no information about it.")

    # ------------------------------------------------------------------- #
    # Section 6 — gain sweep (degeneracy check)
    # ------------------------------------------------------------------- #
    if args.sweep:
        _rule("6. Gain sweep over b_C (reported, NOT minimised)")
        print(f"  {'gain':>6}{'|b|*400':>10}{'solver':>12}{'inl':>6}{'|tvec|':>9}"
              f"{'fit med':>9}{'fit mean':>10}{'fit max':>9}"
              f"{'pre med':>9}{'pre mean':>10}{'pre max':>9}"
              f"{'ho med':>9}{'ho mean':>10}{'ho max':>9}")
        for gain in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            row = score(swing0, sync, gain * b_C, use_ransac=True)
            print(f"  {gain:6.2f}{np.linalg.norm(gain * b_C * 400):10.3f}"
                  f"{row['solver']:>12}{row['inliers']:6d}{row['tvec']:9.3f}"
                  f"{row['fit'][0]:9.2f}{row['fit'][1]:10.2f}{row['fit'][2]:9.2f}"
                  f"{row['pre'][0]:9.2f}{row['pre'][1]:10.2f}{row['pre'][2]:9.2f}"
                  f"{row['ho'][0]:9.2f}{row['ho'][1]:10.2f}{row['ho'][2]:9.2f}")
        print("\n  Reading: `b` is NOT a gauge freedom (unlike k) -- the residuals "
              "move with gain. But the\n  objective is non-convex and its argmin "
              "is not the answer: gain 0.75 posts the best held-out\n  median and "
              "gain 1.50 the best pre-116 median while REGRESSING held-out; "
              "1.25/2.00 collapse\n  into bad PnP minima (|tvec| ~2.6 m). gain=1.0 "
              "is committed because it is the DERIVED value.")

        # --------------------------------------------------------------- #
        # Section 7 — centring sensitivity
        # --------------------------------------------------------------- #
        _rule("7. Centring sensitivity (b = b_C fixed, sweep the zero-sample)")
        print(f"  {'centre':>8}{'fit med':>9}{'fit mean':>10}{'fit max':>9}"
              f"{'inl':>6}{'|tvec|':>9}  {'solver':<12}")
        global PIVOT_SAMPLE  # noqa: PLW0603 -- intentional, restored right after
        orig_pivot_sample = PIVOT_SAMPLE
        for centre in (0, 200, 399):
            PIVOT_SAMPLE = centre
            row = score(swing0, sync, b_C, use_ransac=True)
            print(f"  {centre:8d}{row['fit'][0]:9.2f}{row['fit'][1]:10.2f}"
                  f"{row['fit'][2]:9.2f}{row['inliers']:6d}{row['tvec']:9.3f}  "
                  f"{row['solver']:<12}")
        PIVOT_SAMPLE = orig_pivot_sample
        print("\n  Reading: a constant 3D offset is a gauge freedom in exact "
              "arithmetic (PnP's tvec absorbs\n  it), but the multi-solver PnP "
              "path is non-convex and lands in a different minimum at centre 0\n"
              "  (|tvec| ~2.0 m); centre 399 pushes |tvec| outside the 3.5-4.6 m "
              "player-height anchor band.\n  Centre 200 (the record's midpoint) "
              "is committed as a documented convention, not a free parameter.")

    # ------------------------------------------------------------------- #
    # Section 8 — cross-check and the decision
    # ------------------------------------------------------------------- #
    _rule("8. Cross-check (criterion 4) and the decision")
    if b_A is not None:
        cos_ang = float(np.dot(b_C, b_A) / (np.linalg.norm(b_C) * np.linalg.norm(b_A)))
        angle = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))
        ratio = float(np.linalg.norm(b_A) / np.linalg.norm(b_C))
        print(f"  b_C * 400 (accel, COMMITTED) : {np.round(b_C * 400, 4).tolist()} m  "
              f"|.|={np.linalg.norm(b_C * 400):.4f} m")
        print(f"  b_A * 400 (image, procedure A) : {np.round(b_A * 400, 5).tolist()} m  "
              f"|.|={np.linalg.norm(b_A * 400):.4f} m")
        print(f"  angle between them : {angle:.1f} deg   magnitude ratio "
              f"(|b_A|/|b_C|) : {ratio:.3f}")

    print(f"\n  Gates (from scripts/check_overlay_match.py): median<={GATE_MED:.1f} "
          f"mean<={GATE_MEAN:.1f} max<={GATE_MAX:.1f} inliers>={GATE_INL}")
    print(f"  Recorded baselines: pre-116 (committed b=0) = "
          f"{PRE_BASE[0]:.2f}/{PRE_BASE[1]:.2f}/{PRE_BASE[2]:.2f} px   "
          f"held-out (committed b=0) = "
          f"{HELD_BAR[0]:.2f}/{HELD_BAR[1]:.2f}/{HELD_BAR[2]:.2f} px")

    print(f"\n  {'procedure':<16}{'fit (gate)':<10}{'fit med/mean/max':<24}"
          f"{'pre-116 med/mean/max':<24}{'held-out med/mean/max':<24}")
    for tag, row in results.items():
        print(f"  {tag:<16}{gate_verdict(row):<10}"
              f"{row['fit'][0]:.2f}/{row['fit'][1]:.2f}/{row['fit'][2]:.2f}".ljust(24) +
              f"{row['pre'][0]:.2f}/{row['pre'][1]:.2f}/{row['pre'][2]:.2f}".ljust(24) +
              f"{row['ho'][0]:.2f}/{row['ho'][1]:.2f}/{row['ho'][2]:.2f}")

    print("\n  DECISION: procedure C is committed (config.WRIST_PIVOT_DRIFT_GAIN"
          " = 1.0). It has zero\n  free parameters and is fitted to no "
          "annotation, so the 14 pre-116 reads and the 13\n  held-out reads "
          "both stay pure validation. Procedures A and B are reported above "
          "for\n  comparison only -- per CLAUDE.md and this script's own "
          "docstring, neither is committed\n  even where one posts a "
          "better-looking pre-116 number.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
