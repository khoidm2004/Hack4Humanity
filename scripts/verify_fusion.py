"""Person A — sanity-check + contract generation (PLAN.md §7 & §10 DoD).

Run::

    python scripts/verify_fusion.py

Produces ``data/outputs/swing.npz`` and prints a short metrics report plus a
set of consistency checks.  Exits non-zero on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the `src` package importable when running from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from src import compute_metrics, config, load_csv, reconstruct, to_signal_arrays  # noqa: E402
from src import fusion  # noqa: E402


def run_checks(accel, gyro, swing) -> list[str]:
    """Return a list of human-readable PASS/FAIL strings."""
    checks = []
    fail = False

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fail
        if not ok:
            fail = True
        checks.append(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    # ---- Signal integrity ------------------------------------------------ #
    check("no NaN/Inf in accel", not (np.isnan(accel).any() or np.isinf(accel).any()))
    check("no NaN/Inf in gyro", not (np.isnan(gyro).any() or np.isinf(gyro).any()))
    # The old bound was "0 < |a| < 40 m/s2".  It passed only because the CSV is
    # in g while `config.ACCEL_UNITS` claimed m/s2 and nothing converted: the
    # numbers it saw peaked at 20.9 when the real peak is 205 m/s2.  Now that
    # `load.to_signal_arrays` converts, 40 m/s2 would fail — and it SHOULD,
    # because 40 m/s2 (4 g) was never a defensible bound for a racket strike.
    # The honest bound is the sensor's own range: no axis may exceed its
    # +/-16 g full scale, and |a| cannot exceed that on all three axes at once.
    # Measured: per-axis max 15.96 / 14.96 / 12.38 g, |a| max 205.0 m/s2
    # (20.9 g), with `ax` hard-clipped for samples 181-199 — a 45.7 ms plateau
    # that is centripetal, not an impact spike (Artifacts/analysis.md §2.2d).
    fs_ms2 = config.ACCEL_FULL_SCALE_G * config.G
    axis_max = float(np.abs(accel).max())
    mag_max = float(np.linalg.norm(accel, axis=1).max())
    check(
        f"no accel axis beyond the +/-{config.ACCEL_FULL_SCALE_G:g} g full scale",
        axis_max <= fs_ms2 * 1.01,
        f"max |a_i|={axis_max / config.G:.2f} g",
    )
    check(
        "accel magnitude inside the sensor envelope "
        f"(0 < |a| <= sqrt(3)*{config.ACCEL_FULL_SCALE_G:g} g)",
        0.0 < mag_max <= np.sqrt(3) * fs_ms2 * 1.01,
        f"max={mag_max:.1f} m/s2 ({mag_max / config.G:.1f} g)",
    )

    # ---- Orientation ----------------------------------------------------- #
    check("quaternions unit-norm (±1e-3)", bool(
        np.allclose(np.linalg.norm(swing.quat, axis=1), 1.0, atol=1e-3)))

    # ---- Tip path sanity ------------------------------------------------- #
    radii = np.linalg.norm(swing.tip, axis=1)
    check(
        "pivot tip radius ≈ racket length (constant)",
        bool(np.allclose(radii, config.RACKET_TIP_LEN, atol=1e-4)),
        f"r={radii.mean():.3f}",
    )

    # ---- Lever DIRECTION, the check that was missing --------------------- #
    # The radius check above is direction-BLIND: `[0, 0, L]` and
    # `[-L, 0, 0]` both pass it.  That is how a 90-deg-wrong lever survived
    # three overlay fixes (Artifacts/analysis.md §7).  Two checks now.
    #
    # (A) Consistency: read the lever back out of the output.  Honest caveat —
    #     this would NOT have caught the original bug; with the old hard-coded
    #     literal it would have been tautological.  What it buys is that the
    #     lever is a declared, inspectable constant that cannot drift away
    #     from what `pivot_tip` actually applies.
    lever_unit = np.asarray(config.RACKET_LEVER_BODY, float)
    lever_unit = lever_unit / np.linalg.norm(lever_unit)
    rots = Rotation.from_quat(swing.quat[:, [1, 2, 3, 0]])
    recovered = rots.inv().apply(swing.tip)
    expected = lever_unit * config.RACKET_TIP_LEN
    dev = float(np.abs(recovered - expected).max())
    check(
        "tip reads out along the configured body lever",
        dev < 1e-9,
        f"lever={config.RACKET_LEVER_BODY} max|dev|={dev:.2e} m",
    )

    # (B) The real one: the accelerometer's own centripetal signal is an
    #     independent measurement of the body-frame lever AXIS (no camera, no
    #     annotations, no fusion output).  Compared SIGN-FREE on purpose: the
    #     fit pins the axis, while the lever's SIGN is pinned separately by
    #     `check_overlay_match.py`'s corr(projected u/v, observed u/v) > 0
    #     gates, which a +x lever would fail by mirroring.  Do not "tighten"
    #     this into a signed test.
    #
    #     The 40 deg tolerance is not tuned: the fit's own window-to-window
    #     scatter is 10-20 deg, while the whole +/-y, +/-z family sits at
    #     >= 86 deg.  40 deg sits between them with room on both sides.
    u_fit, r_fit, _bias, n_fit = fusion.fit_centripetal_lever(accel, gyro, lo=205)

    def _axis_angle(u, v):
        c = abs(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
        return float(np.rad2deg(np.arccos(min(c, 1.0))))

    ang = _axis_angle(u_fit, lever_unit)
    check(
        "configured lever axis agrees with the accelerometer centripetal fit "
        "(<= 40 deg, sign-free)",
        ang <= 40.0,
        f"fit u={np.round(u_fit, 3).tolist()} |r|={r_fit:.3f} m "
        f"({n_fit} samples) -> {ang:.1f} deg",
    )
    # Permanent regression evidence that check (B) has teeth: the pre-fix
    # lever, scored against the same fit, every run.
    ang_pz = _axis_angle(u_fit, np.array([0.0, 0.0, 1.0]))
    checks.append(
        f"[INFO] the pre-fix +z lever scores {ang_pz:.1f} deg against this "
        f"same fit -> check (B) would have FAILED it "
        f"({'rejected' if ang_pz > 40.0 else 'NOT REJECTED — investigate'})"
    )

    # ---- COG path is finite & reasonably bounded ------------------------- #
    cog_span = np.linalg.norm(swing.cog.max(axis=0) - swing.cog.min(axis=0))
    check("rest-to-rest COG finite", bool(np.isfinite(swing.cog).all()))
    check(
        "rest-to-rest COG bounded (< 3 m sweep)",
        bool(cog_span < 3.0),
        f"sweep={cog_span:.2f} m",
    )

    # ---- Wrist / pivot translation (the contract field added 2026-10-05) -- #
    # `tip` is the head RELATIVE to the pivot, so the radius check above still
    # holds exactly and the head is `swing.head = pivot + tip`.  Folding
    # `pivot` into `tip` instead would have broken that radius check, both
    # lever-direction checks below, and `racket_pixel_scale`'s L-cancellation
    # argument at once (|tip + 0.907*cog| runs 0.686 -> ~1.5 m) — which is why
    # it is its own field.  See Artifacts/analysis.md §9.3.
    expect_pivot = config.WRIST_PIVOT_COG_SCALE * swing.cog
    pdev = float(np.abs(swing.pivot - expect_pivot).max())
    check("pivot finite", bool(np.isfinite(swing.pivot).all()))
    check("pivot shape matches tip",
          swing.pivot.shape == swing.tip.shape,
          f"{swing.pivot.shape} vs {swing.tip.shape}")
    check(
        "pivot is WRIST_PIVOT_COG_SCALE * cog",
        pdev < 1e-12,
        f"scale={config.WRIST_PIVOT_COG_SCALE:g} max|dev|={pdev:.2e} m",
    )
    pivot_span = float(np.linalg.norm(
        swing.pivot.max(axis=0) - swing.pivot.min(axis=0)))
    check("pivot span bounded (< 3 m)", bool(pivot_span < 3.0),
          f"span={pivot_span:.2f} m")
    head_r = np.linalg.norm(swing.head - swing.pivot, axis=1)
    check(
        "head - pivot is still the racket lever (radius preserved)",
        bool(np.allclose(head_r, config.RACKET_TIP_LEN, atol=1e-9)),
        f"r={head_r.mean():.6f}",
    )
    checks.append(
        f"[INFO] wrist translation now modelled: pivot span {pivot_span:.2f} m "
        f"(was 0.00 m, wrist pinned at the origin). The head is `pivot + tip`; "
        f"`|tip|` is unchanged at {config.RACKET_TIP_LEN} m by construction."
    )

    # ---- Impact ---------------------------------------------------------- #
    metrics = compute_metrics(swing)
    check(
        "impact_idx in range",
        bool(0 <= swing.impact_idx < len(swing.t)),
        f"idx={swing.impact_idx} t={swing.t[swing.impact_idx]:.3f}s",
    )
    check(
        "peak head speed plausible (2..60 m/s)",
        bool(2 <= metrics["peak_head_speed_mps"] <= 60),
        f"v={metrics['peak_head_speed_mps']} m/s ({metrics['peak_head_speed_kmh']} km/h)",
    )
    check("peak g-force > 0.5 g", bool(metrics["peak_gforce"] > 0.5),
          f"{metrics['peak_gforce']} g")
    return checks


def main() -> int:
    print("=" * 60)
    print("Person A — verify_fusion")
    print("=" * 60)

    df = load_csv()
    accel, gyro = to_signal_arrays(df)
    print(f"Loaded {len(df)} samples @ {config.FS} Hz "
          f"(t = {df['t'].iloc[-1]:.3f} s)")

    swing = reconstruct(accel, gyro)

    checks = run_checks(accel, gyro, swing)

    # ---- Generate the contract file ------------------------------------- #
    swing.save()
    print(f"\nWrote contract -> {config.SWING_NPZ}")

    # ---- Metrics report -------------------------------------------------- #
    print("\n--- Metrics ---")
    for k, v in compute_metrics(swing).items():
        print(f"  {k:24s} {v}")

    print("\n--- Consistency checks ---")
    for line in checks:
        print(f"  {line}")

    ok = not any("[FAIL]" in line for line in checks)
    print("\nRESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
