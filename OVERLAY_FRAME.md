> **SUPERSEDED (2026-10-04).** The "What the fit achieves, and what it cannot"
> table below and its arc-mismatch reasoning were read off a reconstruction
> built on two since-fixed defects: a scrambled initial quaternion (171.8°
> off, `src/fusion.py:103`) and — the one that actually produced the "2.01×
> arc, 427° of turning" numbers — `config.SLOWMO_SCALE = 1.0` on footage that
> is really ~240 fps slow motion played at 30 fps, which compared 0.96 s of
> IMU motion against 0.12 s of real video time. Neither is a camera-frame
> problem; both are corrected in `FUSION_NOTES.md`, which is now the current
> write-up for the arc/turning numbers. After the quaternion fix,
> `corr(world +Z, OBSERVED head v)` stays **positive** (+0.649 before, +0.725
> after — not a sign flip, since the orientation fix is a constant
> left-multiply that the annotated-frame window and `_order_correlations`
> below both re-absorb); `ord_v`/`ord_u`, the actual hard gates, hold at
> +0.956/+0.998. This file's coordinate-frame derivation (Z-up world, Y-down
> camera, the bridging rotation) is unaffected and still the reference for
> that part.

# Overlay coordinate frames

Two coordinate conventions meet at the video-overlay projection in
`app.py:build_overlay_camera`, and nothing converted between them until this
fix — that mismatch was the cause of the "upside down" racket-path bug.

> **Status.** Everything down to "Scope" is the original write-up of the
> orientation fix, kept verbatim. The sections after it record what solving
> the pose for real against the footage then showed — including that the
> empirical check below turned out to be circular. Read both.

## The two conventions

1. **Fusion world frame — Z-up.** `src/fusion.py:125` aligns measured
   gravity onto `[0, 0, -1]`, so world "down" is −Z and **world up is +Z**.
   `pivot_tip` (`src/fusion.py:146-165`) returns `swing.tip` in this frame.
2. **OpenCV camera frame — Y-down, Z-forward.** `cv2.projectPoints` (and
   therefore `src/camera.py`'s `Camera`/`project_points`) assumes +X right,
   **+Y down the image**, +Z along the viewing direction (depth).

`build_overlay_camera` previously passed `rvec = [0, 0, 0]` (identity) into
`camera.Camera.from_pnps`, feeding the Z-up world straight into the Y-down
camera with no basis change. World +Z (up) landed on the camera's Z axis —
pure depth, invisible in the 2D projection — while screen-vertical ended up
driven by world +Y instead, with the wrong sign. That is the "upside down /
vertically mirrored" path.

## The bridging rotation

The fix is a rotation about the camera's X axis so that:

- world **+Z (up)** → camera **−Y** (up on screen)
- world **+Y** → camera **+Z** (depth)
- world **+X** → camera **+X** (right, unchanged)

`rvec = [θ, 0, 0]` with `cv2.Rodrigues` as used by `Camera.from_pnps` worked
out empirically to need **θ = +π/2** (not −π/2, despite that being the sign
a naive "standard Z-up→Y-down conversion" derivation suggests).

### Empirical evidence (not just the a-priori derivation)

Projecting `data/outputs/swing.npz`'s `tip` through the camera `app.py`
builds (1280×720, FOV 60°, `tvec=[0,0,10]`), for both candidate signs:

```
rvec = [-pi/2, 0, 0]:
  corr(+Z up, screen v) = +0.999   <-- WRONG sign: +Z still drives v positively (down)
  corr(+Y   , screen v) = +0.013
  corr(+X   , screen u) = +0.999

rvec = [+pi/2, 0, 0]:
  corr(+Z up, screen v) = -0.999   <-- correct: world up moves screen v negative (up)
  corr(+Y   , screen v) = +0.044   <-- near zero, as expected
  corr(+X   , screen u) = +0.999   <-- unmirrored (no left/right flip introduced)
```

`+π/2` is the sign that satisfies the acceptance criterion
(`corr(world +Z, screen v) < 0` with large magnitude, world +Z dominant over
world +Y) and was confirmed visually in the running app (the yellow path now
sweeps low→high through the impact frame on `swing_angle_1.mp4`, with no
left/right mirroring introduced). `-π/2` was rejected because it reproduces
the original bug (screen v still driven positively by world +Z — equivalent
to looking at the swing from the opposite side along the camera's roll).

This rotation is applied as the **base** rotation in
`build_overlay_camera` (`app.py`), before the sidebar yaw/elevation/zoom/pan
`nudge()` adjustments are composed on top (`src/camera.py:nudge`, which does
`rx @ ry @ r` — a non-identity base `r` composes the same way as identity
did, so the sliders still work as incremental adjustments on top of the
corrected base).

## Scope

This is a projection-only fix: `src/fusion.py`'s world convention,
`data/outputs/swing.npz`, and the sync tables are unchanged. Only the base
`rvec` passed into `camera.Camera.from_pnps` inside
`app.py:build_overlay_camera` changed, from `[0, 0, 0]` to `[pi/2, 0, 0]`.

---

# Update (2026-10-04): the pose is now solved, not guessed

The convention above — a Z-up world cannot be fed to a Y-down camera without
a basis change — still holds and still matters. What changed is that
`app.py` no longer *writes that basis change out by hand*. For
`swing_angle_1.mp4` the full rotation **and** translation now come from
`cv2.solvePnP` on 14 real 2D↔3D correspondences
(`src/overlay_calib.py`), and the basis conversion is simply part of the
rotation that fit recovers. The guessed `tvec = [0, 0, 10]` is gone too.

* **Solved pose** (1280×720, FOV 60°, `SOLVEPNP_ITERATIVE`):
  `rvec ≈ [-123.8°, +4.4°, -2.9°]`, `tvec ≈ [0.174, 0.346, 2.937] m`,
  camera 2.96 m from the wrist.
* The legacy `[+π/2, 0, 0] / [0, 0, 10]` pose is kept **only** as the
  fallback for clips with no annotations (`swing_angle_2.mp4`), where
  `app.py` now says so in the UI rather than presenting a guess as a
  calibration.

## The `corr(+Z, v) < 0` check was circular — replaced

The empirical check above tested the projection against the *assumption*
that world +Z is up. It never consulted the footage, so it could only ever
confirm the assumption it started from.

The 13 hand-annotated racket-head pixels settle it without any camera model
at all. Correlating `tip[:, 2]` at those frames against the **observed**
racket-head `v`:

```
corr(world +Z, OBSERVED head v) = +0.649
```

**World +Z moves the racket head *down* the image.** Every solver
(`SQPNP`, `ITERATIVE`, `EPNP`, RANSAC+LM) at every FOV from 40° to 100°
independently recovers a rotation with `(R @ [0,0,1])_y > 0` — i.e. all of
them agree with the annotations and disagree with the assumption.

The reason is the sign of gravity. An accelerometer at rest measures
*specific force*, which points **up** at +1 g, not down. Aligning that
measured vector onto `[0, 0, -1]` therefore puts world **+Z along DOWN**,
not up. `src/fusion.py` is out of scope for this task and is unchanged; the
consequence is simply that "+Z is up" is the wrong label for that axis.

`scripts/check_overlay_match.py` §5 replaces the old check with one that
cannot be circular, and it is a hard failure gate:

```
corr(projected v, observed v) at the annotated frames  > 0
corr(projected u, observed u) at the annotated frames  > 0
```

i.e. *the projection must reproduce the vertical and horizontal ordering of
the racket head as annotated in the footage*. The solved pose gives +0.893
and +0.975. `src/camera.py:solve_pose` applies the same test as a validity
filter and **rejects** any pose that fails it, however low its median
residual — which it must, since the lowest-median RANSAC fit at FOV 60°
(38.3 px) is vertically mirrored (`corr = -0.105`) and would have shipped a
backwards path.

## What the fit achieves, and what it cannot

| | before | after |
|---|---|---|
| projected path extent | 136 × 93 px | 483 × 373 px |
| observed racket-head extent | — | 558 × 305 px |
| median reprojection error | not measurable (pose fabricated) | **55.8 px** |
| max reprojection error | — | 106.9 px |
| camera distance | 10 m (guessed) | 2.96 m (solved) |

**The 50 px target is not met, and no *physically plausible* rigid camera
pose can meet it.** The precise statement matters, because the FOV sweep in
§3 does dip under the target at its extreme end: at FOV 100° the all-point
fit reaches 49.3 px (and the RANSAC fit 26.4 px, but that median is over
inliers only — it discards the very points the criterion is about). Those
rows buy their residual by pulling the camera to **|tvec| = 1.53 m**, down
from 4.57 m at FOV 40°. A 1.5 m camera cannot produce this footage, which
frames the whole player plus a wide hall; the solver is absorbing the
arc-length mismatch below through extreme perspective distortion. That is
overfitting, not calibration, which is why the committed `CALIB_FOV_DEG` is
the physically sensible 60° (55.8 px, |tvec| = 2.96 m) and not the sweep
argmin. Taking the 100° row to declare the target met would be exactly the
number-tuning `CLAUDE.md` forbids.

Three independent measurements establish the real limit, all printed by
`scripts/check_overlay_match.py` §8:

1. **Arc-length mismatch (decisive).** The racket measures at most 195 px
   between the annotated wrist and head pixels, so at `RACKET_TIP_LEN =
   0.686 m` the image scale is **285 px/m** — no PnP involved. The
   reconstructed `tip` travels **5.11 m** of arc across the swing window,
   which is **1454 px** of image arc. The racket head visibly travels
   **722 px**. That is **2.01×**. The net turn of the racket is 99° in the
   reconstruction against 156° as imaged, and the reconstructed path
   contains 427° of total turning — reversals the footage does not show. A
   rigid pose is, to first order, a similarity on the image plane: it can
   translate, rotate and scale a path, but it cannot change how much arc the
   path contains. So the projected path is a double loop where the video
   shows a single sweep.
2. **Camera distance.** PnP puts the camera at 2.96 m; the player's pixel
   height gives 4.08 m and the racket's apparent length gives 3.89 m (those
   two agree, so the pixel scale is sound). PnP is 1.38× too close — pulling
   the camera in is its only freedom for absorbing (1), and the ratio is the
   same at every FOV.
3. **Wrist translation.** `src/fusion.py:pivot_tip` pins the wrist at the
   world origin (`|tip|` is constant at 0.686 m), but the annotated wrist
   pixel moves **220 × 150 px** (258 px max pairwise ≈ 0.91 m) across the
   swing. None of that motion is in `tip`, and no rigid pose can put it
   back.

Per `Artifacts/TASK.md` these are reported, not tuned away. In particular
the FOV was **not** moved to improve the number: the sweep's own argmin sits
at the 100° end of the scanned range (median 26.4 px, camera 1.44 m), which
is what an unidentifiable parameter looks like — across the whole 40–100°
range the all-point median moves by only ~9 px, less than the ±12–25 px
uncertainty on the annotations. `CALIB_FOV_DEG` keeps the 60° the app
already assumed, which **costs** 29 px of median.

Fixing (1) and (3) means giving the wrist a per-frame translation, or
revisiting the orientation integration in `src/fusion.py`. Both are out of
scope here — this task is projection-only — and are noted as future work
rather than attempted.

## Other changes in this pass

* **Animated dot, half-frame bias.** `SyncedData.imu_idx_for_frame` argmins
  over the *rounded* `frame_index`, so for a frame holding ~13.9 IMU samples
  it returns the **first** one — consistently half a frame (~7 samples)
  early. Near impact the tip moves ~0.45 m inside one frame bin, so the dot
  sat visibly behind the racket. `app.py` now uses the new
  `SyncedData.imu_idx_for_frame_centered`, which argmins over `frame_exact`.
  The old method is untouched.
* **The one-point pan anchor is off by default.** It used to slide the whole
  path until the projected impact point hit the detected ball pixel. With a
  pose fitted to 14 correspondences that translates every other frame by one
  point's residual. It survives as the sidebar checkbox *"anchor impact to
  ball pixel (legacy)"*, default off. Acceptance criterion 4 is now met by
  the fit instead, and the check script reports that residual as a number
  (101.4 px — the ball sits on the string bed ~0.09 m nearer the hand than
  `tip`, so this point carries a built-in bias). Switching the anchor off
  also means `get_impact_pixel` — a full video decode, the ~60 s first
  render — only runs when the box is ticked.
* **Slider sensitivity.** `nudge` rotates the camera about its own centre
  and leaves `tvec` alone. That centre is now 3 m from the swing instead of
  the fabricated 10 m, so yaw/elevation move the path roughly 3× further per
  degree than before. The ranges were deliberately *not* shrunk to hide
  this; the sidebar carries a `help=` note instead.
