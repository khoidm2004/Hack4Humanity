# Overlay coordinate frames

Two coordinate conventions meet at the video-overlay projection in
`app.py:build_overlay_camera`, and nothing converted between them until this
fix — that mismatch was the cause of the "upside down" racket-path bug.

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
