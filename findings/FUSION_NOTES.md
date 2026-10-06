# Fusion & time-base notes

Why the projected racket path did not look like the swing, what was actually
wrong, and what is still wrong. Supersedes the arc-mismatch reasoning in
`OVERLAY_FRAME.md` and corrects `SYNC_METHOD.md`.

**Headline:** the path's "2× too much arc" was **not** a reconstruction defect.
`data/video/swing_angle_*.mp4` are **~240 fps slow-motion clips played back at
30 fps**, and `config.SLOWMO_SCALE` was set to `1.0`. The comparison was putting
0.96 s of IMU motion against 0.12 s of real video time. No change inside
`src/fusion.py` could have fixed it.

## 1. The capture frame rate

Reproduce with `uv run python scripts/measure_capture_fps.py`.

**(a) Container metadata — exhausted, settles nothing.** Both clips carry
`mdhd` timescale 600 with `stts` deltas of 20, i.e. a declared **30.000 fps
playback** rate, and **no capture-rate tag at all** — no `udta`, `meta` or
`ilst` atom, no `com.apple.quicktime` key. (`cv2`'s 29.8678 for angle_1 is the
single double-length frame 99, which lies outside the swing window 116–145.)
A container cannot answer this question; the picture itself has to.

**(b) The tennis ball's own diameter and free-fall.** The ball is a known
0.067 m sphere in free flight, so it supplies both the pixel scale and the
frame interval with no IMU and no PnP involved. A window is accepted only if
the parabola residual ≤ 0.5 px, the diameter std ≤ 5 %, and the diameter
≥ 12 px — a window failing any of these does not measure a frame rate:

| clip | window | n | D mean | D std | \|y″\| | resid | verdict |
|---|---|---|---|---|---|---|---|
| angle_1 | 100–130 | 31 | 24.04 px | 0.35 | 0.06212 | 0.18 px | **ACCEPT → 238 fps** |
| angle_1 | 132–146 | 15 | 26.08 px | 4.08 | 0.05289 | 0.45 px | REJECT: D std 16 % |
| angle_2 | 140–165 | 26 | 14.46 px | 0.47 | 0.04286 | 0.22 px | **ACCEPT → 222 fps** |
| angle_2 | 185–210 | 26 | 10.31 px | 1.10 | 0.10765 | 0.37 px | REJECT: D std 11 %, D < 12 px |
| angle_2 | 195–220 | 26 | 8.97 px | 1.37 | 0.06569 | 0.36 px | REJECT: D std 15 %, D < 12 px |

Frames 80–130 are also rejected (parabola residual 37.4 px) because the ball is
still being tossed before ~frame 100 — the rejection is the criterion working,
not a failure.

**(c) The reductio.** For the clip to be real-time 30 fps, the ball's own
parabola (`y″ = 0.06212 px/frame²` at 359 px/m) would require gravity to be
**0.156 m/s²**. The 30 fps hypothesis fails by a factor of 63.

**Verdict.** Two independent clean windows, in two different clips, give **238
and 222 fps** — both *lower* bounds, since the HSV mask erodes a bright small
ball (D = 22.2 / 24.0 / 26.0 px → 229 / 238 / 248 fps). Consumer slow-motion
capture modes are 120, 240, 480, 960 fps; the measured band excludes 120 and
480 by roughly 2× each. **240 fps is the only standard mode consistent with the
measurement**, so `CAPTURE_FPS = 240.0` and `SLOWMO_SCALE = 240/30 = 8.0`. The
repo's original `SLOWMO_SCALE = 8.33` (250 fps) was approximately right.

**What is *not* evidence:** the PnP reprojection residual as a function of
assumed fps is **flat** — 21.5 px at 210, 21.1 at 240, 23.9 at 300 — and its
apparent improvement past ~280 fps is the correspondence window collapsing to a
handful of samples, not a frame-rate measurement. The rate comes from the ball.

## 2. The time-base fix

`SLOWMO_SCALE`: `1.0` → `8.0`. The rescale is applied **at load time** in
`src/synced_data.py:_apply_slowmo`, about the ball-contact anchor:

```
frame_exact_new = anchor + (frame_exact_old - anchor) * SLOWMO_SCALE
```

Anchoring on contact (`config.SYNC_ANCHOR_FRAME["angle_1"] = 130.70344`) matters:
that single instant is the one real correspondence `scripts/run_sync.py` ever
fitted, so rescaling about it preserves the measured offset exactly and changes
only the *rate*. **The sync tables on disk are untouched** — this is why the fix
lives in `SyncedData` rather than regenerating the CSVs.

Effect on the window the overlay compares:

| | before | after |
|---|---|---|
| IMU samples mapped to frames 116–145 | 0–398 (399 of 400) | **175–225** (51 of 400) |
| real time that window spans | 0.96 s | **0.12 s** |

## 3. The windowed path draw

`src/camera.py:draw_overlay` projected **all 400 samples** as one static
polyline, so every frame showed the same ~8×-too-long path regardless of how
good the reconstruction or pose was. It now draws a trail around the displayed
sample (`config.PATH_TRAIL_FRAMES = 12`, `PATH_LEAD_FRAMES = 4`, converted at
`FS / CAPTURE_FPS` = 1.733 samples per frame → 21 behind, 7 ahead), faded
old→new so the stroke reads as direction of travel.

| projected extent | px |
|---|---|
| all 400 samples (what was drawn before) | **880 × 1013** |
| the windowed trail (what is drawn now) | **552 × 301** |
| the racket head's observed extent | **558 × 305** |

The windowed figure matches the observation to 1–2 % on a quantity PnP does not
directly optimise. This is the change the complaint "it still doesn't illustrate
the swing motion" actually hinged on.

## 4. The two real `src/fusion.py` defects

Both are genuine bugs. **Neither moves the arc** — they are fixed on their own
merits, and that they changed nothing about the arc is itself the evidence that
the arc was never a fusion problem.

**Quaternion ordering (`src/fusion.py:103`).** `_align_to_gravity` returns the
contract's `(w,x,y,z)` order; `Rotation.from_quat` reads `(x,y,z,w)`. The
initial orientation was built from a scrambled quaternion — the intended `R0`
maps measured gravity onto `[0,0,-1]`, the scrambled one mapped it onto
`[-0.075, 0.179, 0.981]`, **171.8° out**. scipy renormalises, so nothing ever
raised. Because the loop right-multiplies, this was a pure *left*
multiplication: a rigid rotation of the whole path, changing arc length and
tangent turning by **exactly zero** (identical to 15 significant figures), which
the solved camera pose then silently absorbed. Verified fixed by
`R0 @ g_sensor → [0, 0, -1]`.

**Accelerometer units (`src/load.py`).** `data/raw_data.csv` is in **g** — `ax`
hard-clips at 15.961504 for samples 181–199, the part's ±16 g full scale — while
`config.ACCEL_UNITS` declared `m/s2` and nothing converted. `tip`/`quat` are
immune (`estimate_gravity` renormalises, and the complementary filter is off),
but `rest_to_rest_cog` was double-integrating numbers 9.80665× too small (`cog`
span **0.161 m → 1.633 m**) and `metrics.peak_gforce` was dividing g by g.
Converted once at the boundary; the CSV is unchanged.

Consequently `scripts/verify_fusion.py`'s *"accel magnitude within physics range
(0 < a < 40 m/s²)"* check was **passing only because of the mismatch** — it saw
20.9 when the real peak is 205 m/s². 40 m/s² (4 g) was never a defensible bound
for a racket strike. Replaced with the sensor's own envelope: no axis beyond
±16 g, magnitude within √3 × 16 g. Measured per-axis max 15.96 / 14.96 / 12.38 g.

## 5. Results

| metric | before | after | target / observed |
|---|---|---|---|
| arc ratio | 2.01× | **0.51×** | 0.64 expected (not 1.0) |
| real tangent turning (dense, over window) | 805° | **196°** | — |
| tangent turning at the 13 annotated frames | 390° | **180°** | 235° observed |
| median reprojection, **re-solved** pose | 55.8 px | **21.1 px** | 50 px target → **PASS** |
| median reprojection, stale committed pose | 55.8 px | ≈293 px | report only — see below |
| projected extent (windowed) | 483 × 373 | **552 × 301** | 558 × 305 observed |
| projected extent (all 400 samples) | 483 × 373 | 880 × 1013 | why windowing matters |
| `cog` span | 0.161 m | **1.633 m** | < 3 m check |
| `verify_fusion.py` | 10 checks | **11 checks, all PASS** | — |

**On the two medians.** The ≈293 px figure is the *committed pre-fix pose*
evaluated on the new correspondences. It is reported for completeness but is not
a meaningful measure of this work: that pose was solved *against* the old
reconstruction, so it cannot fairly measure a new one. 21.1 px is the honest
number — the same fitting procedure, re-run.

## 6. What is still wrong

- **Arc ratio 0.51 against an expected 0.64 — RETRACTED.** 0.64 was computed
  from the *annotated-window* (f116-145) wrist spread (258 px) only. Over the
  full follow-through f116-220 the wrist travels much further — see §8/§9 for
  the measured figure and why the true number is "not a fixed 0.64" but a bound
  that depends on which window you ask about. `overlay_calib.fullspan_arc_bound`
  now reports the full-span figure alongside `model_vs_video_arc`'s
  annotated-window one; neither is to be chased by scaling anything.
- **`FS = 416` is still an assumption**, never measured — there is no static
  segment in the record to calibrate against (`|a|` over samples 0–24 averages
  0.763 g, not 1.000). At 240 fps capture the 400-sample record extends past
  the clip at both ends: **365 of 400 samples map inside angle_1's 226 frames,
  so 35 are marked `frame_index = -1`** (measured, not estimated — an earlier
  draft of this note and of `Artifacts/plan.md` said "~22", which was an
  arithmetic slip; the window the overlay actually uses, samples 175–225, is
  unaffected). Expected for an impact-centred capture that outlasts the clip,
  but it is the one loose end in an otherwise self-consistent picture.
- **The pivot lever is three different numbers — RETRACTED, see §10.1.**
  `RACKET_TIP_LEN = 0.686` (butt→tip), the measured wrist→annotated-head rim
  0.544 m, and the unused `L_HANDLE = 0.20` implying 0.486 all looked like
  competing lengths. They collapse: the lever is an exact gauge freedom of
  this overlay (changing it moves no pixel, §10.1), `L_HANDLE` is dead code
  that double-counts the wrist, and 0.544 was a `racket_pixel_scale` window
  artifact. Measured from the real wrist the ratio is 0.977 ⇒ 0.702 m —
  `RACKET_TIP_LEN = 0.686` is right to 2.3%. Left unchanged, same as before,
  but for the reason in §10.1, not the one above.
- **`racket_pixel_scale()` reads 285 px/m** against the ball's 359 px/m — this
  looked like "exactly the 0.686/0.544 lever ratio above", but 0.544 was
  itself produced by this function's own f116-145 window; over all 26 reads
  it reads 375 px/m, agreeing with the ball to 4.5% (§10.1, §10.4). The
  window bug is real and still left as-is, out of scope this round.
- **The complementary filter was latent and broken — now DELETED.** The
  `alpha > 0` branch used to assign `_rotation_aligning(...)` to `rot_accum`,
  which *replaces* rather than blends — discarding all accumulated heading at
  every sample (measured: seed 137 deg of heading, run the branch, the
  component about gravity comes out 0.0000 deg). It has been removed from
  `src/fusion.py:integrate_orientation`, which now raises `NotImplementedError`
  on any nonzero `alpha` instead of silently destroying heading. See §8.
- **`metrics.peak_gforce` now reads ~16.7 g** where it used to read ~2.1 g. That
  is the unit fix landing, not a regression.

## 7. Corrections issued

- `SYNC_METHOD.md` — two correction boxes. Its scale scan concluded "real time,
  slow motion decisively excluded", which is wrong and is why `SLOWMO_SCALE` was
  changed from 8.33 to 1.0 in the first place. The scan **could not** have found
  the answer: `src/sync/align.py:181` searches `np.arange(0.5, 2.01, 0.02)`, so
  8× was never in the search space, and its `slow_excluded` test
  (`plateau[1] < 2.0`) is satisfied by construction for *any* value that scan
  can return. The measured offset remains correct — it was anchored on a single
  instant, which the rescale preserves. Only the *rate* was wrong.
- `OVERLAY_FRAME.md` — superseding note. Its arc-mismatch argument and its
  `corr(world +Z, observed head v)` reasoning were both read off a
  reconstruction built on a scrambled `q0` and a 1.0× time base.

## 8. The racket lever pointed down the wrong body axis

`src/fusion.py:pivot_tip` built the racket lever as
`local = np.array([0.0, 0.0, config.RACKET_TIP_LEN])` — asserting the racket
points along the sensor's **+z** axis. Nothing in the repo ever measured that;
it was an assumption inherited from `PLAN.md`, and the sensor is taped to the
string bed, so its axes are set by how it was taped, not by any physical law.

**Evidence it is wrong, and what it actually is.** The accelerometer's own
centripetal signal is an independent measurement with no camera involved: a
sensor at body-frame radius `r` reads `f = (ww^T - |w|^2 I) r + b`, linear in
`r`. Least-squares fits over three different sample windows
(`fusion.fit_centripetal_lever`, Artifacts/analysis.md §7.1) land at:

| window | `|r|` (m) | unit direction | angle from +z |
|---|---|---|---|
| post-impact 205-399 | 0.213 | `[-0.997 +0.039 +0.061]` | 86.5 deg |
| all unclipped | 0.285 | `[-0.970 -0.065 -0.234]` | 103.5 deg |
| high-omega unclipped | 0.189 | `[-0.957 -0.251 -0.145]` | ~82-90 deg |

Every window is within ~15-20 deg of **-x** and ~90 deg off the modelled +z.
The three windows' off-axis components point in three *different* directions
(+z-ish, -z-ish, -y-ish) rather than agreeing on a consistent lean, which is
why the committed lever is **clean -x** (`config.RACKET_LEVER_BODY = (-1, 0,
0)`) rather than the slightly oblique direction any single window's raw fit
would suggest — a genuinely oblique mount would make the windows agree on
*which way* it leans, and they do not.

A held-out video grid search (pose fitted on the 13 committed annotations
only, scored on racket image-angle against the f146-220 reads, analysis
§7.2) optimises at `[-0.981, +0.173, +0.087]` — 11 deg from the committed
-x and 8 deg from the accelerometer's post-impact answer — **independent
confirmation from a channel the accelerometer fit never saw**. Its optimum
was deliberately **not** adopted: committing the direction that scores best
on the held-out set would be fitting to it and would destroy its value as
independent evidence (it also scores 5.3 deg mean image-angle error against
10.1 deg for pure -x, which is the price of that choice, paid knowingly).

**Measured before/after** (this worktree, `scripts/report_followthrough.py`
and `scripts/check_overlay_match.py`; analysis's own pre-fix figures in
parentheses where they differ):

| quantity | before (`+z`) | after (`-x`, committed) |
|---|---|---|
| fit-13+contact reprojection median | 21.1 px (analysis: 23.7) | 13.7 px (analysis: 12.9) |
| held-out f146-220 median | 297.9 px (analysis: 324) | 184.1 px (analysis: 184 — exact) |
| mean racket image-angle error, f146-220 | 114.4 deg (analysis: 116.7) | 9.1 deg (analysis: 9.1 — exact) |
| windowed extent f116-145 | 552 x 301 px (analysis: 546x272) | 565 x 306 px (analysis: 565x306 — exact) |
| annotated-window arc ratio | 0.509 (exact match) | 0.540 (exact match) |
| in-frame over f112-225 | 100% | 100% (not evidence of the fix — see below) |

`check_overlay_match.py`'s median (13.7 px) replaced the recorded baseline:
`overlay_calib.RECORDED_BASELINE_PX` tightened **21.5 -> 13.7**, never
loosened; `BASELINE_SLACK_PX` stayed `10.0`.

**Criterion 1 is not evidence the fix worked.** Both the `+z` and `-x` levers
stay 100% in-frame over f112-225 — the runaway that leaves the frame under
`+z` does so around f232, past this clip's last frame (225). The fix is
visible in criteria 4-6, not 1.

**Criterion 2, measured honestly.** `scripts/report_followthrough.py` prints
`step[f] = |proj[f] - proj[f-1]|`, the literal single-video-frame projected
step. By that definition the `-x` lever already settles under 15 px/frame
starting at f180 and is at 3.7 px/frame by f212 (f204=6.6, f208=5.2,
f212=3.7, f216=2.3, f220=1.1, f224=0.7). This differs substantially from
`Artifacts/analysis.md` §7.4's reported "30.0 px/frame at f208" / "10.4 at
f220": the projected `(u,v)` positions this script reproduces at those exact
frames match analysis's table to 0.1 px, so the trajectory and pose are not
in question — only the step metric is computed differently (analysis's
table rows are 12-18 video frames apart, and its "step" column is closer to
the point-to-point distance between those sparse rows than to a true
single-frame derivative). Reported as measured, not reconciled further;
either way the lever fix passes criterion 2 by a wide margin.

**The `alpha > 0` branch — deleted, not fixed.** It ended
`rot_accum = _rotation_aligning(g_sensor, g_world)`, which *replaces* the
integrated attitude instead of blending into it: `_rotation_aligning` returns
the minimal rotation between two vectors, so its axis is perpendicular to
gravity by construction and carries zero heading. Measured: seed 137 deg of
pure heading, run the branch, and the component about gravity comes out
0.0000 deg. `src/fusion.py:integrate_orientation` now raises
`NotImplementedError` on any nonzero `alpha` instead of leaving this as a
silent loaded gun. Not rewritten: there is no static window anywhere in this
record to validate a gravity correction against, and every alpha measured
end-to-end was worse than `alpha=0.0` (0.01 -> 33.4 px fit / 429 px held-out;
0.05 -> 28.9 / 270; 0.4 -> 24.7 / 277; 0.002 -> pose rejected outright).

**Direction-sensitive checks.** `scripts/verify_fusion.py` gained two: (A) a
consistency check reading the lever back out of `tip`/`quat` — honestly
documented as **not** the check that would have caught the original bug
(with the old hard-coded literal it would have been tautological); and (B)
the real one, comparing the configured lever against
`fusion.fit_centripetal_lever`'s axis (sign-free, <= 40 deg). The pre-fix
`+z` lever scores 86.5 deg against the same fit and would have been
rejected by check (B) — the permanent `[INFO]` line in `verify_fusion.py`'s
output proves this on every run.

**The wrist-translation verdict (deliverable 7) — SUPERSEDED, see §10.** The
paragraph below recommended the contract change as a follow-up; that
follow-up is this document's own §10. Kept for history, not deleted.

> `scripts/report_followthrough.py --wrist-sweep` evaluates
> `tip = k * swing.cog + tip_for_lever(quat, lever)` against the held-out reads
> for `k` in `[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]`. Only `k=0.2` clears the
> fixed rule (held-out median improves >= 20% vs `k=0`, fit-13 median worsens
> <= 2 px, mean |angle error| does not worsen): held-out median 184.1 -> 139.5
> px (24.2% better), fit-13 median 13.7 -> 12.5 px (better, not worse), mean
> |angle error| 9.1 -> 7.2 deg (better, not worse). **Per the fixed rule this
> is a VERDICT: helps (k=0.2), but it is NOT landed in this PR** — adding `cog`
> to `tip` breaks the `|tip| = RACKET_TIP_LEN` invariant that
> `verify_fusion.py`'s radius check and `racket_pixel_scale`'s L-cancellation
> argument both rest on, which is a contract change belonging in its own task.
> Reported to the coordinator as a follow-up recommendation.

**Visual check — SUPERSEDED by §10.6, see there for the current read.** The
paragraph below described the state before the wrist translation landed
(2026-10-05's task); kept for history, not deleted.

> Visual check (`scripts/render_overlay_frames.py --frames 119 138 150 175 200
> 224`, all six read): the dot sits on or very near the racket head at f119,
> f138 and f150, and close (slightly off the rim) at f175. At **f200 and f224
> the dot is visibly off the racket head** — the racket has wrapped up near
> the player's shoulder while the dot sits near the hip. This is the omitted
> wrist translation showing up visually, not a regression: the pivot model
> pins the wrist at the world origin, and the real wrist travels far enough
> by that point in the follow-through that no rigid pose can put it back (see
> the wrist-translation verdict above). Reported, not hidden.

## 9. Retractions

Four claims from an earlier draft of the task, falsified by direct
measurement in this worktree (`Artifacts/analysis.md`):

| retracted claim | correction |
|---|---|
| "a retrace is not what a camera error looks like" | **FALSE.** The real racket head retraces **530 px leftward** over f150-212 (`Artifacts/analysis.md` §3). A reversal-free path over that span is *wrong*. |
| "the post-impact gyro is string ringing, so integrating it is the bug" | **FALSE.** 65% of post-impact gyro energy is below 5 Hz and only 5.4% above 100 Hz, where the ~165 Hz string band lives; `reconstruct` already low-passes the fusion inputs at 25 Hz. With the lever corrected, that same gyro reproduces the real racket orientation to 9.1 deg over the whole 0.96 s record, through a 16 g clipped impact, with zero gravity correction (§4, §7.3). |
| "samples 180-199 are dead linear, possibly synthetic data" | **FALSE.** Genuine LSM6DSOX output. Gyro values sit on the part's exact 0.07 deg/s LSB grid (1164/1200 to <1e-4; the 36 that miss are off by <=1.4e-3, the CSV's 2-decimal rounding) and accel on its 0.488 mg grid — the ST sensitivities at +/-2000 dps / +/-16 g exactly, which interpolated samples would not hit. The 2nd-difference noise floor is 0.23-0.25 deg/s over s0-59 against a datasheet prediction of 0.19. Per-axis nothing is linear; the "constant +41.6" is the flat top of a sigmoid's derivative (§5). |
| "re-enable `COMP_FILTER_ALPHA` to bound drift" | **FALSE as a fix.** Worse at every alpha measured, and the branch itself carried zero heading. It is now deleted (§6, §8). |

Also retracted: this note's own §6 claim **"arc ratio 0.51 against an
expected 0.64."** 0.64 was computed from the *annotated-window* (f116-145)
wrist spread (258 px) only; it says nothing about the follow-through. Over
the fuller f116-220 span the measured max-pairwise wrist excursion is 302.7
px (`overlay_calib.fullspan_arc_bound`'s `wrist_max_px`, this worktree) —
**not** the ~366 px figure an earlier analysis pass quoted for the same
span, which on inspection was the image-plane bounding-box diagonal
(`max - min` per axis, then combined) rather than a true pairwise maximum
over all point pairs; the two are different quantities and happen to differ
here by about 20%. `fullspan_arc_bound` computes the literal pairwise
maximum, as its own docstring specifies, so this is a correction to the
analysis's arithmetic, not a code change. Using the measured 302.7 px, the
full-span ratio is 0.820 against a `ratio_expected` of 0.789 — both reported
by `scripts/report_followthrough.py` §7, neither chased by scaling anything.
`FUSION_NOTES.md` §6's bullet is updated above to point at this instead of
restating a single fixed target.

## 10. The omitted wrist translation

Task: "the racket head is still not precise", "the animated dot is affected
when the ball contacts the racket", and "from frame 116 forward the dot is
significantly off target" (user report, after PR #9 merged at `4553c27`).
All three traced to one omission, below — not to `RACKET_TIP_LEN`, which an
earlier draft of this task blamed.

### 10.1 The retraction that re-scoped this task

`RACKET_TIP_LEN` is an exact **gauge freedom** of this overlay.
`pivot_tip` pinned the wrist at the origin, so `tip = L·R(t)u`. Scaling
`L -> kL` scales the whole 3D point set about the origin, and PnP absorbs it
exactly via `tvec -> k·tvec`: `K(R(kp) + kt) = k·K(Rp + t)` is the same
homogeneous point, hence the same pixel, for every `k`. Measured in this
worktree: `L = 0.470` (the length the first draft proposed) reproduces
`L = 0.686`'s reprojection to **1.7e-13 px** over all 400 samples, and an
L-sweep from 0.213 to 1.372 m produces only **two** distinct outcomes — the
two RANSAC inlier sets, not a response to `L`.

That falsifies the "three competing lever lengths" §6 used to list:
`L_HANDLE = 0.20` is dead code whose own comment already reads *wrist -> grip
end*, so subtracting it from butt -> tip double-counts the wrist; and the
0.544/0.545 m figure was a `racket_pixel_scale` window artifact — it maxes
over the f116-145 annotated window (195.2 px), where the racket is never
fronto-parallel, while over all 26 reads the peak is 257.4 px at f150, giving
375 px/m, which agrees with the ball's own 359 px/m to **4.5%**, not the 20%
the first draft reported. Measured from the **real** (annotated) wrist rather
than the model's pinned pivot pixel, the apparent-length ratio is 0.977 =>
**wrist -> tip = 0.702 m**: `RACKET_TIP_LEN = 0.686` is right to 2.3%, and
is **unchanged** by this task.

### 10.2 What the old 1.46 ratio was really measuring

The arc-ratio argument used the model's **pinned pivot pixel** (688.2, 406.7,
constant for every sample) as one end of its reference vector, and the
**annotated wrist pixel** (which moves) as the other end's observation. The
gap between those two points is 61.5-179.3 px over the 26 annotated reads,
median 117.9 px. The ratio was measuring that gap — the missing wrist
translation — with `RACKET_TIP_LEN` algebraically cancelled out of it by the
same homogeneous-point argument as §10.1. It was real, just misattributed.

### 10.3 The contract change

`Swing` gained a `pivot: np.ndarray | None = None` field (`src/swing.py`),
last in the dataclass's field order (it has a default; the six fields before
it do not) and defaulting to `np.zeros_like(tip)` in `__post_init__` — the
pre-2026-10-05 pinned-wrist behaviour, and what an older `swing.npz` with no
`pivot` array loads as. `tip` keeps its old meaning exactly: the head
**relative to the pivot**, so `|tip| == RACKET_TIP_LEN` for every sample,
unconditionally. A derived `Swing.head` property returns `pivot + tip` — the
head's WORLD position, which is what every consumer that draws or measures
the racket head actually wants.

`pivot` was **not** folded into `tip`. Folding `k·cog` into `tip` would have
broken, simultaneously: `verify_fusion.py`'s radius check (`|tip + 0.907·cog|`
runs 0.686 -> ~1.5 m, not constant), both of its lever-direction checks (they
read `tip` directly), and `overlay_calib.racket_pixel_scale`'s
L-cancellation argument — for no benefit, since `Swing.head` gives every
consumer the world position in one extra attribute access. Seven consumers
were updated from `swing.tip` to `swing.head`: `overlay_calib`'s
`build_correspondences`/`model_vs_video_arc`/`fullspan_arc_bound`,
`camera.draw_overlay`, `plot.figure_3d`, `app.py`'s impact anchor, and
`check_overlay_match.py`/`report_followthrough.py`'s reprojection and
image-angle code. `scripts/verify_fusion.py` is the one deliberate holdout:
its radius check and both lever-direction checks still read `swing.tip`,
because `tip` — not `head` — is what those invariants are about.

### 10.4 `k` — both routes, one committed

`pivot = wrist_pivot(cog) = config.WRIST_PIVOT_COG_SCALE * cog`
(`src/fusion.py`). `cog` (`rest_to_rest_cog`) is detrended about the origin,
so its amplitude is set by that detrending, not by physics — `k` has to be
calibrated, not assumed, and the held-out f146-220 reads must never be used
to choose it (that is exactly the circularity `build_correspondences`'s guard
exists to prevent).

**Route 1 — amplitude matching (COMMITTED).** Scale `cog`'s max-pairwise
displacement over the annotated window (f116-145, 258.118 px of wrist
travel) to match the same window's `cog` displacement, using only
fit-window reads:

```
wrist_pixel_spread()[2]            = 258.118 px   (fit reads only)
racket_pixel_scale()[0]             = 284.611 px/m (committed value)
  -> observed wrist travel         =   0.906914 m
cog max-pairwise, samples 175-225  =   0.999390 m
  -> k = 0.906914 / 0.999390        =   0.907468  -> committed 0.907
```

Declared sensitivity: `racket_pixel_scale` maxes over f116-145 only (284.6
px/m); over all 26 reads it reads 375.2 px/m (§10.1), which would give
`k = 0.688` instead. Both ends sit inside route 2's flat basin below.
Widening that window is deliberately **out of scope** this round (it touches
`racket_distance_m` and both arc ratios, and raises an unsettled question
about using held-out reads in a reported metric) — left as-is, noted here.

**Route 2 — joint PnP over the fit set (REPORTED, NOT COMMITTED).** `k` swept
as a free parameter alongside the pose, scored on the 13 + contact
correspondences only (held-out used purely to validate, never to choose):

```
    k     med    mean     max  inl  |tvec|   held-med   solver
 0.000   13.71   23.92   91.16   11   2.400     184.1    ransac+lm
 0.100   19.81   22.92   61.23   11   1.532     416.3    ransac+lm   <- BAD MINIMUM
 0.200   12.52   21.29   79.17   11   2.843     139.5    ransac+lm   <- fit-median ARGMIN, bad answer
 0.300   17.60   20.07   48.20   13   2.983      89.9    iterative
 0.400   16.61   19.05   45.90   12   3.196      70.7    iterative
 0.500   16.18   18.20   43.97   13   3.405      72.6    iterative
 0.600   15.88   17.51   42.36   13   3.612      62.3    iterative
 0.700   15.69   16.95   41.01   13   3.816      71.9    iterative
 0.750   15.63   16.71   40.42   13   3.917      76.5    iterative
 0.800   15.58   16.50   39.87   13   4.017      76.8    iterative
 0.850   15.56   16.31   39.38   13   4.118      80.1    iterative
 0.907   15.56   16.13   38.88   13   4.232      78.9    iterative   <- COMMITTED
 0.950   15.56   16.01   38.53   13   4.317      83.5    iterative
 1.000   15.59   15.89   38.16   13   4.417      86.7    iterative
 1.250   15.38   15.55   36.83   13   4.911      82.8    iterative
 1.500   15.11   15.50   36.36   13   5.441      74.0    iterative
 2.000   15.64   15.65   46.25   13   4.064      394.6    ransac+lm   <- BAD MINIMUM
 3.000   14.79   15.92   40.41   13   5.291      409.5    iterative   <- BAD MINIMUM, best fit median of all!
```

Mean and max fall **monotonically** out to `k = 1.5`, so the fit-only
objective has no interior minimum — route 2 cannot identify `k` by itself,
which is *why* route 1 is committed. `k = 0.10`, `2.0` and `3.0` are bad PnP
minima (held-out 416 / 395 / 410 px); `k = 3.0` posts the **lowest** fit
median of every cell tried (14.79 px) with a held-out median of 410 px — an
argmin over the fit set is actively misleading here. What route 2 does say:
the committed `k = 0.907` sits inside a broad flat basin (0.6-1.1, fit median
15.55-15.69 px), which is **agreement**, not fitting. The held-out reads were
never used to pick `k`; they are the validation in §10.5.

### 10.5 Before / after, all eight numbers

| statistic | k=0 (before) | k=0.907 (committed) | criterion | bar | verdict |
|---|---|---|---|---|---|
| fit median (px) | 13.71 | 15.56 | 3 | mean/max/inliers, not median alone | rose — expected, see below |
| fit mean (px) | 23.92 | 16.13 | 3 | improve | PASS |
| fit max (px) | 91.16 | 38.88 | 3 | improve | PASS |
| RANSAC inliers | 11/14 | 13/14 | 3 | improve | PASS |
| held-out median (px) | 184.1 | 78.9 | 1 | <= 100 px | PASS |
| mean \|image-angle error\| (deg) | 9.1 | 8.5 | 5 | <= 9.1 deg | PASS |
| \|tvec\| (m) | 2.400 | 4.232 | 2 | 3.5-4.6 m (anchor 4.075) | PASS |
| ratio A (pinned/projected pivot tail) | 1.460 | 0.982 | 4 | \|A-1\| <= 0.10 | PASS |
| ratio B (real-wrist tail) | 0.998 | 1.056 | — | near 1 (sanity) | OK |

The **median rising** 13.71 -> 15.56 px is the robust fit shedding its two
worst outliers, both at ball contact: f130 head 91.2 -> 38.9 px, the ball
contact itself 72.8 -> 22.2 px. Every other statistic improves sharply. This
is why acceptance criterion 3 was rewritten as a multi-statistic gate
(`overlay_calib.RECORDED_BASELINE_MEAN_PX = 16.2`, `_MAX_PX = 39.0`,
`_INLIERS = 13`, each tightened the same way `RECORDED_BASELINE_PX` was in
PR #9) rather than `RECORDED_BASELINE_PX` itself being loosened:
`RECORDED_BASELINE_PX` is still literally `13.7` and `BASELINE_SLACK_PX` is
still `10.0` — 15.56 still passes that original gate (15.56 < 23.7) — the new
gates catch what the median alone would hide.

All nine numbers above are reproduced verbatim by
`scripts/report_followthrough.py` (the committed run and `--k 0`) and
`scripts/check_overlay_match.py`, run after `scripts/verify_fusion.py`
regenerated `data/outputs/swing.npz` in a separate process (the
`solved_pose` `lru_cache` trap — see Known traps).

### 10.6 The IMPACT-ring fix

`src/camera.py:draw_overlay` drew the IMPACT ring at a fixed 12 px radius,
the same as the animated dot's white ring. `impact_idx = 199` maps to
`frame_exact 130.13`, and the animated dot at f130 **is** sample 199 — so at
exactly that frame the two 12 px rings were **exactly concentric** (measured
0.0 px apart), and the dot's white ring, drawn second, overdrew the blue
IMPACT ring. That is the "when the ball contacts the racket, the animated
dot is affected" symptom: a marker **changing appearance**, not moving — pure
rendering, zero geometric content. The fix enlarges the IMPACT ring to 22 px
(radius only; its true projected centre and the "IMPACT" label offset are
unchanged) so the dot stays legible inside it at every frame, including f130.
Confirmed visually in `Artifacts/overlay_evidence/after_render_f130.png` and
`app_pivot_f130.png`: a 12 px white-ringed red dot sits visibly inside a 22
px blue ring at f130, two distinguishable markers.

Recorded, not fixed: `impact_idx = 199` (frame_exact 130.13) disagrees with
`CONTACT_FRAME_EXACT = 130.70344` by 0.58 frames (2.4 ms) — `impact_idx` is
detected from the gyro/jerk peak, `CONTACT_FRAME_EXACT` from the video ball
track, and they need not agree exactly. Left as an open item.

At f200/f224 (`Artifacts/overlay_evidence/after_render_f200.png`,
`after_render_f224.png`, `app_pivot_f200.png`, `app_pivot_f224.png`, all
read): the dot now sits close to the racket head up near the player's
shoulder through the follow-through, rather than off near the hip as in the
pre-pivot renders §8 described (superseded there, not deleted). This is the
wrist translation showing up visually, matching §10.5's held-out numbers.

### 10.7 Two consequences to be honest about

(a) `overlay_calib.model_vs_video_arc`'s `ratio_expected` (0.642 annotated
window, 0.789 full span) was the bound for a model whose pivot **could not
move** — `pivot_tip` alone produces only the rotational part of the head's
motion, so 1.0 was unreachable by construction. With `Swing.pivot` modelling
the translation, the target genuinely **is** 1.0, and the measured ratio
moved 0.540 -> **0.904** (annotated window) and 0.820 -> **1.319** (full
span, now overshooting). The dict key name (`ratio_expected`) is kept for
continuity with existing callers and this document's own history; its
meaning has changed, documented at the three sites in `src/overlay_calib.py`
that print or compute it.

(b) `racket_pixel_scale`'s L-cancellation argument — that `L` divides out of
the arc ratio exactly — is now only **partial**: `arc3d` is `L·(rotational
arc) + (pivot arc)`, and `pivot` does not scale with `L`, so the old exact
cancellation only holds in the `pivot = 0` limit. A side effect: `L` becomes
*weakly* identifiable from the ratio once pivot is in play (not from
reprojection — PnP still absorbs any `L` exactly via `tvec`, §10.1 still
holds there). At the committed `k = 0.907`, the fit median moves only
15.17 -> 16.13 px across `L = 0.40-1.00` m while the ratio moves
0.901 -> 1.143, landing at ratio = 1.000 around **L ~= 0.60-0.69 m** —
consistent with §10.1's independent 0.702 m. Not acted on; `RACKET_TIP_LEN`
is unchanged.

### 10.8 Two stale numbers, checked against what is actually on disk

TASK deliverable 7 asked this task to correct two stale numbers. Checked,
not assumed:

(i) **"366 px ~= 1.285 m" full-span wrist travel** — already corrected in
this worktree before this task started: §9's final paragraph above already
states the measured 302.7 px and explains the 366 figure was a bounding-box
diagonal, not a pairwise maximum. The only place the 366 figure still
appeared was `overlay_calib.fullspan_arc_bound`'s own docstring, which this
task fixed to read 302.7 px (§5d of the implementation).

(ii) **`|tvec| = 1.63 m`** — `grep -rn "1\.63"` over every `*.md`/`*.py` in
the repo returns four hits, and all four are the `cog` **span** (1.633 m),
which is correct and unrelated to `|tvec|`. There is no stale `|tvec| = 1.63`
anywhere on disk; TASK's figure traced to a stale draft, not to committed
text. The genuinely stale distance figure is `OVERLAY_FRAME.md:165`'s
"2.96 m (solved)", in a historical before/after table from before PR #9's
lever-axis fix — noted here as **superseded** by this task's `|tvec| =
4.232 m`, rather than rewritten in place (that table documents its own
point in history, same policy as §9's retractions above).

### 10.9 Still out of scope, carried forward

- `racket_pixel_scale`'s f116-145 window (§10.1, §10.4) — resolves the ball
  cross-check from 20% to 4.5% and would move `k` from 0.907 to 0.688; left
  unfixed, both values sit inside route 2's basin.
- `CALIB_FOV_DEG = 60` — FOV 80/90/100 give fit medians 10.63/10.82/11.90 px
  against 60's 13.71; a broad shallow basin, not flat, so ratio A (§10.5) is
  not 100% attributable to `k` alone.
- PR #9's Evidence A is weaker than written: with `r_true` exactly along
  `-x`, the centripetal-only fit (`fusion.fit_centripetal_lever`) can return
  axes 15-64 deg off because it omits `alpha x r`, so `verify_fusion.py`'s
  40 deg check rests on an estimator whose own axis error can exceed 40 deg.
  The axis conclusion survives on Evidences B and C, which are independent
  of this estimator. **Not contrary evidence — the axis is not reopened.**

## 11. The pivot's missing linear drift term

Follow-up to §10: after that PR merged, the overlay tracks the racket head
well inside roughly f108-152 and comes off before and after. The "after" side
(previously "f146-220 held-out") is §10's already-disclosed gap, unchanged by
this section. The "before" side is new and is the subject of this section.

### 11.1 The symptom and the measurement

The pre-116 residual ramp (`Artifacts/analysis.md` §2.2, re-verified here via
`scripts/dump_swing_frames.py --verify`) runs 145.4 px at f90, falling
smoothly to 6.7 px at f115 (≈5.5 px per video frame — a ramp, not a kink).
An initial by-eye estimate of f112's offset (~150 px) was wrong: a proper
gridded read puts it at 19.2 px — the dot is already essentially on the
racket there. The measured good window is **≈ f108-152**, not f114-143 as
first assumed.

### 11.2 Four falsified premises

Four assumptions made going into this task turned out to be wrong, caught by
measurement rather than carried forward (`Artifacts/analysis.md` §0):

| assumption | verdict |
|---|---|
| "the good window (114-143) is almost exactly the PnP fit window" | **FALSE.** No kink at f116 or f145; f130 (the fit window's own temporal centre) is its *worst* fit-window frame at 38.9 px. |
| "a single rigid pose can only cancel error locally" | **FALSE.** One static pose over f90-145 reconciles both spans (fit 10.6 px / pre-116 10.3 px) — but no single pose reconciles f116-145 with f146-220. The two ends are different problems. |
| "`GRAVITY_REF_SAMPLES`'s reference window could let gyro drift accumulate" | **FALSIFIED.** `q0` is a pure left-multiply so PnP absorbs it exactly, and sweeping `GRAVITY_REF_SAMPLES` in {10,25,50,100} moves every residual by ≤0.1 px. |
| "mean out-of-plane angle error 12.3°" (the post-145 gap) | **STALE.** That was the `k=0.75` intermediate from the prior round; the committed figure is 8.5°. |

### 11.3 It is the pivot, not the orientation

Decomposing each pre-116 head error into (pivot error) + (lever/orientation
error) puts **80-90% of the error on the pivot**. The racket's own *image
angle* (orientation only, translation-immune) is actually *better* pre-116
(5.06° mean) than inside the fit window (11.13°) or the follow-through
(8.5°) — this is not an orientation problem. The projected-pivot-to-
annotated-wrist travel ratio runs 0.56x before the fit window, 1.03x during
it, 2.16x after it, 0.19x net across the whole record (`Artifacts/analysis.md`
§3) — the pivot undershoots exactly where the overlay visibly comes off the
racket.

### 11.4 The mechanism

`fusion.rest_to_rest_cog`'s final step, `_detrend_rest_to_rest`, removes a
linear trend from the double-integrated position so the path starts and ends
near the origin — standard practice for a ~1 s strapdown integration with no
position aiding. Measured directly from `data/raw_data.csv`: the removed
ramp is **1.404 m** (`[-1.2967, +0.1899, -0.5044]`) against a surviving
detrended `cog` net displacement of only **0.782 m** — the detrend removes
*more* net displacement than it keeps.

The detrend is load-bearing, not a bug to delete outright: scoring the
pose with the position detrend skipped entirely (pivot built straight from
the undetrended double-integrated position, no linear term restored) blows
the held-out median up into the hundreds of pixels — measured here at 430.5
px with the robust solver (596-658 px with the plain solvers) against the
committed 78.9 px. The fix is to give back a *modelled* ramp on top of the
still-detrended `cog`, not to remove the detrend.

### 11.5 Three procedures, one committed

Scored with the pose **re-solved** from the committed 13+contact set (the
deployed configuration), b x 400 and all three spans (med/mean/max px):

| procedure | `b`x400 (m) | `|b|`x400 | fit | pre-116 | held-out | verdict |
|---|---|---|---|---|---|---|
| committed `b=0` (pre-this-section) | — | 0 | 15.56/16.13/38.88 | 50.50/59.92/145.40 | 78.93/74.57/156.56 | baseline |
| **C** accel-derived, COMMITTED | `[-1.176,+0.172,-0.457]` | 1.274 | 15.78/16.38/38.69 | 23.17/30.35/82.05 | **72.56/72.07/113.58** | all gates pass, all 3 held-out stats improve |
| A two-step (pose fixed, `b` on pre-116 only) | `[-1.273,+0.148,-1.130]` | 1.709 | 15.89/16.67/38.88 | 22.29/28.11/74.03 | 76.83/74.50/123.95 | passes; held-out mean improves only 0.07 px |
| B joint `(rvec,tvec,b)`, seed 0 | `[-1.433,+0.535,-0.712]` | 1.687 | 15.68/16.48/38.25 | 17.85/24.34/68.51 | 73.48/72.42/124.93 | passes |
| B joint, **seeded at C's `b`** | `[-1.739,-0.024,-0.455]` | 1.798 | 15.09/16.49/38.91 | 7.31/13.56/44.66 | **97.37/91.00/145.58** | **held-out median AND mean regress** |

All five rows are printed by `scripts/fit_pivot_drift.py`. **C is committed
because it has no objective to minimise.** Procedures A and B both involve a
least-squares fit against annotated pixels, and B's own two rows show the
fit landing in a different local minimum depending on its seed: the seed
that posts the *best* pre-116 number (7.3 px median) is the one that makes
the *never-touched* held-out set worse. This is the same lesson as PR #9's
FOV sweep (§4-§5 above) and PR #10's `k`-sweep (§10.4): a single optimizer
run on a non-convex objective is a lead, not a result. C sidesteps that
entirely — it is a closed-form derivation from the accelerometer, so there
is no local minimum to land in the wrong one of.

### 11.6 Two independent channels

`b_C` (accelerometer, committed) = `[-1.176, +0.172, -0.457] m`.
`b_A` (image, 14 hand-read pre-116 pixels, pose held fixed) =
`[-1.273, +0.148, -1.130] m`. **Angle between them: 20.4°. Magnitude ratio:
1.342.** Both re-derived independently in this worktree
(`scripts/fit_pivot_drift.py` §1/§8), not copied from any prior draft.

This is **20.4°, not the 11.8° an earlier joint-fit `b` had claimed** — that
figure came from a joint fit that does not reproduce under independent
re-verification, and its accompanying claim ("improves on all three held-out
statistics") is **retracted for procedure B**. It holds for procedure C
instead, by a different (non-fitted) route — see §11.5's table.

### 11.7 Gain and centring sweeps

Both reported by `scripts/fit_pivot_drift.py --sweep`, neither argmin taken:

```
gain  fit med/mean/max     pre-116 med      held-out med/mean/max
0.00  15.56/16.13/38.88    50.50            78.93/74.57/156.56
0.50  15.80/16.19/38.65    35.66            72.45/67.74/117.10
0.75  15.66/16.26/38.55    28.20            65.20/68.35/111.55
1.00  15.78/16.38/38.69    23.17            72.56/72.07/113.58   <- COMMITTED
1.25  16.52/16.89/50.62   255.22           432.51/385.12/635.94  <- BAD PnP MINIMUM
1.50  15.49/16.61/38.41    10.39            98.54/87.07/131.91   <- held-out REGRESSES
2.00  16.42/16.73/50.04   271.65           425.95/392.23/642.43  <- BAD PnP MINIMUM
```

Gain 0.75 posts the best held-out median and gain 1.50 the best pre-116
median; 1.50 regresses the held-out set and 1.25/2.00 collapse into bad PnP
minima (`|tvec|` ~2.6 m). `gain=1.0` is committed because it is the
*derived* value — `b = k x removed_slope`, not an argmin.

```
centre   fit med  |tvec|
0        16.57    2.007 m   (bad PnP minimum)
200      15.78    4.357 m   <- COMMITTED
399      15.73    4.743 m   (outside the 3.5-4.6 m player-height anchor band)
```

A constant 3D offset is a gauge freedom in exact arithmetic, but the
multi-solver PnP path is non-convex and the centring changes which minimum
it lands in. 200 (the record's midpoint) is a documented convention, not a
free parameter.

### 11.8 Where the pre-116 reads live and why

The 14 new pre-116 annotations (`src/overlay_calib.py`,
`RACKET_HEAD_PX_PRE116_ANGLE_1` / `WRIST_PX_PRE116_ANGLE_1`) live in their
**own, separate dicts**, not folded into `RACKET_HEAD_PX_ANGLE_1` /
`WRIST_PX_ANGLE_1`. Measured: folding them in moves route-1 `k` from
0.9075 to **1.124** purely as a side effect (`racket_pixel_scale` maxes
`|head-wrist|` and `wrist_cog_scale` takes its window from those same
dicts), dragging `racket_distance_m`, `model_vs_video_arc` and
`fullspan_arc_bound` with it — a second, confounded change riding along
inside a task about the drift term. Kept separate, `k` is bit-identical
before and after (`0.907468`, printed by `check_overlay_match.py` §8(b)).

They are also never fed to `solved_pose`'s correspondence set by default:
`build_correspondences` gained an `include_pre116` flag, default `False`,
used only by `scripts/fit_pivot_drift.py`. Widening the PnP fit window with
these reads was measured (`Artifacts/analysis.md` §4.2) to buy ~12 px on
pre-116 at a cost to the held-out set of 8 px (4 points) to a **doubling to
177 px** (all 14) — a bad trade, deliberately not taken.

### 11.9 Honest costs, all of them

- Fit window: median 15.56 -> **15.78** px, mean 16.13 -> **16.38** px (both
  inside their gates, both slightly worse); max 38.88 -> 38.69 px (slightly
  better); inliers unchanged 13/14.
- Held-out image-angle: 8.51° -> **9.29°**, which trips PR #10's own printed
  9.1° line in `report_followthrough.py` — reported, not tuned; that line is
  not one of this round's 8 acceptance criteria and the script always exits
  0 (it is a reporting script, not a gate, by its own docstring).
- Fit-window image-angle: 11.13° -> 12.48°.
- `|tvec|`: 4.232 -> 4.357 m (inside the 3.5-4.6 m player-height anchor
  band).
- `model_vs_video_arc` ratio: 0.904 -> 0.943 (closer to 1.0, better);
  full-span ratio: 1.319 -> 1.305.
- Pivot span: 1.481 -> **1.784 m** (gate is < 3.0 m, comfortably inside; it
  would fail at `|b|x400` ~ 3.46, the magnitude the earlier, non-reproducing
  joint-fit `b` had produced — a useful independent plausibility check that
  that fit's magnitude was too large).
- Nothing here fixes the follow-through's orientation error — §10's
  disclosed gap is unchanged in kind, improved only incidentally in
  magnitude (held-out max 156.6 -> 113.6 px).

### 11.10 The camera is not perfectly static

Two independent background patches agree on ~8-9 px of leftward camera drift
over f90->f220 (`Artifacts/analysis.md` §5). Smaller than the ±15-25 px
annotation uncertainty, so not the mechanism here, but it puts a ~9 px floor
under any static-pose approach to this footage. Noted, not modelled.

### 11.11 Still out of scope, carried forward

- `racket_pixel_scale`'s f116-145 window (§10.1, §10.4, §11.8).
- `CALIB_FOV_DEG = 60` (§10.9).
- The follow-through orientation error (§10, unchanged in kind by this
  section).
- Whether the f130 contact correspondence still earns its place
  (`Artifacts/analysis.md` §9, open question 5).
