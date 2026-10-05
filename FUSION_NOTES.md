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
- **The pivot lever is three different numbers.** `RACKET_TIP_LEN = 0.686`
  (butt→tip), the measured wrist→annotated-head rim 0.544 m, and the unused
  `L_HANDLE = 0.20` implying 0.486. The sensor is on the *strings*, at a fourth,
  undocumented distance. Deliberately left unchanged: `L` cancels exactly out of
  the arc ratio, and changing it in one place without the other would silently
  break that cancellation. It does affect every metric distance.
- **`racket_pixel_scale()` reads 285 px/m** against the ball's 359 px/m —
  exactly the 0.686/0.544 lever ratio above. Left as-is for the same reason.
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

**The wrist-translation verdict (deliverable 7).**
`scripts/report_followthrough.py --wrist-sweep` evaluates
`tip = k * swing.cog + tip_for_lever(quat, lever)` against the held-out reads
for `k` in `[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]`. Only `k=0.2` clears the
fixed rule (held-out median improves >= 20% vs `k=0`, fit-13 median worsens
<= 2 px, mean |angle error| does not worsen): held-out median 184.1 -> 139.5
px (24.2% better), fit-13 median 13.7 -> 12.5 px (better, not worse), mean
|angle error| 9.1 -> 7.2 deg (better, not worse). **Per the fixed rule this
is a VERDICT: helps (k=0.2), but it is NOT landed in this PR** — adding `cog`
to `tip` breaks the `|tip| = RACKET_TIP_LEN` invariant that
`verify_fusion.py`'s radius check and `racket_pixel_scale`'s L-cancellation
argument both rest on, which is a contract change belonging in its own task.
Reported to the coordinator as a follow-up recommendation.

Visual check (`scripts/render_overlay_frames.py --frames 119 138 150 175 200
224`, all six read): the dot sits on or very near the racket head at f119,
f138 and f150, and close (slightly off the rim) at f175. At **f200 and f224
the dot is visibly off the racket head** — the racket has wrapped up near
the player's shoulder while the dot sits near the hip. This is the omitted
wrist translation showing up visually, not a regression: the pivot model
pins the wrist at the world origin, and the real wrist travels far enough
by that point in the follow-through that no rigid pose can put it back (see
the wrist-translation verdict above). Reported, not hidden.

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
