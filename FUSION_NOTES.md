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

- **Arc ratio 0.51 against an expected 0.64.** 0.64 — not 1.0 — is what a
  *correct* wrist-pinned pivot model should give, because `pivot_tip` holds the
  wrist at the origin and the repo has measured 258 px (~0.72 m) of wrist travel
  it structurally cannot represent. The remaining ~20 % is, by measured size:
  the post-impact excursion-and-return (`Artifacts/analysis.md` §4.5), the
  13-point annotation polyline's own undercount of the observed arc, and the
  pivot really being a point on a translating forearm.
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
- **The complementary filter is latent and broken.** The `alpha > 0` branch
  (`src/fusion.py:108-115`) assigns `_rotation_aligning(...)` to `rot_accum`,
  which *replaces* rather than blends — discarding all accumulated yaw at every
  sample. Dead at `COMP_FILTER_ALPHA = 0.0`. Not touched; do not enable it
  without rewriting it.
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
