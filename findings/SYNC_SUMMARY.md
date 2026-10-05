# Sync Summary

A short-form summary of how `data/raw_data.csv` was synchronized to the two
tennis-swing videos, and a glossary of every variable/column involved. For
the full technical writeup (evidence, numbers, why each method choice was
made), see `SYNC_METHOD.md` — this file is the quick-reference companion to
it, not a replacement.

## What "synced" means here

The IMU sensor (`data/raw_data.csv`) and the cameras (`data/video/*.mp4`)
were recorded independently, with no shared clock. Syncing means finding,
for each video, the one number (an **offset**, in seconds) that converts an
IMU sample's own time into that video's time, so the correct video frame can
be looked up for any CSV row.

## How it was done, in order

1. **Probe both videos** (`scripts/probe_videos.py`) — decode every frame
   (never trust container metadata), measure the real frame rate, frame
   count, and whether the frame interval is constant (CFR) or not (VFR).
2. **Find the impact instant on the IMU side** (`scripts/run_sync.py` →
   `src/sync/imu_events.py`) — a tennis ball striking the racket produces an
   unmistakable signature (accelerometer rail-clip followed by violent
   high-frequency ringing, a near-instant gyro reversal). Two independent
   detectors locate it to confirm it isn't a fluke.
3. **Find the matching instant on the video side**
   (`src/sync/video_motion.py`) — track the ball's image position; its speed
   has a sharp kink at the moment of contact (slow approach, fast departure).
   Fitting a line to each side and intersecting them gives a sub-frame
   contact time.
4. **Offset = video contact time − CSV impact time.** Applied as a flat
   shift (no drift, no scale) to every row of that video's synced CSV.
5. **Cross-check** with a second, independent method (correlating the IMU's
   gyro signal against the video's frame-to-frame motion) — used to catch
   mistakes, not to compute the answer. Where it disagreed, the disagreement
   itself was measured and explained rather than smoothed over (see
   `SYNC_METHOD.md`).

## Result

| video | synced? | offset (CSV → video) | honest error bar |
|---|---|---|---|
| `swing_angle_1.mp4` | **yes** | **+3.9093 s** (+117.28 frames @ 30 fps) | **±1 frame** (±33.3 ms, ±14 CSV samples) |
| `swing_angle_2.mp4` | **no** | — | different take; no ball strike exists in this clip to sync to |

## Variable & column glossary

### Raw input — `data/raw_data.csv`

| name | meaning |
|---|---|
| `index` | Row counter, 0–399. **Not** a timestamp — there is no timestamp column in this file. |
| `ax, ay, az` | Accelerometer, 3 axes, in **g**. `ax` is clipped (saturated) at the sensor's ±16 g rail for 18 samples near impact — a hardware limit, not signal. |
| `gx, gy, gz` | Gyroscope, 3 axes, in **degrees/second**. Not saturated anywhere in this file. |

### Derived / intermediate (computed during sync, not stored in a file by themselves)

| name | meaning |
|---|---|
| `fs` / sample rate | **416 Hz** — an *assumption* from `CLAUDE.md`/`TASK.md`, not a measurement (there's no timestamp column to check it against). The rate-check below is consistent with it but can't independently prove it. |
| `t_sample` (as a concept) | `index / 416.0` — a CSV row's own time, in seconds, starting at 0. (Also the literal column name in the output CSVs — see below.) |
| `impact_index` | The CSV row where ball-racket contact happens — **200** (of 400), found by two independent detectors (gyro-jerk spike; high-frequency accelerometer energy onset) that agree exactly. |
| `t_impact_csv` | `impact_index / 416.0` = **0.4808 s** — the impact instant, in CSV-clock time. |
| `motion_onset_index` | Sample **54** — where gyro magnitude first leaves its resting baseline (the start of the backswing, used as a second reference point, not for the offset itself). |
| `measured_fps` | The video's real frame rate, **measured by decoding every frame and taking the median inter-frame time gap** — not read from (and not trusting) container metadata. 30.000 fps for both videos here. |
| `frame_times` | Per-frame timestamp array for a video (seconds), built the same measured way; used instead of `frame_number / fps` because `swing_angle_1.mp4` drops one frame partway through, which would make that simpler formula wrong after the drop. |
| `t_contact_video` | The video-clock instant of ball-racket contact, found from the ball-trajectory kink — **4.3901 s** (frame 130.70) for `swing_angle_1.mp4`. |
| `offset` | `t_contact_video − t_impact_csv` = **+3.9093 s** — the one number that converts any CSV-side time into that video's time. Computed once per video; `swing_angle_2.mp4` has none (not syncable). |

### Output — `data/synced_data_angle_1.csv`, `data/synced_data_angle_2.csv`

Each file has the original seven raw columns above, plus:

| column | meaning |
|---|---|
| `t_sample` | `index / 416.0` — the row's time on the **IMU's own clock**, seconds. |
| `t_video` | `t_sample + offset` — the same instant, expressed on **that video's clock**, seconds. Empty for all rows in `swing_angle_2.mp4`'s file, since no offset exists for it. |
| `frame_exact` | `t_video` converted to a fractional frame position using that video's real `frame_times` array (not a plain `t_video * fps`, for the reason noted under `frame_times` above). |
| `frame_index` | `frame_exact` rounded to the nearest actual video frame. **`-1`** means the row's time falls outside the video (or, for `swing_angle_2.mp4`, that there's no valid mapping at all — all 400 rows are `-1` there, by design, not by error). |

All 400 rows are kept in both output files — nothing is dropped or clipped, even rows that map outside a video's frame range.

## Known limitations (see `SYNC_METHOD.md` for full detail)

1. `swing_angle_2.mp4` cannot be synced — it's a different take with no ball-strike event to anchor to. This is a property of the footage, not a gap in the method.
2. The error bar is ±1 video frame (±33 ms). The trajectory-fit math alone resolves contact tighter (~±0.2 frames), but unknown rolling-shutter/exposure effects keep the honest bound at a full frame.
3. A secondary cross-correlation check lands 3 frames away from the chosen offset instead of agreeing within ±1 frame — explained (not hidden) as an expected bias from the two signals peaking at physically different instants, with an independent check (the trajectory kink) confirming the chosen offset instead.
4. The 416 Hz sample rate is an assumption, not a measurement, for the reason above.

## Reproduce it

```bash
uv sync
uv run python scripts/probe_videos.py    # measure the videos
uv run python scripts/run_sync.py        # detect events, align, write synced CSVs
uv run python scripts/verify_sync.py     # write before/during/after verification panels
```
