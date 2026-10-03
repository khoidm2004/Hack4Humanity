# IMU / video synchronisation

How `data/raw_data.csv` was aligned to the footage in `data/video/`, so the
result can be reproduced and audited.

```
uv sync
uv run python scripts/probe_videos.py    # measure the videos
uv run python scripts/run_sync.py        # detect events, align, write synced CSVs
uv run python scripts/verify_sync.py     # write the before/during/after panels
```

All numbers below are produced by `scripts/run_sync.py` and written in
machine-readable form to `Artifacts/sync_evidence/sync_result.json`. The plots
and frames cited are in `Artifacts/sync_evidence/`.

## Result in one line

| video | syncable | offset (CSV -> video) | error bar |
|---|---|---|---|
| `swing_angle_1.mp4` | **yes** | **+3.9093 s** (+117.28 frames at 30 fps) | **±1 frame = ±33.3 ms = ±14 CSV samples** |
| `swing_angle_2.mp4` | **no** — different take, contains no ball strike | — | — |

The mapping is a pure offset, with no drift or scale term:

```
t_video = index / 416.0 + offset
frame   = nearest frame in the video's per-frame PTS array to t_video
```

Applied to every row in `data/synced_data_angle_1.csv` and
`data/synced_data_angle_2.csv`.

## The three questions that had to be answered first

**1. Is either video slow motion? No — both are real time.**
Both decode at a measured 30.000 fps (from the median inter-frame PTS diff, not
from a metadata field). Scanning a time-scale factor through the
IMU/video cross-correlation puts the best scale at 0.84x with a plateau of
0.70–1.04x; 1.0 is inside that plateau and 2x-or-greater time expansion is
decisively excluded. See "Rate check" below for what that test can and cannot
resolve.

**2. Are the two videos the same take? No — they are different takes.**
This matters, and the evidence is unambiguous:

* `swing_angle_1.mp4` shows the player standing with his arms out, a ball
  arriving low, and **one forehand drive** — racquet sweeping through at frames
  128–135, ball leaving fast to the right.
* `swing_angle_2.mp4` shows the player from behind for all 246 frames
  **bouncing and tapping a ball with the racquet** and then walking away. There
  is no swing and no ball strike anywhere in it.

`scripts/probe_videos.py` writes a numbered contact sheet of **every** frame of
each clip to `Artifacts/sync_evidence/probe/contactsheet_swing_angle_{1,2}.png`;
one look at each settles it. The two camera positions
are also on opposite sides of the player with different furniture in shot, so
these are two separate recordings from the same session, not two cameras on one
take.

Measured, not just eyeballed: the ball's image speed in angle 2 goes from
7.0 px/frame to 0.2 px/frame across its largest speed change, and the two-line
fit to its trajectory (x and y independently) disagrees by 170 frames — there is
no kink to find, because nothing is struck. The CSV is a single impact-centred
window around a forehand, so there is no event in angle 2 to align it to.
**`swing_angle_2.mp4` therefore cannot be synchronised to this CSV.** That is a
property of the data, not a failure of the method.

**3. Is there a visually identifiable sync event? Yes, in angle 1.**
Ball-racquet contact is bracketed to frames 130–131 by eye
(`Artifacts/sync_evidence/contact/angle_1_contact_f*.png`) and refined to frame
**130.70** by the ball-trajectory fit below. At 30 fps the honest bound is
**±1 frame**.

## The CSV side

`data/raw_data.csv` has no timestamp column, so the time base is
`t_sample = index / 416.0` (416 Hz from `CLAUDE.md`/`TASK.md` — an assumption,
see "Rate check"). 400 samples = 0.9615 s.

**The capture is an impact-triggered window, not a free-running recording.** The
impact sits at sample 200 of 400, i.e. dead centre, ±200 samples (±0.48 s) around
the trigger. This is why a 0.96 s CSV maps into the middle of a 7.6 s video:
the offset places a short window inside a long one. Expected, not a bug.

**Impact index: 200 (t = 0.4808 s), from two independent detectors that agree
exactly.**

* *Primary — gyro jerk.* `|diff(gyro)|` spikes when the racquet is struck and
  its angular rate reverses within two samples. Peak, refined back to onset.
* *Confirming — high-frequency accelerometer energy.* The racquet frame rings at
  ~100 Hz after contact. A causal 50 Hz Butterworth high-pass, vector magnitude,
  first crossing of 20 % of peak.

Both return sample 200. Neither is hardcoded; `run_sync.py` asserts the result
lands within 5 samples of the 200 visible in `data/raw_data_visualized.png`.

Two detector subtleties worth recording, because both initially moved the answer
by 25 samples:

* The confirming detector must use **only non-saturated accelerometer channels**
  (`ay`, `az` here). `ax` is clipped, and the high-pass turns the hard corner
  where it leaves the rail into a false onset ~25 samples early.
* It must use a **causal** filter. `sosfiltfilt` is zero-phase but non-causal and
  leaks the impact's energy backwards in time; and a Hilbert "envelope" of an
  already-rectified magnitude is not meaningful and smears the onset further.

**Motion onset: sample 54 (t = 0.1298 s)**, where gyro magnitude leaves its
resting baseline. Swing duration onset -> impact = 0.3510 s.

**Saturation.** `ax` is **clipped at +15.962 g** for 18 consecutive samples,
indices 182–199 — the accelerometer's ±16 g rail, not signal. This is why the
cross-correlation uses gyro rather than accelerometer. No gyroscope channel is
saturated: the extremes reached are `gx` +1231.8 / −501.3, `gy` +543.6 /
−1285.7, `gz` +1228.3 / −565.9 °/s, and no value repeats more than three times
anywhere in the file, so there is no rail.

## The video side

Measured by decoding every frame, never by trusting container metadata
(`scripts/probe_videos.py`, report in `Artifacts/sync_evidence/probe_report.json`):

| | `swing_angle_1.mp4` | `swing_angle_2.mp4` |
|---|---|---|
| resolution | 1280 x 720 | 1280 x 720 |
| `CAP_PROP_FRAME_COUNT` | 226 | 246 |
| frames actually decoded | **226** | **246** |
| `CAP_PROP_FPS` | 29.8678 | 30.0 |
| fps used (median PTS diff) | **30.000** | **30.000** |
| duration | 7.567 s | 8.200 s |
| CFR / VFR | **near-CFR: 1 irregular interval of 225** | **CFR: 0 irregular of 245** |

`swing_angle_1.mp4` **drops one frame** — a single 66.67 ms interval between
frames 99 and 100 — which is exactly why `CAP_PROP_FPS` reports 29.8678 (the
average over the gap) rather than 30. Consequently `t = i / fps` is wrong for
every frame after 99, and the per-frame PTS array is carried and interpolated
instead. The dropped frame is before the swing, so it does not sit between the
motion onset and contact, but the mapping does not rely on that.

Two implementation details that cost real accuracy if got wrong:

* `CAP_PROP_POS_MSEC` must be read **after** `cap.read()`, not before. Read
  before, it returns the previous frame's value and produces a duplicate
  timestamp at index 1.
* Frame-time convention: frame `i` is taken to represent the instant at its PTS,
  and CSV rows are mapped to the **nearest** frame. Rows falling outside the
  decoded video would get `frame_index = -1`; for angle 1 none do.

### Motion energy

Per-frame motion energy is the **fraction of pixels differing from the previous
frame by more than 40 grey levels** (after downscaling to 320 px wide and a
Gaussian blur), not the usual mean absolute difference. Mean absolute difference
does not work on this footage: both clips are handheld and H.264-encoded, and the
mean is dominated by whole-frame camera shake plus a quantisation step at every
GOP keyframe, which appears as a spurious peak **every 12 frames** in both files
and buries the swing. Counting only large-amplitude pixel changes rejects both.
Global-translation compensation by phase correlation was also tried and moved the
answer by less than one frame, so it is not carried.

### Contact frame, to sub-frame precision

The ball is segmented in HSV and tracked across the swing
(`Artifacts/sync_evidence/ball_track_angle_1.png`). Its image trajectory is two
straight lines meeting at contact: slow before (2.2 px/frame), fast after
(35.6 px/frame). Fitting a line to each side of the speed jump and intersecting
them gives the contact instant between frames:

* from `x(frame)`: **130.80**
* from `y(frame)`: **130.61**
* the two independent fits agree to **0.20 frames**

**Contact frame = 130.70, t_video = 4.3901 s.** This is consistent with, and
tighter than, the visual read of frames 130–131.

## Offset, and the checks on it

**Primary estimate — event matching.**

```
offset = t_contact_video - t_impact_csv
       = 4.3901 s        - 0.4808 s      = +3.9093 s   (+117.28 frames)
```

**Cross-correlation check — DISAGREES by 3.0 frames, and here is why.**
IMU gyro magnitude (not accelerometer: `ax` is clipped) low-passed to the video's
Nyquist, resampled onto the video frame grid, z-scored against the video motion
energy, offset scanned at quarter-frame resolution:

* best correlation **r = 0.902** at offset **+4.0093 s**
* that is **+3.00 frames** from the event-based offset — outside the ±1 frame
  criterion.

This is reported rather than averaged away, because the cause is measurable and
is not an error in the event alignment. The two signals peak at *different
physical instants*:

* the video motion energy peaks **+4.30 frames after** contact (the ball is
  fastest and the racquet sweeps widest just after it leaves the strings);
* the IMU gyro magnitude peaks **0.43 frames before** the impact sample.

So a correlation that lines those two peaks up is expected to sit about
**+4.73 frames** late. The observed disagreement is +3.00 frames — the same sign,
the same order, and smaller than the predicted bias. In other words the
cross-correlation is consistent with the event-based offset once the known
feature mismatch is accounted for, and a 30 fps changed-pixel signal correlated
against an impact-dominated gyro signal simply cannot resolve this to ±1 frame.

**The offset is therefore the event-based one. The two were not averaged.**

**Rate check (is a scale term needed? No).**
No drift or scale parameter is fitted: over 0.96 s, any plausible clock drift is
far below one frame, and fitting a second parameter on a short window with one
event would absorb noise rather than clock error. Instead a scale factor is
*scanned* through the cross-correlation, re-optimising the offset at each scale:

* best scale **0.84x**, with a plateau (within 0.02 correlation of the best) of
  **0.70–1.04x**.

1.0 lies inside that plateau and slow motion is excluded, so the offset-only
model stands. What this test can resolve is honestly ±~17 %: it decisively
answers "is this real-time footage at 30 fps and 416 Hz, or is one of those
grossly wrong", and it does not and cannot measure a few percent of drift.

A duration-based variant of this check — comparing the width of the motion burst
on each side — was tried and dropped as invalid: the IMU capture is an
impact-centred window that **ends while the racquet is still ringing**, so its
burst is truncated by the window edge and its width is not comparable to the
video's.

**The 416 Hz figure is an assumption, not a measurement.** It is stated in
`TASK.md`/`CLAUDE.md` and there is no timestamp column to check it against. The
scale scan above is the only test applied to it, and it is consistent with
416 Hz; it would not catch a 5 % error. Treat a future rate disagreement as a
finding, not something to silently correct for.

## Output files

`data/synced_data_angle_1.csv` and `data/synced_data_angle_2.csv`: the original
seven columns plus

| column | meaning |
|---|---|
| `t_sample` | `index / 416.0`, the CSV-side time in seconds |
| `t_video` | `t_sample + offset`, the time on that video's clock |
| `frame_exact` | fractional frame index, interpolated on the PTS array |
| `frame_index` | nearest integer frame, or `-1` if outside the video |

**All 400 rows are kept in both files.** Nothing is dropped or clipped.

* angle 1: all 400 rows land inside the video, on frames **116–145**.
  `frame_index` is monotonically non-decreasing.
* angle 2: **all 400 rows carry `frame_index = -1` and empty `t_video` /
  `frame_exact`**, because no offset exists for that video (different take). The
  file is present so the deliverable set is complete and the status is explicit
  in the data itself — it is not an alignment.

## Verification

`scripts/verify_sync.py` writes five panels to
`Artifacts/sync_evidence/verify/`, each with the video frame on the left and all
six IMU traces on the right carrying a cursor and dot at the sample that maps to
that frame.

| CSV sample | frame | what the frame shows | what the traces show |
|---|---|---|---|
| 0 | 116 | ball approaching low, racquet at rest by his side | flat resting baseline |
| 100 | 123 | racquet taken back behind the hip | `gx`/`gz` at their most negative (backswing) |
| 200 | 131 | **ball on the strings** | `ax` dropping off the 16 g rail, `gy` jumping from −1250 to ~+100, `ay`/`az` ringing begins **exactly at the cursor** |
| 225 | 133 | ball leaving to the right | ringdown at full amplitude |
| 399 | 145 | follow-through, ball out of shot | ringdown decayed |

This matches `data/raw_data_visualized.png` feature for feature: the plot's
flat-topped `ax` region, its `gy` discontinuity and the onset of `ay`/`az`
ringing all land on the frames where the ball is actually on the strings.

**One interpretation flagged.** The acceptance criterion asks for "a dot from the
synced data on a few sample frames". There is no 3D reconstruction yet, so there
is no physical position in the image at which to place a dot — that
reconstruction is the next task and depends on this one. The dot is therefore
placed on the IMU traces beside the frame rather than on the frame itself, which
is what can actually be checked today. If what was wanted is a dot drawn *on* the
video, that needs the reconstruction first.

## Known limitations

1. **`swing_angle_2.mp4` is unsynchronised**, and cannot be synchronised to this
   CSV — it is a different take with no impact event in it.
2. The ±1 frame error bar is the real bound. The ball-trajectory fit locates
   contact to ~±0.2 frames, but exposure time and rolling shutter are unknown and
   can bias a motion-blurred centroid by up to half a frame, so the quoted bar
   stays at one frame (±33 ms, ±14 CSV samples).
3. The cross-correlation does not independently confirm the offset to ±1 frame
   (see above). The independent confirmation that *does* hold is the ball
   trajectory kink, which agrees with the visual frame bracket.
4. `SYNC_METHOD.md` is hand-written; the authoritative numbers are regenerated
   into `Artifacts/sync_evidence/sync_result.json` on every `run_sync.py`.
