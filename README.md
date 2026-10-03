# Hack For Humanity — IMU Racket → 3D Overlay on Video

Reconstruct a tennis swing from raw IMU (416 Hz) and draw a **sensor-derived
path onto the judge's 8 s slow-motion video** using **path-projection** — no
computer-vision tracking, no ML, and only rough time sync (not pixel-perfect).

The persuasive point for the judges: the curve is computed from the IMU sensor,
not inferred by an AI from the video.

## What it does

- `scripts/verify_fusion.py` — reads `data/raw_data.csv`, reconstructs the
  racket swing (orientation via gyro integration + complementary filter,
  racket-head path via the pivot model, sensor path via rest-to-rest double
  integration, ball-contact impact detection) and writes the shared contract
  `data/outputs/swing.npz` plus the headline metrics.
- `app.py` — a Streamlit dashboard that overlays the projected IMU path on the
  slow-motion video (animated dot at the impact moment + impact marker), a
  free-rotation Plotly 3D view, signal plots, and a metrics panel.  A
  **side-by-side fallback** (video next to the 3D view, no calibration) is
  included as a safety net.

## Project

Reconstruct a 3D representation of a tennis shot (swing) from raw IMU sensor
data alone — accelerometer + gyroscope readings, no cameras or pre-labeled
pose data.

- `data/raw_data.csv` — raw IMU samples (`ax,ay,az,gx,gy,gz`): 3-axis
  accelerometer + 3-axis gyroscope, 400 samples at a 416Hz sampling rate.
- `data/raw_data_visualized.png` — a plotted view of that signal.
- `data/video/*.mp4` — reference footage of the actual tennis shots, for
  visual ground-truth comparison.

See `pre-hackathon.md` for the full technical plan (pipeline, model choices,
execution timeline).

## Development setup

This project uses [`uv`](https://docs.astral.sh/uv/).


```bash
# 1. Install uv, then install the locked environment
uv sync

# 2. Run Python inside the project environment
uv run python scripts/verify_fusion.py
```

Alternatively, activate the venv (`.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` on macOS/Linux) and run `python ...` directly.

## Single-command run (judge demo)

```bash
uv run python scripts/verify_fusion.py   # regenerate swing.npz (person A)
uv run streamlit run app.py              # open the dashboard (person B)
```

The dashboard opens in the browser. Use the sidebar to:

- pick the camera (angle 1 or 2),
- adjust the rough sync (`scale` ≈ 8.33, `offset`) until the dot lands at the
  swing moment,
- nudge the projected path onto the racket (yaw / elevation / zoom / pan),
- toggle path / animated dot / impact marker layers,
- toggle the **side-by-side** fallback (video next to the 3D view) if the
  calibration is ever off.

## Reproduce the contract from scratch

```bash
uv run python scripts/verify_fusion.py
uv run python scripts/mock_swing.py   # optional: synthetic swing for dev
```

## Project layout

```
data/raw_data.csv            # raw IMU, 400 samples x 6 channels @ 416 Hz
data/outputs/swing.npz       # shared contract (swing dataclass), easy to reload
data/video/*.mp4             # 2 slow-motion camera angles
src/
  config.py                  # fs, units, racket geometry, filter/mode settings
  load.py                    # CSV -> DataFrame with time axis t = index/416
  swing.py                   # Swing dataclass (the A<->B contract) + save/load
  fusion.py                  # gravity ref, orientation, pivot tip, rest-to-rest COG, impact
  metrics.py                 # peak head speed (w*r), g-force, RPM, swing duration
  video.py                   # OpenCV frame/metadata I/O + rough imu->video map
  camera.py                  # solvePnP calibration, project_path, draw_overlay, nudge
  plot.py                    # Plotly 3D trajectory + 6-channel signals
app.py                       # Streamlit dashboard (person B)
scripts/verify_fusion.py     # person A deliverable: sanity check + writes swing.npz
scripts/mock_swing.py        # synthetic swing for parallel dev / fallback
```

## Notes

- **Slow motion**: the IMU is *not* re-sampled; the Video frame index is
  `frame = (t_imu × 8.33 + offset) × fps`.  Rough sync is enough — the path is
  a static shape, only the dot moves with time, so a few frames of sync error
  still looks correct (path-projection strategy).
- **Calibration**: the camera is calibrated from one frame by clicking ≥4
  non-collinear racket points (`cv2.solvePnP`).  If that is degenerate, use the
  sidebar sliders to nudge alignment — no calibration board required.
- Intrinsics are estimated from an assumed FOV (~60°) and refined with sliders.

See `PLAN.md` for the full plan, work breakdown and the "Definition of Done".
