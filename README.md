# Hack For Humanity — Tennishot

Reconstruct a tennis swing from raw IMU (416 Hz) and draw a **sensor-derived
path onto the judge's 8 s slow-motion video** using **path-projection** — no
computer-vision tracking, no ML, and only rough time sync (not pixel-perfect).

The persuasive point for the judges: the curve is computed from the IMU sensor,
not inferred by an AI from the video.

## Project pipelie
**Sensor fusion** — Sync data
- Estimate gravity reference, remove gravity + string-ringing noise.
- Integrate gyro into orientation (quaternions) with complementary filtering.
- Compute racket-head path via the **pivot model** and sensor path via**rest-to-rest** double integration.
- Detect the impact (ball contact).
- Output the shared **`Swing` contract** (`swing.npz`).
**Overlay** — draw the IMU path onto video:
- Read video metadata/frames (OpenCV), calibrate camera from one frame (`solvePnP`).
- Rough IMU↔video sync (offset + scale, or exact `frame_index` table).
- Project the 3D path and animate a dot along it, marking impact.
**Dashboard (app)** — Streamlit UI: play/scrub with the path overlaid, a free-rotation 3D view, signal plots, and a metrics panel (head speed, g-force, RPM). Includes a side-by-side fallback.

## Single-command run (judge demo)

```bash
uv run python scripts/verify_fusion.py   # regenerate swing.npz 
uv run streamlit run app.py              # open the dashboard 
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
app.py                       # Streamlit dashboard ()
scripts/verify_fusion.py     # sanity check + writes swing.npz
scripts/mock_swing.py        # synthetic swing for parallel dev / fallback
```