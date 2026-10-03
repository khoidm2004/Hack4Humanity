# Hack For Humanity — Tennishot

Reconstruct a tennis swing from raw IMU (416 Hz) and draw a **sensor-derived
path onto the judge's 8 s slow-motion video** using **path-projection** — no
computer-vision tracking, no ML, and only rough time sync (not pixel-perfect).

The persuasive point for the judges: the curve is computed from the IMU sensor,
not inferred by an AI from the video.

## Project pipeline
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

## Physics & computations

All sensor-fusion math lives in **`src/fusion.py`** (results are wrapped by
`src/metrics.py`). Raw data: 400 samples of `ax,ay,az,gx,gy,gz` at 416 Hz
(`dt = 1/416 s`), accel in m/s², gyro in deg/s.

### 1. Gravity reference — `estimate_gravity`
```
g_ref = mean( accel[0:GRAVITY_REF_SAMPLES] )     # steady-ish window
g     = g_ref / ||g_ref||  *  9.80665             # normalise to |g|
```

### 2. Orientation from gyro — `integrate_orientation`
```
ω[t] = deg2rad( gx,gy,gz )[t]          # deg/s -> rad/s
θ[t] = ||ω[t]|| · dt                    # angle swept in one sample
dq[t] = [cos(θ/2), (ω/||ω||)·sin(θ/2)]  # delta quaternion
R[t]  = R[t-1] ⊗ dq[t]                  # accumulate in world frame (quat product)
q[t]  = quat(R[t])                      # stored as (w, x, y, z)
```
An optional complementary filter nudges roll/pitch toward the accel-measured
gravity direction (weight `alpha`; default `0.0` = pure gyro integration).

### 3. Racket-head path (pivot model) — `pivot_tip`
```
tip(t) = R(t) · [0, 0, L] ,   L = RACKET_TIP_LEN = 0.686 m
```
The wrist is assumed nearly fixed, so the head tip traces the surface of a
sphere of radius `L` around the wrist — the primary overlay curve.

### 4. Sensor path (rest-to-rest) — `rest_to_rest_cog`
```
a_world = R(t) · a_sensor                      # rotate accel to world frame
a_lin   = HP(a_world)                          # high-pass (~0.8 Hz) removes gravity
v[t]    = Σ a_lin[i]·dt  ,  detrend → v[0]≈v[N]≈0
p[t]    = Σ v[i]·dt       ,  detrend → path centred at origin
```

### 5. Impact (ball contact) — `detect_impact`
```
t_gyro = argmax ||gx,gy,gz||          # peak angular speed
jerk   = ||Δaccel / dt||              # magnitude of accel jerk
impact = argmax(jerk) in a ±12-sample window around t_gyro
```

### 6. Metrics — `compute_metrics`
```
v_head   = ||ω|| · L            m/s        % peak racket-head speed  (ω in rad/s)
g_force  = ||a|| / 9.80665                 % peak specific force
RPM      = ||ω|| · 60 / (2π)
duration = t[impact] − t[swing onset]
```
