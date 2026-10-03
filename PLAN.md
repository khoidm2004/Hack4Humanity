# Hack4Humanity — MVP Plan: IMU Racket → 3D Overlay on Video

> Version 3.0 · 2 people · Stack: Streamlit + Plotly + NumPy/SciPy + opencv-python-headless
> Goal: reconstruct a tennis swing from raw IMU (416 Hz) and **draw a sensor-derived path onto the judge's 8 s slow-motion video** using **path-projection** — **no CV tracking**, no ML, and **only rough time sync** (not pixel-perfect).

## 1. Context & Goal

A browser app: the slow-motion swing video plus a **colored curve projected onto the video** tracing the true racket-head trajectory, with a dot animating along it at the right moment, next to a few telemetry numbers (head speed, force, impact moment). The persuasive point for judges: **the curve is computed from the IMU sensor, not inferred by AI from the video.**

**Overlay style = Path-projection** (robust, barely depends on sync). We do *not* force the curve to stick to the moving racket frame-by-frame.

## 2. Confirmed Data Facts

| Fact | Value | Design consequence |
|---|---|---|
| `data/raw_data.csv` | 400 samples × 6 channels (ax..gz), no nulls | a single swing |
| Sampling rate | 416 Hz → dt ≈ 0.0024 s | total ≈ **0.96 s of real time** |
| Accel units | ≈ **m/s²** (mean magnitude ~9.27 ≈ g) | gravity must be removed |
| Gyro units | **deg/s** (peak ~2906) | integrate → quaternion |
| Video | 2 camera angles, 8 s **slow-motion** ≈ 8.33× | rough time mapping, do not re-sample IMU |
| CSV ↔ video relation | CSV = the whole swing, video is slowed down | `video_time = imu_time × 8.33 + offset` |
| Scope | Pure visualization + sensor fusion, **no ML** | — |

## 3. ⭐ Sync Strategy — Why rough sync is enough

The 8.33× slow motion amplifies *any* timing error by 8.33×. Forcing the curve to track the racket per-frame means a small mistake makes it visibly float away from the racket. So we sidestep that:

| Mode | Sync needed | Role |
|---|---|---|
| **B. Path-projection** (PRIMARY) | rough only: offset + scale 8.33, soft impact anchor | The curve is a **static shape** projected from sensor data + camera pose → shape stays correct regardless of timing error; sync only drives the dot |
| **C. Side-by-side** (FALLBACK) | essentially none (shared time axis, one click to align starts) | video next to the 3D view — 100% safe |
| **A. Frame-glued tracking** (STRETCH) | pixel-perfect | only if time remains and sync is solid |

**Principle:** the path is static and always correct; only the dot moves with time. A few frames of sync error still looks fine.

## 4. Architecture & Data Flow

```
CSV ──► src/load.py ──► df (time axis, dt=1/416)      [numpy + pandas]
              │
              ▼
        src/fusion.py                                  [numpy + scipy]
   • gravity reference from the rest segment of accel
   • orientation = integrate gyro → quaternion (SciPy Rotation)
     + complementary filter (accel corrects roll/pitch)
   • translation:
       (a) PIVOT MODEL: tip = pivot + R(t)·[0,0,L]        ← primary overlay
       (b) REST-TO-REST: double-integrate gravity-removed accel +
           drift correction (start/end velocity = 0; high-pass 0.5–1 Hz)  ← "from the data"
       (c) impact detection (|gyro| peak / jerk)
              │
              ▼  Swing dataclass → data/outputs/swing.npz  (CONTRACT)
              │
 src/video.py ◄──► src/camera.py
   [opencv]     │        [opencv]
   read frames, │        calibrate from 1 frame (click ≥4 points on racket)
   fps,          │        projectPoints: 3D → pixel
   rough imu→video│     draw_overlay: path curve + animated dot + impact marker
              │
              ▼
 src/plot.py ──────────► app.py ──────────► judge demo
 [plotly]             [streamlit]
 free-rotation 3D      sidebar · tabs · play/scrub · metrics panel
 + signal plots
```

## 5. Shared Contract (the only coupling point)

Exchange file: **`data/outputs/swing.npz`** (produced by the sensor phase, consumed by the overlay phase). Frozen at kickoff.

```python
# src/swing.py
@dataclass
class Swing:
    t: np.ndarray        # (N,) seconds
    quat: np.ndarray     # (N,4) w,x,y,z — racket orientation in world frame
    tip: np.ndarray      # (N,3) racket head in world coords (m)      [pivot model]
    cog: np.ndarray      # (N,3) sensor point in world coords (m)      [rest-to-rest]
    gyro_deg: np.ndarray # (N,3)
    accel: np.ndarray    # (N,3)
    impact_idx: int      # impact sample index

# Minimum function signatures
load_csv(path)                        -> pd.DataFrame      # pandas
reconstruct(df, fs=416)               -> Swing             # scipy
compute_metrics(Swing)                -> dict
calibrate(frame, obj_pts, img_pts)    -> Camera   # opencv: ≥4 clicked points, 1 frame
project_path(Camera, tip)             -> curve2d  # opencv: projectPoints over whole trajectory
draw_overlay(frame, Camera, tip, t_video, opts) -> frame   # path + dot + impact
map_imu_to_video(swing, fps, scale, offset)       -> frame_idx array   # rough
```

## 6. Tech Stack — What Each Library Is For

Only **5 libraries**. Four are already in `requirements.txt`; we add exactly **one**: `opencv-python-headless`.

### 6.1 Library → Responsibility Map

| Library | Status | Used for | Files |
|---|---|---|---|
| **pandas** | ✅ present | read CSV, time column, null checks | `load.py` |
| **NumPy** | ✅ present | 3D arrays, angle integration, rotation math | `fusion.py`, `metrics.py`, `camera.py` |
| **SciPy** | ✅ present | gravity high-pass, quaternion `Rotation`, Savitzky–Golay, integration | `fusion.py` |
| **OpenCV** | ⚠️ to add | frame extraction, 1-frame `solvePnP` calibration, `projectPoints`, drawing | `video.py`, `camera.py` |
| **Plotly** | ✅ present | free-rotation 3D view, 6-channel signal plots | `plot.py` |
| **Streamlit** | ✅ present | app shell: sidebar, tabs, play/scrub, metrics | `app.py` |

### 6.2 Responsibility Details

**pandas — READ & CLEAN**
- `pd.read_csv('data/raw_data.csv')` → DataFrame.
- Add time column `t = index / 416`; verify no NaN/±inf; declare units in `config.py`.

**NumPy — NUMERICS (foundation)**
- `[x,y,z]` vectors, 3×3 rotation matrices, `np.cumsum` for gyro angle accumulation.
- Save/load `swing.npz` (`np.savez` / `np.load`).

**SciPy — SENSOR FUSION (core)**
- `scipy.signal.butter` + `sosfilt` (high-pass 0.5–1 Hz) → strip gravity from accel.
- `scipy.spatial.transform.Rotation` → per-step `from_rotvec(omega*dt)` accumulated into quaternion.
- `scipy.signal.savgol_filter` to smooth noise; integration + rest-to-rest drift correction.

**OpenCV — VIDEO & OVERLAY**
- `cv2.VideoCapture` → fps, `n_frames`, `frame(i)`.
- `cv2.solvePnP` → calibrate **from one frame** using ≥4 clicked racket points + known racket dimensions → camera pose.
- `cv2.projectPoints` → project the entire `tip[:,3]` trajectory into `curve2d`.
- `cv2.polylines`, `cv2.circle`, `cv2.putText` → path, animated dot, impact marker, labels.
- "headless" = static frame reading, no GUI window.

**Plotly — FIGURES**
- `Scatter3d` → racket 3D model + head trajectory (judges can rotate freely).
- `Scatter` → 6 signal channels vs time, with impact highlighted.
- Frames/slider → synchronized replay.

**Streamlit — APP SHELL**
- `st.sidebar`: camera select, align mode (sliders / click-calibrate), layer toggles, play speed, sync offset (rough).
- `st.image` → render the annotated frame (allows exact frame scrubbing, which `st.video` cannot do).
- `st.slider` scrub; `st.tabs` ("Swing + Video" / "Metrics"); `st.metric` telemetry cards.
- `@st.cache_data` → cache fusion results and frame metadata so the demo stays smooth.

## 7. Work Plan — Consolidated Tasks (both halves, phase by phase)

### 📦 Phase 0 — Setup & Shared Contract
- [x] Add `opencv-python-headless` (`uv add`) and commit the updated `uv.lock`
- [x] Create `src/__init__.py` and the package skeleton
- [x] Freeze the `Swing` contract / `data/outputs/swing.npz` schema
- [x] Confirm data facts in `config.py`: fs=416, units (accel m/s², gyro deg/s), racket geometry (L=0.686, handle=0.20)

### 🧠 Phase 1 — Sensor Fusion Core
- [x] `src/config.py`: fs, units, racket geometry, mode flag (pivot / rest-to-rest)
- [x] `src/load.py`: read CSV → DataFrame with time axis `t = index/416` (pandas)
- [x] `src/fusion.py`:
  - [x] Estimate the gravity reference from the rest segment of accel
  - [x] Integrate gyro → quaternion orientation (SciPy Rotation) + complementary filter (accel corrects roll/pitch)
  - [x] Pivot model: `tip = pivot + R·[0,0,L]` (primary overlay curve)
  - [x] Rest-to-rest: gravity-removed double integration + drift correction (data-driven curve)
  - [x] Impact detection (`|gyro|` peak / jerk)
- [x] `src/metrics.py`: peak tip speed (ω×r), peak g-force, RPM, swing duration
- [x] `scripts/verify_fusion.py`: sanity check (smooth orientation, impact found) → writes `swing.npz`
- ✅ **Deliverable:** reproducible `swing.npz` + metrics from a single command

### 🎬 Phase 2 — Video, Overlay & Dashboard
- [ ] `scripts/mock_swing.py`: synthetic 3D trajectory so overlay work can start before real fusion is ready
- [ ] `src/video.py`: read fps + `n_frames`, `frame(i)`, `map_imu_to_video` (rough: scale 8.33 + offset) — OpenCV
- [ ] `src/camera.py`: `calibrate` from 1 frame (≥4 clicked points) → `project_path` → `draw_overlay` (path + dot + impact marker) + slider nudge — OpenCV
- [ ] `src/plot.py`: Plotly 3D racket + trajectory + 6-channel signal panels
- [ ] `app.py`: Streamlit — sidebar (camera, align mode, layers, play speed, sync offset), play/scrub, metrics panel
- [ ] `README.md`: run instructions + judge demo script
- ✅ **Deliverable:** working app (developed in parallel using the mock trajectory)

### 🤝 Phase 3 — Integration & Finalize
- [ ] Plug the real `swing.npz` into the app
- [ ] Tune offset/scale so the dot animates at the right moment (rough sync, not pixel-perfect)
- [ ] Verify the path lands on the racket in one calibrated frame + correct dot timing
- [ ] End-to-end run of the whole pipeline; fix drift / alignment issues
- [ ] Ensure the side-by-side fallback works
- [ ] Demo rehearsal + finalize README

## 8. Important Technical Notes

- **Slow motion**: do not re-sample the IMU. `frame = (t_imu × 8.33 + offset) × fps`. Rough mapping is enough.
- **Path-projection**: calibrate the camera **from one frame** (click ≥4 non-collinear points on the racket). The static curve's shape is always correct; timing error does not distort it.
- **Rest-to-rest**: assume start/end velocity = 0 to suppress drift. High-pass 0.5–1 Hz before double integration; Savitzky–Golay if noisy.
- **Pivot model**: the wrist is nearly fixed during a short swing → `tip = pivot + R·[0,0,L]`, which reliably lands on the racket.
- **solvePnP**: needs ≥4 non-collinear points (butt, throat, head tip, plus points around the head rim). Two collinear points are degenerate → fall back to sliders (yaw/elevation/zoom/pan).
- **Intrinsics**: estimate from an assumed FOV (~60°) and refine with sliders; no calibration board needed.
- **Yaw drift** (no magnetometer): only a slight rotation about the vertical axis over 1 s — acceptable, and it affects translation rather than the head-tip shape in the pivot model.

## 9. Risks & Fallbacks

| Risk | Fallback |
|---|---|
| Path does not land on the racket | slider nudge; re-calibrate from one frame; fall back to side-by-side |
| Sync drift (amplified 8.33× by slow motion) | **Avoided by path-projection** — shape stays correct, only the dot shifts slightly; adjust rough offset |
| Video missing / unexpected fps | read metadata; demo on a blank canvas with the Plotly 3D view still works |
| Heavy integration drift | pivot model is the primary overlay; rest-to-rest only demonstrates the data |
| Running out of time | drop to side-by-side (mode C) — still demo-worthy; prioritise overlay + sync over metrics |

## 10. Definition of Done

- [ ] `python scripts/verify_fusion.py` produces `swing.npz`
- [ ] Streamlit app scrubs/plays with the correct video fps
- [ ] **Sensor-derived path drawn on the video (path-projection) with the dot animating at the swing moment** (main milestone)
- [ ] Impact marked on both the video and the Plotly 3D view
- [ ] Metrics panel: peak speed, g-force, RPM, duration
- [ ] Side-by-side fallback works (safety net)
- [ ] README with a single command to run
- [ ] (Stretch) Frame-glued tracking / MP4 export / static HTML / switch between the 2 cameras