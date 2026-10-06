# Tennishot

A product of the **Hack For Humanity Hackathon**.

## Goal

Reconstruct a **3D representation of a tennis shot (swing)** from raw IMU
sensor data alone — no computer-vision tracking, no ML, no pose estimation.
A single 416 Hz accelerometer + gyroscope recording from a sensor taped to
the racket is enough to recover the racket head's path through space, the
moment of ball contact, and swing metrics like head speed and spin rate.

The curve drawn over the judge's reference video is **computed from the IMU
sensor**, not inferred by an AI from the footage — the video is sync'd
alongside it only as a rough visual check, with path-projection and only
approximate time alignment (not pixel-perfect tracking).

## Tech stack

| area | tool |
|---|---|
| language / env | Python 3.11, [`uv`](https://docs.astral.sh/uv/) for dependency management |
| numerics | NumPy, SciPy (`Rotation`, filtering, `least_squares`) |
| data handling | pandas |
| video / camera | OpenCV (`solvePnP`, frame I/O) |
| visualization | Plotly (3D trajectory, signal plots), Streamlit (dashboard) |
| (scaffolded, not yet in the fusion pipeline) | CatBoost, PyTorch/torchvision, ONNX, scikit-learn — reserved for a future learned model; the shipped pipeline is pure signal processing, no trained model involved |

See `pyproject.toml` for exact versions.

## How the swing is calculated and visualized

All fusion math lives in `src/fusion.py`; the result is a shared `Swing`
contract (`data/outputs/swing.npz`) that both the metrics and the overlay
read from. In short:

1. **Orientation** — integrate the gyroscope into a quaternion at every
   sample (`integrate_orientation`), seeded from a gravity reference
   (`estimate_gravity`).
2. **Racket head position** — the head traces a sphere of fixed radius
   around the wrist: `tip(t) = R(t) · (L · RACKET_LEVER_BODY)`, where the
   lever direction was *measured* from the accelerometer's own centripetal
   signal, not assumed (`pivot_tip`).
3. **Wrist translation** — the wrist itself isn't fixed in space. Its path
   comes from double-integrating the high-pass-filtered acceleration
   (`rest_to_rest_cog`), restoring a linear drift term the integration
   detrend would otherwise discard (`wrist_pivot`). The racket head's
   position in the world is then `pivot + tip` (`Swing.head`).
4. **Impact detection** — the ball-contact instant is found from the peak
   angular speed, refined by the local peak in acceleration jerk
   (`detect_impact`).
5. **Metrics** — head speed, g-force, RPM and swing duration are derived
   from the same signals (`src/metrics.py`).
6. **Visualization** — `src/camera.py` calibrates a camera pose from a
   handful of known racket-head pixels in one reference frame (`solvePnP`),
   then projects the 3D path onto the video and animates a dot along it in
   sync (`src/plot.py` renders the free-rotation 3D view and signal plots);
   `app.py` is the Streamlit dashboard that ties it together.

Full derivations, measured constants, and the evidence behind each modeling
choice are written up in `findings/FUSION_NOTES.md` and the earlier
`findings/*.md` files — this section is the short version.

## Running it locally

```bash
# 1. install dependencies into a local .venv (first time only)
uv sync

# 2. build the swing contract from the raw sensor data
uv run python scripts/verify_fusion.py

# 3. launch the dashboard
uv run streamlit run app.py
```

The dashboard opens in your browser. Use the sidebar to:

- pick the camera angle (1 or 2),
- adjust the rough sync (`scale` ≈ 8, `offset`) until the dot lands at the
  swing moment,
- nudge the projected path onto the racket (yaw / elevation / zoom / pan),
- toggle the path / animated dot / impact marker layers,
- toggle the **side-by-side** fallback (video next to the 3D view) if the
  calibration is ever off.

To regenerate `swing.npz` from scratch after changing the fusion code, or to
produce a synthetic swing for development without the real sensor data:

```bash
uv run python scripts/verify_fusion.py
uv run python scripts/mock_swing.py   # optional: synthetic swing for dev
```

## Project layout

```
data/raw_data.csv            # raw IMU, 400 samples x 6 channels @ 416 Hz
data/outputs/swing.npz       # shared contract (Swing dataclass), easy to reload
data/video/*.mp4             # 2 slow-motion camera angles
src/
  config.py                  # fs, units, racket geometry, filter/mode settings
  load.py                    # CSV -> DataFrame with time axis t = index/416
  swing.py                   # Swing dataclass (the fusion<->overlay contract) + save/load
  fusion.py                  # gravity ref, orientation, pivot tip, wrist path, impact
  metrics.py                 # peak head speed (w*r), g-force, RPM, swing duration
  video.py                   # OpenCV frame/metadata I/O + rough imu->video map
  camera.py                  # solvePnP calibration, project_path, draw_overlay, nudge
  overlay_calib.py           # committed pixel annotations + pose solving
  plot.py                    # Plotly 3D trajectory + 6-channel signals
app.py                       # Streamlit dashboard
scripts/verify_fusion.py     # sanity check + writes swing.npz
scripts/mock_swing.py        # synthetic swing for parallel dev / fallback
findings/                    # investigative write-ups and measured evidence
```
