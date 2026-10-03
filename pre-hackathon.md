# Pre-Hackathon Plan — Sports Telemetry & Computer Vision

> **Status:** PRE-HACKATHON / BEFORE OFFICIAL SPECIFICATION  
> **Time budget:** ~8 hours  
> **Goal:** Build a fast, reliable MVP for IMU-based 3D swing reconstruction.  
> **Important:** This plan is intentionally written before the detailed challenge specification, dataset schema, target format, evaluation metric, and hardware are known.

---

## 0. Working Interpretation of the Challenge

Spoilered challenge:

> **Sports Telemetry & Computer Vision**  
> Tackle real hardware and data bottlenecks in athletic performance. You’ll be working with raw IMU sensor data to reconstruct complex 3D swings from motion data alone.

Current interpretation:

```text
Raw IMU
   ↓
Cleaning / calibration / preprocessing
   ↓
Temporal sequence
   ↓
ML model
   ↓
3D motion / trajectory / pose
   ↓
Visualization + sports telemetry
```

The exact target is unknown.

Possible targets:

- 3D trajectory
- 3D orientation
- equipment trajectory
- limb/joint pose
- combination of position + orientation

**Do not lock the architecture until the target is known.**

---

# 1. First Principle: 8 Hours Means MVP First

Priority:

1. Correct data pipeline
2. Working model
3. Working reconstruction
4. Working streaming inference
5. Visual demo
6. Deployment optimization
7. Extra polish

Do not spend hours on:

- complex backend
- frontend framework
- distributed systems
- MLOps
- large model experimentation
- sophisticated sensor-fusion frameworks
- TensorRT before the model works

The ideal result is:

```text
IMU → clean → window → model → 3D swing → demo
```

working end-to-end.

---

# 2. What We Need to Discover Immediately

When the official challenge/data is available, spend the first ~15–20 minutes inspecting it.

## Data

- [ ] File format
- [ ] Number of IMUs
- [ ] Sensor placement
- [ ] Sampling rate
- [ ] Timestamp format
- [ ] Accelerometer
- [ ] Gyroscope
- [ ] Magnetometer
- [ ] Quaternion/orientation
- [ ] Missing values
- [ ] Ground-truth 3D data
- [ ] Sequence boundaries

## Target

Determine exactly what must be reconstructed:

- [ ] XYZ trajectory?
- [ ] Orientation?
- [ ] Full 3D pose?
- [ ] Equipment trajectory?
- [ ] Joint positions?
- [ ] Multiple outputs?

## Evaluation

- [ ] RMSE
- [ ] MAE
- [ ] angular error
- [ ] trajectory error
- [ ] FPS
- [ ] latency
- [ ] memory
- [ ] hardware-specific metric

## Hardware

- [ ] Device
- [ ] CPU
- [ ] GPU
- [ ] RAM
- [ ] OS
- [ ] CUDA
- [ ] TensorRT
- [ ] PyTorch support
- [ ] How IMU data reaches the device

## Demo

- [ ] Offline replay?
- [ ] Live sensor?
- [ ] 3D visualization?
- [ ] Computer-vision visualization?
- [ ] Edge deployment?
- [ ] API/UI requirement?

---

# 3. IMU Data Handling

## 3.1 Preferred Signals

If quaternion/orientation is provided and appears trustworthy:

**Use it.**

But do not automatically throw away raw IMU.

Preferred input when available:

```text
accelerometer:
ax ay az

gyroscope:
gx gy gz

orientation:
qx qy qz qw
```

Possible feature shape:

```text
T × 10
```

If quaternion is unavailable:

```text
T × 6
```

for accel + gyro.

If magnetometer is available:

```text
T × 9
```

or combine it with the other signals as appropriate.

The internal model interface should remain:

```text
[B, T, F]
```

where `F` is configurable.

---

# 4. Quaternion Handling

If quaternion exists:

- [ ] Check quaternion magnitude
- [ ] Normalize if necessary
- [ ] Check temporal sign flips
- [ ] Keep a consistent quaternion convention

Important:

```text
q and -q
```

represent the same rotation.

For consecutive samples:

```python
if dot(q[t], q[t-1]) < 0:
    q[t] = -q[t]
```

This prevents artificial discontinuities that can confuse a temporal model.

Do not implement a complicated orientation estimator if a reliable quaternion is already supplied.

---

# 5. Data Cleaning

Keep this simple.

## Timestamp

Handle with:

- **pandas** — inspection, sorting, missing timestamps
- **NumPy** — numerical processing

Tasks:

- [ ] Sort by timestamp
- [ ] Detect duplicate timestamps
- [ ] Estimate sampling interval
- [ ] Detect large gaps
- [ ] Split sequences around major gaps

---

## Missing Values

Handle with:

- **pandas**
- **NumPy**

Rules:

```text
small gap → interpolation
large gap → split/discard sequence
invalid sequence → discard
```

Do not spend time building a complicated imputation model.

---

## Invalid Values

Check:

```text
NaN
+inf
-inf
```

and obvious physically impossible values.

Handle with:

- **NumPy**
- simple clipping/filtering where justified

---

## Normalization

Use:

- **scikit-learn StandardScaler**

Fit only on training data.

Save the scaler and reuse exactly the same scaler during inference.

```text
train → fit scaler
validation/test → transform only
inference → transform only
```

---

# 6. Coordinate Systems

This is one of the most important unknowns.

Determine whether the target is:

- sensor/local coordinates
- body coordinates
- world/global coordinates
- relative trajectory
- absolute position

Do not assume that `XYZ` means global world position.

If the task requires absolute position, expect drift and harder reconstruction.

If it only requires relative 3D swing shape, the problem may be considerably easier.

---

# 7. Sequence Construction

Use the simplest practical streaming pattern:

```text
sliding window + ring buffer
```

Example:

```text
window = 128 samples
stride = 32 samples
```

but choose the actual values after checking sampling rate and swing duration.

Concept:

```text
0 ───────────── 128
       32 ───────────── 160
              64 ───────────── 192
```

Training:

```text
raw sequence
    ↓
sliding windows
    ↓
[B, T, F]
```

Inference:

```text
incoming IMU
     ↓
ring buffer
     ↓
last T samples
     ↓
model
```

Implementation:

- **Python `deque`** for simplest MVP
- NumPy circular buffer if performance requires it

No need for a complicated streaming architecture.

---

# 8. Baseline Model

Before committing to an LSTM, establish a fast baseline if time allows.

## CatBoost baseline

Use:

- **CatBoost**

Convert each window into lightweight statistical/motion features:

```text
mean
std
min
max
range
energy
magnitude
first value
last value
delta
```

For example:

```text
acceleration magnitude
gyro magnitude
mean ax
std ax
max gyro
...
```

Purpose:

- sanity check
- determine whether the dataset contains useful signal
- provide a fast baseline
- give a reference against the sequence model

This is not necessarily the final model.

---

# 9. Main Model Candidate: Small LSTM

LSTM is a reasonable first sequence model.

Input:

```text
[B, T, F]
```

Basic architecture:

```text
Input
  ↓
LSTM
  ↓
Dropout
  ↓
Linear
  ↓
Target
```

Start small:

```text
hidden_size = 64 or 128
layers = 1–2
```

Do not start with a large recurrent network.

---

## Training

Use:

- **PyTorch**
- Adam
- learning rate around `1e-3` as initial baseline
- early stopping

Loss depends on target.

For XYZ regression:

```text
MSELoss
```

or:

```text
SmoothL1Loss
```

if outliers are problematic.

---

# 10. LSTM vs CatBoost

Expected behavior:

### CatBoost

Usually:

- faster to train
- very fast inference
- excellent on engineered tabular features
- does not naturally understand sequence order

### LSTM

Usually:

- more training computation
- naturally models temporal structure
- better suited to raw sequential IMU input
- inference can still be very fast when the model is small

Training time is not automatically a problem.

A small LSTM on a modest dataset can train quickly.

The main factors are:

```text
number of sequences
×
sequence length
×
hidden size
×
number of layers
×
epochs
```

Therefore:

**Do not assume CatBoost is automatically the final choice.**

Use CatBoost as a quick sanity baseline and LSTM as the first serious temporal candidate.

---

# 11. Other Models — Only If Needed

Do not build all alternatives.

Priority:

```text
1. Small LSTM
2. GRU
3. TCN
4. Small Transformer
```

Only investigate another model if:

- LSTM performs poorly
- training is unexpectedly slow
- inference latency is problematic
- the dataset structure strongly suggests another architecture

With an 8-hour limit, one good sequence model is better than four unfinished models.

---

# 12. 3D Reconstruction

After prediction:

```text
model output
    ↓
denormalization
    ↓
optional smoothing
    ↓
3D visualization
```

Possible lightweight smoothing:

- **SciPy Savitzky-Golay filter**
- simple exponential moving average

Do not over-smooth fast athletic motion.

If the target is orientation rather than XYZ, convert/render it appropriately instead of forcing everything into XYZ.

---

# 13. Streaming Inference

MVP architecture:

```text
IMU
 ↓
Ring Buffer
 ↓
Window Ready?
 ↓
Preprocessing
 ↓
Model
 ↓
3D Prediction
 ↓
Visualization
```

Measure:

- preprocessing latency
- model latency
- total latency
- inference frequency/FPS

The same preprocessing used during training must be used during inference.

---

# 14. Why Sliding Window + Buffer Is Still Fine

Do not waste hackathon time looking for a replacement just because it sounds old.

The following remains a perfectly valid streaming pattern:

```text
ring buffer
+
fixed temporal context
+
causal/streaming model
```

Possible later improvements include:

- stateful RNN
- causal TCN
- overlapping window aggregation
- temporal smoothing
- incremental inference

But the MVP should start with:

```text
deque → fixed window → inference
```

It is simple, debuggable, and sufficient for an 8-hour prototype.

---

# 15. Visualization / Demo

Assumption:

The company may expect something similar to a pose/3D-motion demo.

The goal is not a beautiful frontend.

The goal is:

```text
IMU sequence
     ↓
3D swing reconstruction
     ↓
visually obvious result
```

Fastest options:

- **Plotly**
- **Matplotlib 3D**

If an interactive UI is useful:

- **Streamlit + Plotly**

Possible demo:

```text
┌───────────────────────────────────┐
│       3D Swing Reconstruction     │
│                                   │
│             ●                     │
│           /                       │
│         ●                         │
│       /                           │
│     ●                             │
│                                   │
├───────────────────────────────────┤
│ Swing duration: XX s              │
│ Peak velocity: XX                 │
│ Peak acceleration: XX             │
│ Inference latency: XX ms          │
└───────────────────────────────────┘
```

---

# 16. Edge Hardware

Hardware is currently unknown.

If an edge device is provided:

First inspect:

```text
CPU
GPU
RAM
OS
Python
PyTorch
CUDA
TensorRT
```

Then test the existing PyTorch model.

Do NOT begin the hackathon by trying to install/compile everything for TensorRT.

Priority:

```text
working PyTorch inference
        >
working ONNX
        >
working TensorRT
```

If TensorRT works easily:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
```

Benchmark:

```text
PyTorch latency
ONNX latency
TensorRT latency
```

If the edge environment becomes a time sink, use the laptop for the working demo and optimize only after the end-to-end pipeline is functional.

---

# 17. ONNX / TensorRT

Export only after the model is frozen.

Validation order:

```text
PyTorch output
      ↓
ONNX output
      ↓
compare numerical difference
      ↓
TensorRT output
      ↓
benchmark
```

Do not optimize a model whose preprocessing or target handling is still changing.

---

# 18. Recommended Project Structure

Keep it small:

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── windowing.py
│   ├── models.py
│   ├── train.py
│   ├── inference.py
│   └── visualize.py
│
├── configs/
│   └── config.yaml
│
├── checkpoints/
│
├── demo.py
├── requirements.txt
└── README.md
```

Keep configuration for:

```yaml
window_size:
stride:
sampling_rate:
features:
hidden_size:
num_layers:
model:
```

This allows the real challenge schema to be inserted without rewriting the whole project.

---

# 19. 8-Hour Execution Plan

## 00:00–00:20 — Inspect

- [ ] Dataset
- [ ] Schema
- [ ] Target
- [ ] Sampling rate
- [ ] Sensor signals
- [ ] Quaternion availability
- [ ] Metric
- [ ] Hardware
- [ ] Demo requirements

**Do not train yet.**

---

## 00:20–01:20 — Data pipeline

- [ ] Loader
- [ ] Timestamp handling
- [ ] Missing values
- [ ] Invalid values
- [ ] Quaternion normalization/sign handling
- [ ] Scaling
- [ ] Train/validation split

Milestone:

```text
raw data → clean tensors
```

---

## 01:20–02:00 — Sequence pipeline

- [ ] Sliding window
- [ ] Stride
- [ ] PyTorch Dataset
- [ ] DataLoader

Milestone:

```text
[B, T, F]
```

works correctly.

---

## 02:00–02:45 — Fast baseline

- [ ] Statistical feature extraction
- [ ] CatBoost baseline if appropriate
- [ ] Validation metric

Milestone:

```text
We know whether the input contains useful predictive signal.
```

---

## 02:45–04:30 — Main model

- [ ] Small LSTM
- [ ] Train
- [ ] Validate
- [ ] Tune only:
  - window size
  - hidden size
  - learning rate
  - loss

Stop tuning once a reasonable model works.

---

## 04:30–05:15 — Reconstruction

- [ ] Denormalization
- [ ] 3D reconstruction
- [ ] Optional smoothing
- [ ] Error inspection

Milestone:

```text
Prediction → visible 3D motion
```

---

## 05:15–06:15 — Streaming

- [ ] Ring buffer
- [ ] Sliding-window inference
- [ ] Real/replayed data
- [ ] Latency measurement

Milestone:

```text
stream → prediction
```

---

## 06:15–07:00 — Demo

- [ ] 3D visualization
- [ ] Replay/live mode
- [ ] Basic telemetry
- [ ] Minimal UI if needed

---

## 07:00–07:30 — Deployment optimization

Only if the hardware environment is cooperative:

- [ ] ONNX
- [ ] numerical validation
- [ ] TensorRT
- [ ] latency benchmark

---

## 07:30–08:00 — Finalize

- [ ] Fix obvious bugs
- [ ] Freeze model
- [ ] Prepare demo sequence
- [ ] README
- [ ] Architecture diagram
- [ ] Benchmark table
- [ ] Short technical explanation

---

# 20. Final Benchmark Table

Keep one table:

| Model | Input | Validation Metric | Params | Train Time | CPU Latency | GPU Latency |
|---|---|---:|---:|---:|---:|---:|
| CatBoost | engineered window features | | | | | |
| LSTM-small | raw sequence | | | | | |
| LSTM-optimized | raw sequence | | | | | |

Use the table to justify the final model.

---

# 21. What NOT to Build

With only 8 hours:

- [ ] No React frontend
- [ ] No complicated backend
- [ ] No Kafka
- [ ] No database unless required
- [ ] No cloud architecture
- [ ] No MLOps
- [ ] No custom CUDA kernels
- [ ] No giant Transformer
- [ ] No complex sensor-fusion framework
- [ ] No large hyperparameter search
- [ ] No unnecessary microservices

---

# 22. MVP Definition

The MVP is successful if:

```text
IMU data
   ↓
cleaning
   ↓
windowing
   ↓
trained model
   ↓
3D reconstruction
   ↓
streaming/replay
   ↓
visual demo
```

works reliably.

Minimum demo:

1. Load or replay an IMU sequence.
2. Process it through the same streaming pipeline.
3. Reconstruct the 3D swing.
4. Display the reconstruction.
5. Display basic telemetry.
6. Report inference latency.

---

# 23. Important Unknowns After Official Spec Arrives

Immediately update this document with:

## Data

- [ ] Exact schema
- [ ] Number of sensors
- [ ] Sensor locations
- [ ] Sampling frequency
- [ ] Units
- [ ] Quaternion availability
- [ ] Ground truth

## Target

- [ ] Exact output tensor
- [ ] Coordinate system
- [ ] Position vs orientation vs pose
- [ ] Absolute vs relative motion

## Evaluation

- [ ] Official metric
- [ ] Required accuracy
- [ ] Runtime constraints
- [ ] Hardware constraints

## Hardware

- [ ] Exact device
- [ ] Sensor interface
- [ ] GPU/CPU
- [ ] CUDA
- [ ] TensorRT
- [ ] OS/Python

## Demo

- [ ] Offline replay or live input
- [ ] Required visualization
- [ ] Required CV component
- [ ] Required UI
- [ ] Edge requirement

---

# 24. Current Default Technology Stack

| Purpose | Tool |
|---|---|
| Data inspection | pandas |
| Numerical processing | NumPy |
| Signal filtering | SciPy |
| Scaling | scikit-learn |
| Fast baseline | CatBoost |
| Main sequence model | PyTorch LSTM |
| Windowing | NumPy / Python |
| Streaming buffer | `collections.deque` |
| Visualization | Plotly |
| Optional UI | Streamlit |
| Export | ONNX |
| Edge optimization | TensorRT |

---

# 25. Core Rule

Do not decide:

> "We are using LSTM."

Decide:

> "We have a clean `[B, T, F]` sequence pipeline and a clear target. LSTM is the first serious temporal baseline."

Likewise, do not decide:

> "We need TensorRT."

Decide:

> "The model works. Now we check whether the provided hardware benefits from ONNX/TensorRT."

The only thing that should be fixed before the official specification is the **pipeline philosophy**, not the exact model.

---

# Current Status

- [ ] Official detailed specification
- [ ] Dataset
- [ ] Dataset schema
- [ ] Exact target
- [ ] Coordinate system
- [ ] Evaluation metric
- [ ] Hardware
- [ ] Demo requirement
- [ ] Final model selection

**Next action when the challenge opens: inspect the data and specification before writing the final model.**
