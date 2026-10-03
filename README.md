# Hack For Humanity Hackathon

## Project

Reconstruct a 3D representation of a tennis shot (swing) from raw IMU sensor
data alone — accelerometer + gyroscope readings, no cameras or pre-labeled
pose data.

- `data/raw_data.csv` — raw IMU samples (`ax,ay,az,gx,gy,gz`).
- `data/raw_data_visualized.png` — a plotted view of that signal.
- `data/video/*.mp4` — reference footage of the actual tennis shots, for
  visual ground-truth comparison.

See `pre-hackathon.md` for the full technical plan (pipeline, model choices,
execution timeline).

## Development Setup

This project uses [`uv`](https://docs.astral.sh/uv/) (Astral's Python package and dependency manager) for managing dependencies and the Python environment.

### Prerequisites

1. **Install uv**: Follow the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Hack4Hum
   ```

2. **Initialize the environment**:
   ```bash
   uv sync
   ```
   This command creates the virtual environment (`.venv`) and installs all locked dependencies from `uv.lock`.

### Running Commands

To run Python commands within the project environment, use:

```bash
uv run python <script.py>
```

Alternatively, you can activate the virtual environment:

```bash
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

Then run Python commands normally:

```bash
python <script.py>
```