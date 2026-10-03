# Hack For Humanity Hackathon

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