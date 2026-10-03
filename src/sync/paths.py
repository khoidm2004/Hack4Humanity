"""Canonical repo paths, so the three CLI entry points agree on where things live."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_CSV = DATA_DIR / "raw_data.csv"
VIDEO_DIR = DATA_DIR / "video"

EVIDENCE_DIR = REPO_ROOT / "Artifacts" / "sync_evidence"
PROBE_DIR = EVIDENCE_DIR / "probe"
CONTACT_DIR = EVIDENCE_DIR / "contact"
VERIFY_DIR = EVIDENCE_DIR / "verify"

SYNC_METHOD_MD = REPO_ROOT / "SYNC_METHOD.md"
SYNC_RESULT_JSON = EVIDENCE_DIR / "sync_result.json"

# (label, video file, synced-csv output) for each camera angle.
ANGLES = [
    ("angle_1", VIDEO_DIR / "swing_angle_1.mp4", DATA_DIR / "synced_data_angle_1.csv"),
    ("angle_2", VIDEO_DIR / "swing_angle_2.mp4", DATA_DIR / "synced_data_angle_2.csv"),
]

# Sample rate is an *assumption* carried from CLAUDE.md / TASK.md, not a
# measurement. Phase E3 checks it for contradiction rather than correcting it.
IMU_SAMPLE_RATE_HZ = 416.0
