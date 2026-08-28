from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILENAME = "MetroPT3(AirCompressor).csv"
TIMESTAMP_COL = "timestamp"
ANALOGUE_COLS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
    "Oil_temperature", "Motor_current",
]
DIGITAL_COLS = ["COMP"]
REQUIRED_COLS = [TIMESTAMP_COL, *ANALOGUE_COLS, *DIGITAL_COLS]

WINDOW_SECONDS = 3600
STEP_SECONDS = 1800
# The UCI CSV is sampled more sparsely than a strict 1 Hz assumption. A gap
# larger than 30 s is treated as a discontinuity; window coverage is derived
# from each segment's observed median cadence in features.py.
MAX_GAP_SECONDS = 30
FAILURE_HORIZON_HOURS = 12

# Broad physical sanity bounds. They are intentionally permissive and are
# validation guards, not learned operating limits.
ANALOGUE_RANGES = {
    "TP2": (-0.5, 12.0),
    "TP3": (-0.5, 12.0),
    "H1": (-0.5, 12.0),
    "DV_pressure": (-1.0, 3.0),
    "Reservoirs": (-0.5, 12.0),
    "Oil_temperature": (-20.0, 150.0),
    "Motor_current": (-1.0, 30.0),
}

# Failure intervals published with the UCI MetroPT-3 dataset.
FAILURE_INTERVALS = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00", "air_leak"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00", "air_leak"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00", "air_leak"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00", "air_leak"),
]
