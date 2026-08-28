from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .config import ARTIFACT_DIR, RAW_FILENAME
from .features import build_windows
from .labels import add_failure_labels
from .modeling import train_and_evaluate
from .validation import validate_and_segment


def run_training_pipeline(csv_path: str | Path) -> dict:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    raw = pd.read_csv(csv_path)
    valid, report = validate_and_segment(raw, quarantine_path=ARTIFACT_DIR / "quarantine.csv")
    windows = build_windows(valid)
    if windows.empty:
        raise RuntimeError(
            "No feature windows were produced. Inspect timestamp cadence, segmentation, "
            "window length and validation quarantine counts."
        )
    labeled = add_failure_labels(windows)
    labeled.to_csv(ARTIFACT_DIR / "window_features.csv", index=False)
    model, metrics, features = train_and_evaluate(
        labeled,
        model_path=ARTIFACT_DIR / "model.joblib",
        metrics_path=ARTIFACT_DIR / "metrics.json",
    )
    summary = {
        "source": str(csv_path),
        "validation": report.__dict__,
        "windows": len(labeled),
        "observed_cadence_seconds": sorted(labeled["cadence_seconds"].dropna().unique().tolist()),
        "features": features,
        "metrics": metrics.__dict__,
    }
    (ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def default_data_path() -> Path:
    return Path("data") / RAW_FILENAME
