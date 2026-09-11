from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    MAX_GAP_SECONDS,
    STEP_SECONDS,
    TIMESTAMP_COL,
    WINDOW_SECONDS,
)
from .features import build_windows
from .labels import add_failure_labels
from .splits import FINAL_EVENT_TEST_START, chronological_split
from .validation import range_diagnostics, validate_and_segment


DEFAULT_HORIZONS = (1, 3, 6, 12)


def _iso(value: Any) -> str | None:
    timestamp = pd.Timestamp(value)
    return None if pd.isna(timestamp) else timestamp.isoformat()


def _finite_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _cadence_summary(timestamps: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(timestamps, errors="coerce")
    deltas = parsed.diff().dt.total_seconds()
    positive = deltas[deltas > 0]
    adjacent_valid = parsed.notna() & parsed.shift().notna()
    non_increasing = adjacent_valid & (deltas <= 0)

    if positive.empty:
        percentiles: dict[str, float | None] = {
            key: None for key in ("p50", "p90", "p99", "p99_9", "max")
        }
        common: list[dict[str, float | int]] = []
        median_seconds = None
    else:
        values = positive.to_numpy(dtype=float)
        q = np.percentile(values, [50, 90, 99, 99.9])
        percentiles = {
            "p50": _finite_float(q[0]),
            "p90": _finite_float(q[1]),
            "p99": _finite_float(q[2]),
            "p99_9": _finite_float(q[3]),
            "max": _finite_float(values.max()),
        }
        common = [
            {"seconds": float(seconds), "count": int(count)}
            for seconds, count in positive.value_counts().head(10).items()
        ]
        median_seconds = percentiles["p50"]

    return {
        "parsed": int(parsed.notna().sum()),
        "unparseable": int(parsed.isna().sum()),
        "duplicate_timestamps": int(
            (parsed.duplicated(keep="first") & parsed.notna()).sum()
        ),
        "non_increasing_adjacent_pairs": int(non_increasing.sum()),
        "start": _iso(parsed.min()),
        "end": _iso(parsed.max()),
        "positive_delta_seconds": percentiles,
        "most_common_positive_deltas": common,
        "median_seconds": median_seconds,
    }


def _segment_summary(valid: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment_id, segment in valid.groupby("segment_id", sort=True):
        timestamps = pd.to_datetime(segment[TIMESTAMP_COL])
        positive = timestamps.diff().dt.total_seconds()
        positive = positive[positive > 0]
        rows.append(
            {
                "segment_id": int(segment_id),
                "rows": int(len(segment)),
                "start": _iso(timestamps.min()),
                "end": _iso(timestamps.max()),
                "duration_hours": _finite_float(
                    (timestamps.max() - timestamps.min()).total_seconds() / 3600
                ),
                "median_cadence_seconds": (
                    _finite_float(positive.median()) if not positive.empty else None
                ),
            }
        )
    return rows


def _resampling_candidates(
    median_seconds: float | None,
    window_seconds: int,
) -> list[dict[str, int]]:
    if median_seconds is None:
        return []
    observed_floor = max(1, int(math.ceil(median_seconds)))
    candidates = sorted(
        seconds for seconds in {observed_floor, 5, 10, 30, 60} if seconds >= observed_floor
    )
    return [
        {
            "interval_seconds": seconds,
            "timesteps_per_window": int(math.ceil(window_seconds / seconds)),
        }
        for seconds in candidates
    ]


def audit_dataframe(
    raw: pd.DataFrame,
    *,
    source: dict[str, Any] | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    test_start: str | pd.Timestamp = FINAL_EVENT_TEST_START,
    max_gap_seconds: int = MAX_GAP_SECONDS,
    window_seconds: int = WINDOW_SECONDS,
    step_seconds: int = STEP_SECONDS,
) -> dict[str, Any]:
    """Describe the exact evidence available to the temporal experiment."""
    cadence = _cadence_summary(raw[TIMESTAMP_COL]) if TIMESTAMP_COL in raw else {}
    valid, validation = validate_and_segment(raw, max_gap_seconds=max_gap_seconds)
    windows = build_windows(
        valid,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )

    diagnostics = range_diagnostics(raw)
    signal_rows = json.loads(diagnostics.to_json(orient="records"))
    missing_counts = {
        column: int(raw[column].isna().sum())
        for column in raw.columns
        if raw[column].isna().any()
    }

    horizon_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        if windows.empty:
            horizon_rows.append(
                {
                    "horizon_hours": int(horizon),
                    "predictive_windows": 0,
                    "positive_windows": 0,
                    "negative_windows": 0,
                    "quarantined_failure_windows": 0,
                    "split_error": "No complete feature windows were produced",
                }
            )
            continue
        labeled = add_failure_labels(windows, horizon_hours=float(horizon))
        predictive = labeled.loc[~labeled["in_failure"]].copy()
        row: dict[str, Any] = {
            "horizon_hours": int(horizon),
            "predictive_windows": int(len(predictive)),
            "positive_windows": int(predictive["failure_within_horizon"].sum()),
            "negative_windows": int((predictive["failure_within_horizon"] == 0).sum()),
            "quarantined_failure_windows": int(labeled["in_failure"].sum()),
        }
        try:
            train, test = chronological_split(predictive, test_start=test_start)
        except ValueError as exc:
            row["split_error"] = str(exc)
        else:
            row["train_windows"] = int(len(train))
            row["train_positive"] = int(train["failure_within_horizon"].sum())
            row["test_windows"] = int(len(test))
            row["test_positive"] = int(test["failure_within_horizon"].sum())
            row["latest_train_window_end"] = _iso(train["window_end"].max())
            row["earliest_test_window_start"] = _iso(test["window_start"].min())
            row["timestamp_intervals_disjoint"] = bool(
                train["window_end"].max() <= test["window_start"].min()
            )
        horizon_rows.append(row)

    return {
        "schema_version": 1,
        "source": source or {},
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "missing_values": missing_counts,
        "timestamp": cadence,
        "validation": validation.__dict__,
        "gap_policy_seconds": int(max_gap_seconds),
        "segments": _segment_summary(valid),
        "signal_percentiles_raw": signal_rows,
        "window_policy": {
            "history_seconds": int(window_seconds),
            "step_seconds": int(step_seconds),
            "feature_windows": int(len(windows)),
        },
        "horizons": horizon_rows,
        "sequence_resampling_candidates": _resampling_candidates(
            cadence.get("median_seconds"),
            window_seconds,
        ),
        "resampling_decision": "pending_review",
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    source = {
        "filename": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }
    return audit_dataframe(pd.read_csv(path, low_memory=False), source=source)


def write_audit_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
