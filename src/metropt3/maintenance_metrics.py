from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _alert_runs(
    frame: pd.DataFrame,
    alert_mask: pd.Series,
    *,
    expected_step_seconds: int,
) -> list[int]:
    ordered = frame.assign(alert_flag=alert_mask.astype(bool)).copy()
    ordered["window_start"] = pd.to_datetime(ordered["window_start"])
    ordered = ordered.sort_values(["segment_id", "window_start"])
    runs: list[int] = []
    current = 0
    previous_segment: Any = None
    previous_start: pd.Timestamp | None = None
    for row in ordered.itertuples(index=False):
        start = row.window_start
        continuous = (
            previous_segment == row.segment_id
            and previous_start is not None
            and (start - previous_start).total_seconds() <= expected_step_seconds
        )
        if not continuous or not row.alert_flag:
            if current:
                runs.append(current)
            current = 0
        if row.alert_flag:
            current += 1
        previous_segment = row.segment_id
        previous_start = start
    if current:
        runs.append(current)
    return runs


def evaluated_days(frame: pd.DataFrame, *, expected_step_seconds: int) -> float:
    seconds = 0.0
    for _, segment in frame.groupby("segment_id"):
        starts = pd.to_datetime(segment["window_start"])
        seconds += max(
            float((starts.max() - starts.min()).total_seconds()) + expected_step_seconds,
            float(expected_step_seconds),
        )
    return seconds / 86400


def maintenance_metrics(
    frame: pd.DataFrame,
    *,
    threshold: float,
    expected_step_seconds: int = 1800,
    event_column: str = "fold",
) -> dict[str, float | int | None]:
    required = {
        "segment_id",
        "window_start",
        "true_label",
        "probability",
        "hours_to_next_failure",
        event_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Metric input is missing: " + ", ".join(sorted(missing)))
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    labels = frame["true_label"].astype(int)
    probabilities = pd.to_numeric(frame["probability"], errors="coerce")
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError("probability must contain finite values in [0, 1]")
    predictions = probabilities.ge(threshold)
    both_classes = labels.nunique() == 2
    false_runs = _alert_runs(
        frame,
        predictions & labels.eq(0),
        expected_step_seconds=expected_step_seconds,
    )
    warning_runs = _alert_runs(
        frame,
        predictions & labels.eq(1),
        expected_step_seconds=expected_step_seconds,
    )
    days = evaluated_days(frame, expected_step_seconds=expected_step_seconds)

    event_leads: list[float] = []
    event_count = 0
    detected_events = 0
    for _, event in frame.groupby(event_column, sort=False):
        positive = event.loc[event["true_label"].eq(1)]
        if positive.empty:
            continue
        event_count += 1
        warnings = positive.loc[positive["probability"].ge(threshold)]
        if not warnings.empty:
            detected_events += 1
            event_leads.append(float(warnings["hours_to_next_failure"].max()))

    return {
        "windows": int(len(frame)),
        "positive_windows": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(labels, probabilities)) if both_classes else None
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "event_count": event_count,
        "detected_events": detected_events,
        "event_recall": detected_events / event_count if event_count else None,
        "mean_first_warning_lead_hours": (
            float(np.mean(event_leads)) if event_leads else None
        ),
        "minimum_first_warning_lead_hours": (
            float(np.min(event_leads)) if event_leads else None
        ),
        "false_alert_episodes": len(false_runs),
        "evaluated_days": days,
        "false_alerts_per_day": len(false_runs) / days if days else None,
        "warning_alert_windows": int((predictions & labels.eq(1)).sum()),
        "warning_episodes": len(warning_runs),
        "mean_warning_run_windows": (
            float(np.mean(warning_runs)) if warning_runs else 0.0
        ),
        "max_warning_run_windows": max(warning_runs, default=0),
    }


def threshold_table(
    development_predictions: pd.DataFrame,
    thresholds: Iterable[float],
    *,
    expected_step_seconds: int = 1800,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold": float(threshold),
                **maintenance_metrics(
                    development_predictions,
                    threshold=float(threshold),
                    expected_step_seconds=expected_step_seconds,
                ),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> float:
    """Select on development evidence: F2, then alert burden, then specificity."""
    required = {"threshold", "f2", "false_alerts_per_day"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError("Threshold table is missing: " + ", ".join(sorted(missing)))
    if table.empty:
        raise ValueError("Threshold table cannot be empty")
    ranked = table.sort_values(
        ["f2", "false_alerts_per_day", "threshold"],
        ascending=[False, True, False],
    )
    return float(ranked.iloc[0]["threshold"])
