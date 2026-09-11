from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "xgboost": "XGBoost",
    "tcn": "TCN",
    "attention_tcn": "Attention-TCN",
}
EVENT_LABELS = {
    "may_failure": "May development",
    "june_failure": "June development",
    "july_holdout": "July holdout",
}


def load_explorer_evidence(
    evidence_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    root = Path(evidence_root)
    metrics = pd.read_csv(root / "metrics.csv")
    eventwise = pd.read_csv(root / "eventwise_summary.csv")
    thresholds = json.loads((root / "thresholds.json").read_text(encoding="utf-8"))
    if set(metrics["model"].unique()) != set(MODEL_LABELS):
        raise ValueError("Metrics do not contain the frozen three-model comparison")
    if set(eventwise["event"].unique()) != set(EVENT_LABELS):
        raise ValueError("Event-wise evidence does not contain May, June and July")
    return metrics, eventwise, thresholds


def summarize_metric_rows(
    metrics: pd.DataFrame, *, model: str, horizon_hours: int
) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["model"].eq(model)
        & metrics["horizon_hours"].eq(horizon_hours)
    ]
    if selected.empty:
        raise ValueError(f"No metrics for {model} at {horizon_hours} hours")
    return (
        selected.groupby(["threshold_policy", "threshold"], as_index=False)
        .agg(
            average_precision=("average_precision", "mean"),
            ap_lift_over_prevalence=("ap_lift_over_prevalence", "mean"),
            roc_auc=("roc_auc", "mean"),
            prevalence=("prevalence", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            false_alert_episodes_per_day=("false_alert_episodes_per_day", "mean"),
            event_detection_rate=("held_out_event_detected", "mean"),
            first_alert_lead_hours=("first_alert_lead_hours", "mean"),
        )
        .sort_values("threshold_policy")
    )


def threshold_for(
    thresholds: list[dict],
    *,
    model: str,
    horizon_hours: int,
    policy: str,
) -> float:
    if policy == "reference_0.5":
        return 0.5
    matches = [
        row
        for row in thresholds
        if row["model"] == model and row["horizon_hours"] == horizon_hours
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one threshold for {model} h{horizon_hours}")
    return float(matches[0]["operational"]["threshold"])


def load_probability_trace(
    evidence_root: str | Path,
    *,
    model: str,
    horizon_hours: int,
    event: str,
) -> pd.DataFrame:
    root = Path(evidence_root)
    if event == "july_holdout":
        path = root / "holdout" / f"{model}_h{horizon_hours}.csv.gz"
        frame = pd.read_csv(path)
    else:
        path = root / "development" / f"{model}_h{horizon_hours}.csv.gz"
        frame = pd.read_csv(path)
        frame = frame.loc[frame["fold"].eq(event)]
    if frame.empty:
        raise ValueError(f"No trace for {model}, h{horizon_hours}, {event}")
    frame["window_end"] = pd.to_datetime(frame["window_end"])
    return frame


def event_failure_time(config_path: str | Path, event: str) -> pd.Timestamp:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if event == "july_holdout":
        return pd.Timestamp(config["split"]["final_failure_start"])
    matches = [
        fold for fold in config["split"]["development_folds"] if fold["name"] == event
    ]
    if len(matches) != 1:
        raise ValueError(f"No configured failure time for {event}")
    return pd.Timestamp(matches[0]["validation_end"])
