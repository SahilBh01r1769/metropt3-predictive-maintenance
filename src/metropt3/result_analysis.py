from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EVENT_ORDER = ("may_failure", "june_failure", "july_holdout")


def ranking_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Calculate threshold-free metrics for one event, model, horizon and seed."""
    required = {"true_label", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Prediction trace is missing: " + ", ".join(sorted(missing)))
    labels = pd.to_numeric(frame["true_label"], errors="coerce")
    probabilities = pd.to_numeric(frame["probability"], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int)).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if labels.nunique() != 2:
        raise ValueError("Ranking metrics require both classes")

    labels = labels.astype(int)
    prevalence = float(labels.mean())
    positive_scores = probabilities[labels.eq(1)]
    negative_scores = probabilities[labels.eq(0)]
    average_precision = float(average_precision_score(labels, probabilities))
    return {
        "windows": int(len(frame)),
        "positive_windows": int(labels.sum()),
        "prevalence": prevalence,
        "average_precision": average_precision,
        "ap_lift_over_prevalence": average_precision / prevalence,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "positive_score_median": float(positive_scores.median()),
        "negative_score_median": float(negative_scores.median()),
        "median_score_gap": float(
            positive_scores.median() - negative_scores.median()
        ),
    }


def _identity_from_name(path: Path) -> tuple[str, int]:
    stem = path.name.removesuffix(".csv.gz")
    model, horizon = stem.rsplit("_h", 1)
    return model, int(horizon)


def build_eventwise_metrics(evidence_root: str | Path) -> pd.DataFrame:
    """Build May, June and July ranking evidence from committed prediction traces."""
    root = Path(evidence_root)
    rows: list[dict[str, float | int | str]] = []

    for path in sorted((root / "development").glob("*.csv.gz")):
        model, horizon = _identity_from_name(path)
        frame = pd.read_csv(path)
        for (seed, fold), selected in frame.groupby(["seed", "fold"], sort=True):
            rows.append(
                {
                    "model": model,
                    "horizon_hours": horizon,
                    "seed": int(seed),
                    "event": str(fold),
                    **ranking_metrics(selected),
                }
            )

    for path in sorted((root / "holdout").glob("*.csv.gz")):
        model, horizon = _identity_from_name(path)
        frame = pd.read_csv(path)
        for seed, selected in frame.groupby("seed", sort=True):
            rows.append(
                {
                    "model": model,
                    "horizon_hours": horizon,
                    "seed": int(seed),
                    "event": "july_holdout",
                    **ranking_metrics(selected),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No event prediction traces were found")
    result["event"] = pd.Categorical(
        result["event"], categories=EVENT_ORDER, ordered=True
    )
    return result.sort_values(
        ["event", "horizon_hours", "model", "seed"]
    ).reset_index(drop=True)


def summarize_eventwise_metrics(eventwise: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed sensitivity without treating seeds as independent events."""
    required = {
        "model",
        "horizon_hours",
        "seed",
        "event",
        "prevalence",
        "average_precision",
        "ap_lift_over_prevalence",
        "roc_auc",
        "median_score_gap",
    }
    missing = required.difference(eventwise.columns)
    if missing:
        raise ValueError("Event metrics are missing: " + ", ".join(sorted(missing)))
    summary = (
        eventwise.groupby(
            ["event", "horizon_hours", "model"], observed=True, sort=False
        )
        .agg(
            seeds=("seed", "nunique"),
            prevalence=("prevalence", "mean"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            ap_lift_mean=("ap_lift_over_prevalence", "mean"),
            ap_lift_std=("ap_lift_over_prevalence", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            median_score_gap_mean=("median_score_gap", "mean"),
        )
        .reset_index()
    )
    numeric = summary.select_dtypes(include=[np.number]).columns
    summary[numeric] = summary[numeric].astype(float)
    summary["seeds"] = summary["seeds"].astype(int)
    return summary
