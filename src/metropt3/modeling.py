from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


NON_FEATURE_COLS = {
    "segment_id", "window_start", "window_end", "rows", "cadence_seconds",
    "failure_within_horizon", "in_failure", "hours_to_next_failure",
}


@dataclass
class EvaluationMetrics:
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None
    train_windows: int
    test_windows: int
    positive_train: int
    positive_test: int


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def purge_overlapping_training_windows(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Remove training windows that touch the test interval.

    Feature extraction uses half-open windows: ``window_start <= timestamp <
    window_end``. Keeping only training windows whose end is at or before the
    earliest test-window start therefore guarantees that no raw timestamp can
    contribute to both sides of the split.
    """
    required = {"window_start", "window_end"}
    missing = required.difference(train.columns) | required.difference(test.columns)
    if missing:
        raise ValueError(
            "Window bounds are required for overlap purging: "
            + ", ".join(sorted(missing))
        )
    if train.empty or test.empty:
        raise ValueError("Both provisional splits need at least one window")

    test_interval_start = pd.to_datetime(test["window_start"]).min()
    train_ends = pd.to_datetime(train["window_end"])
    return train.loc[train_ends <= test_interval_start].copy()


def chronological_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    event_context_days: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a leakage-safe chronological split.

    When target labels are present, prefer a final-event holdout: the test set
    begins several days before the final positive pre-failure window. This
    keeps the last failure episode unseen during training and, unlike a naive
    tail split, usually preserves both classes in the test set. If that split
    is not viable, fall back to the final ``test_fraction`` of time.
    """
    required = {"window_start", "window_end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Window bounds are required for chronological splitting: "
            + ", ".join(sorted(missing))
        )

    ordered = df.copy()
    ordered["window_start"] = pd.to_datetime(ordered["window_start"])
    ordered["window_end"] = pd.to_datetime(ordered["window_end"])
    if (ordered["window_end"] <= ordered["window_start"]).any():
        raise ValueError("Each window_end must be later than window_start")
    ordered = ordered.sort_values("window_end").reset_index(drop=True)
    if len(ordered) < 5:
        raise ValueError("Need at least 5 windows for evaluation")

    if "failure_within_horizon" in ordered.columns:
        positives = ordered.loc[ordered["failure_within_horizon"].astype(int) == 1, "window_end"]
        if not positives.empty:
            test_start = pd.Timestamp(positives.max()) - pd.Timedelta(days=event_context_days)
            train = ordered.loc[ordered["window_end"] < test_start].copy()
            test = ordered.loc[ordered["window_end"] >= test_start].copy()
            if not train.empty and not test.empty:
                train = purge_overlapping_training_windows(train, test)
                if (
                    len(train) >= 5
                    and len(test) >= 2
                    and train["failure_within_horizon"].nunique() == 2
                    and test["failure_within_horizon"].nunique() == 2
                ):
                    return train, test

    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * (1 - test_fraction)))))
    train = ordered.iloc[:cut].copy()
    test = ordered.iloc[cut:].copy()
    train = purge_overlapping_training_windows(train, test)
    if train.empty:
        raise ValueError("Overlap purge removed every training window")
    return train, test


def train_and_evaluate(
    windows: pd.DataFrame,
    *,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, EvaluationMetrics, list[str]]:
    data = windows.loc[~windows["in_failure"]].dropna(subset=["failure_within_horizon"]).copy()
    train, test = chronological_split(data)
    cols = feature_columns(data)
    if not cols:
        raise ValueError("No numeric feature columns found")
    y_train = train["failure_within_horizon"].astype(int)
    y_test = test["failure_within_horizon"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError("Training split has only one class; adjust horizon/windowing or dataset range")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train[cols], y_train)
    pred = model.predict(test[cols])
    prob = model.predict_proba(test[cols])[:, 1]
    both_classes = y_test.nunique() == 2
    metrics = EvaluationMetrics(
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, prob)) if both_classes else None,
        average_precision=float(average_precision_score(y_test, prob)) if both_classes else None,
        train_windows=len(train),
        test_windows=len(test),
        positive_train=int(y_train.sum()),
        positive_test=int(y_test.sum()),
    )

    if model_path:
        path = Path(model_path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": cols}, path)
    if metrics_path:
        path = Path(metrics_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return model, metrics, cols


def predict_risk(model_bundle: dict, feature_frame: pd.DataFrame) -> np.ndarray:
    model = model_bundle["model"]
    cols = model_bundle["features"]
    missing = [c for c in cols if c not in feature_frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    return model.predict_proba(feature_frame[cols])[:, 1]
