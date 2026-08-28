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
    "segment_id", "window_start", "window_end", "rows",
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


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("window_end").reset_index(drop=True)
    if len(ordered) < 5:
        raise ValueError("Need at least 5 windows for evaluation")
    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * (1 - test_fraction)))))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


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
