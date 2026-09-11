from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metropt3.config import FAILURE_INTERVALS
from metropt3.features import build_windows
from metropt3.labels import add_failure_labels
from metropt3.splits import chronological_split, feature_columns
from metropt3.validation import validate_and_segment


EVENT_NAMES = {"may_failure": 0, "june_failure": 1, "july_holdout": 2}


def _regime_membership(windows: pd.DataFrame) -> pd.Series:
    failures = pd.to_datetime([row[0] for row in FAILURE_INTERVALS])
    next_failure = pd.to_datetime(windows["window_end"])
    deltas = pd.DataFrame(
        {
            name: (failure - next_failure).dt.total_seconds() / 3600
            for name, failure in zip(EVENT_NAMES, failures[1:], strict=True)
        },
        index=windows.index,
    )
    membership = pd.Series("normal", index=windows.index, dtype="object")
    for event, column in zip(EVENT_NAMES, deltas.columns, strict=True):
        membership.loc[deltas[column].between(0, 24, inclusive="both")] = event
    # Keep a clean normal reference away from every published failure onset.
    near_any_failure = deltas.min(axis=1).between(0, 48, inclusive="both")
    membership.loc[near_any_failure & membership.eq("normal")] = "near_failure_excluded"
    return membership


def event_regime_effects(windows: pd.DataFrame) -> pd.DataFrame:
    labelled = add_failure_labels(windows, horizon_hours=12)
    labelled = labelled.loc[~labelled["in_failure"]].copy()
    labelled["regime"] = _regime_membership(labelled)
    normal = labelled.loc[labelled["regime"].eq("normal")]
    if normal.empty:
        raise ValueError("No clean normal windows available for regime comparison")
    rows = []
    for feature in feature_columns(labelled):
        baseline = normal[feature].dropna()
        baseline_median = float(baseline.median())
        baseline_iqr = float(baseline.quantile(0.75) - baseline.quantile(0.25))
        scale = baseline_iqr if baseline_iqr > 0 else 1.0
        for event in EVENT_NAMES:
            selected = labelled.loc[labelled["regime"].eq(event), feature].dropna()
            if selected.empty:
                continue
            event_median = float(selected.median())
            rows.append(
                {
                    "event": event,
                    "feature": feature,
                    "normal_median": baseline_median,
                    "normal_iqr": baseline_iqr,
                    "event_median": event_median,
                    "robust_standardized_difference": (event_median - baseline_median) / scale,
                    "event_windows": int(len(selected)),
                }
            )
    return pd.DataFrame(rows)


def event_regime_summary(effects: pd.DataFrame) -> pd.DataFrame:
    pivot = effects.pivot(index="feature", columns="event", values="robust_standardized_difference")
    for event in EVENT_NAMES:
        if event not in pivot:
            pivot[event] = np.nan
    pivot = pivot[list(EVENT_NAMES)]
    values = pivot.to_numpy(dtype=float)
    pivot["direction_consistency"] = np.abs(np.sign(values).sum(axis=1)) / len(EVENT_NAMES)
    pivot["all_events_same_direction"] = pivot["direction_consistency"].eq(1.0)
    return pivot.reset_index()


def xgboost_gain_importance(windows: pd.DataFrame) -> pd.DataFrame:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Install requirements-experiment.txt for XGBoost analysis") from exc
    labelled = add_failure_labels(windows, horizon_hours=1)
    labelled = labelled.loc[~labelled["in_failure"]].dropna(subset=["failure_within_horizon"])
    train, _ = chronological_split(labelled)
    columns = feature_columns(labelled)
    y = train["failure_within_horizon"].astype(int)
    if y.nunique() != 2:
        raise ValueError("XGBoost importance requires both classes in the training interval")
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)),
        random_state=42,
        n_jobs=1,
    )
    model.fit(train[columns], y)
    # Passing a DataFrame preserves column names in the booster. XGBoost therefore
    # returns names such as ``TP2_mean`` rather than positional ``f0`` keys.
    gains = model.get_booster().get_score(importance_type="gain")
    if not gains:
        raise RuntimeError("XGBoost produced no feature gains")
    return pd.DataFrame(
        {
            "feature": columns,
            "gain": [float(gains.get(column, 0.0)) for column in columns],
        }
    ).sort_values("gain", ascending=False, ignore_index=True)


def analyze(csv_path: str | Path, output_dir: str | Path) -> dict[str, str | int]:
    raw = pd.read_csv(csv_path)
    valid, report = validate_and_segment(raw)
    windows = build_windows(valid)
    effects = event_regime_effects(windows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output / "event_regime_effects.csv", index=False)
    event_regime_summary(effects).to_csv(output / "event_regime_summary.csv", index=False)
    xgboost_gain_importance(windows).to_csv(output / "xgboost_gain_importance.csv", index=False)
    summary = {
        "validated_rows": int(report.valid_rows),
        "segments": int(report.segments),
        "windows": int(len(windows)),
        "event_regime_effects": str(output / "event_regime_effects.csv"),
        "event_regime_summary": str(output / "event_regime_summary.csv"),
        "xgboost_gain_importance": str(output / "xgboost_gain_importance.csv"),
        "interpretation": "descriptive analysis only; does not alter frozen model evidence",
    }
    (output / "diagnostic_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze event precursor regimes from raw MetroPT-3 data")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("evidence/event_regime"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.csv, args.output), indent=2))


if __name__ == "__main__":
    main()
