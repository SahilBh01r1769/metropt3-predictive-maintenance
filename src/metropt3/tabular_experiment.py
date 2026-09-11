from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .labels import add_failure_labels
from .maintenance_metrics import maintenance_metrics, select_threshold, threshold_table
from .modeling import feature_columns, purge_overlapping_training_windows
from .temporal_features import temporal_feature_columns


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tabular_experiment.json"
METADATA_COLUMNS = [
    "segment_id",
    "window_start",
    "window_end",
    "hours_to_next_failure",
    "failure_within_horizon",
]


@dataclass(frozen=True)
class EventFold:
    name: str
    validation_start: pd.Timestamp
    failure_start: pd.Timestamp


def load_tabular_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["horizon_hours"] != [6, 12, 24, 48]:
        raise ValueError("The tabular study horizons must remain 6/12/24/48 hours")
    if config["models"] != [
        "dummy",
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]:
        raise ValueError("The model progression has changed")
    folds = event_folds(config)
    if [fold.name for fold in folds] != [
        "may_failure",
        "june_failure",
        "july_holdout",
    ]:
        raise ValueError("The chronological event folds have changed")
    if any(fold.validation_start >= fold.failure_start for fold in folds):
        raise ValueError("Each validation block must precede its failure")
    if any(
        earlier.failure_start >= later.validation_start
        for earlier, later in zip(folds, folds[1:])
    ):
        raise ValueError("Event folds must be chronological and non-overlapping")
    if config["thresholds"]["selection_data"] != "development_events_only":
        raise ValueError("Thresholds may only use development events")
    return config


def event_folds(config: dict[str, Any]) -> list[EventFold]:
    return [
        EventFold(
            fold["name"],
            pd.Timestamp(fold["validation_start"]),
            pd.Timestamp(fold["failure_start"]),
        )
        for fold in config["event_folds"]
    ]


def chronological_event_split(
    labeled_windows: pd.DataFrame,
    fold: EventFold,
    *,
    temporal_context_hours: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = labeled_windows.loc[~labeled_windows["in_failure"]].copy()
    data["window_start"] = pd.to_datetime(data["window_start"])
    data["window_end"] = pd.to_datetime(data["window_end"])
    provisional_train = data.loc[data["window_end"] < fold.validation_start]
    validation = data.loc[
        data["window_end"].ge(fold.validation_start)
        & data["window_end"].le(fold.failure_start)
    ]
    if provisional_train.empty or validation.empty:
        raise ValueError(f"Event fold {fold.name} has an empty partition")
    if temporal_context_hours < 0:
        raise ValueError("temporal_context_hours must be non-negative")
    training = purge_overlapping_training_windows(provisional_train, validation)
    validation_context_start = validation["window_start"].min() - pd.Timedelta(
        hours=temporal_context_hours
    )
    training = training.loc[training["window_end"] <= validation_context_start].copy()
    if training.empty:
        raise ValueError(f"Event fold {fold.name} purge removed all training windows")
    if training["window_end"].max() > validation_context_start:
        raise AssertionError("Training overlaps the validation feature-history interval")
    return training, validation


def feature_set_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    all_columns = feature_columns(frame)
    temporal = set(temporal_feature_columns(frame))
    current = [column for column in all_columns if column not in temporal]
    if feature_set == "current_window":
        return current
    if feature_set == "temporal_context":
        return current + sorted(temporal)
    raise ValueError(f"Unknown feature set: {feature_set}")


def build_tabular_model(
    name: str,
    *,
    seed: int,
    parameters: dict[str, Any],
    positive_weight: float,
) -> Any:
    if name == "dummy":
        return Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="prior"))]
        )
    if name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(parameters["C"]),
                        class_weight="balanced",
                        max_iter=int(parameters["max_iter"]),
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(parameters["n_estimators"]),
                        min_samples_leaf=int(parameters["min_samples_leaf"]),
                        max_features=parameters["max_features"],
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost requires requirements-experiment.txt") from exc
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        **parameters,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        tree_method="hist",
                        scale_pos_weight=positive_weight,
                        random_state=seed,
                        n_jobs=2,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {name}")


def _prediction_frame(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    fold: str,
    horizon: int,
    model: str,
    feature_set: str,
) -> pd.DataFrame:
    result = validation[METADATA_COLUMNS].copy()
    result = result.rename(columns={"failure_within_horizon": "true_label"})
    result["probability"] = np.asarray(probabilities, dtype=float)
    result["fold"] = fold
    result["horizon_hours"] = horizon
    result["model"] = model
    result["feature_set"] = feature_set
    return result


def _native_importance(model: Pipeline, columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)[0])
        kind = "absolute_standardized_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        kind = "native_tree_importance"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])
    return pd.DataFrame(
        {"feature": columns, "importance": values, "importance_type": kind}
    ).sort_values("importance", ascending=False)


def run_tabular_experiment(
    feature_frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_tabular_config() if config is None else config
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["random_seed"])
    prediction_frames = []
    fold_rows = []

    for horizon in config["horizon_hours"]:
        labeled = add_failure_labels(feature_frame, horizon_hours=float(horizon))
        for feature_set in config["feature_sets"]:
            columns = feature_set_columns(labeled, feature_set)
            for model_name in config["models"]:
                for fold in event_folds(config):
                    training, validation = chronological_event_split(labeled, fold)
                    y_train = training["failure_within_horizon"].astype(int)
                    if y_train.nunique() != 2:
                        raise ValueError(
                            f"{fold.name}/h{horizon} training data is not binary"
                        )
                    positive_weight = float(y_train.eq(0).sum() / y_train.eq(1).sum())
                    model = build_tabular_model(
                        model_name,
                        seed=seed,
                        parameters=config["model_parameters"].get(model_name, {}),
                        positive_weight=positive_weight,
                    )
                    x_train = training[columns].replace([np.inf, -np.inf], np.nan)
                    x_validation = validation[columns].replace([np.inf, -np.inf], np.nan)
                    started = time.perf_counter()
                    model.fit(x_train, y_train)
                    fit_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    probabilities = model.predict_proba(x_validation)[:, 1]
                    prediction_seconds = time.perf_counter() - started
                    predictions = _prediction_frame(
                        validation,
                        probabilities,
                        fold=fold.name,
                        horizon=int(horizon),
                        model=model_name,
                        feature_set=feature_set,
                    )
                    prediction_frames.append(predictions)
                    fold_rows.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
                            "horizon_hours": int(horizon),
                            "fold": fold.name,
                            "train_windows": len(training),
                            "positive_train_windows": int(y_train.sum()),
                            "fit_seconds": fit_seconds,
                            "prediction_seconds": prediction_seconds,
                            "threshold_policy": "reference_0.5",
                            "threshold": 0.5,
                            **maintenance_metrics(predictions, threshold=0.5),
                        }
                    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    threshold_rows = []
    selected_rows = []
    event_rows = []
    comparison_rows = []
    group_columns = ["model", "feature_set", "horizon_hours"]
    starts = config["thresholds"]
    candidates = np.arange(
        float(starts["candidate_start"]),
        float(starts["candidate_stop"]) + float(starts["candidate_step"]) / 2,
        float(starts["candidate_step"]),
    )
    for identity, group in predictions.groupby(group_columns, sort=False):
        model_name, feature_set, horizon = identity
        development = group.loc[group["fold"].isin(config["development_events"])]
        final = group.loc[group["fold"].eq(config["final_event"])]
        analysis = threshold_table(development, candidates)
        for key, value in zip(group_columns, identity):
            analysis[key] = value
        threshold_rows.append(analysis)
        selected = select_threshold(analysis)
        selected_rows.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "horizon_hours": horizon,
                "threshold": selected,
                "selection_scope": "may_and_june_development_predictions_only",
            }
        )
        for fold_name, event in group.groupby("fold", sort=False):
            event_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "horizon_hours": horizon,
                    "fold": fold_name,
                    "threshold": selected,
                    **maintenance_metrics(event, threshold=selected),
                }
            )
        development_metrics = maintenance_metrics(development, threshold=selected)
        final_metrics = maintenance_metrics(final, threshold=selected)
        comparison_rows.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "horizon_hours": horizon,
                "selected_threshold": selected,
                **{f"development_{key}": value for key, value in development_metrics.items()},
                **{f"holdout_{key}": value for key, value in final_metrics.items()},
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    best_index = comparison["development_average_precision"].astype(float).idxmax()
    best = comparison.loc[best_index]
    best_key = (best["model"], best["feature_set"], int(best["horizon_hours"]))
    interpretable = comparison.loc[comparison["model"].ne("dummy")]
    interpretation = interpretable.loc[
        interpretable["development_average_precision"].astype(float).idxmax()
    ]
    interpretation_key = (
        interpretation["model"],
        interpretation["feature_set"],
        int(interpretation["horizon_hours"]),
    )

    horizon_rows = []
    for horizon, candidates_for_horizon in comparison.groupby("horizon_hours"):
        winner = candidates_for_horizon.loc[
            candidates_for_horizon["development_average_precision"].astype(float).idxmax()
        ]
        horizon_rows.append(winner.to_dict())

    predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(output / "fold_results.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv(
        output / "threshold_analysis.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(output / "selected_thresholds.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output / "event_metrics.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(output / "horizon_comparison.csv", index=False)
    labeled = add_failure_labels(
        feature_frame, horizon_hours=float(interpretation_key[2])
    )
    final_fold = next(
        fold for fold in event_folds(config) if fold.name == config["final_event"]
    )
    final_train, _ = chronological_event_split(labeled, final_fold)
    best_columns = feature_set_columns(labeled, interpretation_key[1])
    y_train = final_train["failure_within_horizon"].astype(int)
    best_model = build_tabular_model(
        interpretation_key[0],
        seed=seed,
        parameters=config["model_parameters"].get(interpretation_key[0], {}),
        positive_weight=float(y_train.eq(0).sum() / y_train.eq(1).sum()),
    )
    best_model.fit(
        final_train[best_columns].replace([np.inf, -np.inf], np.nan), y_train
    )
    importance = _native_importance(best_model, best_columns)
    importance.insert(0, "model", interpretation_key[0])
    importance.insert(1, "feature_set", interpretation_key[1])
    importance.insert(2, "horizon_hours", interpretation_key[2])
    importance.to_csv(output / "feature_importance.csv", index=False)
    joblib.dump(
        {
            "model": best_model,
            "features": best_columns,
            "identity": interpretation_key,
            "selection_scope": "pooled May and June development predictions",
        },
        output / "best_tabular_model.joblib",
    )
    summary = {
        "study_id": config["study_id"],
        "best_selected_on": "pooled May and June development average precision",
        "best_model": best_key[0],
        "best_feature_set": best_key[1],
        "best_horizon_hours": best_key[2],
        "best_threshold": float(best["selected_threshold"]),
        "interpretation_model": interpretation_key[0],
        "interpretation_feature_set": interpretation_key[1],
        "interpretation_horizon_hours": interpretation_key[2],
        "final_event": config["final_event"],
        "final_event_used_for_selection": False,
        "prediction_rows": len(predictions),
    }
    (output / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
