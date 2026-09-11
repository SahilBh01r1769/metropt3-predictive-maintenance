from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .experiment_config import load_experiment_config
from .experiment_runner import CellSpec, cell_checkpoint_valid, false_alert_episodes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluated_days(frame: pd.DataFrame, step_seconds: int) -> float:
    durations = []
    for _, segment in frame.groupby("segment_id"):
        starts = pd.to_datetime(segment["window_start"])
        duration = (starts.max() - starts.min()).total_seconds() + step_seconds
        durations.append(max(duration, step_seconds))
    return float(sum(durations) / 86400)


def _metric_row(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    policy: str,
    threshold: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    labels = frame["true_label"].astype(int)
    probabilities = frame["probability"].astype(float)
    predictions = frame[prediction_column].astype(int)
    prevalence = float(labels.mean())
    both_classes = labels.nunique() == 2
    average_precision = (
        float(average_precision_score(labels, probabilities)) if both_classes else None
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    false_episodes = false_alert_episodes(
        frame,
        prediction_column,
        expected_step_seconds=int(config["windows"]["step_seconds"]),
    )
    days = _evaluated_days(frame, int(config["windows"]["step_seconds"]))
    positive_alerts = frame.loc[
        frame[prediction_column].eq(1) & frame["true_label"].eq(1)
    ]
    if positive_alerts.empty:
        detected = False
        lead_time = None
    else:
        detected = True
        lead_time = float(positive_alerts["hours_to_next_failure"].max())
    return {
        "threshold_policy": policy,
        "threshold": threshold,
        "windows": len(frame),
        "positive_windows": int(labels.sum()),
        "prevalence": prevalence,
        "average_precision": average_precision,
        "ap_lift_over_prevalence": (
            average_precision / prevalence
            if average_precision is not None and prevalence > 0
            else None
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_alert_episodes": false_episodes,
        "evaluated_days": days,
        "false_alert_episodes_per_day": false_episodes / days if days else None,
        "held_out_event_detected": detected,
        "first_alert_lead_hours": lead_time,
    }


def validate_and_summarize_experiment(
    output_root: str | Path,
    *,
    output_csv: str | Path | None = None,
    report_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    require_complete: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    experiment = load_experiment_config() if config is None else config
    root = Path(output_root)
    errors: list[str] = []
    rows = []
    identities: dict[tuple[int, int], set[tuple[str, int]]] = {}
    complete_cells = 0
    revisions: set[str] = set()

    for model in experiment["models"]["order"]:
        for horizon in experiment["targets"]["horizon_hours"]:
            group = root / model / f"h{horizon}"
            threshold_path = group / "threshold.json"
            predictions_path = group / "holdout_predictions.csv"
            if not threshold_path.exists() or not predictions_path.exists():
                if require_complete:
                    errors.append(f"Missing finalized group: {model}/h{horizon}")
                continue
            threshold_record = json.loads(threshold_path.read_text(encoding="utf-8"))
            if threshold_record.get("holdout_predictions_sha256") != _sha256(
                predictions_path
            ):
                errors.append(f"Finalized prediction hash mismatch: {model}/h{horizon}")
            predictions = pd.read_csv(predictions_path)
            for seed in experiment["training"]["seeds"]:
                cell = group / f"seed{seed}"
                manifest_path = cell / "manifest.json"
                history_path = cell / "history.json"
                if not manifest_path.exists() or not history_path.exists():
                    errors.append(f"Missing cell evidence: {model}/h{horizon}/seed{seed}")
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                history = json.loads(history_path.read_text(encoding="utf-8"))
                spec = CellSpec(model, int(horizon), int(seed))
                if not cell_checkpoint_valid(cell, spec, experiment):
                    errors.append(f"Checkpoint hash mismatch: {model}/h{horizon}/seed{seed}")
                    continue
                if manifest.get("status") != "complete" or not history:
                    errors.append(f"Incomplete cell evidence: {model}/h{horizon}/seed{seed}")
                    continue
                if manifest.get("code_revision"):
                    revisions.add(manifest["code_revision"])
                else:
                    errors.append(f"Missing code revision: {model}/h{horizon}/seed{seed}")
                if manifest.get("dataset_sha256") != experiment["audit_evidence"][
                    "dataset_sha256"
                ]:
                    errors.append(f"Dataset mismatch: {model}/h{horizon}/seed{seed}")
                selected = predictions.loc[predictions["seed"].eq(seed)].copy()
                if selected.empty:
                    errors.append(f"Missing predictions: {model}/h{horizon}/seed{seed}")
                    continue
                probabilities = pd.to_numeric(selected["probability"], errors="coerce")
                if probabilities.isna().any() or not probabilities.between(0, 1).all():
                    errors.append(f"Invalid probabilities: {model}/h{horizon}/seed{seed}")
                    continue
                identity = set(
                    zip(selected["window_id"].astype(str), selected["true_label"].astype(int))
                )
                identity_key = (int(horizon), int(seed))
                if identity_key in identities and identities[identity_key] != identity:
                    errors.append(f"Window/label mismatch at h{horizon}/seed{seed}")
                else:
                    identities[identity_key] = identity
                if model == "attention_tcn" and seed == experiment["reporting"][
                    "primary_seed_for_window_traces"
                ]:
                    weights_path = cell / "attention_weights.npy"
                    if not weights_path.exists():
                        errors.append(f"Missing attention weights: h{horizon}/seed{seed}")
                    else:
                        weights = np.load(weights_path)
                        if weights.shape != (
                            len(selected),
                            int(experiment["sequence"]["timesteps"]),
                        ) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
                            errors.append(f"Invalid attention weights: h{horizon}/seed{seed}")
                base = {
                    "model": model,
                    "horizon_hours": int(horizon),
                    "seed": int(seed),
                    "development_fit_seconds": manifest["timing_seconds"][
                        "development_fit"
                    ],
                    "final_fit_seconds": manifest["timing_seconds"]["final_fit"],
                    "prediction_seconds": manifest["timing_seconds"][
                        "holdout_prediction"
                    ],
                    "parameter_count": manifest.get("parameter_count"),
                    "artifact_bytes": manifest["artifact_bytes"],
                }
                for column, policy, threshold in (
                    (
                        "reference_prediction",
                        "reference_0.5",
                        float(threshold_record["reference_threshold"]),
                    ),
                    (
                        "operational_prediction",
                        "development_selected",
                        float(threshold_record["operational"]["threshold"]),
                    ),
                ):
                    rows.append(
                        {
                            **base,
                            **_metric_row(
                                selected,
                                column,
                                policy=policy,
                                threshold=threshold,
                                config=experiment,
                            ),
                        }
                    )
                complete_cells += 1

    expected_cells = (
        len(experiment["models"]["order"])
        * len(experiment["targets"]["horizon_hours"])
        * len(experiment["training"]["seeds"])
    )
    if require_complete and complete_cells != expected_cells:
        errors.append(f"Expected {expected_cells} complete cells; found {complete_cells}")
    if complete_cells and len(revisions) != 1:
        errors.append("Completed cells do not share one code revision")
    metrics = pd.DataFrame(rows)
    report = {
        "valid": not errors,
        "complete_cells": complete_cells,
        "expected_cells": expected_cells,
        "metric_rows": len(metrics),
        "code_revision": next(iter(revisions)) if len(revisions) == 1 else None,
        "errors": errors,
    }
    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(destination, index=False)
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors and require_complete:
        raise ValueError("Experiment evidence validation failed: " + "; ".join(errors))
    return metrics, report
