from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Protocol
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score

from .experiment_config import load_experiment_config
from .labels import add_failure_labels
from .modeling import feature_columns, purge_overlapping_training_windows
from .sequences import SequenceDataset, fit_sequence_normalizer, split_sequence_indices
from .temporal_models import build_temporal_model, temporal_parameter_count


PREDICTION_COLUMNS = {
    "model",
    "horizon_hours",
    "seed",
    "window_id",
    "segment_id",
    "window_start",
    "window_end",
    "true_label",
    "probability",
    "hours_to_next_failure",
}


@dataclass(frozen=True)
class CellSpec:
    model: str
    horizon_hours: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model}/h{self.horizon_hours}/seed{self.seed}"


@dataclass
class TrainingResult:
    development_predictions: pd.DataFrame
    holdout_predictions: pd.DataFrame
    history: list[dict[str, Any]]
    best_iterations: list[int]
    development_fit_seconds: float
    final_fit_seconds: float
    prediction_seconds: float
    parameter_count: int | None
    model_complexity: dict[str, int]
    normalizer_manifest: dict[str, Any] | None = None
    attention_weights: np.ndarray | None = None


class CellTrainer(Protocol):
    def train(
        self,
        spec: CellSpec,
        dataset: SequenceDataset,
        config: dict[str, Any],
        artifact_path: Path,
    ) -> TrainingResult: ...


def _config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _labeled_dataset(dataset: SequenceDataset, horizon_hours: int) -> SequenceDataset:
    metadata = add_failure_labels(dataset.metadata, horizon_hours=horizon_hours)
    if metadata["in_failure"].any():
        raise AssertionError("Sequence dataset still contains active-failure windows")
    return SequenceDataset(dataset.values, metadata, dataset.channels)


def _development_splits(
    metadata: pd.DataFrame, config: dict[str, Any]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    index_by_id = {
        window_id: index for index, window_id in enumerate(metadata["window_id"])
    }
    splits = []
    for fold in config["split"]["development_folds"]:
        start = pd.Timestamp(fold["validation_start"])
        end = pd.Timestamp(fold["validation_end"])
        provisional_train = metadata.loc[metadata["window_end"] < start]
        validation = metadata.loc[
            (metadata["window_end"] >= start) & (metadata["window_end"] < end)
        ]
        if provisional_train.empty or validation.empty:
            raise ValueError(f"Development fold {fold['name']} is empty")
        training = purge_overlapping_training_windows(provisional_train, validation)
        train_indices = np.asarray(
            [index_by_id[value] for value in training["window_id"]], dtype=int
        )
        validation_indices = np.asarray(
            [index_by_id[value] for value in validation["window_id"]], dtype=int
        )
        if metadata.iloc[train_indices]["window_end"].max() > metadata.iloc[
            validation_indices
        ]["window_start"].min():
            raise AssertionError("Development raw timestamp intervals overlap")
        splits.append((fold["name"], train_indices, validation_indices))
    return splits


def _prediction_frame(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    probabilities: np.ndarray,
    **extra: Any,
) -> pd.DataFrame:
    selected = metadata.iloc[indices]
    frame = pd.DataFrame(
        {
            "window_id": selected["window_id"].to_numpy(),
            "segment_id": selected["segment_id"].to_numpy(),
            "window_start": selected["window_start"].to_numpy(),
            "window_end": selected["window_end"].to_numpy(),
            "true_label": selected["failure_within_horizon"].astype(int).to_numpy(),
            "probability": np.asarray(probabilities, dtype=float),
            "hours_to_next_failure": selected["hours_to_next_failure"].to_numpy(),
        }
    )
    for key, value in extra.items():
        frame[key] = value
    return frame


def _validate_predictions(frame: pd.DataFrame, *, development: bool) -> None:
    required = set(PREDICTION_COLUMNS)
    if development:
        required.add("fold")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Prediction trace is missing: " + ", ".join(sorted(missing)))
    probabilities = pd.to_numeric(frame["probability"], errors="coerce")
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    duplicate_columns = ["window_id", "fold"] if development else ["window_id"]
    if frame.duplicated(duplicate_columns).any():
        raise ValueError("Prediction trace contains duplicate window identities")


class XGBoostCellTrainer:
    def train(self, spec, dataset, config, artifact_path):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost requires requirements-experiment.txt") from exc

        metadata = dataset.metadata
        columns = feature_columns(metadata)
        if not columns:
            raise ValueError("No engineered summary features are available")
        model_config = config["models"]["xgboost"]
        dev_frames = []
        history = []
        best_iterations = []
        development_fit_seconds = 0.0

        for fold_name, train_indices, validation_indices in _development_splits(
            metadata, config
        ):
            y_train = metadata.iloc[train_indices]["failure_within_horizon"].astype(int)
            y_validation = metadata.iloc[validation_indices][
                "failure_within_horizon"
            ].astype(int)
            if y_train.nunique() < 2 or y_validation.nunique() < 2:
                raise ValueError(f"Development fold {fold_name} is not binary")
            started = time.perf_counter()
            model = XGBClassifier(
                n_estimators=int(model_config["maximum_estimators"]),
                learning_rate=float(model_config["learning_rate"]),
                max_depth=int(model_config["max_depth"]),
                min_child_weight=float(model_config["min_child_weight"]),
                subsample=float(model_config["subsample"]),
                colsample_bytree=float(model_config["colsample_bytree"]),
                reg_lambda=float(model_config["reg_lambda"]),
                objective=model_config["objective"],
                eval_metric=model_config["evaluation_metric"],
                tree_method=model_config["tree_method"],
                scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
                early_stopping_rounds=int(model_config["early_stopping_rounds"]),
                random_state=spec.seed,
                n_jobs=2,
            )
            model.fit(
                metadata.iloc[train_indices][columns],
                y_train,
                eval_set=[(metadata.iloc[validation_indices][columns], y_validation)],
                verbose=False,
            )
            development_fit_seconds += time.perf_counter() - started
            best_iteration = int(model.best_iteration) + 1
            best_iterations.append(best_iteration)
            probabilities = model.predict_proba(
                metadata.iloc[validation_indices][columns]
            )[:, 1]
            dev_frames.append(
                _prediction_frame(
                    metadata,
                    validation_indices,
                    probabilities,
                    fold=fold_name,
                )
            )
            history.append(
                {
                    "fold": fold_name,
                    "best_iteration": best_iteration,
                    "validation_average_precision": float(
                        average_precision_score(y_validation, probabilities)
                    ),
                }
            )

        train_indices, test_indices = split_sequence_indices(
            dataset, test_start=config["split"]["final_holdout_start"]
        )
        y_train = metadata.iloc[train_indices]["failure_within_horizon"].astype(int)
        final_iterations = int(np.median(best_iterations))
        final_model = XGBClassifier(
            n_estimators=final_iterations,
            learning_rate=float(model_config["learning_rate"]),
            max_depth=int(model_config["max_depth"]),
            min_child_weight=float(model_config["min_child_weight"]),
            subsample=float(model_config["subsample"]),
            colsample_bytree=float(model_config["colsample_bytree"]),
            reg_lambda=float(model_config["reg_lambda"]),
            objective=model_config["objective"],
            eval_metric=model_config["evaluation_metric"],
            tree_method=model_config["tree_method"],
            scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
            random_state=spec.seed,
            n_jobs=2,
        )
        started = time.perf_counter()
        final_model.fit(metadata.iloc[train_indices][columns], y_train, verbose=False)
        final_fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probabilities = final_model.predict_proba(metadata.iloc[test_indices][columns])[:, 1]
        prediction_seconds = time.perf_counter() - started
        joblib.dump({"model": final_model, "features": columns}, artifact_path)

        return TrainingResult(
            development_predictions=pd.concat(dev_frames, ignore_index=True),
            holdout_predictions=_prediction_frame(
                metadata, test_indices, probabilities
            ),
            history=history,
            best_iterations=best_iterations,
            development_fit_seconds=development_fit_seconds,
            final_fit_seconds=final_fit_seconds,
            prediction_seconds=prediction_seconds,
            parameter_count=None,
            model_complexity={
                "trees": len(final_model.get_booster().get_dump()),
                "leaves": int(
                    sum(
                        tree.count("leaf=")
                        for tree in final_model.get_booster().get_dump()
                    )
                ),
            },
        )


class TorchCellTrainer:
    def train(self, spec, dataset, config, artifact_path):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("Temporal training requires requirements-experiment.txt") from exc

        def seed_everything(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_config = config["training"]
        early_config = training_config["early_stopping"]
        metadata = dataset.metadata
        dev_frames = []
        history = []
        best_iterations = []
        development_fit_seconds = 0.0

        def fit_model(train_indices, validation_indices=None, epochs=None, fold=None):
            seed_everything(spec.seed)
            train_ids = metadata.iloc[train_indices]["window_id"]
            normalizer = fit_sequence_normalizer(dataset, train_ids)
            normalized = normalizer.transform(dataset).values
            y_train = metadata.iloc[train_indices]["failure_within_horizon"].to_numpy(
                dtype=np.float32
            )
            positives = float(y_train.sum())
            if positives == 0 or positives == len(y_train):
                raise ValueError("Temporal training split must contain both classes")
            model = build_temporal_model(
                spec.model, input_channels=normalized.shape[2], config=config
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training_config["learning_rate"]),
                weight_decay=float(training_config["weight_decay"]),
            )
            loss_function = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor((len(y_train) - positives) / positives, device=device)
            )
            tensors = TensorDataset(
                torch.from_numpy(normalized[train_indices]),
                torch.from_numpy(y_train),
            )
            generator = torch.Generator().manual_seed(spec.seed)
            loader = DataLoader(
                tensors,
                batch_size=int(training_config["batch_size"]),
                shuffle=True,
                generator=generator,
            )
            best_score = -np.inf
            best_epoch = 0
            best_state = None
            stale_epochs = 0
            maximum_epochs = epochs or int(training_config["maximum_epochs"])
            epoch_history = []
            for epoch in range(1, maximum_epochs + 1):
                model.train()
                losses = []
                for batch_values, batch_labels in loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch_values.to(device))
                    loss = loss_function(logits, batch_labels.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(training_config["gradient_clip_norm"])
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                record = {"fold": fold, "epoch": epoch, "train_loss": float(np.mean(losses))}
                if validation_indices is not None:
                    probabilities, _ = predict(model, normalized, validation_indices, False)
                    labels = metadata.iloc[validation_indices][
                        "failure_within_horizon"
                    ].astype(int)
                    if labels.nunique() < 2:
                        raise ValueError(f"Development fold {fold} is not binary")
                    score = float(average_precision_score(labels, probabilities))
                    record["validation_average_precision"] = score
                    if score > best_score + float(early_config["minimum_delta"]):
                        best_score = score
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                        best_epoch = epoch
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                epoch_history.append(record)
                if validation_indices is not None and stale_epochs >= int(
                    early_config["patience"]
                ):
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            selected_epoch = best_epoch if validation_indices is not None else len(epoch_history)
            return model, normalizer, normalized, epoch_history, selected_epoch

        def predict(model, normalized, indices, attention):
            model.eval()
            with torch.no_grad():
                values = torch.from_numpy(normalized[indices]).to(device)
                if attention:
                    logits, weights = model(values, return_attention=True)
                    return torch.sigmoid(logits).cpu().numpy(), weights.cpu().numpy()
                logits = model(values)
                return torch.sigmoid(logits).cpu().numpy(), None

        for fold_name, train_indices, validation_indices in _development_splits(
            metadata, config
        ):
            started = time.perf_counter()
            model, _, normalized, fold_history, iterations = fit_model(
                train_indices, validation_indices, fold=fold_name
            )
            development_fit_seconds += time.perf_counter() - started
            best_iterations.append(iterations)
            history.extend(fold_history)
            probabilities, _ = predict(model, normalized, validation_indices, False)
            dev_frames.append(
                _prediction_frame(
                    metadata,
                    validation_indices,
                    probabilities,
                    fold=fold_name,
                )
            )

        train_indices, test_indices = split_sequence_indices(
            dataset, test_start=config["split"]["final_holdout_start"]
        )
        final_epochs = int(np.median(best_iterations))
        started = time.perf_counter()
        model, normalizer, normalized, final_history, _ = fit_model(
            train_indices, epochs=final_epochs, fold="final_refit"
        )
        final_fit_seconds = time.perf_counter() - started
        history.extend(final_history)
        capture_attention = (
            spec.model == "attention_tcn"
            and spec.seed == config["reporting"]["primary_seed_for_window_traces"]
        )
        started = time.perf_counter()
        probabilities, attention_weights = predict(
            model, normalized, test_indices, capture_attention
        )
        prediction_seconds = time.perf_counter() - started
        torch.save(
            {
                "model": spec.model,
                "state_dict": model.state_dict(),
                "normalizer": normalizer.to_manifest(),
                "channels": list(dataset.channels),
            },
            artifact_path,
        )
        return TrainingResult(
            development_predictions=pd.concat(dev_frames, ignore_index=True),
            holdout_predictions=_prediction_frame(
                metadata, test_indices, probabilities
            ),
            history=history,
            best_iterations=best_iterations,
            development_fit_seconds=development_fit_seconds,
            final_fit_seconds=final_fit_seconds,
            prediction_seconds=prediction_seconds,
            parameter_count=temporal_parameter_count(model),
            model_complexity={"trainable_parameters": temporal_parameter_count(model)},
            normalizer_manifest=normalizer.to_manifest(),
            attention_weights=attention_weights,
        )


class DefaultCellTrainer:
    def train(self, spec, dataset, config, artifact_path):
        if spec.model == "xgboost":
            return XGBoostCellTrainer().train(
                spec, dataset, config, artifact_path.with_suffix(".joblib")
            )
        return TorchCellTrainer().train(
            spec, dataset, config, artifact_path.with_suffix(".pt")
        )


def _cell_complete(cell_path: Path, spec: CellSpec, config_sha: str) -> bool:
    manifest_path = cell_path / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "complete" or manifest.get("cell") != asdict(spec):
        return False
    if manifest.get("config_sha256") != config_sha:
        return False
    file_hashes = manifest.get("file_sha256", {})
    required = {
        "development_predictions.csv",
        "holdout_probabilities.csv",
        "history.json",
    }
    if not required.issubset(file_hashes):
        return False
    if len([name for name in file_hashes if name.startswith("model.")]) != 1:
        return False
    for relative, expected_hash in file_hashes.items():
        path = cell_path / relative
        if not path.exists() or _sha256(path) != expected_hash:
            return False
    return True


def run_cell(
    spec: CellSpec,
    dataset: SequenceDataset,
    output_root: str | Path,
    *,
    config: dict[str, Any] | None = None,
    trainer: CellTrainer | None = None,
) -> str:
    experiment = load_experiment_config() if config is None else config
    if spec not in experiment_cells(experiment):
        raise ValueError(f"Cell is outside the frozen experiment matrix: {spec.key}")
    config_sha = _config_sha256(experiment)
    cell_path = Path(output_root) / spec.key
    if _cell_complete(cell_path, spec, config_sha):
        return "skipped"
    if cell_path.exists():
        raise RuntimeError(f"Invalid completed checkpoint requires review: {cell_path}")

    temporary = cell_path.parent / f".{cell_path.name}-{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        labeled = _labeled_dataset(dataset, spec.horizon_hours)
        result = (trainer or DefaultCellTrainer()).train(
            spec, labeled, experiment, temporary / "model"
        )
        for frame in (
            result.development_predictions,
            result.holdout_predictions,
        ):
            frame["model"] = spec.model
            frame["horizon_hours"] = spec.horizon_hours
            frame["seed"] = spec.seed
        _validate_predictions(result.development_predictions, development=True)
        _validate_predictions(result.holdout_predictions, development=False)
        result.development_predictions.to_csv(
            temporary / "development_predictions.csv", index=False
        )
        result.holdout_predictions.to_csv(
            temporary / "holdout_probabilities.csv", index=False
        )
        (temporary / "history.json").write_text(
            json.dumps(result.history, indent=2), encoding="utf-8"
        )
        if result.attention_weights is not None:
            np.save(temporary / "attention_weights.npy", result.attention_weights)
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        model_files = [path for path in files if path.name.startswith("model.")]
        if len(model_files) != 1:
            raise ValueError("Trainer must write exactly one model artifact")
        final_train, final_test = split_sequence_indices(
            labeled, test_start=experiment["split"]["final_holdout_start"]
        )
        development_counts = {
            name: {"train": len(train), "validation": len(validation)}
            for name, train, validation in _development_splits(
                labeled.metadata, experiment
            )
        }
        if spec.model == "xgboost":
            inputs = {"feature_columns": feature_columns(labeled.metadata)}
            model_parameters = experiment["models"]["xgboost"]
        else:
            inputs = {"sequence_channels": list(labeled.channels)}
            model_parameters = {
                "encoder": experiment["models"]["shared_tcn_encoder"],
                "head": experiment["models"][spec.model],
            }
        manifest = {
            "status": "complete",
            "cell": asdict(spec),
            "config_sha256": config_sha,
            "dataset_sha256": experiment["audit_evidence"]["dataset_sha256"],
            "inputs": inputs,
            "model_parameters": model_parameters,
            "normalization_fit_scope": experiment["sequence"]["normalization"][
                "fit_scope"
            ],
            "split": {
                "development": development_counts,
                "final_holdout_start": experiment["split"]["final_holdout_start"],
                "final_train_windows": len(final_train),
                "final_test_windows": len(final_test),
                "raw_timestamp_overlap_purged": True,
            },
            "best_iterations": result.best_iterations,
            "timing_seconds": {
                "development_fit": result.development_fit_seconds,
                "final_fit": result.final_fit_seconds,
                "holdout_prediction": result.prediction_seconds,
            },
            "parameter_count": result.parameter_count,
            "model_complexity": result.model_complexity,
            "artifact_bytes": model_files[0].stat().st_size,
            "normalizer": result.normalizer_manifest,
            "package_versions": _package_versions(),
            "file_sha256": {path.name: _sha256(path) for path in files},
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, cell_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "completed"


def experiment_cells(config: dict[str, Any]) -> list[CellSpec]:
    return [
        CellSpec(model, int(horizon), int(seed))
        for model in config["models"]["order"]
        for horizon in config["targets"]["horizon_hours"]
        for seed in config["training"]["seeds"]
    ]


def run_experiment(
    dataset: SequenceDataset,
    output_root: str | Path,
    *,
    config: dict[str, Any] | None = None,
    trainer: CellTrainer | None = None,
    cells: list[CellSpec] | None = None,
) -> dict[str, int]:
    experiment = load_experiment_config() if config is None else config
    counts = {"completed": 0, "skipped": 0}
    for spec in cells or experiment_cells(experiment):
        status = run_cell(
            spec,
            dataset,
            output_root,
            config=experiment,
            trainer=trainer,
        )
        counts[status] += 1
    for model in experiment["models"]["order"]:
        for horizon in experiment["targets"]["horizon_hours"]:
            finalize_model_horizon(output_root, model, int(horizon), experiment)
    return counts


def finalize_model_horizon(
    output_root: str | Path,
    model: str,
    horizon_hours: int,
    config: dict[str, Any],
) -> bool:
    """Apply one development-selected threshold after every seed is complete."""
    group_path = Path(output_root) / model / f"h{horizon_hours}"
    development_frames = []
    holdout_frames = []
    config_sha = _config_sha256(config)
    for seed in config["training"]["seeds"]:
        spec = CellSpec(model, horizon_hours, int(seed))
        cell_path = Path(output_root) / spec.key
        if not _cell_complete(cell_path, spec, config_sha):
            return False
        development_frames.append(
            pd.read_csv(cell_path / "development_predictions.csv")
        )
        holdout_frames.append(pd.read_csv(cell_path / "holdout_probabilities.csv"))

    development = pd.concat(development_frames, ignore_index=True)
    holdout = pd.concat(holdout_frames, ignore_index=True)
    selected = select_operational_threshold(development, config)
    holdout["reference_prediction"] = (
        holdout["probability"] >= float(config["thresholds"]["reference"])
    ).astype(int)
    holdout["operational_prediction"] = (
        holdout["probability"] >= selected["threshold"]
    ).astype(int)

    predictions_temp = group_path / f".predictions-{uuid.uuid4().hex}.tmp"
    summary_temp = group_path / f".threshold-{uuid.uuid4().hex}.tmp"
    holdout.to_csv(predictions_temp, index=False)
    summary = {
        "model": model,
        "horizon_hours": horizon_hours,
        "selection_scope": config["thresholds"]["operational_selection_scope"],
        "seeds": config["training"]["seeds"],
        "reference_threshold": config["thresholds"]["reference"],
        "operational": selected,
        "holdout_predictions_sha256": _sha256(predictions_temp),
    }
    summary_temp.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(predictions_temp, group_path / "holdout_predictions.csv")
    os.replace(summary_temp, group_path / "threshold.json")
    return True


def false_alert_episodes(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    expected_step_seconds: int | None = None,
) -> int:
    ordered = frame.sort_values(["segment_id", "window_start"]).copy()
    ordered["window_start"] = pd.to_datetime(ordered["window_start"])
    predicted = ordered[prediction_column].astype(bool)
    groups = ordered["segment_id"]
    previous = predicted.groupby(groups).shift(fill_value=False)
    after_gap = pd.Series(False, index=ordered.index)
    if expected_step_seconds is not None:
        previous_start = ordered.groupby("segment_id")["window_start"].shift()
        after_gap = (
            ordered["window_start"] - previous_start
        ).dt.total_seconds() > expected_step_seconds
    new_run = predicted & (~previous | after_gap)
    return int((new_run & ordered["true_label"].eq(0)).sum())


def select_operational_threshold(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> dict[str, float]:
    threshold_config = config["thresholds"]
    candidates = np.arange(
        threshold_config["candidate_start"],
        threshold_config["candidate_stop"] + threshold_config["candidate_step"] / 2,
        threshold_config["candidate_step"],
    )
    best = None
    for threshold in candidates:
        candidate = predictions.copy()
        candidate["prediction"] = candidate["probability"] >= threshold
        f2 = float(
            fbeta_score(
                candidate["true_label"], candidate["prediction"], beta=2, zero_division=0
            )
        )
        false_alerts = false_alert_episodes(
            candidate,
            "prediction",
            expected_step_seconds=int(config["windows"]["step_seconds"]),
        )
        rank = (f2, -false_alerts, float(threshold))
        if best is None or rank > best[0]:
            best = (rank, threshold, f2, false_alerts)
    assert best is not None
    return {
        "threshold": float(round(best[1], 10)),
        "development_f2": best[2],
        "development_false_alert_episodes": int(best[3]),
    }
