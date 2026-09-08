from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ANALOGUE_COLS, DIGITAL_COLS, ROOT


DEFAULT_EXPERIMENT_CONFIG = ROOT / "configs" / "temporal_experiment.json"
EXPECTED_MODELS = ["xgboost", "tcn", "attention_tcn"]
EXPECTED_HORIZONS = [1, 3, 6, 12]


def tcn_receptive_field(encoder: dict[str, Any]) -> int:
    """Return receptive field in timesteps for the frozen causal TCN stack."""
    kernel_size = int(encoder["kernel_size"])
    convolutions = int(encoder["convolutions_per_block"])
    dilations = [int(value) for value in encoder["dilations"]]
    return 1 + convolutions * (kernel_size - 1) * sum(dilations)


def validate_experiment_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment configuration schema")

    horizons = config["targets"]["horizon_hours"]
    if horizons != EXPECTED_HORIZONS:
        raise ValueError(f"Horizons must remain {EXPECTED_HORIZONS}")

    model_order = config["models"]["order"]
    if model_order != EXPECTED_MODELS:
        raise ValueError(f"Model order must remain {EXPECTED_MODELS}")

    sequence = config["sequence"]
    expected_steps = config["windows"]["history_seconds"] // sequence[
        "resample_interval_seconds"
    ]
    if config["windows"]["history_seconds"] % sequence[
        "resample_interval_seconds"
    ]:
        raise ValueError("Resampling interval must divide the history exactly")
    if sequence["timesteps"] != expected_steps:
        raise ValueError("Sequence timesteps do not match history and resampling interval")

    expected_channels = [*ANALOGUE_COLS, *DIGITAL_COLS]
    if sequence["channels"] != expected_channels:
        raise ValueError("Sequence channels must match the validated feature evidence")
    if sequence["normalization"]["fit_scope"] != "training_only":
        raise ValueError("Sequence normalization must be fitted on training data only")

    models = config["models"]
    if models["tcn"]["encoder"] != "shared_tcn_encoder":
        raise ValueError("TCN must use the shared encoder")
    if models["attention_tcn"]["encoder"] != "shared_tcn_encoder":
        raise ValueError("Attention-TCN must use the same encoder as TCN")
    if not models["shared_tcn_encoder"].get("causal"):
        raise ValueError("The temporal encoder must remain causal")
    if tcn_receptive_field(models["shared_tcn_encoder"]) < sequence["timesteps"]:
        raise ValueError("TCN receptive field must cover the full input history")

    split = config["split"]
    holdout_start = pd.Timestamp(split["final_holdout_start"])
    final_failure = pd.Timestamp(split["final_failure_start"])
    if holdout_start >= final_failure:
        raise ValueError("Final holdout must begin before the held-out failure")
    if not split.get("purge_raw_timestamp_overlap"):
        raise ValueError("Raw timestamp overlap purge cannot be disabled")
    fold_bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for fold in split["development_folds"]:
        start = pd.Timestamp(fold["validation_start"])
        end = pd.Timestamp(fold["validation_end"])
        if start >= end or end >= holdout_start:
            raise ValueError("Development folds must be ordered before final holdout")
        fold_bounds.append((start, end))
    if any(left[1] > right[0] for left, right in zip(fold_bounds, fold_bounds[1:])):
        raise ValueError("Development folds cannot overlap")

    training = config["training"]
    seeds = training["seeds"]
    if len(seeds) != len(set(seeds)) or training["primary_seed"] not in seeds:
        raise ValueError("Training seeds must be unique and include the primary seed")
    if config["reporting"]["primary_seed_for_window_traces"] != training[
        "primary_seed"
    ]:
        raise ValueError("Window-trace seed must match the primary training seed")

    thresholds = config["thresholds"]
    if thresholds["operational_selection_scope"] != (
        "pooled_development_fold_predictions_only"
    ):
        raise ValueError("Operational thresholds must use development predictions only")
    if not (
        0 < thresholds["candidate_start"]
        < thresholds["candidate_stop"]
        < 1
    ):
        raise ValueError("Threshold candidate range must stay inside (0, 1)")


def load_experiment_config(
    path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_experiment_config(config)
    evidence_path = ROOT / config["audit_evidence"]["summary_path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence["audit_code_revision"] != config["audit_evidence"][
        "audit_code_revision"
    ]:
        raise ValueError("Audit revision does not match its frozen evidence")
    if evidence["dataset"]["sha256"] != config["audit_evidence"][
        "dataset_sha256"
    ]:
        raise ValueError("Dataset hash does not match its frozen audit evidence")
    return config
