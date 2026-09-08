from copy import deepcopy

import numpy as np

from metropt3.experiment_config import (
    load_experiment_config,
    tcn_receptive_field,
    validate_experiment_config,
)


def test_frozen_configuration_covers_the_full_sequence_history():
    config = load_experiment_config()

    assert config["sequence"]["resample_interval_seconds"] == 30
    assert config["sequence"]["timesteps"] == 120
    assert config["models"]["order"] == ["xgboost", "tcn", "attention_tcn"]
    assert tcn_receptive_field(config["models"]["shared_tcn_encoder"]) == 125
    assert config["training"]["seeds"] == [17, 42, 89]


def test_configuration_rejects_a_sequence_length_mismatch():
    config = load_experiment_config()
    changed = deepcopy(config)
    changed["sequence"]["timesteps"] = 119

    with np.testing.assert_raises_regex(ValueError, "timesteps"):
        validate_experiment_config(changed)


def test_configuration_rejects_a_different_attention_encoder():
    config = load_experiment_config()
    changed = deepcopy(config)
    changed["models"]["attention_tcn"]["encoder"] = "larger_encoder"

    with np.testing.assert_raises_regex(ValueError, "same encoder"):
        validate_experiment_config(changed)


def test_configuration_rejects_holdout_preprocessing_or_threshold_tuning():
    config = load_experiment_config()

    normalization_changed = deepcopy(config)
    normalization_changed["sequence"]["normalization"]["fit_scope"] = "all_data"
    with np.testing.assert_raises_regex(ValueError, "training data only"):
        validate_experiment_config(normalization_changed)

    threshold_changed = deepcopy(config)
    threshold_changed["thresholds"]["operational_selection_scope"] = "final_holdout"
    with np.testing.assert_raises_regex(ValueError, "development predictions only"):
        validate_experiment_config(threshold_changed)


def test_configuration_rejects_overlapping_development_folds():
    config = load_experiment_config()
    changed = deepcopy(config)
    changed["split"]["development_folds"][1]["validation_start"] = (
        "2020-05-29 12:00:00"
    )

    with np.testing.assert_raises_regex(ValueError, "cannot overlap"):
        validate_experiment_config(changed)
