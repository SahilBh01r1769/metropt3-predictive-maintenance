import json

import numpy as np
import pandas as pd
import pytest

from metropt3.experiment_config import load_experiment_config
from metropt3.experiment_runner import (
    CellSpec,
    TrainingResult,
    experiment_cells,
    false_alert_episodes,
    run_cell,
    run_experiment,
    select_operational_threshold,
)
from metropt3.sequences import SequenceDataset
from metropt3.temporal_models import build_temporal_model


class FakeTrainer:
    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail

    def train(self, spec, dataset, config, artifact_path):
        self.calls += 1
        artifact_path.with_suffix(".fake").write_bytes(b"model")
        if self.fail:
            raise RuntimeError("interrupted")

        metadata = dataset.metadata
        common = {
            "window_id": metadata["window_id"].iloc[:2].to_numpy(),
            "segment_id": metadata["segment_id"].iloc[:2].to_numpy(),
            "window_start": metadata["window_start"].iloc[:2].to_numpy(),
            "window_end": metadata["window_end"].iloc[:2].to_numpy(),
            "true_label": [0, 1],
            "probability": [0.2, 0.8],
            "hours_to_next_failure": [10.0, 1.0],
        }
        development = pd.DataFrame({**common, "fold": ["fold-a", "fold-a"]})
        return TrainingResult(
            development_predictions=development,
            holdout_predictions=pd.DataFrame(common),
            history=[{"epoch": 1}],
            best_iterations=[1, 1],
            development_fit_seconds=1.0,
            final_fit_seconds=2.0,
            prediction_seconds=0.1,
            parameter_count=12,
            model_complexity={"trainable_parameters": 12},
        )


def synthetic_dataset():
    starts = pd.to_datetime(
        [
            "2020-05-20 00:00",
            "2020-05-20 01:00",
            "2020-05-20 02:00",
            "2020-05-20 03:00",
            "2020-05-20 04:00",
            "2020-05-23 00:00",
            "2020-05-29 22:00",
            "2020-05-30 10:00",
            "2020-06-05 08:00",
            "2020-07-08 12:00",
            "2020-07-08 13:00",
            "2020-07-08 14:00",
            "2020-07-08 15:00",
        ]
    )
    metadata = pd.DataFrame(
        {
            "window_id": [f"window-{index}" for index in range(len(starts))],
            "segment_id": 0,
            "window_start": starts,
            "window_end": starts + pd.Timedelta(hours=1),
            "in_failure": False,
        }
    )
    values = np.ones((len(starts), 4, 2), dtype=np.float32)
    return SequenceDataset(values, metadata, ("sensor", "coverage"))


def test_experiment_matrix_is_exactly_three_models_by_four_horizons_by_three_seeds():
    config = load_experiment_config()

    cells = experiment_cells(config)

    assert len(cells) == 36
    assert {cell.model for cell in cells} == {"xgboost", "tcn", "attention_tcn"}
    assert {cell.horizon_hours for cell in cells} == {1, 3, 6, 12}
    assert {cell.seed for cell in cells} == {17, 42, 89}

    with pytest.raises(ValueError, match="outside the frozen experiment matrix"):
        run_cell(
            CellSpec("transformer", 24, 100),
            synthetic_dataset(),
            ".unused",
            config=config,
            trainer=FakeTrainer(),
        )


def test_cell_checkpoint_is_atomic_and_hash_verified(tmp_path):
    config = load_experiment_config()
    spec = CellSpec("tcn", 3, 42)
    trainer = FakeTrainer()

    first = run_cell(
        spec, synthetic_dataset(), tmp_path, config=config, trainer=trainer
    )
    second = run_cell(
        spec, synthetic_dataset(), tmp_path, config=config, trainer=trainer
    )

    assert first == "completed"
    assert second == "skipped"
    assert trainer.calls == 1
    cell_path = tmp_path / spec.key
    manifest = json.loads((cell_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["cell"] == {"model": "tcn", "horizon_hours": 3, "seed": 42}
    assert manifest["parameter_count"] == 12
    assert manifest["artifact_bytes"] == 5

    (cell_path / "history.json").write_text("tampered")
    with pytest.raises(RuntimeError, match="requires review"):
        run_cell(spec, synthetic_dataset(), tmp_path, config=config, trainer=trainer)


def test_interrupted_cell_leaves_no_completed_or_temporary_checkpoint(tmp_path):
    config = load_experiment_config()
    spec = CellSpec("attention_tcn", 6, 17)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_cell(
            spec,
            synthetic_dataset(),
            tmp_path,
            config=config,
            trainer=FakeTrainer(fail=True),
        )

    assert not (tmp_path / spec.key).exists()
    assert not list((tmp_path / spec.model / f"h{spec.horizon_hours}").glob("*.tmp"))


def test_runner_counts_completed_and_resumed_cells(tmp_path):
    config = load_experiment_config()
    cells = [CellSpec("xgboost", 1, 17), CellSpec("tcn", 1, 17)]
    trainer = FakeTrainer()

    first = run_experiment(
        synthetic_dataset(), tmp_path, config=config, trainer=trainer, cells=cells
    )
    second = run_experiment(
        synthetic_dataset(), tmp_path, config=config, trainer=trainer, cells=cells
    )

    assert first == {"completed": 2, "skipped": 0}
    assert second == {"completed": 0, "skipped": 2}


def test_complete_seed_group_gets_one_development_selected_threshold(tmp_path):
    config = load_experiment_config()
    cells = [CellSpec("xgboost", 1, seed) for seed in config["training"]["seeds"]]

    run_experiment(
        synthetic_dataset(),
        tmp_path,
        config=config,
        trainer=FakeTrainer(),
        cells=cells,
    )

    group = tmp_path / "xgboost" / "h1"
    threshold = json.loads((group / "threshold.json").read_text())
    predictions = pd.read_csv(group / "holdout_predictions.csv")
    assert threshold["selection_scope"].endswith("development_fold_predictions_only")
    assert set(predictions["seed"]) == {17, 42, 89}
    assert {"reference_prediction", "operational_prediction"}.issubset(
        predictions.columns
    )


def test_false_alert_episode_resets_on_gap_and_segment_boundary():
    frame = pd.DataFrame(
        {
            "segment_id": [0, 0, 0, 1],
            "window_start": pd.to_datetime(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 00:30",
                    "2020-01-01 02:00",
                    "2020-01-01 02:30",
                ]
            ),
            "true_label": [0, 0, 0, 0],
            "prediction": [1, 1, 1, 1],
        }
    )

    assert false_alert_episodes(
        frame, "prediction", expected_step_seconds=1800
    ) == 3


def test_threshold_selection_prefers_fewer_alerts_then_higher_threshold():
    config = load_experiment_config()
    config["thresholds"].update(
        {"candidate_start": 0.4, "candidate_stop": 0.6, "candidate_step": 0.1}
    )
    predictions = pd.DataFrame(
        {
            "segment_id": [0, 0, 0, 0],
            "window_start": pd.date_range("2020-01-01", periods=4, freq="30min"),
            "true_label": [0, 0, 1, 1],
            "probability": [0.1, 0.5, 0.8, 0.9],
        }
    )

    selected = select_operational_threshold(predictions, config)

    assert selected["threshold"] == 0.6
    assert selected["development_f2"] == 1.0
    assert selected["development_false_alert_episodes"] == 0


def test_temporal_model_factory_rejects_unfrozen_model_before_importing_torch():
    with pytest.raises(ValueError, match="Unsupported temporal model"):
        build_temporal_model("transformer", input_channels=9, config={})
