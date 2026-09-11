import json

import pandas as pd
import pytest

from metropt3.labels import add_failure_labels
from metropt3.maintenance_metrics import maintenance_metrics, select_threshold, threshold_table
from metropt3.tabular_experiment import (
    EventFold,
    chronological_event_split,
    load_tabular_config,
)


def metric_frame(probabilities, labels, *, folds=None):
    count = len(labels)
    starts = pd.date_range("2020-01-01", periods=count, freq="30min")
    return pd.DataFrame(
        {
            "segment_id": [0] * count,
            "window_start": starts,
            "true_label": labels,
            "probability": probabilities,
            "hours_to_next_failure": list(reversed(range(1, count + 1))),
            "fold": folds or ["event"] * count,
        }
    )


def test_horizon_labels_expand_at_expected_boundaries():
    failure = pd.Timestamp("2020-05-29 23:30:00")
    windows = pd.DataFrame(
        {
            "window_start": [failure - pd.Timedelta(hours=31)],
            "window_end": [failure - pd.Timedelta(hours=30)],
        }
    )
    assert add_failure_labels(windows, horizon_hours=24)["failure_within_horizon"].iloc[0] == 0
    assert add_failure_labels(windows, horizon_hours=48)["failure_within_horizon"].iloc[0] == 1


def test_event_fold_is_chronological_and_raw_intervals_are_purged():
    ends = pd.date_range("2020-01-01 01:00", periods=10, freq="30min")
    frame = pd.DataFrame(
        {
            "segment_id": [0] * 10,
            "window_start": ends - pd.Timedelta(hours=1),
            "window_end": ends,
            "failure_within_horizon": [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            "in_failure": [False] * 10,
        }
    )
    fold = EventFold("test", ends[7], ends[9])
    train, validation = chronological_event_split(
        frame, fold, temporal_context_hours=1
    )
    assert train["window_end"].max() <= validation["window_start"].min()
    assert train["window_end"].max() <= (
        validation["window_start"].min() - pd.Timedelta(hours=1)
    )
    assert train["window_end"].max() < fold.validation_start
    assert validation["window_end"].max() <= fold.failure_start


def test_maintenance_metrics_count_event_detection_lead_and_alert_runs():
    frame = metric_frame(
        [0.8, 0.7, 0.1, 0.9, 0.8, 0.2],
        [0, 0, 0, 1, 1, 1],
    )
    result = maintenance_metrics(frame, threshold=0.5)
    assert result["event_recall"] == 1.0
    assert result["mean_first_warning_lead_hours"] == 3.0
    assert result["false_alert_episodes"] == 1
    assert result["warning_episodes"] == 1
    assert result["max_warning_run_windows"] == 2


def test_threshold_selection_uses_only_the_supplied_development_predictions():
    development = metric_frame([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    table = threshold_table(development, [0.2, 0.5, 0.95])
    selected = select_threshold(table)
    unrelated_test = metric_frame([0.99, 0.99], [0, 1])
    unrelated_test["probability"] = 1 - unrelated_test["probability"]
    assert select_threshold(table) == selected
    assert selected == 0.5


def test_tabular_configuration_rejects_reordered_event_folds(tmp_path):
    config_path = tmp_path / "config.json"
    config = load_tabular_config()
    config["event_folds"] = list(reversed(config["event_folds"]))
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="chronological event folds"):
        load_tabular_config(config_path)
