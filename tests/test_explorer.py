import json

import pandas as pd
import pytest

from metropt3.explorer import (
    filter_eventwise_rows,
    filter_metric_rows,
    event_failure_time,
    load_event_regime_evidence,
    summarize_metric_rows,
    threshold_for,
)


def test_threshold_is_loaded_from_development_selection():
    thresholds = [
        {
            "model": "xgboost",
            "horizon_hours": 3,
            "operational": {"threshold": 0.38},
        }
    ]
    assert threshold_for(
        thresholds,
        model="xgboost",
        horizon_hours=3,
        policy="development_selected",
    ) == pytest.approx(0.38)
    assert threshold_for(
        thresholds,
        model="xgboost",
        horizon_hours=3,
        policy="reference_0.5",
    ) == pytest.approx(0.5)


def test_metric_summary_preserves_threshold_policies():
    frame = pd.DataFrame(
        {
            "model": ["xgboost"] * 4,
            "horizon_hours": [1] * 4,
            "seed": [17, 42, 17, 42],
            "threshold_policy": ["reference_0.5"] * 2
            + ["development_selected"] * 2,
            "threshold": [0.5, 0.5, 0.3, 0.3],
            "average_precision": [0.1] * 4,
            "ap_lift_over_prevalence": [2.0] * 4,
            "roc_auc": [0.6] * 4,
            "prevalence": [0.05] * 4,
            "balanced_accuracy": [0.5] * 4,
            "precision": [0.0, 0.0, 0.1, 0.2],
            "recall": [0.0, 0.0, 1.0, 1.0],
            "false_alert_episodes_per_day": [0, 0, 2, 4],
            "held_out_event_detected": [False, False, True, True],
            "first_alert_lead_hours": [float("nan"), float("nan"), 1.0, 2.0],
        }
    )
    result = summarize_metric_rows(frame, model="xgboost", horizon_hours=1)
    operational = result.loc[
        result["threshold_policy"].eq("development_selected")
    ].iloc[0]
    assert operational["precision"] == pytest.approx(0.15)
    assert operational["false_alert_episodes_per_day"] == pytest.approx(3.0)
    assert operational["event_detection_rate"] == pytest.approx(1.0)


def test_failure_times_come_from_frozen_config(tmp_path):
    config = {
        "split": {
            "final_failure_start": "2020-07-15 14:30:00",
            "development_folds": [
                {
                    "name": "may_failure",
                    "validation_end": "2020-05-29 23:30:00",
                }
            ],
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert event_failure_time(path, "july_holdout") == pd.Timestamp(
        "2020-07-15 14:30"
    )
    assert event_failure_time(path, "may_failure") == pd.Timestamp(
        "2020-05-29 23:30"
    )


def test_metric_filter_defaults_to_all_seeds_and_can_select_one():
    frame = pd.DataFrame(
        {
            "model": ["xgboost", "xgboost", "tcn"],
            "horizon_hours": [1, 1, 1],
            "seed": [17, 42, 17],
            "threshold_policy": ["reference_0.5"] * 3,
        }
    )
    aggregate = filter_metric_rows(frame, model="xgboost", horizon_hours=1)
    assert aggregate["seed"].tolist() == [17, 42]
    selected = filter_metric_rows(frame, model="xgboost", horizon_hours=1, seed=42)
    assert selected["seed"].tolist() == [42]


def test_event_filter_preserves_only_requested_evidence():
    frame = pd.DataFrame(
        {
            "model": ["xgboost", "tcn", "tcn"],
            "horizon_hours": [1, 1, 3],
            "event": ["may_failure", "july_holdout", "july_holdout"],
        }
    )
    selected = filter_eventwise_rows(
        frame, model="tcn", horizon_hours=1, event="july_holdout"
    )
    assert len(selected) == 1
    assert selected.iloc[0].to_dict() == {
        "model": "tcn",
        "horizon_hours": 1,
        "event": "july_holdout",
    }


def test_empty_filter_selection_fails_clearly():
    frame = pd.DataFrame(
        {"model": ["xgboost"], "horizon_hours": [1], "event": ["may_failure"]}
    )
    with pytest.raises(ValueError, match="No event-wise evidence"):
        filter_eventwise_rows(frame, model="tcn")


def test_event_regime_loader_validates_required_columns(tmp_path):
    pd.DataFrame(
        {
            "feature": ["TP2_mean"],
            "may_failure": [0.2],
            "june_failure": [0.1],
            "july_holdout": [4.5],
            "direction_consistency": [1.0],
        }
    ).to_csv(tmp_path / "event_regime_summary.csv", index=False)
    pd.DataFrame(
        {
            "event": ["may_failure", "june_failure", "july_holdout"],
            "feature": ["TP2_mean"] * 3,
            "robust_standardized_difference": [0.2, 0.1, 4.5],
            "event_windows": [38, 48, 33],
        }
    ).to_csv(tmp_path / "event_regime_effects.csv", index=False)
    summary, effects = load_event_regime_evidence(tmp_path)
    assert summary["feature"].tolist() == ["TP2_mean"]
    assert set(effects["event"]) == {"may_failure", "june_failure", "july_holdout"}
