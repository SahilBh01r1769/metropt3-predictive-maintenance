import json

import pandas as pd
import pytest

from metropt3.explorer import event_failure_time, summarize_metric_rows, threshold_for


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
