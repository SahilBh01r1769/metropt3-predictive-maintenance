import pandas as pd
import pytest

from metropt3.result_analysis import (
    build_eventwise_metrics,
    ranking_metrics,
    summarize_eventwise_metrics,
)


def test_ranking_metrics_exposes_score_direction_and_lift():
    frame = pd.DataFrame(
        {
            "true_label": [0, 0, 1, 1],
            "probability": [0.1, 0.2, 0.8, 0.9],
        }
    )
    result = ranking_metrics(frame)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["ap_lift_over_prevalence"] == pytest.approx(2.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["median_score_gap"] == pytest.approx(0.7)


def test_ranking_metrics_rejects_unscorable_or_invalid_traces():
    with pytest.raises(ValueError, match="both classes"):
        ranking_metrics(pd.DataFrame({"true_label": [0, 0], "probability": [0.1, 0.2]}))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ranking_metrics(pd.DataFrame({"true_label": [0, 1], "probability": [0.1, 1.2]}))


def test_event_summary_keeps_seed_variation_separate_from_event_count():
    eventwise = pd.DataFrame(
        {
            "model": ["tcn", "tcn", "tcn"],
            "horizon_hours": [1, 1, 1],
            "seed": [17, 42, 89],
            "event": ["may_failure"] * 3,
            "prevalence": [0.1] * 3,
            "average_precision": [0.2, 0.3, 0.4],
            "ap_lift_over_prevalence": [2.0, 3.0, 4.0],
            "roc_auc": [0.6, 0.7, 0.8],
            "median_score_gap": [0.1, 0.2, 0.3],
        }
    )
    summary = summarize_eventwise_metrics(eventwise).iloc[0]
    assert summary["seeds"] == 3
    assert summary["average_precision_mean"] == pytest.approx(0.3)
    assert summary["average_precision_std"] == pytest.approx(0.1)
    assert summary["ap_lift_mean"] == pytest.approx(3.0)


def test_eventwise_builder_keeps_development_events_separate_from_holdout(tmp_path):
    development = tmp_path / "development"
    holdout = tmp_path / "holdout"
    development.mkdir()
    holdout.mkdir()
    pd.DataFrame(
        {
            "seed": [17, 17, 17, 17],
            "fold": ["may_failure", "may_failure", "june_failure", "june_failure"],
            "true_label": [0, 1, 0, 1],
            "probability": [0.1, 0.9, 0.2, 0.8],
        }
    ).to_csv(development / "tcn_h1.csv.gz", index=False)
    pd.DataFrame(
        {
            "seed": [17, 17],
            "true_label": [0, 1],
            "probability": [0.3, 0.7],
        }
    ).to_csv(holdout / "tcn_h1.csv.gz", index=False)

    result = build_eventwise_metrics(tmp_path)

    assert result["event"].astype(str).tolist() == [
        "may_failure",
        "june_failure",
        "july_holdout",
    ]
    assert result["average_precision"].tolist() == [1.0, 1.0, 1.0]
