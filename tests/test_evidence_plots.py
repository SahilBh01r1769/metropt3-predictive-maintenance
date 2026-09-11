from pathlib import Path

import pandas as pd
import pytest

from metropt3.evidence_plots import event_timeline_data, load_evidence


def test_evidence_loader_rejects_incomplete_metrics(tmp_path: Path):
    pd.DataFrame({"model": ["xgboost"]}).to_csv(tmp_path / "metrics.csv", index=False)
    pd.DataFrame({"event": ["may_failure"]}).to_csv(
        tmp_path / "eventwise_summary.csv", index=False
    )
    with pytest.raises(ValueError, match="Metrics evidence is missing"):
        load_evidence(tmp_path)


def test_timeline_aggregates_seeds_without_using_post_failure_rows():
    trace = pd.DataFrame(
        {
            "hours_to_next_failure": [2, 2, 1, 1, -1, 30],
            "probability": [0.2, 0.4, 0.6, 0.8, 0.99, 0.99],
            "seed": [17, 42, 17, 42, 17, 17],
        }
    )
    result = event_timeline_data(trace, max_lead_hours=24)
    assert result["hours_to_next_failure"].tolist() == [2, 1]
    assert result["probability_mean"].tolist() == pytest.approx([0.3, 0.7])
