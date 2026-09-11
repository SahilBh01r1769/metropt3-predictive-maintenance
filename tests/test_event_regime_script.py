import pandas as pd
import pytest

from scripts.analyze_event_regimes import event_regime_summary


def test_event_regime_summary_reports_consistent_direction():
    effects = pd.DataFrame(
        {
            "event": ["may_failure", "june_failure", "july_holdout"],
            "feature": ["f"] * 3,
            "robust_standardized_difference": [3.0, 2.0, 1.0],
        }
    )
    result = event_regime_summary(effects)
    assert result.loc[0, "direction_consistency"] == pytest.approx(1.0)
    assert bool(result.loc[0, "all_events_same_direction"])
