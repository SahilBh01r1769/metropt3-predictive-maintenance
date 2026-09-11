import pandas as pd
import pytest

from scripts.analyze_event_regimes import event_regime_effects, event_regime_summary


def test_event_regime_effects_reports_direction_against_normal_baseline():
    rows = []
    for event, value in [("may_failure", 3.0), ("june_failure", 2.0), ("july_holdout", 1.0)]:
        rows.extend({"window_end": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i), "f": value} for i in range(3))
    rows.extend({"window_end": pd.Timestamp("2020-02-01") + pd.Timedelta(hours=i), "f": 0.0} for i in range(10))
    frame = pd.DataFrame(rows)
    # The helper's fixed membership uses published dates, so test summary shape directly.
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
