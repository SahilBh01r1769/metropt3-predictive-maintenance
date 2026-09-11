import numpy as np
import pandas as pd
import pytest

from metropt3.temporal_features import add_temporal_context


def windows(values, *, segment_ids=None):
    count = len(values)
    ends = pd.date_range("2020-01-01 01:00:00", periods=count, freq="30min")
    frame = pd.DataFrame(
        {
            "segment_id": segment_ids or [0] * count,
            "window_start": ends - pd.Timedelta(hours=1),
            "window_end": ends,
            "TP2_mean": values,
        }
    )
    return frame


def test_lags_and_rolling_statistics_do_not_cross_segments():
    frame = windows([1.0, 2.0, 100.0, 101.0], segment_ids=[0, 0, 1, 1])

    result = add_temporal_context(
        frame,
        signal_columns=["TP2_mean"],
        rolling_hours=[3],
        minimum_history_windows=1,
    )

    assert result.loc[1, "temporal__TP2_mean__lag1"] == 1.0
    assert np.isnan(result.loc[2, "temporal__TP2_mean__lag1"])
    assert result.loc[3, "temporal__TP2_mean__mean_3h"] == 100.0


def test_rolling_baseline_uses_only_prior_windows():
    original = windows([1.0, 2.0, 3.0, 4.0])
    changed_future = original.copy()
    changed_future.loc[3, "TP2_mean"] = 10000.0

    first = add_temporal_context(
        original,
        signal_columns=["TP2_mean"],
        rolling_hours=[3],
        minimum_history_windows=1,
    )
    second = add_temporal_context(
        changed_future,
        signal_columns=["TP2_mean"],
        rolling_hours=[3],
        minimum_history_windows=1,
    )

    columns = [column for column in first if column.startswith("temporal__")]
    pd.testing.assert_frame_equal(first.loc[:2, columns], second.loc[:2, columns])
    assert first.loc[2, "temporal__TP2_mean__mean_3h"] == pytest.approx(1.5)


def test_slope_uses_elapsed_time_and_never_a_future_value():
    frame = windows([0.0, 1.0, 2.0, 200.0])

    result = add_temporal_context(
        frame,
        signal_columns=["TP2_mean"],
        rolling_hours=[3],
        minimum_history_windows=1,
    )

    assert result.loc[2, "temporal__TP2_mean__slope_3h"] == pytest.approx(2.0)


def test_invalid_temporal_feature_inputs_fail_explicitly():
    with pytest.raises(ValueError, match="segment_id"):
        add_temporal_context(
            windows([1.0, 2.0]).drop(columns="segment_id"),
            signal_columns=["TP2_mean"],
        )
    with pytest.raises(ValueError, match="unique"):
        duplicate = windows([1.0, 2.0])
        duplicate.loc[1, "window_end"] = duplicate.loc[0, "window_end"]
        add_temporal_context(duplicate, signal_columns=["TP2_mean"])
