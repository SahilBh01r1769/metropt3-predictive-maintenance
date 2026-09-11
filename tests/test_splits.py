import numpy as np
import pandas as pd

from metropt3.splits import chronological_split


def window_frame(ends, **columns):
    ends = pd.DatetimeIndex(ends)
    return pd.DataFrame(
        {
            "window_start": ends - pd.Timedelta(hours=1),
            "window_end": ends,
            **columns,
        }
    )


def raw_timestamps(windows, freq):
    return {
        timestamp
        for row in windows.itertuples()
        for timestamp in pd.date_range(
            row.window_start,
            row.window_end,
            freq=freq,
            inclusive="left",
        )
    }


def test_chronological_split_preserves_time_order():
    frame = window_frame(
        pd.date_range("2020-01-01", periods=10, freq="h"),
        failure_within_horizon=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    )
    train, test = chronological_split(frame, test_start="2020-01-01 08:00:00")
    assert train["window_end"].max() < test["window_end"].min()


def test_split_uses_fixed_boundary_independently_of_labels():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    target = np.zeros(40, dtype=int)
    target[[5, 6, 18, 19, 33, 34]] = 1
    frame = window_frame(dates, failure_within_horizon=target)
    frame["window_start"] = frame["window_end"] - pd.Timedelta(days=2)
    boundary = pd.Timestamp("2020-02-02")
    train, test = chronological_split(frame, test_start=boundary)

    relabeled = frame.copy()
    relabeled["failure_within_horizon"] = 1 - relabeled["failure_within_horizon"]
    relabeled_train, relabeled_test = chronological_split(
        relabeled, test_start=boundary
    )

    assert train["window_end"].max() <= test["window_start"].min()
    assert raw_timestamps(train, "h").isdisjoint(raw_timestamps(test, "h"))
    assert train["window_end"].tolist() == relabeled_train["window_end"].tolist()
    assert test["window_end"].tolist() == relabeled_test["window_end"].tolist()


def test_split_purges_every_raw_timestamp_shared_with_test_interval():
    starts = pd.date_range("2020-01-01", periods=10, freq="30min")
    frame = pd.DataFrame(
        {
            "window_start": starts,
            "window_end": starts + pd.Timedelta(hours=1),
        }
    )
    train, test = chronological_split(frame, test_start="2020-01-01 04:30:00")
    assert raw_timestamps(train, "10min").isdisjoint(raw_timestamps(test, "10min"))
    assert train["window_end"].max() <= test["window_start"].min()
    assert len(train) == 6


def test_split_rejects_windows_without_raw_time_bounds():
    frame = pd.DataFrame(
        {"window_end": pd.date_range("2020-01-01", periods=10, freq="h")}
    )
    with np.testing.assert_raises_regex(ValueError, "Window bounds"):
        chronological_split(frame, test_start="2020-01-01 08:00:00")


def test_fixed_boundary_does_not_fall_back_when_one_side_is_empty():
    frame = window_frame(pd.date_range("2020-01-01", periods=10, freq="h"))
    with np.testing.assert_raises_regex(ValueError, "must leave windows on both sides"):
        chronological_split(frame, test_start="2021-01-01")
