import numpy as np
import pandas as pd

from metropt3.modeling import chronological_split, train_and_evaluate


def window_frame(ends, **columns):
    ends = pd.DatetimeIndex(ends)
    return pd.DataFrame({
        "window_start": ends - pd.Timedelta(hours=1),
        "window_end": ends,
        **columns,
    })


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
    df = window_frame(
        pd.date_range("2020-01-01", periods=10, freq="h"),
        in_failure=False,
        failure_within_horizon=[0,0,0,1,0,0,1,0,1,0],
        f1=np.arange(10, dtype=float),
    )
    train, test = chronological_split(df, 0.2)
    assert train["window_end"].max() < test["window_end"].min()


def test_split_holds_out_final_positive_episode_when_viable():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    target = np.zeros(40, dtype=int)
    target[[5, 6, 18, 19, 33, 34]] = 1
    df = window_frame(
        dates,
        in_failure=False,
        failure_within_horizon=target,
        f1=np.arange(40, dtype=float),
    )
    df["window_start"] = df["window_end"] - pd.Timedelta(days=2)
    train, test = chronological_split(df, event_context_days=7)
    assert train["window_end"].max() <= test["window_start"].min()
    assert raw_timestamps(train, "h").isdisjoint(raw_timestamps(test, "h"))
    assert train["failure_within_horizon"].sum() >= 2
    assert test["failure_within_horizon"].sum() >= 1
    assert pd.Timestamp("2020-02-04") in set(test["window_end"])


def test_split_purges_every_raw_timestamp_shared_with_test_interval():
    starts = pd.date_range("2020-01-01", periods=10, freq="30min")
    df = pd.DataFrame({
        "window_start": starts,
        "window_end": starts + pd.Timedelta(hours=1),
        "in_failure": False,
        "f1": np.arange(10, dtype=float),
    })

    train, test = chronological_split(df, test_fraction=0.2)

    assert raw_timestamps(train, "10min").isdisjoint(
        raw_timestamps(test, "10min")
    )
    assert train["window_end"].max() <= test["window_start"].min()
    assert len(train) == 7  # the eighth provisional train window overlapped


def test_split_rejects_windows_without_raw_time_bounds():
    df = pd.DataFrame({
        "window_end": pd.date_range("2020-01-01", periods=10, freq="h"),
        "f1": np.arange(10, dtype=float),
    })

    with np.testing.assert_raises_regex(ValueError, "Window bounds"):
        chronological_split(df)


def test_training_returns_metrics():
    n = 40
    y = np.array(([0, 1] * 20), dtype=int)
    df = window_frame(
        pd.date_range("2020-01-01", periods=n, freq="h"),
        in_failure=False,
        failure_within_horizon=y,
        sensor_mean=y + np.linspace(0, 0.1, n),
        sensor_std=0.2 + y * 0.1,
    )
    _, metrics, cols = train_and_evaluate(df)
    assert metrics.train_windows > metrics.test_windows
    assert "sensor_mean" in cols
    assert 0 <= metrics.balanced_accuracy <= 1
