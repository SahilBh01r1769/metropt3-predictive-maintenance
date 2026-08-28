import numpy as np
import pandas as pd

from metropt3.modeling import chronological_split, train_and_evaluate


def test_chronological_split_preserves_time_order():
    df = pd.DataFrame({
        "window_end": pd.date_range("2020-01-01", periods=10, freq="h"),
        "in_failure": False,
        "failure_within_horizon": [0,0,0,1,0,0,1,0,1,0],
        "f1": np.arange(10, dtype=float),
    })
    train, test = chronological_split(df, 0.2)
    assert train["window_end"].max() < test["window_end"].min()


def test_training_returns_metrics():
    n = 40
    y = np.array(([0, 1] * 20), dtype=int)
    df = pd.DataFrame({
        "window_end": pd.date_range("2020-01-01", periods=n, freq="h"),
        "in_failure": False,
        "failure_within_horizon": y,
        "sensor_mean": y + np.linspace(0, 0.1, n),
        "sensor_std": 0.2 + y * 0.1,
    })
    _, metrics, cols = train_and_evaluate(df)
    assert metrics.train_windows > metrics.test_windows
    assert "sensor_mean" in cols
    assert 0 <= metrics.balanced_accuracy <= 1
