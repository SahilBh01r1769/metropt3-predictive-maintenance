import numpy as np
import pandas as pd

from metropt3.features import build_windows
from metropt3.labels import add_failure_labels
from metropt3.validation import validate_and_segment


def frame(start="2020-04-17 10:00:00", seconds=7200):
    ts = pd.date_range(start, periods=seconds, freq="s")
    comp = (np.arange(seconds) % 120 < 60).astype(int)
    return pd.DataFrame({
        "timestamp": ts,
        "TP2": 2.0 + comp * 7,
        "TP3": 8.8,
        "H1": 8.0,
        "DV_pressure": 0.1,
        "Reservoirs": 8.7,
        "Oil_temperature": 65.0,
        "Motor_current": 3.0 + comp * 5,
        "COMP": comp,
    })


def test_gap_creates_new_segment_and_windows_do_not_cross_it():
    df = frame(seconds=4000)
    df.loc[2000:, "timestamp"] += pd.Timedelta(seconds=20)
    valid, report = validate_and_segment(df)
    assert report.segments == 2
    windows = build_windows(valid, window_seconds=1200, step_seconds=600)
    assert not windows.empty
    assert windows.groupby("segment_id").size().shape[0] == 2


def test_invalid_row_is_quarantined():
    df = frame(seconds=100)
    df.loc[3, "Oil_temperature"] = 999
    valid, report = validate_and_segment(df)
    assert report.quarantined_rows == 1
    assert len(valid) == 99


def test_failure_horizon_label_before_known_failure():
    df = frame(start="2020-04-17 10:00:00", seconds=7200)
    valid, _ = validate_and_segment(df)
    windows = build_windows(valid, window_seconds=3600, step_seconds=1800)
    labeled = add_failure_labels(windows, horizon_hours=12)
    assert labeled["failure_within_horizon"].max() == 1
    assert not labeled["in_failure"].any()
