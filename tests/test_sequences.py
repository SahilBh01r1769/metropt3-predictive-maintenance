import numpy as np
import pandas as pd
import pytest

from metropt3.config import ANALOGUE_COLS, DIGITAL_COLS
from metropt3.sequences import (
    OBSERVATION_CHANNEL,
    SequenceDataset,
    build_sequence_dataset,
    fit_sequence_normalizer,
    split_sequence_indices,
)


SOURCE_CHANNELS = [*ANALOGUE_COLS, *DIGITAL_COLS]
SEQUENCE_CHANNELS = (*SOURCE_CHANNELS, OBSERVATION_CHANNEL)


def source_segment(segment_id, start, *, value, seconds=3600):
    timestamps = pd.date_range(start, periods=seconds // 10, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "segment_id": segment_id,
            **{channel: float(value) for channel in SOURCE_CHANNELS},
        }
    )
    frame["COMP"] = (np.arange(len(frame)) % 6 < 3).astype(float)
    return frame


def window(segment_id, start, *, in_failure=False, cadence_seconds=10.0):
    start = pd.Timestamp(start)
    return {
        "segment_id": segment_id,
        "window_start": start,
        "window_end": start + pd.Timedelta(hours=1),
        "cadence_seconds": cadence_seconds,
        "in_failure": in_failure,
        "failure_within_horizon": 0,
    }


def test_builder_creates_one_sequence_for_each_non_failure_window():
    source = source_segment(0, "2020-04-01", value=2.0)
    windows = pd.DataFrame(
        [
            window(0, "2020-04-01"),
            window(0, "2020-04-01", in_failure=True),
        ]
    )

    dataset = build_sequence_dataset(source, windows)

    assert dataset.values.shape == (1, 120, 9)
    assert dataset.channels == SEQUENCE_CHANNELS
    assert len(dataset.metadata) == 1
    assert not dataset.metadata["in_failure"].any()
    np.testing.assert_allclose(dataset.values[0, :, -1], 1.0)
    np.testing.assert_allclose(dataset.values[0, :, 0], 2.0)
    np.testing.assert_allclose(dataset.values[0, ::2, 7], 1.0)
    np.testing.assert_allclose(dataset.values[0, 1::2, 7], 0.0)


def test_builder_never_mixes_continuity_segments():
    source = pd.concat(
        [
            source_segment(0, "2020-04-01", value=1.0),
            source_segment(1, "2020-04-02", value=9.0),
        ],
        ignore_index=True,
    )
    windows = pd.DataFrame(
        [window(0, "2020-04-01"), window(1, "2020-04-02")]
    )

    dataset = build_sequence_dataset(source, windows)

    np.testing.assert_allclose(dataset.values[0, :, 0], 1.0)
    np.testing.assert_allclose(dataset.values[1, :, 0], 9.0)


def test_builder_marks_empty_bins_and_limits_within_window_filling():
    source = source_segment(0, "2020-04-01", value=2.0)
    seconds = (source["timestamp"] - source["timestamp"].min()).dt.total_seconds()
    source = source.loc[
        ~seconds.between(300, 329) & ~seconds.between(600, 659)
    ].copy()

    dataset = build_sequence_dataset(
        source, pd.DataFrame([window(0, "2020-04-01")])
    )

    assert dataset.values[0, 10, -1] == 0.0
    assert dataset.values[0, 20, -1] == 0.0
    assert dataset.values[0, 21, -1] == 0.0
    assert dataset.values[0, 10, 0] == 2.0
    assert dataset.values[0, 20, 0] == 2.0
    assert np.isnan(dataset.values[0, 21, 0])
    assert not np.isnan(dataset.values[0, 10, 7])
    assert not np.isnan(dataset.values[0, 20, 7])
    assert np.isnan(dataset.values[0, 21, 7])


def test_builder_rejects_duplicate_feature_windows():
    source = source_segment(0, "2020-04-01", value=2.0)
    repeated = window(0, "2020-04-01")

    with pytest.raises(ValueError, match="must be unique"):
        build_sequence_dataset(source, pd.DataFrame([repeated, repeated]))


def test_normalizer_uses_only_named_training_windows():
    metadata = pd.DataFrame(
        {
            "window_id": ["train-a", "train-b", "test"],
            "window_start": pd.date_range("2020-01-01", periods=3, freq="h"),
            "window_end": pd.date_range("2020-01-01 01:00", periods=3, freq="h"),
        }
    )
    values = np.empty((3, 2, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    values[0, :, :] = 1.0
    values[1, :, :] = 3.0
    values[2, :, :] = 1000.0
    values[2, 0, 0] = np.nan
    dataset = SequenceDataset(values, metadata, SEQUENCE_CHANNELS)

    normalizer = fit_sequence_normalizer(dataset, ["train-a", "train-b"])
    transformed = normalizer.transform(dataset)

    assert normalizer.training_window_count == 2
    assert normalizer.fallback_values[0] == 2.0
    assert normalizer.means[0] == 2.0
    assert normalizer.means[7] == 0.0
    assert normalizer.means[8] == 0.0
    np.testing.assert_allclose(transformed.values[:2, :, 0].mean(), 0.0)
    assert transformed.values[2, 1, 0] > 100
    assert transformed.values[2, 0, 0] == 0.0
    np.testing.assert_allclose(transformed.values[:, :, 7], values[:, :, 7])
    np.testing.assert_allclose(transformed.values[:, :, 8], values[:, :, 8])


def test_normalizer_rejects_unknown_or_repeated_training_ids():
    metadata = pd.DataFrame({"window_id": ["a"]})
    values = np.ones((1, 2, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    dataset = SequenceDataset(values, metadata, SEQUENCE_CHANNELS)

    with pytest.raises(ValueError, match="Unknown training window IDs"):
        fit_sequence_normalizer(dataset, ["missing"])
    with pytest.raises(ValueError, match="must be unique"):
        fit_sequence_normalizer(dataset, ["a", "a"])


def test_sequence_split_preserves_raw_timestamp_disjointness():
    starts = pd.date_range("2020-07-08 00:00", periods=7, freq="h")
    metadata = pd.DataFrame(
        {
            "window_id": [f"window-{index}" for index in range(7)],
            "window_start": starts,
            "window_end": starts + pd.Timedelta(hours=1),
        }
    )
    values = np.ones((7, 2, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    dataset = SequenceDataset(values, metadata, SEQUENCE_CHANNELS)

    train_indices, test_indices = split_sequence_indices(
        dataset, test_start="2020-07-08 05:00"
    )

    train = metadata.iloc[train_indices]
    test = metadata.iloc[test_indices]
    assert train["window_end"].max() <= test["window_start"].min()
    assert "window-4" in set(test["window_id"])
    assert set(train["window_id"]).isdisjoint(test["window_id"])
