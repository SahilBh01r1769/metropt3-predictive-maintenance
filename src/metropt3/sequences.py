from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ANALOGUE_COLS, DIGITAL_COLS, TIMESTAMP_COL
from .experiment_config import load_experiment_config
from .modeling import FINAL_EVENT_TEST_START, chronological_split


OBSERVATION_CHANNEL = "bin_observed_fraction"


@dataclass(frozen=True)
class SequenceDataset:
    """Ordered sensor tensors and their auditable feature-window metadata."""

    values: np.ndarray
    metadata: pd.DataFrame
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("Sequence values must have shape (windows, steps, channels)")
        if self.values.shape[0] != len(self.metadata):
            raise ValueError("Sequence and metadata window counts differ")
        if self.values.shape[2] != len(self.channels):
            raise ValueError("Sequence channel count does not match channel names")
        if "window_id" not in self.metadata:
            raise ValueError("Sequence metadata must contain window_id")
        if self.metadata["window_id"].duplicated().any():
            raise ValueError("window_id values must be unique")


@dataclass(frozen=True)
class SequenceNormalizer:
    """Training-fitted missing-value and scaling parameters."""

    channels: tuple[str, ...]
    fallback_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    standardized_channels: tuple[str, ...]
    training_window_count: int
    training_window_ids_sha256: str

    def transform(self, dataset: SequenceDataset) -> SequenceDataset:
        if dataset.channels != self.channels:
            raise ValueError("Dataset channels do not match fitted normalizer")

        values = dataset.values.astype(np.float32, copy=True)
        for channel_index, fallback in enumerate(self.fallback_values):
            missing = np.isnan(values[:, :, channel_index])
            values[:, :, channel_index][missing] = fallback

        for channel in self.standardized_channels:
            channel_index = self.channels.index(channel)
            values[:, :, channel_index] = (
                values[:, :, channel_index] - self.means[channel_index]
            ) / self.scales[channel_index]

        return SequenceDataset(
            values=values,
            metadata=dataset.metadata.copy(),
            channels=dataset.channels,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "channels": list(self.channels),
            "fallback_values": self.fallback_values.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "standardized_channels": list(self.standardized_channels),
            "training_window_count": self.training_window_count,
            "training_window_ids_sha256": self.training_window_ids_sha256,
        }


def _window_id(segment_id: int, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{segment_id}:{start.isoformat()}:{end.isoformat()}"


def _prepare_windows(windows: pd.DataFrame) -> pd.DataFrame:
    required = {"segment_id", "window_start", "window_end", "in_failure"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError("Missing window columns: " + ", ".join(sorted(missing)))

    prepared = windows.copy()
    prepared["window_start"] = pd.to_datetime(prepared["window_start"])
    prepared["window_end"] = pd.to_datetime(prepared["window_end"])
    if prepared[["window_start", "window_end"]].isna().any().any():
        raise ValueError("Window bounds cannot be missing")
    if (prepared["window_end"] <= prepared["window_start"]).any():
        raise ValueError("Each window_end must be later than window_start")

    prepared = prepared.loc[~prepared["in_failure"].astype(bool)].copy()
    prepared["window_id"] = [
        _window_id(int(segment), start, end)
        for segment, start, end in zip(
            prepared["segment_id"],
            prepared["window_start"],
            prepared["window_end"],
        )
    ]
    if prepared["window_id"].duplicated().any():
        raise ValueError("Accepted feature windows must be unique")
    return prepared.sort_values(["window_end", "window_start"]).reset_index(drop=True)


def build_sequence_dataset(
    source: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
) -> SequenceDataset:
    """Resample accepted one-hour windows without crossing continuity segments."""
    experiment = load_experiment_config() if config is None else config
    sequence = experiment["sequence"]
    history_seconds = int(experiment["windows"]["history_seconds"])
    interval_seconds = int(sequence["resample_interval_seconds"])
    timesteps = int(sequence["timesteps"])
    source_channels = tuple(sequence["channels"])
    output_channels = (*source_channels, OBSERVATION_CHANNEL)

    required_source = {TIMESTAMP_COL, "segment_id", *source_channels}
    missing_source = required_source.difference(source.columns)
    if missing_source:
        raise ValueError("Missing source columns: " + ", ".join(sorted(missing_source)))
    if history_seconds != interval_seconds * timesteps:
        raise ValueError("Sequence dimensions do not cover the configured history")

    prepared_windows = _prepare_windows(windows)
    grouped_source = {
        int(segment_id): segment.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        for segment_id, segment in source.groupby("segment_id", sort=False)
    }
    tensors: list[np.ndarray] = []

    analogue_indices = [source_channels.index(channel) for channel in ANALOGUE_COLS]
    comp_index = source_channels.index(DIGITAL_COLS[0])
    maximum_missing = int(sequence["missing_bins"]["maximum_consecutive_bins"])

    for row in prepared_windows.itertuples(index=False):
        segment_id = int(row.segment_id)
        if segment_id not in grouped_source:
            raise ValueError(f"No source observations for segment {segment_id}")

        segment = grouped_source[segment_id]
        times = pd.DatetimeIndex(pd.to_datetime(segment[TIMESTAMP_COL]))
        left = int(times.searchsorted(row.window_start, side="left"))
        right = int(times.searchsorted(row.window_end, side="left"))
        chunk = segment.iloc[left:right].copy()
        if chunk.empty:
            raise ValueError(f"Feature window {row.window_id} contains no source rows")

        offsets = (
            pd.to_datetime(chunk[TIMESTAMP_COL]) - row.window_start
        ).dt.total_seconds()
        bins = (offsets // interval_seconds).astype(int)
        chunk = chunk.loc[(bins >= 0) & (bins < timesteps)].copy()
        chunk["_bin"] = bins.loc[chunk.index].to_numpy()

        aggregated = chunk.groupby("_bin", sort=True)[list(source_channels)].mean()
        aggregated = aggregated.reindex(range(timesteps))

        for channel_index in analogue_indices:
            channel = source_channels[channel_index]
            aggregated[channel] = aggregated[channel].interpolate(
                method="linear",
                limit=maximum_missing,
                limit_area="inside",
            )
        comp_channel = source_channels[comp_index]
        aggregated[comp_channel] = aggregated[comp_channel].ffill(
            limit=maximum_missing
        )

        cadence_seconds = float(getattr(row, "cadence_seconds", np.nan))
        if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
            deltas = times.to_series().diff().dt.total_seconds()
            positive = deltas[deltas > 0]
            if positive.empty:
                raise ValueError(f"Cannot determine cadence for {row.window_id}")
            cadence_seconds = float(positive.median())
        expected_per_bin = max(1.0, interval_seconds / cadence_seconds)
        observed = (
            chunk.groupby("_bin").size().reindex(range(timesteps), fill_value=0)
            / expected_per_bin
        ).clip(upper=1.0)
        aggregated[OBSERVATION_CHANNEL] = observed.astype(float)
        tensors.append(aggregated[list(output_channels)].to_numpy(dtype=np.float32))

    if tensors:
        values = np.stack(tensors)
    else:
        values = np.empty((0, timesteps, len(output_channels)), dtype=np.float32)
    return SequenceDataset(
        values=values,
        metadata=prepared_windows,
        channels=output_channels,
    )


def fit_sequence_normalizer(
    dataset: SequenceDataset,
    training_window_ids: Iterable[str],
) -> SequenceNormalizer:
    ids = list(training_window_ids)
    if not ids:
        raise ValueError("At least one training window is required")
    if len(ids) != len(set(ids)):
        raise ValueError("Training window IDs must be unique")

    index_by_id = {
        window_id: index
        for index, window_id in enumerate(dataset.metadata["window_id"].tolist())
    }
    unknown = sorted(set(ids).difference(index_by_id))
    if unknown:
        raise ValueError("Unknown training window IDs: " + ", ".join(unknown))
    selected = dataset.values[[index_by_id[window_id] for window_id in ids]]

    fallback = np.nanmedian(selected, axis=(0, 1)).astype(np.float32)
    if np.isnan(fallback).any():
        missing_channels = [
            dataset.channels[index]
            for index in np.flatnonzero(np.isnan(fallback))
        ]
        raise ValueError(
            "Training windows contain no observations for: "
            + ", ".join(missing_channels)
        )
    filled = selected.copy()
    for channel_index, value in enumerate(fallback):
        missing = np.isnan(filled[:, :, channel_index])
        filled[:, :, channel_index][missing] = value

    means = np.zeros(len(dataset.channels), dtype=np.float32)
    scales = np.ones(len(dataset.channels), dtype=np.float32)
    for channel in ANALOGUE_COLS:
        channel_index = dataset.channels.index(channel)
        means[channel_index] = float(filled[:, :, channel_index].mean())
        scale = float(filled[:, :, channel_index].std())
        scales[channel_index] = scale if np.isfinite(scale) and scale > 0 else 1.0

    id_digest = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    return SequenceNormalizer(
        channels=dataset.channels,
        fallback_values=fallback,
        means=means,
        scales=scales,
        standardized_channels=tuple(ANALOGUE_COLS),
        training_window_count=len(ids),
        training_window_ids_sha256=id_digest,
    )


def split_sequence_indices(
    dataset: SequenceDataset,
    *,
    test_start: str | pd.Timestamp = FINAL_EVENT_TEST_START,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the fixed, purged chronological split back to tensor indices."""
    train, test = chronological_split(dataset.metadata, test_start=test_start)
    index_by_id = {
        window_id: index
        for index, window_id in enumerate(dataset.metadata["window_id"].tolist())
    }
    train_indices = np.asarray(
        [index_by_id[window_id] for window_id in train["window_id"]], dtype=int
    )
    test_indices = np.asarray(
        [index_by_id[window_id] for window_id in test["window_id"]], dtype=int
    )
    if (
        dataset.metadata.iloc[train_indices]["window_end"].max()
        > dataset.metadata.iloc[test_indices]["window_start"].min()
    ):
        raise AssertionError("Raw timestamp intervals overlap after sequence split")
    return train_indices, test_indices
