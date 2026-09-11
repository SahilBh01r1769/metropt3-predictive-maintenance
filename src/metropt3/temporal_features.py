from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_TEMPORAL_SIGNALS = (
    "TP2_mean",
    "TP3_mean",
    "pressure_diff_mean",
    "DV_pressure_mean",
    "Reservoirs_mean",
    "Oil_temperature_mean",
    "Motor_current_mean",
    "motor_current_volatility",
    "comp_duty_cycle",
)
DEFAULT_ROLLING_HOURS = (3, 6, 12)


def _trailing_slope(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lookback_hours: int,
) -> np.ndarray:
    """Return a causal least-squares slope per hour for each trailing interval."""
    result = np.full(len(values), np.nan, dtype=float)
    seconds = timestamps.astype("int64").to_numpy(dtype=np.int64) / 1e9
    lookback_seconds = lookback_hours * 3600
    for right in range(len(values)):
        left = int(np.searchsorted(seconds, seconds[right] - lookback_seconds, side="left"))
        selected_values = values[left : right + 1]
        selected_seconds = seconds[left : right + 1]
        finite = np.isfinite(selected_values)
        if finite.sum() < 2:
            continue
        x = (selected_seconds[finite] - selected_seconds[finite][-1]) / 3600
        if np.ptp(x) == 0:
            continue
        result[right] = float(np.polyfit(x, selected_values[finite], 1)[0])
    return result


def temporal_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("temporal__")]


def add_temporal_context(
    windows: pd.DataFrame,
    *,
    signal_columns: Iterable[str] = DEFAULT_TEMPORAL_SIGNALS,
    rolling_hours: Iterable[int] = DEFAULT_ROLLING_HOURS,
    minimum_history_windows: int = 2,
) -> pd.DataFrame:
    """Add causal lag, rolling, baseline-distance and trend features.

    Every calculation is performed independently inside one continuity segment. Rolling
    statistics are shifted by one window, so a row's baseline contains only windows
    whose end time is no later than the preceding observation. Slopes include the
    current window but never a future one.
    """
    required = {"segment_id", "window_start", "window_end", *signal_columns}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError("Temporal features require: " + ", ".join(sorted(missing)))
    lookbacks = tuple(sorted({int(hours) for hours in rolling_hours}))
    if not lookbacks or any(hours <= 0 for hours in lookbacks):
        raise ValueError("rolling_hours must contain positive values")
    if minimum_history_windows < 1:
        raise ValueError("minimum_history_windows must be positive")

    output = windows.copy()
    output["window_start"] = pd.to_datetime(output["window_start"])
    output["window_end"] = pd.to_datetime(output["window_end"])
    if (output["window_end"] <= output["window_start"]).any():
        raise ValueError("window_end must be after window_start")
    if output.duplicated(["segment_id", "window_end"]).any():
        raise ValueError("Each segment/window_end pair must be unique")
    output["_input_order"] = np.arange(len(output))

    transformed: list[pd.DataFrame] = []
    for _, segment in output.groupby("segment_id", sort=True):
        segment = segment.sort_values("window_end").copy()
        timestamps = pd.DatetimeIndex(segment["window_end"])
        if not timestamps.is_monotonic_increasing:
            raise AssertionError("Window ordering failed")

        for column in signal_columns:
            values = pd.to_numeric(segment[column], errors="coerce")
            if values.isna().all():
                raise ValueError(f"{column} contains no numeric values")
            prefix = f"temporal__{column}"
            segment[f"{prefix}__lag1"] = values.shift(1)
            segment[f"{prefix}__delta1"] = values - values.shift(1)

            past = pd.Series(values.shift(1).to_numpy(), index=timestamps)
            rolling_means: dict[int, np.ndarray] = {}
            for hours in lookbacks:
                rolling = past.rolling(
                    f"{hours}h",
                    closed="both",
                    min_periods=minimum_history_windows,
                )
                mean = rolling.mean().to_numpy()
                rolling_means[hours] = mean
                segment[f"{prefix}__mean_{hours}h"] = mean
                segment[f"{prefix}__variance_{hours}h"] = rolling.var(ddof=0).to_numpy()
                segment[f"{prefix}__slope_{hours}h"] = _trailing_slope(
                    values.to_numpy(dtype=float), timestamps, hours
                )

            longest = max(lookbacks)
            segment[f"{prefix}__distance_from_{longest}h_baseline"] = (
                values.to_numpy(dtype=float) - rolling_means[longest]
            )
        transformed.append(segment)

    result = pd.concat(transformed, ignore_index=True)
    result = result.sort_values("_input_order").drop(columns="_input_order")
    return result.reset_index(drop=True)
