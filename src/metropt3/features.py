from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANALOGUE_COLS, STEP_SECONDS, TIMESTAMP_COL, WINDOW_SECONDS


def _window_features(chunk: pd.DataFrame) -> dict[str, float]:
    times = pd.to_datetime(chunk[TIMESTAMP_COL])
    elapsed_hours = (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600
    if elapsed_hours <= 0:
        raise ValueError("A feature window needs at least two increasing timestamps")

    out: dict[str, float] = {}
    for col in ANALOGUE_COLS:
        values = chunk[col].to_numpy(dtype=float)
        out[f"{col}_mean"] = float(np.mean(values))
        out[f"{col}_std"] = float(np.std(values))
        out[f"{col}_min"] = float(np.min(values))
        out[f"{col}_max"] = float(np.max(values))
        out[f"{col}_roc_per_hour"] = float(
            (values[-1] - values[0]) / elapsed_hours
        )
    out["pressure_diff_mean"] = float((chunk["TP3"] - chunk["TP2"]).mean())
    out["comp_duty_cycle"] = float(chunk["COMP"].mean())
    out["motor_current_volatility"] = float(chunk["Motor_current"].diff().abs().mean())
    return out


def _observed_cadence_seconds(times: pd.DatetimeIndex) -> float | None:
    if len(times) < 2:
        return None
    diffs = pd.Series(times).diff().dt.total_seconds()
    positive = diffs[diffs > 0]
    if positive.empty:
        return None
    cadence = float(positive.median())
    return cadence if np.isfinite(cadence) and cadence > 0 else None


def build_windows(
    df: pd.DataFrame,
    *,
    window_seconds: int = WINDOW_SECONDS,
    step_seconds: int = STEP_SECONDS,
    min_coverage: float = 0.8,
) -> pd.DataFrame:
    """Create time windows strictly within each segment.

    Coverage is based on each segment's observed median timestamp cadence rather
    than assuming one row per second. ``searchsorted`` avoids scanning the full
    segment for every window and keeps the implementation practical on MetroPT-3.
    """
    if "segment_id" not in df.columns:
        raise ValueError("segment_id is required; run validate_and_segment first")
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be positive")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")

    rows: list[dict] = []
    window_delta = pd.Timedelta(seconds=window_seconds)
    step_delta = pd.Timedelta(seconds=step_seconds)

    for segment_id, segment in df.groupby("segment_id", sort=True):
        segment = segment.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        if segment.empty:
            continue

        times = pd.DatetimeIndex(segment[TIMESTAMP_COL])
        cadence_seconds = _observed_cadence_seconds(times)
        if cadence_seconds is None or cadence_seconds > window_seconds:
            continue

        expected_rows = max(1, int(round(window_seconds / cadence_seconds)))
        min_rows = max(1, int(np.floor(expected_rows * min_coverage)))
        start = times[0]
        last = times[-1]

        while start + window_delta <= last + pd.Timedelta(seconds=cadence_seconds):
            end = start + window_delta
            left = int(times.searchsorted(start, side="left"))
            right = int(times.searchsorted(end, side="left"))
            chunk = segment.iloc[left:right]
            if len(chunk) >= min_rows:
                row = {
                    "segment_id": int(segment_id),
                    "window_start": start,
                    "window_end": end,
                    "rows": int(len(chunk)),
                    "cadence_seconds": cadence_seconds,
                }
                row.update(_window_features(chunk))
                rows.append(row)
            start += step_delta

    return pd.DataFrame(rows)
