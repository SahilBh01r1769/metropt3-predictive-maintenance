from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANALOGUE_COLS, STEP_SECONDS, TIMESTAMP_COL, WINDOW_SECONDS


def _window_features(chunk: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ANALOGUE_COLS:
        values = chunk[col].to_numpy(dtype=float)
        out[f"{col}_mean"] = float(np.mean(values))
        out[f"{col}_std"] = float(np.std(values))
        out[f"{col}_min"] = float(np.min(values))
        out[f"{col}_max"] = float(np.max(values))
        out[f"{col}_roc"] = float((values[-1] - values[0]) / max(1, len(values) - 1))
    out["pressure_diff_mean"] = float((chunk["TP3"] - chunk["TP2"]).mean())
    out["comp_duty_cycle"] = float(chunk["COMP"].mean())
    out["motor_current_volatility"] = float(chunk["Motor_current"].diff().abs().mean())
    return out


def build_windows(
    df: pd.DataFrame,
    *,
    window_seconds: int = WINDOW_SECONDS,
    step_seconds: int = STEP_SECONDS,
    min_coverage: float = 0.8,
) -> pd.DataFrame:
    """Create time windows strictly within each segment.

    ``searchsorted`` is used instead of scanning the entire segment for every
    window so the implementation remains practical on the full MetroPT-3 file.
    """
    if "segment_id" not in df.columns:
        raise ValueError("segment_id is required; run validate_and_segment first")
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be positive")

    rows: list[dict] = []
    min_rows = max(1, int(window_seconds * min_coverage))
    window_delta = pd.Timedelta(seconds=window_seconds)
    step_delta = pd.Timedelta(seconds=step_seconds)

    for segment_id, segment in df.groupby("segment_id", sort=True):
        segment = segment.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        if segment.empty:
            continue
        times = pd.DatetimeIndex(segment[TIMESTAMP_COL])
        start = times[0]
        last = times[-1]
        while start + window_delta <= last + pd.Timedelta(seconds=1):
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
                }
                row.update(_window_features(chunk))
                rows.append(row)
            start += step_delta
    return pd.DataFrame(rows)
