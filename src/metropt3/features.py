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
    """Create time windows strictly within each segment."""
    if "segment_id" not in df.columns:
        raise ValueError("segment_id is required; run validate_and_segment first")
    rows: list[dict] = []
    min_rows = max(1, int(window_seconds * min_coverage))

    for segment_id, segment in df.groupby("segment_id", sort=True):
        segment = segment.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        if segment.empty:
            continue
        times = segment[TIMESTAMP_COL]
        start = times.iloc[0]
        last = times.iloc[-1]
        while start + pd.Timedelta(seconds=window_seconds) <= last + pd.Timedelta(seconds=1):
            end = start + pd.Timedelta(seconds=window_seconds)
            mask = (times >= start) & (times < end)
            chunk = segment.loc[mask]
            if len(chunk) >= min_rows:
                row = {
                    "segment_id": int(segment_id),
                    "window_start": start,
                    "window_end": end,
                    "rows": int(len(chunk)),
                }
                row.update(_window_features(chunk))
                rows.append(row)
            start += pd.Timedelta(seconds=step_seconds)
    return pd.DataFrame(rows)
