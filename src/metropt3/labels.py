from __future__ import annotations

import pandas as pd

from .config import FAILURE_HORIZON_HOURS, FAILURE_INTERVALS


def failure_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"failure_start": pd.Timestamp(start), "failure_end": pd.Timestamp(end), "failure_type": kind}
            for start, end, kind in FAILURE_INTERVALS
        ]
    )


def add_failure_labels(
    windows: pd.DataFrame,
    *,
    horizon_hours: float = FAILURE_HORIZON_HOURS,
) -> pd.DataFrame:
    """Label a window positive when a published failure begins within the future horizon.

    Windows overlapping an active failure are marked ``in_failure`` and excluded from the
    predictive target by callers. This avoids training a pre-failure classifier on data
    captured after the failure has already begun.
    """
    if "window_end" not in windows.columns:
        raise ValueError("window_end is required")
    out = windows.copy()
    out["window_end"] = pd.to_datetime(out["window_end"])
    failures = failure_table()
    horizon = pd.Timedelta(hours=horizon_hours)

    targets: list[int] = []
    in_failure: list[bool] = []
    hours_to_failure: list[float | None] = []

    for end in out["window_end"]:
        active = ((failures["failure_start"] <= end) & (end <= failures["failure_end"])).any()
        future = failures[failures["failure_start"] > end].sort_values("failure_start")
        if future.empty:
            delta_h = None
            target = 0
        else:
            delta = future.iloc[0]["failure_start"] - end
            delta_h = float(delta.total_seconds() / 3600)
            target = int(delta <= horizon)
        targets.append(target)
        in_failure.append(bool(active))
        hours_to_failure.append(delta_h)

    out["failure_within_horizon"] = targets
    out["in_failure"] = in_failure
    out["hours_to_next_failure"] = hours_to_failure
    return out
