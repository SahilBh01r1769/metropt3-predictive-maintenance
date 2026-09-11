from __future__ import annotations

import pandas as pd


NON_FEATURE_COLS = {
    "segment_id",
    "window_id",
    "window_start",
    "window_end",
    "rows",
    "cadence_seconds",
    "failure_within_horizon",
    "in_failure",
    "hours_to_next_failure",
}

FINAL_EVENT_TEST_START = pd.Timestamp("2020-07-08 14:30:00")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def purge_overlapping_training_windows(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Remove training windows whose raw interval touches the test interval."""
    required = {"window_start", "window_end"}
    missing = required.difference(train.columns) | required.difference(test.columns)
    if missing:
        raise ValueError(
            "Window bounds are required for overlap purging: "
            + ", ".join(sorted(missing))
        )
    if train.empty or test.empty:
        raise ValueError("Both provisional splits need at least one window")

    test_interval_start = pd.to_datetime(test["window_start"]).min()
    train_ends = pd.to_datetime(train["window_end"])
    return train.loc[train_ends <= test_interval_start].copy()


def chronological_split(
    frame: pd.DataFrame,
    *,
    test_start: str | pd.Timestamp = FINAL_EVENT_TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at a fixed boundary, then purge overlapping raw time intervals."""
    required = {"window_start", "window_end"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Window bounds are required for chronological splitting: "
            + ", ".join(sorted(missing))
        )

    ordered = frame.copy()
    ordered["window_start"] = pd.to_datetime(ordered["window_start"])
    ordered["window_end"] = pd.to_datetime(ordered["window_end"])
    if (ordered["window_end"] <= ordered["window_start"]).any():
        raise ValueError("Each window_end must be later than window_start")
    ordered = ordered.sort_values("window_end").reset_index(drop=True)
    if len(ordered) < 5:
        raise ValueError("Need at least 5 windows for evaluation")

    boundary = pd.Timestamp(test_start)
    if pd.isna(boundary):
        raise ValueError("test_start must be a valid timestamp")
    train = ordered.loc[ordered["window_end"] < boundary].copy()
    test = ordered.loc[ordered["window_end"] >= boundary].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Fixed holdout boundary {boundary} must leave windows on both sides"
        )
    train = purge_overlapping_training_windows(train, test)
    if train.empty:
        raise ValueError("Overlap purge removed every training window")
    return train, test
