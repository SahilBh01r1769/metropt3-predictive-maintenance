from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANALOGUE_COLS, ANALOGUE_RANGES, DIGITAL_COLS, MAX_GAP_SECONDS, REQUIRED_COLS, TIMESTAMP_COL


@dataclass
class ValidationReport:
    input_rows: int
    valid_rows: int
    quarantined_rows: int
    duplicate_timestamps: int
    segments: int


def validate_and_segment(
    df: pd.DataFrame,
    *,
    max_gap_seconds: int = MAX_GAP_SECONDS,
    quarantine_path: str | Path | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate MetroPT rows and assign segment_id without bridging data gaps."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    input_rows = len(work)
    work[TIMESTAMP_COL] = pd.to_datetime(work[TIMESTAMP_COL], errors="coerce")
    for col in [*ANALOGUE_COLS, *DIGITAL_COLS]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    duplicate_mask = work[TIMESTAMP_COL].duplicated(keep="first") & work[TIMESTAMP_COL].notna()
    invalid = work[TIMESTAMP_COL].isna() | duplicate_mask
    for col in ANALOGUE_COLS:
        lo, hi = ANALOGUE_RANGES[col]
        invalid |= work[col].isna() | ~work[col].between(lo, hi)
    for col in DIGITAL_COLS:
        invalid |= work[col].isna() | ~work[col].isin([0, 1])

    quarantined = work.loc[invalid].copy()
    valid = work.loc[~invalid].copy().sort_values(TIMESTAMP_COL).reset_index(drop=True)

    if quarantine_path is not None and not quarantined.empty:
        path = Path(quarantine_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        quarantined.to_csv(path, index=False)

    if valid.empty:
        valid["segment_id"] = pd.Series(dtype="int64")
        segments = 0
    else:
        gaps = valid[TIMESTAMP_COL].diff().dt.total_seconds().fillna(0)
        valid["segment_id"] = (gaps > max_gap_seconds).cumsum().astype(int)
        segments = int(valid["segment_id"].nunique())

    report = ValidationReport(
        input_rows=input_rows,
        valid_rows=len(valid),
        quarantined_rows=len(quarantined),
        duplicate_timestamps=int(duplicate_mask.sum()),
        segments=segments,
    )
    return valid, report


def range_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Return useful percentiles for tuning validation ranges from real data."""
    rows = []
    for col in ANALOGUE_COLS:
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        p = np.percentile(values, [0.1, 1, 50, 99, 99.9])
        rows.append({"sensor": col, "p0.1": p[0], "p1": p[1], "p50": p[2], "p99": p[3], "p99.9": p[4]})
    return pd.DataFrame(rows)
