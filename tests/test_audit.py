import json

import numpy as np
import pandas as pd

from metropt3.audit import audit_dataframe, file_sha256, write_audit_report


def audit_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=1500, freq="10s")
    timestamps = timestamps.to_series().reset_index(drop=True)
    timestamps.loc[750:] = timestamps.loc[750:] + pd.Timedelta(minutes=2)
    comp = (np.arange(len(timestamps)) % 12 < 6).astype(int)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "TP2": 2.0 + comp * 7,
            "TP3": 8.8,
            "H1": 8.0,
            "DV_pressure": 0.1,
            "Reservoirs": 8.7,
            "Oil_temperature": 65.0,
            "Motor_current": 3.0 + comp * 5,
            "COMP": comp,
        }
    )


def test_audit_records_cadence_segments_windows_and_split_evidence():
    report = audit_dataframe(
        audit_frame(),
        horizons=(1, 3),
        test_start="2020-01-01 03:30:00",
        window_seconds=1200,
        step_seconds=600,
    )

    assert report["timestamp"]["median_seconds"] == 10.0
    assert report["validation"]["segments"] == 2
    assert report["window_policy"]["feature_windows"] > 0
    assert [row["horizon_hours"] for row in report["horizons"]] == [1, 3]
    assert all(row["timestamp_intervals_disjoint"] for row in report["horizons"])
    assert report["sequence_resampling_candidates"][0] == {
        "interval_seconds": 10,
        "timesteps_per_window": 120,
    }
    assert report["resampling_decision"] == "pending_review"


def test_audit_preserves_an_unscorable_fixed_split_as_evidence():
    report = audit_dataframe(
        audit_frame(),
        horizons=(1,),
        test_start="2021-01-01",
        window_seconds=1200,
        step_seconds=600,
    )

    assert "split_error" in report["horizons"][0]
    assert "must leave windows on both sides" in report["horizons"][0]["split_error"]


def test_audit_reports_when_no_complete_windows_exist():
    report = audit_dataframe(
        audit_frame().iloc[:30],
        horizons=(1,),
        window_seconds=3600,
        step_seconds=1800,
    )

    assert report["window_policy"]["feature_windows"] == 0
    assert report["horizons"][0]["split_error"] == (
        "No complete feature windows were produced"
    )


def test_written_audit_is_stable_and_source_hash_is_recordable(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("timestamp,value\n2020-01-01,1\n", encoding="utf-8")
    report = {"schema_version": 1, "source": {"sha256": file_sha256(source)}}

    first = write_audit_report(report, tmp_path / "first.json")
    second = write_audit_report(report, tmp_path / "second.json")

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == report
