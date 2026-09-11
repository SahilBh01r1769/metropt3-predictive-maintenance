from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from metropt3.audit import file_sha256
from metropt3.features import build_windows
from metropt3.tabular_experiment import run_tabular_experiment
from metropt3.tabular_plots import save_tabular_plots
from metropt3.temporal_features import add_temporal_context
from metropt3.validation import validate_and_segment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe temporal tabular comparison."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tabular_experiment"))
    args = parser.parse_args()
    raw = pd.read_csv(args.csv)
    valid, validation = validate_and_segment(raw)
    windows = build_windows(valid)
    temporal = add_temporal_context(windows)
    summary = run_tabular_experiment(temporal, args.output)
    plots = save_tabular_plots(args.output)
    (args.output / "validation.json").write_text(
        json.dumps(validation.__dict__, indent=2), encoding="utf-8"
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {
        "code_revision": revision,
        "dataset_sha256": file_sha256(args.csv),
        "raw_rows": len(raw),
        "validated_rows": len(valid),
        "feature_windows": len(windows),
        "temporal_feature_columns": len(
            [column for column in temporal if column.startswith("temporal__")]
        ),
        "plots": [path.name for path in plots],
    }
    (args.output / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
