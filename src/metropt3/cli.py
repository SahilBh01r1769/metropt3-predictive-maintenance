from __future__ import annotations

import argparse
import json

from .audit import audit_csv, write_audit_report
from .config import ARTIFACT_DIR
from .pipeline import default_data_path, run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="MetroPT-3 predictive-maintenance pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="validate data, build windows, train and evaluate")
    train.add_argument("--csv", default=str(default_data_path()), help="path to MetroPT-3 CSV")
    audit = sub.add_parser("audit", help="measure data and window evidence before modelling")
    audit.add_argument("--csv", default=str(default_data_path()), help="path to MetroPT-3 CSV")
    audit.add_argument(
        "--output",
        default=str(ARTIFACT_DIR / "data_audit.json"),
        help="path for the deterministic JSON report",
    )
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(run_training_pipeline(args.csv), indent=2, default=str))
    elif args.command == "audit":
        report = audit_csv(args.csv)
        output = write_audit_report(report, args.output)
        print(json.dumps({"audit_report": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
