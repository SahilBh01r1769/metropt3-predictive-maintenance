from __future__ import annotations

import argparse
from pathlib import Path

from .audit import audit_csv, write_audit_report
from .config import ARTIFACT_DIR
from .config import RAW_FILENAME


def default_data_path():
    return Path("data") / RAW_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MetroPT-3 experiment inputs")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="measure data and window evidence before modelling")
    audit.add_argument("--csv", default=str(default_data_path()), help="path to MetroPT-3 CSV")
    audit.add_argument(
        "--output",
        default=str(ARTIFACT_DIR / "data_audit.json"),
        help="path for the deterministic JSON report",
    )
    args = parser.parse_args()
    if args.command == "audit":
        import json

        report = audit_csv(args.csv)
        output = write_audit_report(report, args.output)
        print(json.dumps({"audit_report": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
