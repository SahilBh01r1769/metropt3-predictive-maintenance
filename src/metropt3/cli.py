from __future__ import annotations

import argparse
import json

from .pipeline import default_data_path, run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="MetroPT-3 predictive-maintenance pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="validate data, build windows, train and evaluate")
    train.add_argument("--csv", default=str(default_data_path()), help="path to MetroPT-3 CSV")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(run_training_pipeline(args.csv), indent=2, default=str))


if __name__ == "__main__":
    main()
