from __future__ import annotations

import argparse
from pathlib import Path

from metropt3.evidence_plots import generate_evidence_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate portfolio figures from committed experiment evidence"
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("evidence/temporal_experiment"),
    )
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    for path in generate_evidence_plots(args.evidence, args.output):
        print(path)


if __name__ == "__main__":
    main()
