from __future__ import annotations

import argparse
from pathlib import Path

from metropt3.result_analysis import (
    build_eventwise_metrics,
    summarize_eventwise_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute event-wise transfer diagnostics from prediction traces."
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("evidence/temporal_experiment"),
    )
    args = parser.parse_args()
    eventwise = build_eventwise_metrics(args.evidence_root)
    summary = summarize_eventwise_metrics(eventwise)
    eventwise.to_csv(args.evidence_root / "eventwise_metrics.csv", index=False)
    summary.to_csv(args.evidence_root / "eventwise_summary.csv", index=False)
    print(f"Wrote {len(eventwise)} event/seed rows and {len(summary)} summaries")


if __name__ == "__main__":
    main()
