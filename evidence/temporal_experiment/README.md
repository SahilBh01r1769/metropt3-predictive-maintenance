# Temporal experiment evidence

This directory records the held-out comparison produced from commit
`78b8afa2a9c81a8f0dc7b0e9d231f5d6fde335d1` on the audited MetroPT-3 source.

| File | Purpose |
|---|---|
| `development/*.csv.gz` | Per-window probabilities from the two chronological development folds, split by model and horizon |
| `holdout/*.csv.gz` | Per-window July probabilities and both hard-prediction policies, split by model and horizon |
| `thresholds.json` | Operational thresholds selected from pooled development predictions |
| `metrics.csv` | 72 recomputed metric rows: 36 cells under the reference and development-selected thresholds |
| `run_context.json` | Dataset hash, row/window counts, sequence shape and execution device |
| `validation.json` | Validation result produced before the bundle was exported |

The original returned ZIP had SHA-256
`1924b3d5c777d5a281476378336b437c593ca646adb83dd8138c17dace73b7cf`.
The repository keeps its analysis inputs as compressed tables rather than committing
the Colab checkpoint layout or model binaries.

## Checks performed after export

- all 36 model/horizon/seed cells were present;
- 112 files in the returned export matched the hashes stored in their manifests;
- all 12 combined holdout traces matched their stored hashes;
- the 72 metric rows were independently recomputed from the exported predictions;
- window IDs and labels agreed across all three models for each horizon and seed;
- all four primary-seed attention arrays had shape `2032 × 120` and each row summed to one;
- every manifest named the same code revision and audited dataset hash.

The model binaries were deliberately retained in the Colab checkpoint directory and
were not in the returned ZIP. Consequently, the evidence proves the recorded
predictions and metrics are internally consistent, but it cannot recreate predictions
without rerunning training. This boundary is stated explicitly instead of treating a
successful pre-export validation as stronger evidence than it is.
