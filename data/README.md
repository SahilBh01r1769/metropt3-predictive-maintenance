# Dataset

This repository does not commit the MetroPT-3 CSV because it is large.

## Recommended download

From the repository root:

```bash
python scripts/download_data.py
```

The script downloads the official UCI archive and extracts:

```text
data/MetroPT3(AirCompressor).csv
```

You can also download **MetroPT3(AirCompressor).csv** manually from the UCI Machine Learning Repository, dataset ID 791.

- Dataset DOI: `10.24432/C5VW3R`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset contains multivariate telemetry from an Air Production Unit (APU) in a metro train, including pressure, temperature, motor-current and digital control signals. The source documentation describes operational data recorded during 2020 and publishes four air-leak failure periods. Those intervals are represented in `src/metropt3/config.py` and are used only to construct future failure-horizon labels.

The raw dataset remains the authoritative source. Do not commit local copies, derived window files, trained artifacts or quarantine outputs.
