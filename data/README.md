# Dataset

This repository does not commit the MetroPT-3 CSV because it is large.

Download **MetroPT3(AirCompressor).csv** from the UCI Machine Learning Repository (dataset ID 791) and place it here:

```text
data/MetroPT3(AirCompressor).csv
```

Dataset DOI: `10.24432/C5VW3R`

The dataset contains multivariate 1 Hz telemetry from an Air Production Unit (APU) in a metro train, including pressure, temperature, motor-current and digital control signals. The published failure intervals are represented in `src/metropt3/config.py` and are used only to construct future failure-horizon labels.

The raw dataset remains the authoritative source. Do not commit local copies, derived window files, or quarantine outputs.
