# MetroPT-3 Maintenance Dashboard Demo

This directory contains the user-facing Streamlit visualization for the project.

## Purpose

The demo is a condition-monitoring interface, not a replacement for the training pipeline. It lets a user explore sensor telemetry, data quality, segment-safe feature windows and a latest-window health/risk view without downloading the full MetroPT-3 dataset first.

## Data modes

- **Healthy reference** — synthetic telemetry used only to demonstrate the UI.
- **Degradation scenario** — synthetic drifting telemetry used only to demonstrate the UI.
- **Upload MetroPT-style CSV** — validates and visualizes compatible user data.

Synthetic data is never used as model-evaluation evidence.

## Risk display

If `artifacts/model.joblib` is available, the dashboard uses that trained model bundle for the latest compatible feature window.

If no trained artifact is present, the dashboard shows a clearly labeled **heuristic demo health-risk indicator**. That indicator is a visualization aid, not an ML prediction.

The default verified Random Forest baseline is documented in `../RESULTS.md` and currently has weak held-out predictive performance. The demo does not conceal or override that result.

## Run locally

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
streamlit run demo/app.py
```

## Streamlit Community Cloud

The current demo is lightweight enough to deploy directly from the repository without the raw 208 MB dataset or a committed model artifact.

Suggested configuration:

- Repository: `SahilBh01r1769/metropt3-predictive-maintenance`
- Branch: `main`
- Main file path: `demo/app.py`
- Python: 3.11

The public deployment will start with the synthetic reference modes and CSV upload. A model artifact should only be bundled into a hosted demo after its provenance and evaluation are explicitly documented.
