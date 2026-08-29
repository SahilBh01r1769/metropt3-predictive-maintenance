# MetroPT-3 Maintenance Dashboard Demo

This directory contains the user-facing Streamlit condition-monitoring interface for the project.

## Purpose

The demo lets a user explore sensor telemetry, data quality, segment-safe feature windows and a latest-window health/risk view without downloading the full MetroPT-3 dataset first. It is intentionally lightweight enough for public Streamlit hosting while the full training/data pipeline remains available in the repository.

## Data modes

- **Healthy reference** — synthetic telemetry used only to demonstrate the UI.
- **Degradation scenario** — synthetic drifting telemetry used only to demonstrate the UI.
- **Upload MetroPT-style CSV** — validates and visualizes compatible user data.

Synthetic data is never used as model-evaluation evidence.

## Risk display

If `artifacts/model.joblib` is available, the dashboard uses that trained model bundle for the latest compatible feature window.

If no trained artifact is present, the dashboard shows a clearly labeled **heuristic demo health-risk indicator**. That indicator is a visualization aid, not a claimed ML prediction.

The verified Random Forest baseline is documented in `../RESULTS.md`; the hosted interface does not conceal or replace those measured results.

## Run locally

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
streamlit run demo/app.py
```

On this hosted-demo branch, the root entrypoint also works:

```bash
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

Use:

- Repository: `SahilBh01r1769/metropt3-predictive-maintenance`
- Branch: `demo/hosted-maintenance-dashboard`
- Main file path: `streamlit_app.py`
- Python: 3.11

No raw 200+ MB MetroPT CSV is required for the public demo. It starts with the built-in reference scenarios and accepts compatible CSV uploads from the user.

A model artifact should only be bundled into the hosted branch when its provenance and evaluation remain explicitly documented.
