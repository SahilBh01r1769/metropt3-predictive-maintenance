# Temporal experiment explorer

The Streamlit app reads the committed experiment evidence; it does not simulate a
maintenance product or score uploaded sensor data.

Use the controls to compare:

- XGBoost, TCN and Attention-TCN;
- 1, 3, 6 and 12-hour prediction horizons;
- May, June and July failure episodes;
- the fixed `0.5` threshold and the development-selected operational threshold.

The views expose ranking, transfer between failure episodes, probability timelines,
event detection, warning lead time and false-alert burden. Every displayed value is
loaded from `evidence/temporal_experiment`.

Run from the repository root:

```bash
pip install -r requirements.txt
pip install -e .
streamlit run demo/app.py
```

Attention weights are not displayed because the raw arrays were not included in the
compact committed evidence. They must not be reconstructed or presented as causal
explanations.
