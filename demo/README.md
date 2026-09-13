# Temporal experiment explorer

The Streamlit app reads the committed experiment evidence; it does not simulate a
maintenance product or score uploaded sensor data.

Use the compact navigation to move through:

- **Study overview** — the question, controlled design and principal result;
- **Compare models** — aggregate or per-seed ranking across models and horizons;
- **Event transfer** — May, June and July AP lift, probability traces and the
  descriptive sensor-regime diagnostic;
- **Alert trade-offs** — the fixed `0.5` threshold against thresholds selected
  from development evidence only.

The views expose ranking, transfer between failure episodes, probability timelines,
event detection, warning lead time and false-alert burden. Every numerical result is
loaded from `evidence/temporal_experiment` or the committed descriptive diagnostic in
`evidence/event_regime`.

Run from the repository root:

```bash
pip install -r requirements.txt
pip install -e .
streamlit run demo/app.py
```

Attention weights are not displayed because the raw arrays were not included in the
compact committed evidence. They must not be reconstructed or presented as causal
explanations.
