# Verified Baseline Results

These results come from a clean GitHub Actions run on the official UCI MetroPT-3 archive. They are included to make the repository reproducible and to prevent unverified portfolio claims.

## Verification run

- Workflow: `full data smoke`
- Run ID: `33190420834`
- Python: 3.11
- Source: official UCI MetroPT-3 archive downloaded by `scripts/download_data.py`
- Raw rows: **1,516,948**
- Valid rows after validation: **1,515,830**
- Quarantined rows: **1,118**
- Duplicate timestamps: **0**
- Gap-defined segments: **334**
- Observed median segment cadences represented in accepted windows: **10 s and 12 s**
- Segment-safe feature windows: **8,012**
- Model input features: **38**

The same run also verified that the trained artifact can be loaded and that the Streamlit dashboard starts successfully with it present.

## Baseline model

The current baseline is a class-balanced Random Forest evaluated with the repository's final-event chronological holdout policy.

| Metric | Verified value |
|---|---:|
| Balanced accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.4862 |
| Average precision | 0.0111 |
| Training windows | 5,818 |
| Test windows | 2,034 |
| Positive training windows | 72 |
| Positive test windows | 24 |

## Interpretation

This baseline **does not demonstrate useful predictive performance** on the held-out final failure episode. Its ROC-AUC is approximately random and it did not identify the positive test windows at the default classifier threshold.

That negative result is intentionally preserved rather than replaced with an easier random split or an unverified historical metric. It shows that:

1. the 12-hour pre-failure target is extremely imbalanced,
2. one-hour statistical windows plus the current hand-engineered features are not sufficient for this holdout,
3. predictive-maintenance evaluation must be event-aware and chronological, and
4. better performance will require model/target/feature research rather than presentation changes.

## What is verified versus not claimed

Verified:

- official UCI download works from a clean environment,
- validation and quarantine run on the full CSV,
- windows do not cross configured segment boundaries,
- coverage adapts to observed timestamp cadence,
- published failure intervals produce future-horizon labels,
- model artifacts and metrics are generated reproducibly,
- evaluation contains both positive and negative examples,
- the Streamlit dashboard starts with the trained artifact.

Not claimed:

- production-ready failure prediction,
- continuous Remaining Useful Life ground truth,
- an LSTM/BiLSTM result,
- historical RMSE/MAE values from prior experiments,
- a percentage improvement over another model.

## Next modeling experiments

A credible next research phase would compare, under the same event-aware evaluation policy:

- multiple prediction horizons (for example 6 h, 24 h, 48 h),
- rolling-origin evaluation across individual failure episodes,
- stronger temporal features and lag/trend features,
- calibrated tree/boosting baselines,
- sequence models only after the tabular baselines and leakage checks are established,
- threshold selection based on maintenance costs rather than a fixed 0.5 cutoff.

Any future metric should be added here only after it is reproduced on a clean run.
