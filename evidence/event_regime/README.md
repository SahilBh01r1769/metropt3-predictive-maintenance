# Event-regime diagnostic evidence

These tables compare engineered one-hour window summaries in the 24 hours before the
May, June and July failure onsets with clean normal-operation windows. A feature's
effect is its event median minus the normal median, divided by the normal interquartile
range. This is descriptive evidence, not a significance test and not a new predictor.

The extraction matched the frozen audit counts: 1,515,830 validated rows, 334
continuity segments and 8,012 complete windows. It identified 38 May, 48 June and 33
July precursor windows. Only three independent failure episodes are available.

`event_regime_summary.csv` is the compact table used for the heatmap in
`figures/event_regime_shift.png`. Large values for near-constant features should not be
read as causal or reliable effects; the normal interquartile range can be close to
zero.

The first Colab extraction produced an all-zero XGBoost importance file because the
script used positional feature keys while the estimator preserved DataFrame column
names. That file is intentionally not included. The corrected command now maps native
gain values by column name and fails loudly if the estimator returns no gains:

```bash
python scripts/analyze_event_regimes.py --csv data/MetroPT3\(AirCompressor\).csv
```

Importance should be treated as model inspection, not evidence that a feature transfers
between compressor failure episodes.
