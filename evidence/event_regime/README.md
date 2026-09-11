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

The first extraction's importance mapping was ambiguous across XGBoost versions: some
backends return DataFrame column names while others return positional `f0`, `f1`, …
keys. The extractor now supports both and records `importance_key` for every feature.
If all gains are zero with `importance_key=not_used`, the fitted model made no split
using that feature; this is a result to report, not a reason to invent an importance.
If positional keys are present, they are mapped back to the audited feature order.

```bash
python scripts/analyze_event_regimes.py --csv data/MetroPT3\(AirCompressor\).csv
```

Importance should be treated as model inspection, not evidence that a feature transfers
between compressor failure episodes.
