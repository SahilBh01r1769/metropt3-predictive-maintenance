# Experiment setup

This file records the rules used for the final model comparison. Most of these choices were fixed before the full run so I would not keep moving the setup after seeing July.

The actual values live in `configs/temporal_experiment.json`.

## Question

Given one hour of compressor history, does keeping more of the temporal structure help predict an upcoming air leak?

The comparison is:

- XGBoost on engineered one-hour features
- TCN on a 120-step sequence built from the same hour
- Attention-TCN using the same TCN encoder plus attention pooling

The target is tested at 1, 3, 6 and 12 hours before failure.

## Data and windows

The final data audit found:

- 1,516,948 raw rows
- 1,515,830 valid rows
- 334 continuity segments
- 8,012 complete one-hour windows
- 7,846 predictive windows after removing windows that overlap active failures

A gap larger than the configured continuity limit starts a new segment. Windows never cross those gaps.

The sequence version of a one-hour window is resampled to 120 steps of 30 seconds. Analogue channels use limited interpolation for short internal gaps; the compressor state is forward-filled within the same limit. An observation-coverage channel is kept so missingness is not silently erased.

One correction happened before this final experiment: the original rate-of-change feature divided by row count. It was changed to elapsed hours so cadence changes do not alter the meaning of the feature.

## Split

I did not use a random split because the 30-minute step creates heavily overlapping one-hour windows.

May and June are used for development. July is the final holdout for this comparison.

The test-selection boundary is `2020-07-08 14:30:00`, before the July failure. Training windows are purged if any raw timestamp they use also belongs to the later evaluation interval.

That means the two sides are separated by raw observations, not just by feature-row IDs.

The split is not moved when a horizon produces very few positives.

## Why one hour

One hour was chosen because it gives the tabular and sequence models the same bounded history and is long enough to contain short operating changes without making the sequence large.

It was not selected by searching several history lengths. A 3h or 6h history would be a separate follow-up experiment.

## Models

### XGBoost

Uses the existing 38 engineered window features. This is the strong tabular reference.

### TCN

Uses causal 1D convolutions over the ordered sequence. The model only uses samples at or before the end of the current window.

### Attention-TCN

Uses the same TCN encoder and replaces last-step pooling with learned attention pooling.

The point of the third model is to test whether the pooling change helps, not to build the biggest architecture possible.

I did not add an LSTM or Transformer after this. With only a few independent failures, adding more architectures would make the comparison wider without fixing the evidence problem.

## Training

Normalization and missing-value fallback values are fitted on the training partition only.

The three seeds are `17`, `42` and `89`.

Development folds are used for early stopping and for selecting one operational threshold. July is not used for either.

The reference threshold remains `0.5` so it is always possible to compare the tuned alert policy with the default classifier output.

## Metrics

Average precision is the main ranking metric because positives are rare.

I also report positive prevalence and `AP / prevalence` (AP lift). The lift is useful when comparing horizons because a 12-hour target naturally has more positive windows than a 1-hour target.

Other saved metrics include ROC-AUC, Brier score, balanced accuracy, precision, recall, F1 and confusion counts.

For alert behavior I also record:

- false-alert episodes per evaluated day
- whether the held-out event was detected
- first-alert lead time

A false alert is counted as a new positive run on a negative-labelled period; a negative prediction, gap or segment boundary ends the run.

## Saved outputs

Each model/horizon run saves per-window probabilities together with the window timestamps, label, thresholded predictions and time to next failure. The run metadata also records the split, feature/channel names, package versions, model settings and timing information.

Those saved probabilities are what `RESULTS.md`, the plots and the Streamlit explorer use.

If a split check fails, labels disagree between otherwise identical cells, or preprocessing touches the holdout, the run stops rather than continuing with those results.