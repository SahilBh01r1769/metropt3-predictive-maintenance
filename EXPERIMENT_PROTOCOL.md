# Temporal representation experiment protocol

Status: frozen before implementing or running the corrected comparison.

## Question

Given one hour of compressor telemetry, which representation transfers most
usefully to a later documented failure: engineered window summaries, learned
temporal patterns, or learned patterns with attention over the history? The
study also asks whether this changes at `1`, `3`, `6`, and `12` hours before a
failure.

The final July failure is one untouched, held-out episode. It is evidence about
transfer to that episode, not a claim of broad performance across compressors
or failure types.

## Evidence held constant

All cells use the same validated rows, continuity segmentation, one-hour
history windows, 30-minute window step, documented failure intervals, and the
same horizon labels. Rate-of-change is per elapsed time. A feature or sequence
window that overlaps an active failure is quarantined before splitting.

The test boundary is fixed independently of labels and horizon:

- held-out failure start: `2020-07-15 14:30:00`;
- test selection boundary: `2020-07-08 14:30:00`;
- test windows end on or after that boundary;
- a training window is purged when any raw timestamp used by it belongs to the
  test interval;
- no chronological-tail fallback is allowed.

If a cell lacks a class, the run remains an honest unscorable result; the
boundary is not moved to manufacture a metric.

Before the first pilot, a data audit will record the observed sampling cadence,
continuity gaps, sensor columns, and the resampling interval. The interval and
all architecture sizes will then be frozen in versioned configuration before
any final-holdout result is viewed. The audit is data description, not model
selection.

## Predeclared representation ladder

Exactly three models are compared. They form a progression in how much temporal
structure the model is asked to learn rather than a model zoo.

| Level | Model | Input | Purpose |
| --- | --- | --- | --- |
| 1 | XGBoost | engineered one-hour summaries | Strong nonlinear tabular reference: can use interactions in the existing feature engineering. |
| 2 | Temporal Convolutional Network (TCN) | resampled, ordered one-hour sensor sequence | Tests whether local and longer temporal patterns add evidence beyond summaries. |
| 3 | Attention-TCN | same sequence and TCN encoder | Tests whether selectively pooling historical positions changes the ranking or alert burden. |

The TCN uses causal temporal convolutions: the representation at a window end
uses only observations at or before that end. Attention pooling is applied to
the encoded past sequence only. Its weights are diagnostic model focus, not a
causal explanation of a physical failure.

XGBoost receives the existing engineered features, including elapsed-time
rate-of-change. The two sequence models receive the audited raw signal channels
plus explicitly versioned derived channels. Imputation and normalization for
each model are fitted on training data only. No model sees future samples,
failure-period samples, or test-derived normalization statistics.

Graph neural networks are excluded because this dataset does not supply a
defensible graph. Transformers are excluded because four documented failure
episodes are insufficient evidence for that capacity increase. A recurrent
model is not added merely to make a ladder; the controlled question is
summary-versus-convolutional temporal representation-versus-attention pooling.

## Development and final evaluation

The earlier documented failures are used only for rolling-origin development:
to verify the pipeline, select the fixed operational threshold, and reject
broken configurations. The final July episode is not used for architecture,
feature, threshold, or epoch selection. A compact configuration, fixed seed,
and early-stopping rule are written to the experiment configuration before the
full run.

Every model is evaluated at `1`, `3`, `6`, and `12` hours. Changing the horizon
changes only the target label; input history, split, and preprocessing rules do
not change.

Hypotheses:

- Shorter horizons may be easier if this telemetry contains late precursors.
- TCN may beat summary features if the order and duration of changes matter.
- Attention-TCN may improve results only if different portions of the history
  are consistently useful across separate failure episodes.

A collapse, a tie, or a worse attention model is a result: it limits what this
dataset supports.

## Metrics and threshold policy

Average precision is primary because positive windows are rare. Positive
prevalence and `AP / prevalence` are reported to compare horizon difficulty.
The remaining metrics are ROC-AUC when both classes exist, Brier score,
balanced accuracy, precision, recall, F1, and confusion counts.

The reference hard threshold is `0.5` for every cell. One operational threshold
may be selected from earlier rolling-origin development only, using a fixed
rule recorded before the final run; it is then frozen and applied unchanged to
the July holdout. Both reference and operational results are shown, never
quietly substituted for each other.

An alert is a chronological `0 -> 1` prediction transition. A negative
prediction, gap, or segment boundary ends an alert run. A false alert begins on
a negative-labelled window. The report includes false-alert episodes per
evaluated day, held-out-event detection, and first-alert lead time. The latter
is one case outcome per cell, not a population detection rate.

## Required evidence

Each model/horizon run persists a per-window trace with model, horizon,
segment ID, window start/end, true label, probability, reference prediction,
operational prediction, and time to next failure. The run manifest records
dataset identity, audit/resampling configuration, feature or channel names,
normalization fit scope, split boundaries, counts, seeds, package versions,
model parameters, fit/prediction time, parameter count where applicable, and
serialized artifact size.

Result tables compare models only within a horizon. Average precision is not
averaged across horizons with different prevalence. A difference under `0.01`
AP is reported as a practical tie, not as significance. Time and size are
secondary, same-environment measurements.

The run stops rather than reporting results if timestamp disjointness fails,
window IDs differ unexpectedly across cells, labels vary under an otherwise
identical horizon, or normalization includes holdout observations.
