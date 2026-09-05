# Three-model horizon experiment protocol

Status: fixed before running the corrected comparison.

This experiment asks two narrow questions:

1. Does prediction become more reliable when the target moves from 12 hours to
   6, 3 or 1 hour before a documented failure?
2. Under the same evidence, does a linear model, a bagged tree model or a
   boosted tree model rank the held-out windows more usefully?

The final July failure is one held-out episode. It is evidence about transfer to
that episode, not an estimate of general performance across compressors or
failure types.

## Evidence held constant

All 12 model/horizon combinations will use the same validated rows, continuity
segments, one-hour feature windows, 30-minute step and 38-feature definitions.
Rate-of-change features use elapsed hours. Windows overlapping a documented
failure are quarantined before either split is formed.

The test selection boundary is fixed independently of the horizon label:

- held-out failure start: `2020-07-15 14:30:00`;
- test selection boundary: `2020-07-08 14:30:00`, seven days earlier;
- a test window has `window_end >= test selection boundary`;
- training windows must end at or before the earliest test-window start;
- the raw-timestamp disjointness assertion must pass;
- there is no chronological-tail fallback.

If either split lacks a class at a particular horizon, the run is retained and
the affected metric is reported as unavailable. The boundary will not be moved
to manufacture a scorable result.

## Predeclared models

Every estimator receives the same median imputation and standard scaling inside
a scikit-learn `Pipeline`. No hyperparameter search will be run on the held-out
episode.

### Regularized Logistic Regression

Linear baseline and reference for whether the engineered features contain a
stable additive signal.

```text
penalty=l2
C=1.0
solver=liblinear
class_weight=balanced
max_iter=1000
random_state=42
```

### Random Forest

Bagged nonlinear baseline, preserving the existing model configuration.

```text
n_estimators=300
class_weight=balanced_subsample
min_samples_leaf=2
random_state=42
n_jobs=-1
```

### Histogram Gradient Boosting

Boosted nonlinear baseline for sequentially correcting errors without adding a
model family beyond the three declared here.

```text
learning_rate=0.1
max_iter=100
max_leaf_nodes=31
min_samples_leaf=20
l2_regularization=1.0
class_weight=balanced
random_state=42
```

These configurations are frozen for the comparison. A convergence failure or
unsupported parameter is a failed run to record, not permission to tune against
the test episode. A necessary compatibility correction must be documented before
the entire comparison is rerun.

## Horizons and hypotheses

Each model is evaluated at `1`, `3`, `6` and `12` hours. Only the target label
changes; features, split boundary, preprocessing and estimator configuration do
not.

The hypotheses are:

- Shorter horizons will improve average-precision lift over prevalence if the
  current features capture late precursors rather than long degradation trends.
- The nonlinear models will outperform Logistic Regression if interactions
  between pressure, current and compressor duty cycle transfer to the held-out
  episode.
- Histogram Gradient Boosting may approach or exceed Random Forest ranking with
  a smaller serialized artifact. Timing and size are secondary observations,
  not substitutes for predictive evidence.

A collapse at every horizon rejects the idea that horizon length is the main
problem. Similar performance from all three models weakens the claim that model
family, rather than labels or representation, is the bottleneck.

## Threshold and alert policy

The hard-prediction threshold is fixed at `0.5` for every model and horizon. It
will not be selected from the final holdout. Average precision, ROC-AUC and Brier
score will also be reported so conclusions do not depend only on that threshold.

For the operational alert view, one alert begins on a `0 -> 1` transition in
chronological hard predictions. A negative prediction, segment change or
non-contiguous window ends the run. An alert is false when its trigger window's
true label is zero.

```text
false alerts per evaluated day = false alert transitions / test-span days
```

This is a window-level simulation, not a claim about deployed maintenance
operations.

For the one held-out event, detection is positive when at least one predicted-
positive window ends before the failure and within the evaluated horizon. Lead
time is the interval from the first such window end to failure start. A missed
event has no lead-time value. This produces one event outcome per cell, so it
will be shown as a case result rather than a population rate.

## Outputs required from each run

Each of the 12 cells must persist an auditable test-window trace containing:

```text
model
horizon_hours
segment_id
window_start
window_end
y_true
probability
prediction_at_0_5
hours_to_next_failure
```

The run manifest must also record package versions, feature names, model
parameters, split boundaries, row/window counts and random seed. Aggregate
results alone are insufficient because they cannot locate false positives or
reconstruct alert transitions.

Reported metrics are:

- average precision and positive prevalence;
- ROC-AUC when both test classes exist;
- balanced accuracy, precision, recall and F1 at 0.5;
- true-negative, false-positive, false-negative and true-positive counts;
- Brier score;
- false alerts per evaluated day;
- held-out-event detection and lead time where defined;
- fit time, prediction time and serialized artifact size as secondary
  same-run measurements.

## Comparison rules

Average precision is the primary ranking metric because positive windows are
rare. Models are compared within a horizon; raw average precision will not be
averaged across horizons with different prevalence. `AP / prevalence` is
reported as descriptive lift for the horizon study.

Differences smaller than `0.01` average precision are described as practically
tied. No statistical significance claim will be made from one held-out event.
Calibration, hard-prediction behavior and false-alert burden can disqualify an
otherwise higher-ranking model, but they will not be combined into an invented
single score.

Fit time, prediction time and artifact size are compared only within the same
execution environment. They describe implementation cost for this run and are
not hardware-independent benchmarks.

## Stop conditions

The comparison stops and records the reason if timestamp disjointness fails,
the final-event boundary changes between cells, feature columns differ, or a
model receives information unavailable at its window end. Results from a
partially completed or methodologically inconsistent grid will not be promoted
to `RESULTS.md`.
