# What failed

This is an investigation record, not a defense of the baseline. The first full-data
run produced a working pipeline and an unusable predictor. Two correctness fixes
were made afterward, so the old metrics are historical evidence rather than current
performance claims.

## The result that forced the investigation

The historical final-event holdout contained 2,034 windows, including 24 positive
windows. The Random Forest reported:

| Metric | Historical result |
|---|---:|
| Balanced accuracy | 0.5000 |
| ROC-AUC | 0.4862 |
| Average precision | 0.0111 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |

Only 1.18% of the test windows were positive. Average precision was 1.11%, roughly
the same as—and slightly below—the positive prevalence. ROC-AUC was also below
0.5. The model did not merely choose a bad alert threshold; its ranking contained
no useful evidence that the final failure's positive windows belonged above the
negative windows.

## What predictions did it actually make?

The saved metrics establish one fact: none of the 24 positive test windows became
a true-positive hard prediction at the default threshold.

The combination of zero recall and balanced accuracy of 0.5000 is consistent with
predicting every test window as negative. It is not enough to prove that exact
confusion matrix, because only rounded aggregate metrics were retained. A very
small number of false positives could be hidden by rounding. The run did not save
per-window probabilities, predictions, or a confusion matrix, so I cannot honestly
reconstruct more detail after the fact.

That missing trace is itself a failure in the experiment. The pipeline saved a model
and summary scores, but not enough evidence to audit individual decisions.

## Where did false positives concentrate?

This cannot be answered from the retained artifacts. There is no verified record
linking a false alert to its timestamp, continuity segment, sensor values, distance
to a failure, or operating state.

Possible concentrations—compressor transitions, high motor current, pressure
changes, or the edges of continuity segments—remain hypotheses. Presenting any of
them as an observed pattern would invent a result. A corrected evaluation run must
persist at least the window boundaries, true label, predicted label, and probability
before this question can be answered.

## How were true failure windows different from what the model expected?

The ranking metrics provide the only defensible answer so far: the current feature
representation did not assign higher risk to the final episode's positive windows
than to ordinary windows. They do not reveal which feature relationships changed.

The model sees one-hour summaries: mean, standard deviation, minimum, maximum and
endpoint rate of change for seven analogue sensors, plus pressure difference,
compressor duty cycle and motor-current volatility. These features compress each
hour into 38 values. They can discard ordering inside the hour, short transients,
recovery cycles and longer degradation history. This is a plausible explanation for
failure, but it has not yet been isolated experimentally.

Feature importance would not solve this question by itself. It would describe which
variables the forest used in training, not why their relationships failed to transfer
to the held-out event. Answering that requires comparing the distributions and
prediction traces of earlier-event and final-event windows.

## Was the 12-hour horizon too difficult?

It may be, but the baseline does not prove it.

Twelve hours was an initial design choice rather than the winner of a controlled
comparison. At that horizon, only 72 of 5,818 historical training windows and 24 of
2,034 test windows were positive. This gives the classifier little positive evidence,
and a twelve-hour label may include windows before a detectable signal appears.

The next modeling experiment will keep the split, feature set, model configuration
and metrics fixed while changing only the horizon to 1, 3, 6 and 12 hours. If shorter
horizons improve event recall and probability ranking, that will support the claim
that useful precursors occur closer to failure. If all four collapse, horizon length
is not the main problem.

## How many failure episodes are available?

The configuration contains four published air-leak intervals:

| Episode | Approximate duration |
|---|---:|
| 2020-04-18 | 24 hours |
| 2020-05-29 to 2020-05-30 | 6.5 hours |
| 2020-06-05 to 2020-06-07 | 52.5 hours |
| 2020-07-15 | 4.5 hours |

Four episodes are enough to expose a failure of generalization, but not enough to
support a broad claim about air-compressor failures. Holding out the final episode
also means that model selection has very few independent events available.

## Do different failure modes look alike to the model?

The repository cannot answer this. All four intervals are labeled `air_leak`; no
verified subtype or component-level cause is supplied to the model. Duration differs
substantially, but duration is not a failure mode.

The current binary target therefore asks whether one feature representation transfers
across four air-leak episodes. It does not demonstrate generalization across different
mechanical failure types, and the documentation should not imply otherwise.

## Corrections made after the historical run

Two problems mean the published baseline must be rerun before it can be treated as
the result of the current code:

1. Rate of change was divided by observation count. It is now divided by elapsed
   time and reported per hour.
2. Chronological window records could still share raw timestamps across the split.
   Training windows touching the test interval are now explicitly purged.

These corrections improve the validity of the experiment. They do not guarantee a
better metric, and no corrected metric will be claimed until it is produced.

## Current conclusion

The Random Forest baseline failed on the held-out final episode, and the retained
artifacts are insufficient for a detailed error analysis. The useful result is
narrower: aggregate performance exposed no transferable signal at the chosen
horizon, while the investigation uncovered weaknesses in both feature calculation
and evaluation evidence.

The next run must produce auditable predictions, not just another score table. Until
then, claims about false-positive locations, sensor-specific failure signatures, and
the benefit of a shorter horizon remain open questions.
