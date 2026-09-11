# Investigation log

## 2026-09-03 — Rate of change used row position as time

The first version divided each sensor's endpoint change by the number of row
steps in the window. That measures change per observation. It is wrong when
observations are not equally spaced or when cadence changes.

The corrected feature divides endpoint change by the timestamps' elapsed hours
and is named `*_roc_per_hour`. Two windows covering the same change over the
same time now produce the same rate even when they contain different numbers
of observations.

This is still an endpoint slope: it does not describe movement within the
window or prove that the feature predicts a failure.

## 2026-09-06 — One hour was fixed as the observation history

One hour was chosen as a bounded, interpretable history that could be represented both
as engineered summaries and as 120 ordered 30-second sequence steps. It keeps the
representation comparison controlled: all three models observe the same time span.

This was an experimental design choice, not the winner of a history-length search. Now
that July has been inspected, changing the history because of its outcome would be an
exploratory follow-up rather than a correction to the frozen study.

## 2026-09-06 — The final failure became the chronological holdout

Randomly splitting overlapping windows would place near-duplicate time intervals on
both sides of evaluation and mix operating regimes from before and after later
failures. The boundary was therefore fixed before model results at 2020-07-08 14:30,
with training windows purged until no raw timestamp can also occur in the test interval.

May and June remain chronological development folds. July tests transfer to a later
failure episode; it is one case study, not a population estimate.

## 2026-09-06 — Average precision and lift became primary

Positive windows are extremely rare: prevalence ranges from 0.098% at one hour to
1.181% at twelve hours in the July holdout. Accuracy can look strong while predicting
every window as normal, and ROC-AUC alone can obscure the quality of the highest-ranked
warnings.

Average precision focuses on rare-positive ranking. Dividing AP by positive prevalence
adds the random-ranking reference needed to compare horizons with different class
ratios. Balanced accuracy, calibration, confusion counts and alert-level measures are
still retained, but they answer different questions.

There are thousands of windows but very few independent failure episodes. Neither AP
nor repeated seeds removes that evidence limit.
