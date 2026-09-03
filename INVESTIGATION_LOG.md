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
