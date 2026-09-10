# What failed

The final experiment is a negative modeling result. That is not hidden: the three
representations mostly fail to rank windows before the July failure above normal
operation.

## 1. The original Random Forest result was not auditable

The earlier baseline saved aggregate scores but not per-window probabilities. It also
predated two correctness fixes: rate of change was measured per observation rather
than per elapsed time, and raw timestamps could be shared across the temporal split.

Those historical numbers are no longer current evidence. The replacement experiment
stores every development and holdout probability, window identity, label, threshold,
training history and manifest.

## 2. The July episode did not resemble the learned positive ranking

XGBoost is the best model at all four horizons, yet its mean ROC-AUC ranges only from
0.420 to 0.456. Its three-, six- and twelve-hour AP is approximately equal to positive
prevalence. The sequence models are worse: every mean AP lift is below `0.8×`, and
their mean ROC-AUC never exceeds 0.180.

This is stronger evidence than “the 0.5 threshold was wrong.” Ranking metrics do not
depend on that single cutoff. The models generally assign higher risk to ordinary July
windows than to the actual pre-failure windows.

The likely explanation is episode shift combined with too few independent events.
There are many overlapping windows but only four published air-leak intervals. Window
count does not create new independent failure mechanisms.

### The models learned different development episodes

Event-wise replay makes that explanation concrete. At the one-hour horizon, TCN
achieves `14.99×` AP lift on the May development failure but only `2.57×` on June and
`0.75×` on July. Attention-TCN instead reaches `13.60×` on June, compared with `2.18×`
on May and `0.78×` on July. XGBoost is weaker on the development episodes but fails
least severely on July.

This is not a clean progression from simple to complex. It is evidence of unstable,
episode-specific ranking: each sequence model finds a strong pattern in a different
development event, and neither pattern transfers to the held-out event. Three random
seeds test optimization sensitivity, but they do not turn three episodes into a large
sample of failure mechanisms.

## 3. Preserving within-hour order did not help

The TCN sees ordered 30-second aggregates instead of 38 summary statistics. It should
benefit if short transients or ordering inside the hour transfer across events. It does
not: its AP is below prevalence at every horizon.

Possible reasons remain hypotheses rather than findings:

- the useful context may be longer than one hour;
- event-specific operating regimes may dominate a shared precursor;
- thirty-second aggregation may remove useful short transients;
- the development folds may not contain enough distinct positive behavior for neural
  early stopping.

The current experiment cannot distinguish these explanations without adding new
independent failure episodes or changing the predeclared question.

## 4. Attention added complexity, not evidence

Attention-TCN shares the exact encoder used by TCN, so the comparison isolates pooling.
It adds 1,089 trainable parameters but produces lower mean AP than TCN at six and twelve
hours and essentially the same collapse at one and three hours.

Attention weights are retained for inspection, but they are not causal explanations.
When the model's held-out ranking is poor, presenting its high-weight timesteps as a
failure signature would be especially misleading.

## 5. Development-selected thresholds did not transfer cleanly

Thresholds were selected from pooled May/June development predictions by F2, never
from July. They detect the July event in several cells, but the apparent success comes
with low precision and frequent false-alert episodes.

For example, XGBoost detects the event in all seeds at the 12-hour horizon, but mean
precision is 1.18% and the policy produces 1.49 false-alert episodes per evaluated day.
At the default 0.5 threshold, no model detects the event in any cell.

This demonstrates why “event detected” cannot be reported alone. It must be paired
with ranking, precision and alert burden.

## 6. The horizon hypothesis was only partly informative

Absolute AP increases toward twelve hours because positive prevalence also increases.
After normalizing by prevalence, XGBoost's clearest lift occurs at one hour (`1.63×`),
but that setting detects the held-out event in only one of three seeds and produces
3.44 false-alert episodes per day at the operational threshold.

The study therefore does not identify a useful horizon. It shows that closer prediction
is somewhat easier to rank for XGBoost, but still not operationally credible.

## 7. The exported evidence has a reproducibility boundary

Colab validated all checkpoints before export. The returned ZIP contains manifests,
histories, probabilities, combined predictions, metrics and attention weights, but not
the model binaries retained in Drive. The exported traces and metrics were independently
hash-checked and recomputed; prediction recreation still requires retraining.

That limitation is recorded in the evidence directory rather than treating internal
validation as proof that the bundle is fully self-contained.

## Conclusion

The strongest defensible outcome is methodological: corrected elapsed-time features,
active-failure quarantine, fixed temporal holdout, raw-timestamp purge, training-only
normalization, predeclared comparisons and auditable prediction traces all survived a
full-data run.

The modeling conclusion is narrower and negative. None of the three models establishes
a transferable warning signal for the held-out July episode. XGBoost fails least;
TCN and attention add cost without improving the evidence.
