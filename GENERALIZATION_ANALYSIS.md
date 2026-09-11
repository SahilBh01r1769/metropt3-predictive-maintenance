# Generalization analysis

The central result of the study is a cross-event generalization problem: the three representations learn useful structure around different historical failure episodes, but that structure does not transfer reliably to the held-out July event.

This document explains that result in more detail than the main README.

## 1. The original baseline was superseded

The earlier Random Forest baseline saved aggregate scores but not per-window probabilities. It also predated two correctness fixes: rate of change was measured per observation rather than per elapsed time, and raw timestamps could be shared across the temporal split.

Those historical scores are therefore not used as current evidence. The replacement experiment records development and holdout probabilities, window identities, labels, thresholds, training histories and manifests.

## 2. July does not resemble the learned positive ranking

XGBoost is the strongest held-out model at all four horizons, but its mean ROC-AUC remains between 0.420 and 0.456. At the three-, six- and twelve-hour horizons its AP is approximately equal to positive prevalence.

The sequence models transfer less well. Their mean AP lift on July is below `0.8×` at every horizon and their mean ROC-AUC never exceeds 0.180.

This is not simply a threshold problem. Ranking metrics are threshold-independent: the models generally assign higher risk to ordinary July windows than to the actual pre-failure windows.

## 3. The models learn different development episodes

Event-wise replay makes the instability visible. At the one-hour horizon:

| Event | XGBoost AP lift | TCN AP lift | Attention-TCN AP lift |
|---|---:|---:|---:|
| May development | 1.00× | **14.99×** | 2.18× |
| June development | 1.64× | 2.57× | **13.60×** |
| July holdout | **1.63×** | 0.75× | 0.78× |

TCN strongly ranks the May precursor; Attention-TCN strongly ranks June. Neither behavior carries into July.

That pattern is more informative than a simple “deep model failed” conclusion. Added capacity can capture episode-specific temporal structure without establishing a shared warning signature.

Repeated seeds test optimization sensitivity, but they do not create additional independent failure mechanisms. The dataset contains many overlapping windows around only a few documented failures.

## 4. Preserving within-hour order did not improve transfer

The TCN receives ordered 30-second aggregates rather than 38 engineered summaries. It should help if short transients or within-hour ordering are consistent across events.

That advantage does not appear on July: TCN AP remains below prevalence at every horizon.

Several explanations remain plausible rather than proven:

- useful context may extend beyond one hour;
- operating regime may dominate a shared precursor;
- 30-second aggregation may smooth short transients;
- the development folds may contain too little independent positive behavior for stable neural early stopping.

The existing study cannot cleanly distinguish those possibilities without changing the experimental question or adding more independent failures.

## 5. Attention added complexity without improving transfer

Attention-TCN uses the same causal encoder as TCN and changes only the pooling stage. It adds 1,089 trainable parameters but does not improve held-out ranking.

Attention weights are retained as diagnostic model focus, not treated as causal explanations. When held-out ranking is weak, interpreting high-weight timesteps as physical failure causes would be especially misleading.

## 6. Development-selected thresholds do not transfer cleanly

Operational thresholds were selected from pooled May/June development predictions using F2; July was not used for threshold selection.

Those thresholds recover the July event in several cells, but the apparent improvement comes with low precision and recurring false-alert episodes. For example, XGBoost detects July in all seeds at the 12-hour horizon, while mean precision is 1.18% and the policy generates 1.49 false-alert episodes per evaluated day.

At the fixed `0.5` threshold, no model detects the July event.

This is why event detection is reported together with ranking quality, precision and alert burden.

## 7. The horizon comparison does not identify a clearly useful warning range

Absolute AP increases toward twelve hours partly because longer horizons contain more positive windows. AP lift over prevalence is therefore the more useful cross-horizon comparison.

XGBoost shows its clearest lift at one hour (`1.63×`), but that setting detects July in only one of three seeds at the development-selected threshold and produces 3.44 false-alert episodes per day.

The study therefore does not identify one operationally credible warning horizon.

## 8. Sensor-regime diagnostics support an episode-shift interpretation

A follow-up descriptive analysis compares feature behavior in the 24 hours before the May, June and July failures with clean normal-operation windows.

Several feature directions repeat, but the magnitude is not stable. July shows much larger standardized shifts for features including `TP2_mean`, `pressure_diff_mean` and `H1_mean` than May or June.

That does not prove a causal mechanism, but it is consistent with the model evidence: each episode appears to expose a somewhat different precursor regime, making a shared supervised warning pattern difficult to learn from so few independent events.

The diagnostic tables are committed under [`evidence/event_regime`](evidence/event_regime).

## 9. Reproducibility boundary

The exported evidence contains manifests, histories, probabilities, combined predictions, metrics and attention evidence, but not every trained model binary.

The committed traces are sufficient to recompute the reported evaluation results. Exact prediction recreation requires retraining from the frozen configuration.

## Conclusion

The strongest result is not a high predictive score; it is a controlled demonstration of the gap between **fitting temporal patterns** and **learning a transferable warning signature**.

XGBoost generalizes least poorly. TCN and Attention-TCN find stronger episode-specific patterns but do not transfer them to July. With very few independent failure episodes, additional representation complexity does not overcome the evidence limit.

That conclusion is bounded to this MetroPT-3 case study and is the reason the project is presented as a temporal generalization investigation rather than a deployable maintenance model.
