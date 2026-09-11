# Results

Final results for the XGBoost / TCN / Attention-TCN comparison.

Average precision (AP) is the main ranking metric because the positive class is very small. AP lift is `AP / positive prevalence`, so `1.0×` is approximately random-ranking performance for a given horizon.

## July holdout

| Model | Horizon | AP, mean ± SD | AP lift | ROC-AUC |
|---|---:|---:|---:|---:|
| XGBoost | 1 h | 0.001604 ± 0.000433 | **1.63×** | 0.420 |
| XGBoost | 3 h | 0.002958 ± 0.000006 | 1.00× | 0.438 |
| XGBoost | 6 h | 0.005819 ± 0.000150 | 0.99× | 0.438 |
| XGBoost | 12 h | 0.012286 ± 0.001744 | 1.04× | 0.456 |
| TCN | 1 h | 0.000742 ± 0.000001 | 0.75× | 0.005 |
| TCN | 3 h | 0.001736 ± 0.000013 | 0.59× | 0.012 |
| TCN | 6 h | 0.003843 ± 0.000876 | 0.65× | 0.155 |
| TCN | 12 h | 0.007008 ± 0.000616 | 0.59× | 0.180 |
| Attention-TCN | 1 h | 0.000767 ± 0.000047 | 0.78× | 0.040 |
| Attention-TCN | 3 h | 0.001745 ± 0.000020 | 0.59× | 0.016 |
| Attention-TCN | 6 h | 0.003226 ± 0.000011 | 0.55× | 0.008 |
| Attention-TCN | 12 h | 0.006306 ± 0.000044 | 0.53× | 0.025 |

![Held-out performance across horizons](figures/horizon_performance.png)

XGBoost ranks best of the three on July at every horizon, but the held-out result is still weak. ROC-AUC remains below 0.5 throughout, and only the one-hour AP is clearly above prevalence.

The sequence models rank the July pre-failure windows below most ordinary windows. Absolute AP rises at longer horizons because there are more positive windows, so AP lift is the more useful cross-horizon comparison.

## May vs June vs July

The event-by-event replay shows a different pattern:

| Event | XGBoost AP lift | TCN AP lift | Attention-TCN AP lift |
|---|---:|---:|---:|
| May | 1.00× | **14.99×** | 2.18× |
| June | 1.64× | 2.57× | **13.60×** |
| July | **1.63×** | 0.75× | 0.78× |

![Cross-event transfer](figures/cross_event_transfer.png)

TCN ranks the May precursor strongly, while Attention-TCN ranks June strongly. Neither pattern repeats in July.

This suggests that the sequence models learned episode-specific score structure rather than one warning pattern that transfers consistently across failures. That result is also why the project stops at these three models instead of adding another deep architecture.

## Sensor-regime diagnostic

Engineered feature values from the 24 hours before each failure were compared with clean normal-operation windows. Values below are standardized by the normal-window interquartile range.

| Feature | May | June | July |
|---|---:|---:|---:|
| `TP2_mean` | +0.27 | +0.09 | **+4.50** |
| `pressure_diff_mean` | −0.50 | −0.12 | **−5.03** |
| `H1_mean` | −0.46 | −0.13 | **−4.91** |
| `Oil_temperature_mean` | +0.43 | +0.10 | **+1.15** |
| `Motor_current_max` | −0.62 | −0.38 | −0.27 |

![Sensor-regime shifts](figures/event_regime_shift.png)

Some directions repeat, but July is much more extreme on several pressure-related features. With only three evaluated failure episodes, this is not enough to define a universal precursor; it is more useful as evidence that the operating/failure regime changes between events.

The full diagnostic tables are in `evidence/event_regime/`.

An older exported XGBoost feature-importance CSV contained all-zero values with unclear key mapping, so it is not used as evidence here. The current extraction code records the mapping source correctly for future runs.

## Thresholds and alerts

The fixed `0.5` threshold detects no July event for any model/horizon/seed combination.

The table below uses thresholds chosen only from May/June development predictions. “Detected” is the number of seeds that produced at least one true-positive July alert.

| Model | Horizon | Threshold | Detected | False alerts/day | First alert lead |
|---|---:|---:|---:|---:|---:|
| XGBoost | 1 h | 0.30 | 1/3 | 3.44 | 0.53 h |
| XGBoost | 3 h | 0.38 | 3/3 | 1.49 | 2.53 h |
| XGBoost | 6 h | 0.36 | 3/3 | 1.49 | 5.53 h |
| XGBoost | 12 h | 0.35 | 3/3 | 1.49 | 11.53 h |
| TCN | 1 h | 0.95 | 0/3 | 0.00 | — |
| TCN | 3 h | 0.08 | 0/3 | 0.88 | — |
| TCN | 6 h | 0.87 | 0/3 | 0.01 | — |
| TCN | 12 h | 0.05 | 3/3 | 2.24 | 11.53 h |
| Attention-TCN | 1 h | 0.06 | 0/3 | 0.00 | — |
| Attention-TCN | 3 h | 0.05 | 0/3 | 1.83 | — |
| Attention-TCN | 6 h | 0.05 | 1/3 | 2.53 | 5.53 h |
| Attention-TCN | 12 h | 0.06 | 2/3 | 1.02 | 11.53 h |

![Threshold choice and false-alert burden](figures/threshold_alert_burden.png)

Lowering the threshold can make the July event detectable, but the cost is very low precision and frequent false alerts. Event detection therefore has to be read together with alert burden rather than on its own.

## Runtime

These timings come from the same Tesla T4 session. They are useful for relative cost within this run, not as general hardware benchmarks.

| Model | Mean development fit | Mean final fit | Mean prediction | Stored artifact | Parameters |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.21 s | 0.06 s | 0.006 s | 17.6 KB | tree-dependent |
| TCN | 29.26 s | 8.31 s | 0.018 s | 126.9 KB | 29,185 |
| Attention-TCN | 25.13 s | 5.51 s | 0.018 s | 132.6 KB | 30,274 |

![Fit cost versus held-out ranking](figures/fit_cost_vs_ranking.png)

The TCN models are much more expensive to fit and do not improve the held-out ranking. Attention adds 1,089 parameters over the TCN without improving the July result.

## Probability traces

The saved per-window predictions make the event behavior easier to inspect than a single summary score:

![Event-centered probability timelines](figures/event_probability_timelines.png)

The development events show clear model-specific score structure. The July precursor does not reproduce that structure for the sequence models.

## Interpretation

The main findings are:

- overlapping time windows make the dataset look much larger than the number of genuinely independent failure episodes;
- preserving more temporal detail does not automatically improve transfer;
- TCN and Attention-TCN can fit different failure episodes strongly without learning one repeatable warning pattern;
- threshold tuning can improve event detection while still producing an impractical false-alert rate.

Longer input history and additional independent failure episodes are reasonable follow-ups. They are not included in this experiment because the July result is already known; redesigning the models around that outcome would turn the holdout into development feedback.

The raw result files and per-window traces are under `evidence/temporal_experiment/`.