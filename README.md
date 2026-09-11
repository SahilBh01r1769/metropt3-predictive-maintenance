# MetroPT-3 Predictive Maintenance

This started as a fairly standard predictive-maintenance project: take the MetroPT-3 compressor telemetry and see whether upcoming air leaks can be predicted early enough to be useful.

The part that became more interesting was the evaluation. I wanted to know whether a model that looks good around one failure would still behave the same way around a later one, so the final comparison uses the same one-hour history for three models:

- **XGBoost** on engineered summary features
- **TCN** on the ordered sensor sequence
- **Attention-TCN** on the same TCN encoder with attention pooling

I tested 1, 3, 6 and 12-hour warning horizons. May and June are used for development; July is the final holdout for this experiment.

The short version: the deeper models did learn strong patterns, but not the same ones. TCN ranked the May precursor very well, Attention-TCN did the same for June, and both fell apart on July. XGBoost held up best on the later event, although the overall result is still not strong enough to call this a useful maintenance predictor.

**[Open the Streamlit results explorer](https://metropt3-predictive-maintenance.streamlit.app/)**

## Project snapshot

- **1,516,948** raw telemetry rows
- **334** continuity segments after validation and gap handling
- **8,012** complete one-hour windows
- **7,846** predictive windows after removing windows inside active failures
- **3 models × 4 prediction horizons × 3 seeds**
- committed per-window probabilities, thresholds and result tables

The raw dataset is not stored in the repo; `scripts/download_data.py` downloads the official MetroPT-3 archive.

## What I compared

| Model | Input | What I wanted to test |
|---|---|---|
| XGBoost | 38 engineered features from one hour | whether summary statistics are already enough |
| TCN | 120 ordered 30-second steps | whether keeping within-hour order helps |
| Attention-TCN | same sequence and TCN encoder | whether attention pooling adds anything useful |

All three models see the same amount of history. The sequence models use eight sensor/actuator channels plus one observation-coverage channel.

I did not use a random train/test split. These windows overlap heavily in time, so a random split can put almost the same raw observations on both sides. The final setup keeps July later in time and purges training windows that touch the evaluation interval.

## What happened

The one-hour horizon shows the main result most clearly:

| Event | XGBoost AP lift | TCN AP lift | Attention-TCN AP lift |
|---|---:|---:|---:|
| May | 1.00× | **14.99×** | 2.18× |
| June | 1.64× | 2.57× | **13.60×** |
| July | **1.63×** | 0.75× | 0.78× |

![AP lift across failure episodes](figures/cross_event_transfer.png)

The surprising part is not simply that the neural models score poorly on July. It is that each of them looks convincing on a different earlier failure. That made me treat this as an event-to-event transfer problem rather than keep adding model complexity.

Across all four July horizons, XGBoost ranks best of the three, but the held-out performance is still weak. The detailed numbers are in [RESULTS.md](RESULTS.md).

![Held-out horizon comparison](figures/horizon_performance.png)

## Looking at the sensor regimes

After seeing the transfer problem, I added one descriptive check rather than another predictor. It compares engineered features during the 24 hours before the May, June and July failures with normal operating windows.

![Pre-failure feature shifts](figures/event_regime_shift.png)

Some directions repeat, but the size of the shift changes a lot between events. July is especially different for features such as `TP2_mean`, `pressure_diff_mean` and `H1_mean`. With so few independent failures, that is enough to explain why I stopped treating the window count as if it were a large sample of failure behavior.

The underlying tables are in `evidence/event_regime/`.

## Alert thresholds

I also checked what the model scores would look like as maintenance alerts.

At the default `0.5` threshold, none of the models detects the July event. Thresholds picked only from May/June can recover July in some settings, but usually by accepting a lot of false alarms. For example, XGBoost detects July in all three seeds at 3, 6 and 12 hours with about **1.49 false-alert episodes per evaluated day**.

So I report alert detection together with precision, lead time and false-alert burden rather than treating “event detected” as enough.

![Threshold and false-alert trade-off](figures/threshold_alert_burden.png)

## Results explorer

The Streamlit app reads the saved experiment outputs. It is not a simulated live maintenance product.

You can switch between:

- XGBoost, TCN and Attention-TCN
- 1, 3, 6 and 12-hour horizons
- May, June and July
- the fixed `0.5` threshold and the threshold selected on development data

Run it locally with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
streamlit run demo/app.py
```

## Reproducing the experiment

The final experiment settings are in `configs/temporal_experiment.json` and the Colab run is in `notebooks/02_temporal_experiment_colab.ipynb`.

To regenerate the result plots from the committed tables:

```bash
pip install -r requirements-experiment.txt
pip install -e .
python scripts/generate_evidence_plots.py
```

To recompute the event-wise ranking tables from the saved probability traces:

```bash
python scripts/analyze_temporal_results.py
```

The committed outputs are enough to audit the reported metrics. The model binaries were not part of the returned Colab export, so reproducing the exact predictions requires retraining from the saved configuration.

## Repository layout

```text
src/metropt3/                  preprocessing, sequence models and experiment code
configs/                       final experiment settings
evidence/                      saved metrics and probability traces
figures/                       plots built from those results
notebooks/                     data audit, training and follow-up diagnostics
demo/app.py                    Streamlit results explorer
RESULTS.md                     detailed numbers and interpretation
EXPERIMENT_PROTOCOL.md         setup used for the final run
```

## A few limits

There are many overlapping windows but only a few independent failure episodes. This repo therefore says more about **how the models behaved on these separate events** than about expected performance on another compressor.

July was untouched for the final three-model comparison. Since those results are now known, any new feature or architecture designed specifically because July failed would be a follow-up experiment, not another untouched test.

Dataset: **MetroPT-3** public air-compressor telemetry.