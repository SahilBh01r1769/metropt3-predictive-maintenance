# MetroPT-3 Predictive Maintenance

This project studies whether warning patterns learned before earlier failures in the MetroPT-3 air-compressor dataset also work for a later failure.

I started with the usual predictive-maintenance goal of comparing models, but the more interesting result was that the deeper sequence models learned useful patterns around individual failures without transferring well to the July event.

The final comparison uses the same one-hour observation history for three models:

- **XGBoost** on engineered summary features
- **TCN** on the ordered sensor sequence
- **Attention-TCN** using the same temporal encoder with attention pooling

Predictions are evaluated at 1, 3, 6 and 12-hour warning horizons. May and June are used for development; July is the final holdout for the three-model comparison.

**[Open the Streamlit results explorer](https://metropt3-predictive-maintenance.streamlit.app/)**

## Explorer preview

The Streamlit app reads the saved experiment results and lets the comparison be inspected without retraining the models.

![MetroPT experiment explorer walkthrough](assets/metropt-explorer-walkthrough.gif)

<details>
<summary>Static explorer snapshots</summary>

<p align="center">
  <img src="assets/study-overview.png" alt="MetroPT study overview" width="49%">
  <img src="assets/event-transfer.png" alt="MetroPT event transfer comparison" width="49%">
</p>
<p align="center">
  <img src="assets/model-comparison.png" alt="MetroPT model comparison" width="70%">
</p>

</details>

## Experiment setup

The raw telemetry contains **1,516,948 rows**. After validation and gap handling, the data is divided into continuity segments so one-hour windows never cross a large break in the sensor stream.

The final dataset contains:

- 334 continuity segments
- 8,012 complete one-hour windows
- 7,846 predictive windows after excluding windows already inside an active failure
- three models, four warning horizons and three random seeds

All models receive the same amount of history. XGBoost sees 38 engineered features from that hour, while the sequence models receive 120 ordered 30-second steps built from the sensor channels plus an observation-coverage channel.

A random train/test split is avoided because neighboring windows overlap heavily in time. The final setup keeps the later July event separate and purges training windows that would share raw observations with the evaluation period.

The detailed setup is recorded in `EXPERIMENT_PROTOCOL.md`.

## What happened

The one-hour horizon shows the main result clearly:

| Event | XGBoost AP lift | TCN AP lift | Attention-TCN AP lift |
|---|---:|---:|---:|
| May | 1.00× | **14.99×** | 2.18× |
| June | 1.64× | 2.57× | **13.60×** |
| July | **1.63×** | 0.75× | 0.78× |

![AP lift across failure episodes](figures/cross_event_transfer.png)

TCN ranks the May failure strongly and Attention-TCN does the same for June, but neither pattern carries over to July. XGBoost is less impressive on the development failures, yet it is the best of the three on July across all four horizons.

That does **not** mean XGBoost solves the task. Held-out performance is weak overall. The useful result is that thousands of overlapping training windows came from only a small number of independent failure episodes, and the richer temporal models appear to have learned episode-specific warning patterns.

![Held-out horizon comparison](figures/horizon_performance.png)

Full per-horizon metrics are kept in `RESULTS.md` rather than repeated here.

## Looking at the failure episodes

After seeing the transfer failure, I compared the sensor behavior before the May, June and July events with normal-operation windows.

![Pre-failure feature shifts](figures/event_regime_shift.png)

Some feature directions repeat, but the size of the changes differs considerably between failures. July is especially different for several pressure-related features.

This does not prove exactly why the sequence models failed, but it supports the main interpretation: these failure episodes are not interchangeable examples of one stable precursor pattern.

## Alert thresholds

Ranking metrics are only part of predictive maintenance, so I also checked what happens when model scores are turned into alerts.

At the default `0.5` threshold, none of the models detects July. Thresholds chosen only from May and June can recover the event in some settings, but the improvement often comes with frequent false alerts.

![Threshold and false-alert trade-off](figures/threshold_alert_burden.png)

For that reason the project reports event detection together with false-alert burden and lead time instead of presenting one threshold as universally good.

## Running the explorer

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
streamlit run demo/app.py
```

To regenerate the main figures from the committed result tables:

```bash
pip install -r requirements-experiment.txt
pip install -e .
python scripts/generate_evidence_plots.py
```

The repository also contains saved per-window probabilities, thresholds and result tables. The original model binaries were not part of the returned Colab export, so reproducing the exact predictions requires retraining from the saved experiment configuration.

## Repository guide

```text
src/metropt3/          preprocessing, models and experiment code
configs/               final experiment settings
evidence/              saved metrics and probability traces
figures/               plots generated from the saved results
notebooks/             data audit, training and follow-up analysis
demo/app.py            Streamlit results explorer
RESULTS.md             detailed result tables and interpretation
EXPERIMENT_PROTOCOL.md experiment setup
```

## Scope

This is a case study on one compressor dataset with very few independent failure episodes. It is useful for studying transfer between those events, but the results should not be read as expected performance on another compressor or failure type.

Dataset: **MetroPT-3** public air-compressor telemetry.
