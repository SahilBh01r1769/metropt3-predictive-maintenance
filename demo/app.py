from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from metropt3.explorer import (
    EVENT_LABELS,
    MODEL_LABELS,
    event_failure_time,
    load_explorer_evidence,
    load_probability_trace,
    summarize_metric_rows,
    threshold_for,
)

EVIDENCE_ROOT = ROOT / "evidence" / "temporal_experiment"
CONFIG_PATH = ROOT / "configs" / "temporal_experiment.json"

st.set_page_config(page_title="MetroPT-3 Experiment Explorer", layout="wide")
st.title("MetroPT-3 temporal representation study")
st.caption(
    "Explore the committed evidence behind the XGBoost → TCN → Attention-TCN comparison. "
    "This is an experiment viewer, not a live maintenance product."
)


@st.cache_data
def load_data():
    return load_explorer_evidence(EVIDENCE_ROOT)


try:
    metrics, eventwise, thresholds = load_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Committed experiment evidence could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.header("Evidence controls")
    model = st.selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
    horizon = st.selectbox(
        "Prediction horizon", [1, 3, 6, 12], format_func=lambda value: f"{value} h"
    )
    event = st.selectbox(
        "Failure episode", list(EVENT_LABELS), format_func=EVENT_LABELS.get
    )
    threshold_policy = st.radio(
        "Threshold mode",
        ["reference_0.5", "development_selected"],
        format_func={
            "reference_0.5": "0.5 reference",
            "development_selected": "Development-selected",
        }.get,
    )
    st.divider()
    st.caption("All values are loaded from versioned prediction traces and metric tables.")

overview, transfer, timeline, operations = st.tabs(
    ["Experiment overview", "Cross-event transfer", "Probability timeline", "Operational view"]
)

with overview:
    selected = summarize_metric_rows(metrics, model=model, horizon_hours=horizon)
    reference = selected.loc[selected["threshold_policy"].eq("reference_0.5")].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average precision", f"{reference['average_precision']:.4f}")
    c2.metric("AP lift / prevalence", f"{reference['ap_lift_over_prevalence']:.2f}×")
    c3.metric("ROC-AUC", f"{reference['roc_auc']:.3f}")
    c4.metric("Positive prevalence", f"{reference['prevalence']:.3%}")

    july = eventwise.loc[eventwise["event"].eq("july_holdout")].copy()
    july["model_label"] = july["model"].map(MODEL_LABELS)
    fig = px.line(
        july,
        x="horizon_hours",
        y="ap_lift_mean",
        color="model_label",
        markers=True,
        labels={
            "horizon_hours": "Prediction horizon (hours)",
            "ap_lift_mean": "AP lift over prevalence",
            "model_label": "Model",
        },
        title="Held-out July ranking across horizons",
    )
    fig.add_hline(y=1, line_dash="dash", annotation_text="prevalence baseline")
    fig.update_xaxes(tickvals=[1, 3, 6, 12])
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Absolute AP rises as longer horizons create more positive windows. AP lift divides by "
        "prevalence, making the four horizon tasks more comparable."
    )

with transfer:
    event_frame = eventwise.loc[eventwise["horizon_hours"].eq(horizon)].copy()
    event_frame["event_label"] = event_frame["event"].map(EVENT_LABELS)
    event_frame["model_label"] = event_frame["model"].map(MODEL_LABELS)
    fig = px.bar(
        event_frame,
        x="event_label",
        y="ap_lift_mean",
        color="model_label",
        barmode="group",
        error_y="ap_lift_std",
        labels={
            "event_label": "Failure episode",
            "ap_lift_mean": "AP lift over prevalence",
            "model_label": "Model",
        },
        title=f"Cross-event transfer at the {horizon}-hour horizon",
    )
    fig.add_hline(y=1, line_dash="dash", annotation_text="prevalence baseline")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "The development episodes do not identify one consistently superior temporal model. "
        "At one hour, TCN ranks May strongly and Attention-TCN ranks June strongly; neither "
        "ranking transfers to July. XGBoost is weaker on development episodes but fails least "
        "severely on the held-out event."
    )

with timeline:
    trace = load_probability_trace(
        EVIDENCE_ROOT, model=model, horizon_hours=horizon, event=event
    )
    failure_time = event_failure_time(CONFIG_PATH, event)
    trace = trace.loc[
        trace["window_end"].between(failure_time - pd.Timedelta(hours=48), failure_time)
    ]
    aggregated = (
        trace.groupby("window_end", as_index=False)
        .agg(
            probability=("probability", "mean"),
            probability_min=("probability", "min"),
            probability_max=("probability", "max"),
        )
    )
    chosen_threshold = threshold_for(
        thresholds,
        model=model,
        horizon_hours=horizon,
        policy=threshold_policy,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.concat([aggregated["window_end"], aggregated["window_end"].iloc[::-1]]),
            y=pd.concat(
                [aggregated["probability_max"], aggregated["probability_min"].iloc[::-1]]
            ),
            fill="toself",
            fillcolor="rgba(80, 104, 138, 0.16)",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name="Seed range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=aggregated["window_end"],
            y=aggregated["probability"],
            mode="lines",
            name="Mean probability",
            line={"color": "#50688a"},
        )
    )
    fig.add_hline(
        y=chosen_threshold,
        line_dash="dot",
        annotation_text=f"threshold {chosen_threshold:.2f}",
    )
    fig.add_vline(
        x=failure_time.timestamp() * 1000,
        line_dash="dash",
        annotation_text="failure starts",
    )
    fig.update_layout(
        title=f"{MODEL_LABELS[model]} before the {EVENT_LABELS[event]} failure",
        xaxis_title="Window end",
        yaxis_title="Predicted probability",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The line is the mean over three seeds; the shaded region is their minimum-to-maximum range."
    )

with operations:
    selected = summarize_metric_rows(metrics, model=model, horizon_hours=horizon)
    chosen = selected.loc[selected["threshold_policy"].eq(threshold_policy)].iloc[0]
    threshold = threshold_for(
        thresholds,
        model=model,
        horizon_hours=horizon,
        policy=threshold_policy,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold", f"{threshold:.2f}")
    c2.metric("Seeds detecting July", f"{chosen['event_detection_rate']:.0%}")
    c3.metric("False alerts / day", f"{chosen['false_alert_episodes_per_day']:.2f}")
    lead = chosen["first_alert_lead_hours"]
    c4.metric("Mean first-warning lead", "—" if pd.isna(lead) else f"{lead:.2f} h")

    comparison = selected.copy()
    comparison["Threshold policy"] = comparison["threshold_policy"].map(
        {
            "reference_0.5": "0.5 reference",
            "development_selected": "Development-selected",
        }
    )
    display = comparison[
        [
            "Threshold policy",
            "threshold",
            "precision",
            "recall",
            "balanced_accuracy",
            "false_alert_episodes_per_day",
            "event_detection_rate",
            "first_alert_lead_hours",
        ]
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.warning(
        "A detected event is not sufficient evidence of a useful policy. The development-selected "
        "thresholds sometimes recover July only by accepting low precision and frequent false alerts."
    )

if model == "attention_tcn":
    st.divider()
    st.subheader("Attention interpretation boundary")
    st.caption(
        "Attention weights indicate where the model focused within the observed sequence. "
        "They are not causal explanations of compressor failure. Raw attention arrays were "
        "validated in the original export but are not part of the compact committed evidence, "
        "so this explorer does not manufacture an attention visualization."
    )
