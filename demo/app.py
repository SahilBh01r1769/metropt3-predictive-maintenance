from __future__ import annotations

import json
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
    MODEL_COLORS,
    MODEL_LABELS,
    event_failure_time,
    filter_eventwise_rows,
    filter_metric_rows,
    load_event_regime_evidence,
    load_explorer_evidence,
    load_probability_trace,
    summarize_metric_rows,
    threshold_for,
)

EVIDENCE_ROOT = ROOT / "evidence" / "temporal_experiment"
REGIME_ROOT = ROOT / "evidence" / "event_regime"
CONFIG_PATH = ROOT / "configs" / "temporal_experiment.json"
AUDIT_PATH = ROOT / "evidence" / "data_audit_summary.json"
HORIZONS = [1, 3, 6, 12]
SEEDS = [17, 42, 89]
THRESHOLD_LABELS = {
    "reference_0.5": "0.5 reference",
    "development_selected": "Development-selected",
}

st.set_page_config(
    page_title="MetroPT-3 Experiment Explorer",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    :root {
        --page: #F3F1EB; --panel: #FAF8F3; --strong: #FFFFFF;
        --border: #D8D3C8; --text: #26333A; --muted: #68757A;
        --holdout: #B18A50; --warning: #A6534F;
    }
    .stApp { background: var(--page); color: var(--text); }
    [data-testid="stHeader"] { background: rgba(243, 241, 235, .92); }
    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2.2rem; }
    h1, h2, h3, p, label, [data-testid="stCaptionContainer"] { color: var(--text); }
    h1 { font-size: 2rem !important; letter-spacing: -.03em; }
    h2 { font-size: 1.35rem !important; margin-top: 1.8rem !important; }
    h3 { font-size: 1rem !important; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    div[data-testid="stMetric"] {
        background: var(--strong); border: 1px solid var(--border);
        border-radius: 4px; padding: .8rem 1rem;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { color: var(--text); font-size: 1.45rem; }
    div[data-testid="stExpander"], [data-testid="stDataFrame"] {
        border: 1px solid var(--border); border-radius: 4px; background: var(--panel);
    }
    .study-kicker { color: var(--muted); font-size: .75rem; letter-spacing: .12em;
        text-transform: uppercase; font-weight: 700; }
    .question-panel { background: var(--strong); border: 1px solid var(--border);
        border-left: 4px solid var(--holdout); padding: 1.2rem 1.35rem; margin: 1rem 0; }
    .question-panel h2 { margin: 0 !important; font-size: 1.45rem !important; }
    .conclusion-panel { background: #FBF5F3; border: 1px solid #D9BBB7;
        border-left: 4px solid var(--warning); padding: 1rem 1.2rem; margin: 1rem 0; }
    .method-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: .45rem;
        align-items: stretch; margin: 1rem 0 1.25rem; }
    .method-step { background: var(--panel); border: 1px solid var(--border);
        padding: .75rem; min-height: 84px; }
    .method-step b { display: block; font-size: .88rem; margin-bottom: .25rem; }
    .method-step span { color: var(--muted); font-size: .78rem; line-height: 1.3; }
    .holdout-step { background: #FCF8F0; border-color: var(--holdout); }
    .definition { background: var(--panel); border-top: 1px solid var(--border);
        border-bottom: 1px solid var(--border); padding: .65rem .8rem;
        color: var(--muted); font-size: .84rem; margin: .75rem 0 1rem; }
    .section-intro { max-width: 760px; color: var(--muted); margin-bottom: 1rem; }
    @media (max-width: 800px) { .method-strip { grid-template-columns: 1fr; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    metrics, eventwise, thresholds = load_explorer_evidence(EVIDENCE_ROOT)
    regime_summary, regime_effects = load_event_regime_evidence(REGIME_ROOT)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    return metrics, eventwise, thresholds, regime_summary, regime_effects, audit


def chart_layout(fig: go.Figure, *, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#FAF8F3",
        plot_bgcolor="#FAF8F3",
        font={"color": "#26333A", "family": "Arial, sans-serif", "size": 12},
        title={"font": {"size": 16}, "x": 0.01, "xanchor": "left"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 45, "r": 20, "t": 72, "b": 45},
        hoverlabel={"bgcolor": "#FFFFFF", "font_color": "#26333A"},
    )
    fig.update_xaxes(gridcolor="#E8E3DA", zerolinecolor="#D8D3C8")
    fig.update_yaxes(gridcolor="#E8E3DA", zerolinecolor="#D8D3C8")
    return fig


def model_name(value: str) -> str:
    return MODEL_LABELS[value]


def event_name(value: str) -> str:
    return EVENT_LABELS[value]


def aggregate_trace(trace: pd.DataFrame, seed: int | None) -> pd.DataFrame:
    if seed is not None:
        selected = trace.loc[trace["seed"].eq(seed)].copy()
        return selected[["window_end", "probability"]].assign(
            probability_min=selected["probability"],
            probability_max=selected["probability"],
        )
    return trace.groupby("window_end", as_index=False).agg(
        probability=("probability", "mean"),
        probability_min=("probability", "min"),
        probability_max=("probability", "max"),
    )


try:
    metrics, eventwise, thresholds, regime_summary, regime_effects, audit = load_data()
except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Committed experiment evidence could not be loaded: {exc}")
    st.stop()

st.markdown('<div class="study-kicker">MetroPT-3 · controlled temporal study</div>', unsafe_allow_html=True)
st.title("Experiment results explorer")
st.caption(
    "Read-only views over committed metrics and probability traces. No models are retrained and no new predictions are generated."
)

page = st.radio(
    "Explorer section",
    ["Study overview", "Compare models", "Event transfer", "Alert trade-offs"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

if page == "Study overview":
    st.markdown(
        '<div class="question-panel"><div class="study-kicker">Study question</div>'
        '<h2>Do warning patterns learned around the May and June failures transfer to July?</h2></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="method-strip">
  <div class="method-step"><b>One-hour history</b><span>The same observation length for every model.</span></div>
  <div class="method-step"><b>Three representations</b><span>XGBoost summaries → TCN sequence → Attention-TCN.</span></div>
  <div class="method-step"><b>Four horizons</b><span>Warnings 1, 3, 6 or 12 hours before failure.</span></div>
  <div class="method-step"><b>Development</b><span>May and June, with three repeated seeds.</span></div>
  <div class="method-step holdout-step"><b>Final holdout</b><span>July remained untouched during model and threshold choices.</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validated telemetry rows", f"{audit['dataset']['valid_rows']:,}")
    c2.metric("Predictive windows", f"{audit['windows']['predictive_windows']:,}")
    c3.metric("Independent failures", "3", help="May and June development events plus the July holdout event.")
    c4.metric("Model × horizon × seed", "36", help="3 models × 4 horizons × 3 seeds.")
    st.markdown(
        '<div class="definition"><b>Important:</b> 7,846 overlapping windows provide many training examples, '
        'but they come from only three independent failure episodes. Window count is not event count.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="conclusion-panel"><b>Main finding</b><br>'
        'Sequence models ranked individual development events strongly, but those warning patterns did not transfer reliably to July. '
        'XGBoost held up best on July, although holdout performance remained weak.</div>',
        unsafe_allow_html=True,
    )
    july = filter_eventwise_rows(eventwise, event="july_holdout")
    july["Model"] = july["model"].map(MODEL_LABELS)
    fig = px.line(
        july, x="horizon_hours", y="ap_lift_mean", color="Model", markers=True,
        color_discrete_map={MODEL_LABELS[k]: v for k, v in MODEL_COLORS.items()},
        labels={"horizon_hours": "Prediction horizon (hours)", "ap_lift_mean": "AP lift"},
        title="Question: which representation ranked July warning windows best?",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="#68757A", annotation_text="prevalence baseline")
    fig.update_xaxes(tickvals=HORIZONS)
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.caption("AP lift divides average precision by positive prevalence. A value of 1× matches the prevalence baseline.")
    if st.button("Explore the model comparison →", type="primary"):
        st.info("Choose **Compare models** in the navigation above to inspect horizons, seeds and saved values.")

elif page == "Compare models":
    st.header("Compare models")
    st.markdown(
        '<p class="section-intro">Compare ranking quality under the same one-hour history and chronological evidence. '
        'The default aggregates all three seeds; selecting a seed exposes run-to-run variation.</p>',
        unsafe_allow_html=True,
    )
    filters = st.columns([1.2, 1, 1, 1.25])
    model = filters[0].selectbox("Model", [None, *MODEL_LABELS], format_func=lambda v: "All models" if v is None else model_name(v))
    horizon = filters[1].selectbox("Horizon", HORIZONS, format_func=lambda value: f"{value} h")
    seed = filters[2].selectbox("Seed", [None, *SEEDS], format_func=lambda v: "Aggregate (3 seeds)" if v is None else str(v))
    event = filters[3].selectbox("Event", [None, *EVENT_LABELS], format_func=lambda v: "All events" if v is None else event_name(v))
    metric_label = st.selectbox(
        "Primary question", ["AP lift", "Average precision", "ROC-AUC"],
        help="This changes the displayed chart, not the underlying experiment.",
    )
    column_for = {"AP lift": "ap_lift_mean", "Average precision": "average_precision_mean", "ROC-AUC": "roc_auc_mean"}
    y_column = column_for[metric_label]
    selected_events = filter_eventwise_rows(eventwise, model=model, horizon_hours=horizon, event=event)
    selected_events["Model"] = selected_events["model"].map(MODEL_LABELS)
    selected_events["Event"] = selected_events["event"].map(EVENT_LABELS)
    if seed is not None:
        raw = pd.read_csv(EVIDENCE_ROOT / "eventwise_metrics.csv")
        raw = raw.loc[raw["seed"].eq(seed) & raw["horizon_hours"].eq(horizon)]
        if model is not None:
            raw = raw.loc[raw["model"].eq(model)]
        if event is not None:
            raw = raw.loc[raw["event"].eq(event)]
        raw["Model"] = raw["model"].map(MODEL_LABELS)
        raw["Event"] = raw["event"].map(EVENT_LABELS)
        selected_events = raw.rename(columns={
            "ap_lift_over_prevalence": "ap_lift_mean",
            "average_precision": "average_precision_mean",
            "roc_auc": "roc_auc_mean",
        })
    fig = px.bar(
        selected_events, x="Model", y=y_column, color="Model",
        facet_col="Event" if event is None else None,
        color_discrete_map={MODEL_LABELS[k]: v for k, v in MODEL_COLORS.items()},
        labels={y_column: metric_label},
        title=f"Question: how did representation choice affect {metric_label.lower()} at {horizon} h?",
    )
    fig.update_layout(showlegend=False)
    if metric_label == "AP lift":
        fig.add_hline(y=1, line_dash="dash", line_color="#68757A")
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.caption("More complex does not mean consistently better: the neural models peak on different development episodes, while XGBoost transfers least poorly to July.")
    compact = selected_events[["Event", "Model", "prevalence", "average_precision_mean", "ap_lift_mean", "roc_auc_mean"]].copy()
    compact.columns = ["Event", "Model", "Prevalence", "AP", "AP lift", "ROC-AUC"]
    st.dataframe(compact, width="stretch", hide_index=True)
    with st.expander("Inspect saved metric and probability evidence"):
        st.dataframe(filter_metric_rows(metrics, model=model, horizon_hours=horizon, seed=seed), width="stretch", hide_index=True)
        st.caption("Rows are loaded directly from the committed metric table. The saved probability traces are not modified.")

elif page == "Event transfer":
    st.header("Event transfer")
    st.markdown(
        '<p class="section-intro">May and June are development evidence. July is the final holdout: a later episode not used '
        'for model selection, early stopping or operational-threshold selection.</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    horizon = c1.selectbox("Horizon", HORIZONS, format_func=lambda value: f"{value} h", key="transfer_h")
    model = c2.selectbox("Timeline model", list(MODEL_LABELS), format_func=model_name, key="transfer_m")
    seed = c3.selectbox("Timeline seed", [None, *SEEDS], format_func=lambda v: "Aggregate (3 seeds)" if v is None else str(v))
    transfer = filter_eventwise_rows(eventwise, horizon_hours=horizon)
    transfer["Model"] = transfer["model"].map(MODEL_LABELS)
    transfer["Event"] = transfer["event"].map(EVENT_LABELS)
    fig = px.bar(
        transfer, x="Event", y="ap_lift_mean", color="Model", barmode="group", error_y="ap_lift_std",
        color_discrete_map={MODEL_LABELS[k]: v for k, v in MODEL_COLORS.items()},
        category_orders={"Event": list(EVENT_LABELS.values())},
        labels={"ap_lift_mean": "AP lift", "ap_lift_std": "Seed variation"},
        title=f"Question: did {horizon}-hour ranking transfer from development to holdout?",
    )
    fig.add_vrect(x0=1.5, x1=2.5, fillcolor="#B18A50", opacity=0.09, line_width=0)
    fig.add_hline(y=1, line_dash="dash", line_color="#68757A", annotation_text="prevalence baseline")
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.caption("Error bars show variation across three training seeds. The shaded July group is the held-out episode.")
    event = st.segmented_control("Probability timeline event", list(EVENT_LABELS), default="july_holdout", format_func=event_name)
    trace = load_probability_trace(EVIDENCE_ROOT, model=model, horizon_hours=horizon, event=event)
    failure_time = event_failure_time(CONFIG_PATH, event)
    trace = trace.loc[trace["window_end"].between(failure_time - pd.Timedelta(hours=48), failure_time)]
    plotted = aggregate_trace(trace, seed)
    threshold = threshold_for(thresholds, model=model, horizon_hours=horizon, policy="development_selected")
    fig = go.Figure()
    if seed is None:
        fig.add_trace(go.Scatter(
            x=pd.concat([plotted["window_end"], plotted["window_end"].iloc[::-1]]),
            y=pd.concat([plotted["probability_max"], plotted["probability_min"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(82,106,146,.14)", line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip", name="Seed range",
        ))
    fig.add_trace(go.Scatter(
        x=plotted["window_end"], y=plotted["probability"], mode="lines",
        name="Mean probability" if seed is None else f"Seed {seed}", line={"color": MODEL_COLORS[model], "width": 2},
    ))
    fig.add_hline(y=threshold, line_dash="dot", line_color="#A6534F", annotation_text=f"development threshold {threshold:.2f}")
    fig.add_vline(x=failure_time.timestamp() * 1000, line_dash="dash", line_color="#B18A50", annotation_text="failure starts")
    fig.update_layout(title=f"Question: when did {MODEL_LABELS[model]} score the {EVENT_LABELS[event]} precursor?", yaxis_range=[0, 1])
    fig.update_xaxes(title="Window end")
    fig.update_yaxes(title="Predicted probability")
    st.plotly_chart(chart_layout(fig), width="stretch")
    with st.expander("Sensor-regime diagnostic — supporting evidence"):
        features = regime_summary.sort_values("direction_consistency").head(12)["feature"].tolist()
        heat = regime_effects.loc[regime_effects["feature"].isin(features)].pivot(index="feature", columns="event", values="robust_standardized_difference")
        heat = heat.reindex(columns=list(EVENT_LABELS))
        fig = px.imshow(
            heat, color_continuous_scale=[[0, "#A6534F"], [.5, "#FAF8F3"], [1, "#3F7471"]],
            color_continuous_midpoint=0, labels={"color": "Robust shift", "x": "Episode", "y": "Feature"},
            aspect="auto", title="Question: did selected pre-failure sensor summaries move in the same direction?",
        )
        fig.update_xaxes(tickvals=list(EVENT_LABELS), ticktext=list(EVENT_LABELS.values()))
        st.plotly_chart(chart_layout(fig, height=500), width="stretch")
        st.caption("Robust shifts compare each episode's 24-hour precursor windows with clean normal windows. Different regimes are a plausible explanation for weak transfer, not a causal finding.")

elif page == "Alert trade-offs":
    st.header("Alert trade-offs")
    st.markdown(
        '<p class="section-intro">Ranking asks whether warning windows receive higher scores. Alert usefulness also depends on '
        'the selected threshold, false-alert burden and how early a warning appears.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model", list(MODEL_LABELS), format_func=model_name, key="alert_m")
    horizon = c2.selectbox("Horizon", HORIZONS, format_func=lambda value: f"{value} h", key="alert_h")
    selected = summarize_metric_rows(metrics, model=model, horizon_hours=horizon).copy()
    selected["Threshold mode"] = selected["threshold_policy"].map(THRESHOLD_LABELS)
    fig = go.Figure()
    for _, row in selected.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["false_alert_episodes_per_day"]], y=[row["event_detection_rate"]], mode="markers+text",
            text=[row["Threshold mode"]], textposition="top center",
            marker={"size": 15, "color": MODEL_COLORS[model] if row["threshold_policy"] == "reference_0.5" else "#A6534F"},
            name=row["Threshold mode"],
            customdata=[[row["threshold"], row["precision"], row["recall"], row["first_alert_lead_hours"]]],
            hovertemplate="Threshold %{customdata[0]:.2f}<br>False alerts/day %{x:.2f}<br>Event detection %{y:.0%}<br>Precision %{customdata[1]:.3f}<br>Recall %{customdata[2]:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title="Question: what false-alert burden accompanied July event detection?",
        xaxis_title="False-alert episodes per evaluated day", yaxis_title="Seeds detecting the July event",
        yaxis_tickformat=".0%", yaxis_range=[-0.08, 1.12], showlegend=False,
    )
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.caption("A false-alert episode is one continuous run of alerts outside the true pre-failure interval; repeated positive windows in the same run count once.")
    display = selected[["Threshold mode", "threshold", "precision", "recall", "event_detection_rate", "first_alert_lead_hours", "false_alert_episodes_per_day"]].copy()
    display.columns = ["Threshold mode", "Threshold", "Precision", "Recall", "July detection", "First-warning lead (h)", "False alerts/day"]
    st.dataframe(display, width="stretch", hide_index=True)
    st.markdown(
        '<div class="conclusion-panel"><b>Interpretation boundary</b><br>'
        'Operational thresholds were selected from pooled May/June predictions only. Recovering July can still be a poor policy when precision is low or false alerts are frequent.</div>',
        unsafe_allow_html=True,
    )

if page != "Study overview":
    st.divider()
    st.caption(
        "Prediction horizon = how far before failure a window is labelled positive. Holdout = evidence excluded from model and threshold choices. "
        "Attention weights indicate model focus, not causal sensor explanations."
    )
