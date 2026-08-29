from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from metropt3.config import ANALOGUE_COLS, ARTIFACT_DIR
from metropt3.features import build_windows
from metropt3.modeling import predict_risk
from metropt3.validation import range_diagnostics, validate_and_segment

st.set_page_config(page_title="MetroPT-3 Maintenance Dashboard", page_icon="⚙️", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1320px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(45,212,191,.25); border-radius: 18px;
       background: linear-gradient(135deg, rgba(45,212,191,.10), rgba(17,28,46,.45)); margin-bottom: 1.1rem;}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: .45rem 0 0; color: #a9b8cc;}
.pill {display:inline-block; padding:.25rem .65rem; margin:.6rem .35rem 0 0; border-radius:999px;
       border:1px solid rgba(45,212,191,.35); color:#8ce9db; font-size:.78rem;}
.risk-card {padding:1.2rem 1.4rem; border-radius:16px; border:1px solid rgba(148,163,184,.2); background:#111c2e;}
.small-note {color:#94a3b8; font-size:.82rem;}
.explain-card {padding:.9rem 1rem; border-radius:14px; border:1px solid rgba(148,163,184,.16); background:rgba(17,28,46,.55); min-height:104px;}
.explain-kicker {font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#5eead4; font-weight:700;}
.explain-card h4 {margin:.25rem 0 .25rem 0; font-size:1rem;}
.explain-card p {margin:0; color:#9fb0c5; font-size:.84rem; line-height:1.45;}
.section-hint {color:#8fa2b8; font-size:.83rem; margin-top:-.35rem; margin-bottom:.7rem;}
</style>
<div class="hero">
  <h1>MetroPT-3 Predictive Maintenance</h1>
  <p>Turn air-compressor sensor telemetry into condition signals that help identify when equipment may need attention.</p>
  <span class="pill">Multivariate sensor telemetry</span>
  <span class="pill">Gap-safe segmentation</span>
  <span class="pill">Failure-horizon modeling</span>
</div>
""",
    unsafe_allow_html=True,
)

# A compact orientation layer: enough context to understand the workflow
# without turning the dashboard into documentation.
o1, o2, o3 = st.columns(3)
with o1:
    st.markdown(
        """<div class="explain-card"><div class="explain-kicker">Objective</div>
        <h4>Detect changing equipment condition</h4>
        <p>Watch pressure, temperature and electrical behaviour for patterns that may precede maintenance events.</p></div>""",
        unsafe_allow_html=True,
    )
with o2:
    st.markdown(
        """<div class="explain-card"><div class="explain-kicker">Method</div>
        <h4>Summarize telemetry over time windows</h4>
        <p>Validate readings, separate data gaps, then convert each continuous window into condition features.</p></div>""",
        unsafe_allow_html=True,
    )
with o3:
    st.markdown(
        """<div class="explain-card"><div class="explain-kicker">Result</div>
        <h4>Surface a maintenance-risk signal</h4>
        <p>The latest window is summarized as Stable, Watch or High Attention alongside the signals behind it.</p></div>""",
        unsafe_allow_html=True,
    )


def synthetic_frame(kind: str, seconds: int = 7200) -> pd.DataFrame:
    rng = np.random.default_rng(42 if kind == "Healthy reference" else 7)
    t = np.arange(seconds)
    ts = pd.date_range("2020-03-20 08:00:00", periods=seconds, freq="s")
    degrading = kind == "Degradation scenario"
    ramp = np.linspace(0, 1, seconds) if degrading else np.zeros(seconds)
    comp = ((t // 90) % 2).astype(int)
    data = pd.DataFrame(
        {
            "timestamp": ts,
            "TP2": 1.3 + comp * 7.8 + rng.normal(0, 0.10, seconds) + 0.5 * ramp,
            "TP3": 8.7 + rng.normal(0, 0.08, seconds) - 0.7 * ramp,
            "H1": 8.1 + rng.normal(0, 0.09, seconds) - 0.5 * ramp,
            "DV_pressure": 0.05 + rng.normal(0, 0.025, seconds) + 0.7 * ramp,
            "Reservoirs": 8.6 + rng.normal(0, 0.07, seconds) - 0.5 * ramp,
            "Oil_temperature": 62 + rng.normal(0, 0.45, seconds) + 18 * ramp,
            "Motor_current": 3 + comp * 5.5 + rng.normal(0, 0.35, seconds) + 2.4 * ramp,
            "COMP": comp,
        }
    )
    return data


def heuristic_risk(latest: pd.Series) -> float:
    """Transparent demo-only indicator; this is not the trained ML model."""
    signals = [
        np.clip((latest.get("Oil_temperature_mean", 60) - 65) / 35, 0, 1),
        np.clip((latest.get("Motor_current_mean", 5) - 6) / 8, 0, 1),
        np.clip(latest.get("DV_pressure_mean", 0) / 1.5, 0, 1),
        np.clip(abs(latest.get("pressure_diff_mean", 7.5) - 7.5) / 5, 0, 1),
        np.clip(latest.get("motor_current_volatility", 0) / 3, 0, 1),
    ]
    return float(np.mean(signals))


with st.sidebar:
    st.header("Data source")
    mode = st.radio(
        "Choose input",
        ["Healthy reference", "Degradation scenario", "Upload MetroPT-style CSV"],
        help="Use the two built-in scenarios to see how the dashboard responds to stable versus drifting sensor behaviour.",
    )
    st.caption("Built-in scenarios are synthetic and demonstrate the analysis workflow; they are not model-validation data.")
    uploaded = None
    if mode == "Upload MetroPT-style CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
    st.divider()
    st.subheader("Windowing")
    st.caption("A window groups nearby readings so condition is judged from a period of behaviour, not one isolated measurement.")
    window_minutes = st.slider("Window length", 10, 60, 60, 10)
    step_minutes = st.slider("Step", 5, 30, 30, 5, help="How far forward the analysis moves before creating the next window.")

if mode == "Upload MetroPT-style CSV":
    if uploaded is None:
        st.info("Upload a MetroPT-style CSV to begin, or select a built-in reference scenario.")
        st.stop()
    raw = pd.read_csv(uploaded)
else:
    raw = synthetic_frame(mode)

try:
    valid, report = validate_and_segment(raw)
except Exception as exc:
    st.error(f"Input validation failed: {exc}")
    st.stop()

windows = build_windows(
    valid,
    window_seconds=window_minutes * 60,
    step_seconds=step_minutes * 60,
    min_coverage=0.7,
)

st.markdown("### 1. Data preparation")
st.markdown(
    '<div class="section-hint">Before calculating condition, the pipeline checks sensor values and separates discontinuous periods so unrelated readings are not analyzed together.</div>',
    unsafe_allow_html=True,
)
q1, q2, q3, q4 = st.columns(4)
q1.metric("Valid rows", f"{report.valid_rows:,}", help="Readings that passed schema and broad physical sanity checks.")
q2.metric("Quarantined", f"{report.quarantined_rows:,}", help="Rows excluded because values were missing, duplicated or outside broad validation bounds.")
q3.metric("Continuous segments", report.segments, help="Separate continuous stretches of telemetry. Analysis windows never cross a large timestamp gap.")
q4.metric("Feature windows", f"{len(windows):,}", help="Time windows converted from raw readings into summarized condition features.")

if windows.empty:
    st.warning("No complete feature windows could be built from this input.")
    st.stop()

latest = windows.iloc[-1]
model_path = ARTIFACT_DIR / "model.joblib"
if model_path.exists():
    bundle = joblib.load(model_path)
    risk = float(predict_risk(bundle, windows.tail(1))[0])
    risk_label = "Trained model probability"
    risk_note = "Generated by artifacts/model.joblib from the reproducible training pipeline."
else:
    risk = heuristic_risk(latest)
    risk_label = "Demo health-risk indicator"
    risk_note = "Heuristic visualization only — not an ML prediction. Train the repository pipeline to generate a model artifact."

if risk >= 0.67:
    status = "HIGH ATTENTION"
elif risk >= 0.34:
    status = "WATCH"
else:
    status = "STABLE"

st.markdown("### 2. Current condition")
st.markdown(
    '<div class="section-hint">The most recent analysis window is reduced to a condition summary. The nearby measurements show which operating signals contributed context to that result.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2])
with left:
    st.markdown(
        f"""<div class="risk-card"><div class="small-note">{risk_label}</div>
        <h2 style="margin:.35rem 0">{status}</h2><div style="font-size:2.1rem;font-weight:700">{risk:.0%}</div>
        <p class="small-note">{risk_note}</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("#### Latest engineered window")
    st.caption("A snapshot of the latest window after raw telemetry has been summarized into maintenance-oriented features.")
    st.metric("Oil temperature mean", f"{latest['Oil_temperature_mean']:.1f} °C", help="Higher sustained temperature can indicate increasing thermal load or deteriorating operating conditions.")
    st.metric("Motor current mean", f"{latest['Motor_current_mean']:.2f} A", help="Current provides a view of electrical/mechanical load on the compressor motor.")
    st.metric("Compressor duty cycle", f"{latest['comp_duty_cycle']:.0%}", help="Share of the window during which the compressor was active.")
    st.metric("Pressure differential", f"{latest['pressure_diff_mean']:.2f} bar", help="Difference between TP3 and TP2; changes can reveal altered pressure behaviour across the system.")

with right:
    st.markdown("#### Sensor telemetry")
    st.caption("Inspect the original sensor signal over time. Different colors indicate separate continuous segments rather than one uninterrupted series.")
    sensor = st.selectbox(
        "Sensor",
        ANALOGUE_COLS,
        index=5,
        help="Choose any analogue sensor to see the raw behaviour that ultimately feeds the engineered windows.",
    )
    plot_df = valid[["timestamp", sensor, "segment_id"]].copy()
    fig = px.line(plot_df, x="timestamp", y=sensor, color="segment_id", labels={"timestamp": "Time"})
    fig.update_layout(height=430, legend_title_text="Segment", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### 3. Condition over time")
st.markdown(
    '<div class="section-hint">These are not raw readings. Each line tracks a feature calculated from successive windows, making gradual changes easier to see than in second-by-second telemetry.</div>',
    unsafe_allow_html=True,
)
trend_cols = ["Oil_temperature_mean", "Motor_current_mean", "DV_pressure_mean", "pressure_diff_mean"]
trend = windows[["window_end", *trend_cols]].melt("window_end", var_name="feature", value_name="value")
fig2 = px.line(trend, x="window_end", y="value", facet_row="feature", color="feature")
fig2.update_layout(height=650, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, use_container_width=True)

with st.expander("How to read this dashboard"):
    st.markdown(
        """
**Raw telemetry → validated segments → time windows → condition features → risk signal**

- **Raw telemetry** shows what the physical sensors measured.
- **Feature windows** summarize a period of behaviour so short spikes do not dominate the decision.
- **Condition trends** show whether those summaries are drifting over time.
- **Stable / Watch / High Attention** is the final interpretation of the latest window, not a diagnosis of a specific component failure.

For the hosted demo, the built-in scenarios are synthetic. The full repository can train a classifier against the published MetroPT-3 failure intervals using the real dataset.
"""
    )

with st.expander("Data quality and range diagnostics"):
    st.dataframe(range_diagnostics(valid), use_container_width=True, hide_index=True)
    st.caption("Percentiles help tune sanity bounds from observed operating data; they are not failure thresholds.")

with st.expander("How the training pipeline differs from this hosted demo"):
    st.markdown(
        """
- The full pipeline labels windows from the published MetroPT-3 failure intervals.
- Active-failure windows are excluded from the pre-failure target.
- Evaluation is chronological rather than random to reduce temporal leakage.
- A trained `artifacts/model.joblib` is produced only after running on the real dataset.
- This public dashboard can visualize uploaded data without presenting synthetic scenarios as model-performance evidence.
"""
    )
