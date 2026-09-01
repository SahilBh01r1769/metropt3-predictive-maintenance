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

st.set_page_config(page_title="MetroPT-3 Maintenance Dashboard", layout="wide")

st.markdown(
    """
<style>
:root {
  --page: #f1eee8;
  --panel: #e6e0d6;
  --panel-2: #ddd6cb;
  --ink: #29251f;
  --muted: #6c655c;
  --line: #bdb4a8;
  --rust: #82503a;
  --slate: #50616d;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--page);
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
}
[data-testid="stSidebar"] {
  background: #e4ded4;
  border-right: 1px solid var(--line);
}
.block-container { max-width: 1320px; padding-top: 2rem; }
h1, h2, h3 { color: var(--ink) !important; letter-spacing: -0.02em; }
.hero {
  padding: 1.25rem 0 1rem;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  margin-bottom: 1.25rem;
}
.hero h1 { margin: 0; font-size: 2.1rem; }
.hero p { margin: .45rem 0 0; color: var(--muted); max-width: 760px; }
.context-line {
  margin-top: .75rem;
  color: var(--muted);
  font-size: .8rem;
  letter-spacing: .02em;
}
.risk-card {
  padding: 1.1rem 1.2rem;
  border-radius: 3px;
  border: 1px solid var(--line);
  background: var(--panel);
}
.small-note { color: var(--muted); font-size: .82rem; }
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 11px;
  box-shadow: none;
}
.stButton > button,
[data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"] {
  border-radius: 3px !important;
  box-shadow: none !important;
}
.skeleton {
  height: 112px;
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 3px;
  margin: 8px 0 16px;
  animation: skeletonPulse 1.05s ease-in-out infinite;
}
@keyframes skeletonPulse { 0%,100% { opacity: .45; } 50% { opacity: .78; } }
</style>
<div class="hero">
  <h1>MetroPT-3 Predictive Maintenance</h1>
  <p>Explore air-compressor sensor health, segment-safe feature windows and maintenance risk signals.</p>
  <div class="context-line">SENSOR TELEMETRY / GAP-SAFE SEGMENTATION / FAILURE-HORIZON MODELING</div>
</div>
""",
    unsafe_allow_html=True,
)


def loading_placeholder():
    slot = st.empty()
    slot.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)
    return slot


def synthetic_frame(kind: str, seconds: int = 7200) -> pd.DataFrame:
    rng = np.random.default_rng(42 if kind == "Healthy reference" else 7)
    t = np.arange(seconds)
    ts = pd.date_range("2020-03-20 08:00:00", periods=seconds, freq="s")
    degrading = kind == "Degradation scenario"
    ramp = np.linspace(0, 1, seconds) if degrading else np.zeros(seconds)
    comp = ((t // 90) % 2).astype(int)
    return pd.DataFrame(
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
    mode = st.radio("Choose input", ["Healthy reference", "Degradation scenario", "Upload MetroPT-style CSV"])
    st.caption("Built-in scenarios are synthetic and exist only to demonstrate the dashboard workflow.")
    uploaded = None
    if mode == "Upload MetroPT-style CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
    st.divider()
    st.subheader("Windowing")
    window_minutes = st.slider("Window length", 10, 60, 60, 10)
    step_minutes = st.slider("Step", 5, 30, 30, 5)

if mode == "Upload MetroPT-style CSV":
    if uploaded is None:
        st.info("Upload a MetroPT-style CSV to begin, or select a built-in reference scenario.")
        st.stop()
    raw = pd.read_csv(uploaded)
else:
    raw = synthetic_frame(mode)

loading = loading_placeholder()
try:
    valid, report = validate_and_segment(raw)
    windows = build_windows(
        valid,
        window_seconds=window_minutes * 60,
        step_seconds=step_minutes * 60,
        min_coverage=0.7,
    )
except Exception as exc:
    loading.empty()
    st.error(f"Input validation failed: {exc}")
    st.stop()
loading.empty()

q1, q2, q3, q4 = st.columns(4)
q1.metric("Valid rows", f"{report.valid_rows:,}")
q2.metric("Quarantined", f"{report.quarantined_rows:,}")
q3.metric("Continuous segments", report.segments)
q4.metric("Feature windows", f"{len(windows):,}")

if windows.empty:
    st.warning("No complete feature windows could be built from this input.")
    st.stop()

latest = windows.iloc[-1]
model_path = ARTIFACT_DIR / "model.joblib"
model_loading = loading_placeholder()
if model_path.exists():
    bundle = joblib.load(model_path)
    risk = float(predict_risk(bundle, windows.tail(1))[0])
    risk_label = "Trained model probability"
    risk_note = "Generated by artifacts/model.joblib from the reproducible training pipeline."
else:
    risk = heuristic_risk(latest)
    risk_label = "Demo health-risk indicator"
    risk_note = "Heuristic visualization only, not an ML prediction. Train the repository pipeline to generate a model artifact."
model_loading.empty()

if risk >= 0.67:
    status = "HIGH ATTENTION"
elif risk >= 0.34:
    status = "WATCH"
else:
    status = "STABLE"

left, right = st.columns([1, 2])
with left:
    st.markdown(
        f"""<div class="risk-card"><div class="small-note">{risk_label}</div>
        <h2 style="margin:.35rem 0">{status}</h2><div style="font-size:2.1rem;font-weight:700">{risk:.0%}</div>
        <p class="small-note">{risk_note}</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("#### Latest engineered window")
    st.metric("Oil temperature mean", f"{latest['Oil_temperature_mean']:.1f} °C")
    st.metric("Motor current mean", f"{latest['Motor_current_mean']:.2f} A")
    st.metric("Compressor duty cycle", f"{latest['comp_duty_cycle']:.0%}")
    st.metric("Pressure differential", f"{latest['pressure_diff_mean']:.2f} bar")

with right:
    st.markdown("#### Sensor telemetry")
    sensor = st.selectbox("Sensor", ANALOGUE_COLS, index=5)
    plot_df = valid[["timestamp", sensor, "segment_id"]].copy()
    fig = px.line(
        plot_df,
        x="timestamp",
        y=sensor,
        line_group="segment_id",
        labels={"timestamp": "Time"},
        color_discrete_sequence=["#50616d"],
    )
    fig.update_traces(line={"color": "#50616d", "width": 1.4})
    fig.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Window-level condition trends")
trend_cols = ["Oil_temperature_mean", "Motor_current_mean", "DV_pressure_mean", "pressure_diff_mean"]
trend = windows[["window_end", *trend_cols]].melt("window_end", var_name="feature", value_name="value")
fig2 = px.line(
    trend,
    x="window_end",
    y="value",
    facet_row="feature",
    color="feature",
    color_discrete_sequence=["#82503a", "#50616d", "#5f6d59", "#756552"],
)
fig2.update_layout(height=650, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Data quality and range diagnostics"):
    st.dataframe(range_diagnostics(valid), use_container_width=True, hide_index=True)
    st.caption("Percentiles help tune sanity bounds from observed operating data; they are not failure thresholds.")

with st.expander("How the production pipeline differs from this demo"):
    st.markdown(
        """
- The repository training pipeline labels windows from the published MetroPT-3 failure intervals.
- Active-failure windows are excluded from the pre-failure target.
- Evaluation is chronological rather than random to reduce temporal leakage.
- A trained `artifacts/model.joblib` is produced only after running on the real dataset.
- This public dashboard can visualize uploaded data without presenting a synthetic demonstration as production-ready.
"""
    )
