from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ("xgboost", "tcn", "attention_tcn")
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "tcn": "TCN",
    "attention_tcn": "Attention-TCN",
}
EVENT_ORDER = ("may_failure", "june_failure", "july_holdout")
EVENT_LABELS = {
    "may_failure": "May",
    "june_failure": "June",
    "july_holdout": "July",
}
COLORS = {
    "xgboost": "#3d6b57",
    "tcn": "#b36546",
    "attention_tcn": "#50688a",
}


def load_evidence(evidence_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(evidence_root)
    metrics = pd.read_csv(root / "metrics.csv")
    eventwise = pd.read_csv(root / "eventwise_summary.csv")
    required_metrics = {
        "model",
        "horizon_hours",
        "seed",
        "threshold_policy",
        "average_precision",
        "ap_lift_over_prevalence",
        "false_alert_episodes_per_day",
        "held_out_event_detected",
        "development_fit_seconds",
        "final_fit_seconds",
    }
    required_eventwise = {
        "event",
        "horizon_hours",
        "model",
        "ap_lift_mean",
        "ap_lift_std",
    }
    missing = required_metrics.difference(metrics.columns)
    if missing:
        raise ValueError("Metrics evidence is missing: " + ", ".join(sorted(missing)))
    missing = required_eventwise.difference(eventwise.columns)
    if missing:
        raise ValueError(
            "Event-wise evidence is missing: " + ", ".join(sorted(missing))
        )
    return metrics, eventwise


def _finish(fig: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_horizon_performance(eventwise: pd.DataFrame, output: Path) -> Path:
    frame = eventwise.loc[eventwise["event"].eq("july_holdout")]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for model in MODEL_ORDER:
        selected = frame.loc[frame["model"].eq(model)].sort_values("horizon_hours")
        ax.errorbar(
            selected["horizon_hours"],
            selected["ap_lift_mean"],
            yerr=selected["ap_lift_std"].fillna(0),
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.axhline(1, color="#777777", linestyle="--", linewidth=1, label="Prevalence")
    ax.set(
        title="Held-out July ranking across prediction horizons",
        xlabel="Prediction horizon (hours)",
        ylabel="Average precision lift over prevalence",
        xticks=[1, 3, 6, 12],
    )
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    return _finish(fig, output)


def plot_cross_event_transfer(eventwise: pd.DataFrame, output: Path) -> Path:
    frame = eventwise.loc[eventwise["horizon_hours"].eq(1)].copy()
    pivot = frame.pivot(index="event", columns="model", values="ap_lift_mean").reindex(
        index=EVENT_ORDER, columns=MODEL_ORDER
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(EVENT_ORDER))
    width = 0.24
    for index, model in enumerate(MODEL_ORDER):
        ax.bar(
            x + (index - 1) * width,
            pivot[model],
            width,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
    ax.set(
        title="One-hour precursor ranking changes across failure episodes",
        ylabel="Average precision lift over prevalence",
        xticks=x,
        xticklabels=[EVENT_LABELS[event] for event in EVENT_ORDER],
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _finish(fig, output)


def plot_threshold_burden(metrics: pd.DataFrame, output: Path) -> Path:
    grouped = (
        metrics.groupby(
            ["model", "horizon_hours", "threshold_policy"], as_index=False
        )
        .agg(
            false_alerts_per_day=("false_alert_episodes_per_day", "mean"),
            event_detection_rate=("held_out_event_detected", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    styles = {"reference_0.5": "--", "development_selected": "-"}
    labels = {"reference_0.5": "0.5 reference", "development_selected": "Development-selected"}
    for model in MODEL_ORDER:
        for policy in styles:
            selected = grouped.loc[
                grouped["model"].eq(model)
                & grouped["threshold_policy"].eq(policy)
            ].sort_values("horizon_hours")
            axes[0].plot(
                selected["horizon_hours"],
                selected["false_alerts_per_day"],
                marker="o",
                linestyle=styles[policy],
                color=COLORS[model],
                alpha=0.9,
            )
            axes[1].plot(
                selected["horizon_hours"],
                selected["event_detection_rate"],
                marker="o",
                linestyle=styles[policy],
                color=COLORS[model],
                alpha=0.9,
                label=f"{MODEL_LABELS[model]} — {labels[policy]}",
            )
    axes[0].set(title="False-alert burden", ylabel="False-alert episodes/day")
    axes[1].set(title="Held-out event detection", ylabel="Fraction of seeds detecting event")
    for ax in axes:
        ax.set(xlabel="Prediction horizon (hours)", xticks=[1, 3, 6, 12])
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Threshold choice trades missed events for alert burden", y=1.02)
    return _finish(fig, output)


def plot_cost_vs_ranking(metrics: pd.DataFrame, output: Path) -> Path:
    frame = metrics.loc[metrics["threshold_policy"].eq("reference_0.5")].copy()
    summary = (
        frame.groupby("model", as_index=False)
        .agg(
            fit_seconds=("development_fit_seconds", "mean"),
            final_fit_seconds=("final_fit_seconds", "mean"),
            heldout_ap_lift=("ap_lift_over_prevalence", "mean"),
        )
    )
    summary["total_fit_seconds"] = summary["fit_seconds"] + summary["final_fit_seconds"]
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for row in summary.itertuples(index=False):
        ax.scatter(
            row.total_fit_seconds,
            row.heldout_ap_lift,
            s=90,
            color=COLORS[row.model],
            label=MODEL_LABELS[row.model],
        )
        ax.annotate(
            MODEL_LABELS[row.model],
            (row.total_fit_seconds, row.heldout_ap_lift),
            xytext=(6, 5),
            textcoords="offset points",
        )
    ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set(
        title="More fit cost did not improve held-out ranking",
        xlabel="Mean development + final fit time (seconds, log scale)",
        ylabel="Mean July AP lift across horizons and seeds",
    )
    ax.grid(alpha=0.2)
    return _finish(fig, output)


def load_event_trace(
    evidence_root: str | Path,
    *,
    model: str,
    horizon_hours: int,
    event: str,
) -> pd.DataFrame:
    root = Path(evidence_root)
    if event == "july_holdout":
        path = root / "holdout" / f"{model}_h{horizon_hours}.csv.gz"
        frame = pd.read_csv(path)
    else:
        path = root / "development" / f"{model}_h{horizon_hours}.csv.gz"
        frame = pd.read_csv(path)
        frame = frame.loc[frame["fold"].eq(event)]
    frame["window_end"] = pd.to_datetime(frame["window_end"])
    return frame


def event_timeline_data(frame: pd.DataFrame, *, max_lead_hours: float = 24) -> pd.DataFrame:
    required = {"hours_to_next_failure", "probability", "seed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Prediction trace is missing: " + ", ".join(sorted(missing)))
    selected = frame.loc[
        frame["hours_to_next_failure"].notna()
        & frame["hours_to_next_failure"].between(0, max_lead_hours)
    ].copy()
    if selected.empty:
        raise ValueError("No pre-failure windows are available for the requested event")
    return (
        selected.groupby("hours_to_next_failure", as_index=False)
        .agg(probability_mean=("probability", "mean"), probability_std=("probability", "std"))
        .sort_values("hours_to_next_failure", ascending=False)
    )


def plot_event_timelines(evidence_root: Path, output: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.6), sharex=True, sharey=True)
    for ax, event in zip(axes, EVENT_ORDER, strict=True):
        for model in MODEL_ORDER:
            trace = load_event_trace(
                evidence_root, model=model, horizon_hours=1, event=event
            )
            timeline = event_timeline_data(trace)
            ax.plot(
                -timeline["hours_to_next_failure"],
                timeline["probability_mean"],
                color=COLORS[model],
                linewidth=1.8,
                label=MODEL_LABELS[model],
            )
        ax.axvline(0, color="#222222", linestyle="--", linewidth=1)
        ax.set_title(f"{EVENT_LABELS[event]} failure")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_ylabel("Mean predicted probability across seeds")
    axes[-1].set_xlabel("Hours relative to failure start")
    fig.suptitle("One-hour model scores before each failure episode", y=1.01)
    return _finish(fig, output)


def plot_event_regime_heatmap(regime_path: Path, output: Path) -> Path:
    frame = pd.read_csv(regime_path).set_index("feature")
    columns = ["may_failure", "june_failure", "july_holdout"]
    values = frame[columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, max(5.2, len(frame) * 0.18)))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set(
        title="Pre-failure feature shifts versus clean normal windows",
        xlabel="Failure episode",
        ylabel="Engineered feature",
        xticks=np.arange(len(columns)),
        xticklabels=[EVENT_LABELS[column] for column in columns],
        yticks=np.arange(len(frame.index)),
        yticklabels=frame.index,
    )
    ax.tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=ax, label="Robust standardized difference (clipped at ±2)")
    return _finish(fig, output)


def generate_evidence_plots(evidence_root: str | Path, output_dir: str | Path) -> list[Path]:
    root = Path(evidence_root)
    output = Path(output_dir)
    metrics, eventwise = load_evidence(root)
    figures = [
        plot_horizon_performance(eventwise, output / "horizon_performance.png"),
        plot_cross_event_transfer(eventwise, output / "cross_event_transfer.png"),
        plot_threshold_burden(metrics, output / "threshold_alert_burden.png"),
        plot_cost_vs_ranking(metrics, output / "fit_cost_vs_ranking.png"),
        plot_event_timelines(root, output / "event_probability_timelines.png"),
    ]
    regime_path = root.parent / "event_regime" / "event_regime_summary.csv"
    if regime_path.exists():
        figures.append(plot_event_regime_heatmap(regime_path, output / "event_regime_shift.png"))
    return figures
