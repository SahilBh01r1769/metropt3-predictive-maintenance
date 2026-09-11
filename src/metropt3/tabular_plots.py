from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_tabular_plots(output_dir: str | Path) -> list[Path]:
    """Save four diagnostic plots from tabular experiment evidence."""
    root = Path(output_dir)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(root / "model_comparison.csv")
    thresholds = pd.read_csv(root / "threshold_analysis.csv")
    events = pd.read_csv(root / "event_metrics.csv")
    importance = pd.read_csv(root / "feature_importance.csv")
    best = comparison.loc[comparison["development_average_precision"].idxmax()]
    identity = (
        comparison["model"].eq(best["model"])
        & comparison["feature_set"].eq(best["feature_set"])
        & comparison["horizon_hours"].eq(best["horizon_hours"])
    )

    created: list[Path] = []
    figure, axis = plt.subplots(figsize=(8, 5))
    for (model, feature_set), group in comparison.groupby(["model", "feature_set"]):
        group = group.sort_values("horizon_hours")
        axis.plot(
            group["horizon_hours"],
            group["development_average_precision"],
            marker="o",
            label=f"{model} · {feature_set}",
        )
    axis.set(xlabel="Prediction horizon (hours)", ylabel="Development average precision")
    axis.set_title("Horizon and representation comparison")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = plots / "horizon_performance.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path)

    chosen_thresholds = thresholds.loc[
        thresholds["model"].eq(best["model"])
        & thresholds["feature_set"].eq(best["feature_set"])
        & thresholds["horizon_hours"].eq(best["horizon_hours"])
    ]
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        chosen_thresholds["false_alerts_per_day"],
        chosen_thresholds["event_recall"],
        c=chosen_thresholds["threshold"],
        cmap="viridis",
        s=24,
    )
    axis.set(
        xlabel="False-alert episodes per operating day",
        ylabel="Development event recall",
        title="Threshold trade-off for development-selected model",
    )
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Threshold")
    figure.tight_layout()
    path = plots / "threshold_tradeoff.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path)

    chosen_events = events.loc[
        events["model"].eq(best["model"])
        & events["feature_set"].eq(best["feature_set"])
        & events["horizon_hours"].eq(best["horizon_hours"])
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        chosen_events["fold"],
        chosen_events["mean_first_warning_lead_hours"].fillna(0),
    )
    axis.set(ylabel="First-warning lead time (hours)", title="Lead time by failure event")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = plots / "event_lead_time.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path)

    top = importance.head(15).sort_values("importance")
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top["feature"], top["importance"])
    axis.set(xlabel=importance["importance_type"].iloc[0], title="Selected model feature importance")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = plots / "feature_importance.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path)
    return created
