import json
from pathlib import Path


NOTEBOOK = Path("notebooks/03_tabular_experiment_colab.ipynb")


def test_tabular_notebook_is_self_contained_and_exports_required_evidence():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "investigation/temporal-validation" in source
    assert "pip', 'install'" in source
    assert "scripts/download_data.py" in source
    assert "scripts/run_tabular_experiment.py" in source
    for artifact in (
        "horizon_comparison.csv",
        "model_comparison.csv",
        "event_metrics.csv",
        "threshold_analysis.csv",
        "fold_results.csv",
        "feature_importance.csv",
        "best_tabular_model.joblib",
    ):
        assert artifact in source
    assert "metropt_tabular_evidence.zip" in source
    assert "files.download" in source
