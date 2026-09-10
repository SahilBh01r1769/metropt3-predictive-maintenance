import json
from pathlib import Path


NOTEBOOK = Path("notebooks/02_temporal_experiment_colab.ipynb")


def test_experiment_notebook_is_valid_and_preserves_methodology():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "assert torch.cuda.is_available()" in source
    assert "Dataset differs from audited source" in source
    assert "Expected 7,846 predictive windows" in source
    assert "run_experiment(dataset, output_root, config=config)" in source
    assert "validate_and_summarize_experiment" in source
    assert "report['complete_cells'] == 36" in source
    assert "metropt_temporal_evidence.zip" in source
    assert "files.download(str(bundle))" in source
    assert "final_holdout" not in source.lower() or "held-out probabilities" in source
