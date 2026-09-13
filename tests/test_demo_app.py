from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "demo" / "app.py"


def test_overview_smoke_loads_committed_evidence():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Experiment results explorer"
    assert any(metric.label == "Predictive windows" for metric in app.metric)


def test_all_explorer_sections_render():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    for section in ["Compare models", "Event transfer", "Alert trade-offs"]:
        app.radio[0].set_value(section)
        app.run(timeout=20)
        assert not app.exception
        assert any(header.value == section for header in app.header)
