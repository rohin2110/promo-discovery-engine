"""Headless smoke tests for the Streamlit dashboard (streamlit.testing.AppTest)."""
from pathlib import Path

import pytest

from config.settings import LLMSettings, ScoringConfig
from core.pipeline import PromoPipeline
from tests.conftest import make_dataset

APP = str(Path(__file__).resolve().parent.parent / "app.py")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Microdrama Promo Discovery Engine"
    assert [t.label for t in at.tabs] == ["Upload", "Run", "Results", "Quality and review", "Export", "Past runs"]


def test_weight_validation_in_sidebar():
    at = AppTest.from_file(APP, default_timeout=60).run()
    hook = next(n for n in at.sidebar.number_input if n.label == "Hook")
    hook.set_value(50).run()
    assert not at.exception
    assert any("sum to 100" in e.value for e in at.sidebar.error)


def test_results_cards_render(app_cfg):
    res = PromoPipeline(ScoringConfig(), LLMSettings(provider="heuristic"), app_cfg).run(make_dataset(2))
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["result"] = {
        "run_id": res.run_id, "output_dir": str(res.output_dir), "raw_dir": str(res.raw_dir), "backend": res.backend,
        "candidates": res.candidates, "episodes": res.episodes, "parse_report": res.parse_report,
        "qa_report": res.qa_report, "errors": res.errors, "scoring": ScoringConfig(),
    }
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Candidates discovered" in labels and "Eligible promos" in labels and "Hook" in labels
    assert any("heuristic baseline" in w.value for w in at.warning)
