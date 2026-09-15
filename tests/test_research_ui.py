"""UI state and multipage navigation tests; no live market-data calls."""
from pathlib import Path
from unittest.mock import patch

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]


def page():
    return st_testing.AppTest.from_file(str(ROOT / "pages/2_Walk_Forward.py"), default_timeout=60).run()


def demo():
    at = page()
    at.radio(key="wf_source").set_value("Synthetic software demo").run()
    at.button(key="wf_load").click().run()
    at.button(key="wf_run").click().run()
    return at


def test_empty_page_never_requests_data():
    with patch("Utils.research_data.download_daily", side_effect=AssertionError("Unexpected network request")):
        at = page()
    assert not at.exception
    assert not at.metric
    assert at.title[0].value == "Walk-forward Research"


def test_development_does_not_evaluate_holdout():
    with patch("Utils.research_protocol.evaluate_holdout", side_effect=AssertionError("Holdout was peeked")):
        at = demo()
    assert not at.exception
    assert len(at.metric) == 4
    assert "wf_holdout" not in at.session_state
    assert any("NOT VALIDATED" in x.value for x in at.subheader)
    assert any("not market data" in x.value for x in at.warning)


def test_holdout_requires_click_and_repeated_tuning_is_labeled():
    at = demo()
    at.button(key="wf_evaluate").click().run()
    assert not at.exception
    assert "wf_holdout" in at.session_state
    assert len(at.metric) == 8
    assert at.button(key="wf_evaluate").disabled
    at.button(key="wf_run").click().run()
    assert "wf_holdout" not in at.session_state
    assert any("already been opened" in x.value for x in at.warning)


def test_protocol_changes_invalidate_displayed_results():
    at = demo()
    at.number_input(key="wf_slip").set_value(25.0).run()
    assert not at.exception
    assert not at.metric
    assert any("Protocol changed" in x.value for x in at.warning)
    at.button(key="wf_run").click().run()
    assert not at.exception
    assert len(at.metric) == 4


def test_changing_source_requires_reload():
    at = demo()
    at.radio(key="wf_source").set_value("Historical daily data").run()
    assert not at.exception
    assert not at.metric
    assert any("Source settings changed" in x.value for x in at.warning)


def test_main_navigation_works_without_current_market_context():
    with patch("Utils.ui_sections.get_active_stock_context", return_value=None), \
         patch("Utils.ui_sections.render_sidebar", return_value=("SPY", "1y")):
        at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        at.switch_page("pages/2_Walk_Forward.py").run()
    assert not at.exception
    assert at.title[0].value == "Walk-forward Research"
