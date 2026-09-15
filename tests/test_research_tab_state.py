"""State registration smoke test; actual tab selection is verified in the browser.

Streamlit AppTest's Tab representation does not simulate frontend tab selection.
Mutating session_state is not a substitute for that user interaction. The browser
script must still find visible holdout results after clicking Evaluate.
"""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]


def test_stateful_tabs_register_and_holdout_still_evaluates():
    at = st_testing.AppTest.from_file(str(ROOT / "pages/2_Walk_Forward.py"), default_timeout=60).run()
    at.radio(key="wf_source").set_value("Synthetic software demo").run()
    at.button(key="wf_load").click().run()
    at.button(key="wf_run").click().run()
    assert at.session_state["wf_active_tab"] == "Development"
    assert len(at.tabs) == 3
    at.button(key="wf_evaluate").click().run()
    assert not at.exception
    assert "wf_holdout" in at.session_state
    assert at.button(key="wf_evaluate").disabled
    assert len(at.metric) == 8
