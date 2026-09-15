"""Regression for holdout results disappearing behind a reset tab after rerun."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]


def test_holdout_reveal_preserves_active_tab():
    at = st_testing.AppTest.from_file(str(ROOT / "pages/2_Walk_Forward.py"), default_timeout=60).run()
    at.radio(key="wf_source").set_value("Synthetic software demo").run()
    at.button(key="wf_load").click().run()
    at.button(key="wf_run").click().run()
    at.session_state["wf_active_tab"] = "Reserved holdout"
    at.run()
    at.button(key="wf_evaluate").click().run()
    assert not at.exception
    assert at.session_state["wf_active_tab"] == "Reserved holdout"
    assert at.button(key="wf_evaluate").disabled
    assert len(at.metric) == 8
