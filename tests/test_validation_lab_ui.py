"""Standalone page smoke tests. Full navigation still requires browser verification."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
PAGE = Path(__file__).resolve().parents[1] / "pages" / "1_Validation_Lab.py"


def page():
    return st_testing.AppTest.from_file(str(PAGE), default_timeout=30).run()


def test_empty_state_is_network_independent():
    at = page()
    assert not at.exception
    assert at.title[0].value == "Validation Lab"
    assert not at.metric


def test_demo_remains_explicitly_unvalidated():
    at = page()
    at.toggle(key="validation_demo").set_value(True).run()
    assert not at.exception
    assert len(at.metric) == 4
    assert "NOT VALIDATED" in at.subheader[0].value
    assert any("NOT MARKET DATA" in x.value for x in at.warning)
    assert len(at.get("download_button")) == 3


def test_cash_costs_and_delay_recompute_without_exception():
    at = page()
    at.toggle(key="validation_demo").set_value(True).run()
    previous = at.metric[0].value
    at.number_input(key="validation_cash").set_value(5000.0)
    at.number_input(key="validation_slip").set_value(25.0)
    at.number_input(key="validation_delay").set_value(2)
    at.run()
    assert not at.exception
    assert previous != at.metric[0].value


def test_bracket_and_halt_inputs_render():
    at = page()
    at.toggle(key="validation_demo").set_value(True)
    at.number_input(key="validation_stop").set_value(8.0)
    at.number_input(key="validation_target").set_value(12.0)
    at.number_input(key="validation_halt").set_value(5.0)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 3
