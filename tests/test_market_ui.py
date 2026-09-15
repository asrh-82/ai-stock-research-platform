"""Offline Streamlit tests: no live quotes or market network calls."""
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from Utils.market_data import demo_snapshot

APP = "from Utils.market_monte_carlo_ui import render_market_monte_carlo\nrender_market_monte_carlo()"


def synthetic_run():
    app = AppTest.from_string(APP).run()
    app.selectbox(key="market:source").set_value("Synthetic demo")
    app.button(key="market:run").click().run(timeout=30)
    assert not app.exception
    return app


def test_initial_render_does_not_fetch_market_or_fundamental_data():
    with patch("Utils.market_monte_carlo_ui._load_snapshot", side_effect=AssertionError("No implicit fetch")):
        app = AppTest.from_string(APP).run()
        assert not app.exception
        assert not app.error
        assert "market:result" not in app.session_state


def test_synthetic_run_reproducibility_comparison_calibration_export():
    app = synthetic_run()
    first = app.session_state["market:result"]["result"].paths.copy()
    assert any("SYNTHETIC" in item.value for item in app.warning)
    app.button(key="market:run").click().run(timeout=30)
    import numpy as np
    np.testing.assert_array_equal(first, app.session_state["market:result"]["result"].paths)
    app.button(key="market:compare").click().run(timeout=30)
    assert not app.exception
    assert len(app.session_state["market:result"]["comparison"]) == 3
    app.button(key="market:coverage").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["market:result"]["calibration"] is not None
    app.button(key="market:export").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["market:result"]["bundle"].startswith(b"PK")


def test_failed_live_request_clears_previous_result_without_demo_fallback():
    app = synthetic_run()
    app.selectbox(key="market:source").set_value("Live market data")
    with patch("Utils.market_monte_carlo_ui._load_snapshot", side_effect=ValueError("Provider unavailable")):
        app.button(key="market:run").click().run(timeout=30)
    assert not app.exception
    assert any("No simulation produced" in item.value for item in app.error)
    assert "market:result" not in app.session_state


def test_live_mode_uses_requested_ticker_and_not_fundamentals():
    snapshot = demo_snapshot()
    app = AppTest.from_string(APP).run()
    app.text_input(key="market:ticker").set_value("USO")
    with patch("Utils.market_monte_carlo_ui._load_snapshot", return_value=snapshot) as loader:
        app.button(key="market:run").click().run(timeout=30)
    assert not app.exception
    assert loader.call_args.args[0] == "USO"


def test_navigation_preserves_workspaces_and_separates_dcf():
    source = Path("app.py").read_text()
    assert 'pages/0_Stock_Selection.py' in source
    assert 'pages/1_Validation_Lab.py' in source
    assert 'pages/2_Walk_Forward.py' in source
    assert 'pages/4_Market_Monte_Carlo.py' in source
    page = Path("pages/3_Company_Tools.py").read_text()
    assert "render_dcf_uncertainty" in page
    assert "does not inherit manual DCF edits" in page
    assert "render_market_monte_carlo(active_ticker" in page
