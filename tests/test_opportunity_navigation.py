"""The opportunity landing page must preserve the existing workspace routes."""
from pathlib import Path
from unittest.mock import patch

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from Utils.paper_record import empty_ledger


def test_opportunity_home_and_selection_round_trip_without_network():
    root = Path(__file__).resolve().parents[1]
    vault = {"document": empty_ledger(), "ready": True, "ack": None, "error": None}
    with (patch("Utils.paper_vault.vault_component", return_value=vault),
          patch("Utils.opportunity_ui.run_scan", side_effect=AssertionError("network on navigation")) as opportunity,
          patch("Utils.selection_ui.cached_universe", side_effect=AssertionError("network on navigation")) as selection):
        app = st_testing.AppTest.from_file(str(root / "app.py"), default_timeout=45).run()
        assert not app.exception
        assert app.title[0].value == "Opportunities"
        app.switch_page("pages/0_Stock_Selection.py").run()
        assert not app.exception
        assert app.title[0].value == "Stock Selection Workspace"
        app.switch_page("pages/5_Opportunities.py").run()
        assert not app.exception
        assert app.title[0].value == "Opportunities"
        opportunity.assert_not_called()
        selection.assert_not_called()
