"""Backup uploader reruns must retain the open restore panel."""
from pathlib import Path
from unittest.mock import patch

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from Utils.paper_record import empty_ledger


def test_backup_panel_registers_persistent_state():
    result = {"document": empty_ledger(), "ready": True, "ack": None, "error": None}
    root = Path(__file__).resolve().parents[1]
    with patch("Utils.paper_vault.vault_component", return_value=result):
        at = st_testing.AppTest.from_file(str(root / "app.py"), default_timeout=45).run()
        at.switch_page("pages/0_Stock_Selection.py").run()
        assert not at.exception
        assert at.session_state["ws_backups_open"] is False
        at.run()
        assert "ws_backups_open" in at.session_state
