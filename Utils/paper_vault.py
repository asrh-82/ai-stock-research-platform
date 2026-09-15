"""Browser-local persistence with explicit write acknowledgment and conflict checks."""
from pathlib import Path
from uuid import uuid4

import streamlit as st
from streamlit.components.v1 import declare_component

from Utils.paper_record import empty_ledger, validate_ledger

vault_component = declare_component("private_paper_vault", path=str(
    Path(__file__).resolve().parents[1] / "components" / "paper_vault"))


def sync_vault() -> tuple[dict, bool]:
    pending = st.session_state.get("ws_vault_command")
    result = vault_component(command=pending, key="ws_browser_record", default=None)
    if result is not None and isinstance(result, dict):
        if "document" in result:
            try:
                st.session_state["ws_ledger"] = validate_ledger(result["document"])
            except ValueError as exc:
                st.error(f"Saved record failed validation: {exc}")
                return empty_ledger(), False
        if pending and result.get("ack") == pending["id"]:
            st.session_state.pop("ws_vault_command", None)
            pending = None
        if result.get("error"):
            st.warning(result["error"])
        ready = bool(result.get("ready")) and pending is None
    else:
        ready = False
    document = st.session_state.get("ws_ledger", empty_ledger())
    if pending:
        st.caption("Saving your record. A save is confirmed only after browser acknowledgment.")
    return document, ready


def queue_save(document: dict) -> None:
    document = validate_ledger(document)
    if st.session_state.get("ws_vault_command"):
        raise ValueError("A save is already pending.")
    current = st.session_state.get("ws_ledger", empty_ledger())
    st.session_state["ws_vault_command"] = {
        "id": uuid4().hex, "base_revision": current["revision"], "document": document}
