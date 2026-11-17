"""Interactive filing research and the earlier reviewed Kahala case study."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from Utils.filings.pipeline import research_company
from Utils.filings.price_study import MARKET_ERRORS, run_price_study
from Utils.filings.sources import ResearchError
from Utils.research import load_evidence, render_report, resolve_research_subject


def render_research(context):
    active = context["ticker"]
    initial = active
    if active == "MTY.TO" and "kahala" in st.session_state.get("global_lookup_query", "").lower():
        initial = "Kahala Brands"
    st.subheader("Company filing research")
    with st.form(key=f"filing_request_{active}"):
        query = st.text_input("Company or ticker to research", value=initial, max_chars=160)
        include_study = st.checkbox("Include price patterns and a backtest")
        submitted = st.form_submit_button("Generate report", type="primary")
    st.caption(
        "Supports MTY and Kahala Brands, plus SEC issuers with standard USD financials in a 10-K or 10-Q."
    )
    selection = "filing_result"
    if submitted:
        st.session_state.pop(selection, None)
        try:
            with st.spinner("Reading filings and checking reported figures…"):
                report = research_company(query)
        except ResearchError as exc:
            st.error(str(exc))
        else:
            markdown = report.markdown()
            market_audit = issue = None
            if include_study:
                try:
                    with st.spinner("Testing the six fixed trading rules…"):
                        study = run_price_study(report)
                    markdown += "\n" + study.markdown
                    market_audit = study.audit
                except MARKET_ERRORS as exc:
                    issue = f"The filing report is available. The price study could not finish. {exc}"
            st.session_state[selection] = {
                "active": active,
                "ticker": report.ticker,
                "markdown": markdown,
                "audit": report.audit(),
                "market_audit": market_audit,
                "issue": issue,
            }

    result = st.session_state.get(selection)
    if result and result["active"] == active:
        symbol = result["ticker"]
        if result["issue"]:
            st.warning(result["issue"])
        st.download_button(
            "Download generated report", result["markdown"], f"{symbol}_report.md", "text/markdown"
        )
        st.download_button(
            "Download filing evidence", result["audit"], f"{symbol}_evidence.json", "application/json"
        )
        if result["market_audit"]:
            st.download_button(
                "Download price data and trade audit",
                result["market_audit"],
                f"{symbol}_study.zip",
                "application/zip",
            )
        st.markdown(result["markdown"])
    if resolve_research_subject(active):
        with st.expander("Reviewed Kahala case study · November 17, 2026"):
            _render_case_study()


def _render_case_study():
    evidence = load_evidence()
    st.caption(
        f"Evidence verified {evidence['verified_on']}. Parent financials and parent stock are identified throughout."
    )
    report = render_report(evidence, as_of=max(date.today().isoformat(), evidence["verified_on"]))
    st.download_button("Download sourced report", report, "Kahala_Brands_Report.md", "text/markdown")
    st.markdown(report)
    snapshot_path = Path(__file__).resolve().parents[1] / "Data" / "research" / "kahala_study.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        st.subheader("Frozen historical study")
        st.caption(
            f"Verified {snapshot['verified_on']}; {snapshot['holdout_start']} through {snapshot['holdout_end']}. This result is a dated snapshot, not recalculated from current prices."
        )
        rows = []
        for label, key in [
            ("Selected RSI rule", "holdout_metrics"),
            ("MTY buy and hold", "mty_buy_hold_metrics"),
            ("XIU benchmark", "benchmark_metrics"),
        ]:
            value = snapshot[key]
            rows.append(
                {
                    "Portfolio": label,
                    "Return": f"{value['total_return']:.2%}",
                    "Maximum drawdown": f"{value['maximum_drawdown']:.2%}",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.warning(
            "Only three completed final-period trades. Earlier fixed-rule RSI performance was negative. No repeatable trading edge is established."
        )
        st.download_button(
            "Download frozen study summary",
            snapshot_path.read_text(),
            "Kahala_MTY_Study.json",
            "application/json",
        )
