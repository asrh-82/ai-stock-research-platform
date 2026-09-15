"""One connected screen -> research -> paper-record workflow."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from Utils.paper_record import (append_event, empty_ledger, evaluate_cohort, freeze_selection,
                               parse_backup, records)
from Utils.paper_vault import queue_save, sync_vault
from Utils.selection import (BENCHMARK, DEFAULT_UNIVERSE, FORMULA, NY, ScreenRules,
                             canonical, correlation_context, demo_universe, digest,
                             parse_universe, rank_changes, scan, utc_now)
from Utils.selection_data import load_universe


@st.cache_data(ttl=900, show_spinner=False)
def cached_universe(text: str, day: str):
    return load_universe(text)


@st.cache_data(ttl=900, show_spinner=False)
def company_context(ticker: str):
    from Utils.ui_sections import get_active_stock_context
    return get_active_stock_context(ticker, "1y")


def _save(document: dict) -> None:
    queue_save(document)
    st.rerun()


def _open_research() -> None:
    st.session_state["ws_research_symbol"] = st.session_state.get("ws_selected")
    st.session_state["ws_view"] = "Research"


def _screen_table(snapshot: dict, previous: dict | None) -> pd.DataFrame:
    changes = rank_changes(snapshot, previous)
    return pd.DataFrame([{"Rank": r["rank"], "Ticker": r["symbol"],
                          "Change": changes.get(r["symbol"]),
                          "Momentum": r["momentum"], "Percentile": r["percentile"],
                          "Reported close": r["reported_close"], "Volatility": r["volatility"],
                          "Max drawdown": r["drawdown"], "Avg dollar volume": r["dollar_volume"]}
                         for r in snapshot["ranked"]])


def _show_screen(document: dict, ready: bool, snapshot: dict | None) -> None:
    st.subheader("Find the names worth researching")
    left, right = st.columns([3, 1])
    with left:
        source = st.radio("Data source", ["Market data", "Synthetic demo"], horizontal=True, key="ws_source")
        universe = st.text_area("Research universe", DEFAULT_UNIVERSE, key="ws_universe", height=75,
                               help="2–25 tickers. This is your declared current universe, not historical index membership.")
    with right:
        st.caption("BASELINE")
        st.write("12–1 momentum")
        st.caption("252/21-session approximation. The latest 21 sessions are excluded from the rank signal.")
    with st.expander("Eligibility and method"):
        price = st.number_input("Minimum reported price", min_value=0., max_value=100000., value=5., key="ws_min_price")
        liquidity = st.number_input("Minimum 20-session average dollar volume", min_value=0., max_value=1e12,
                                    value=2_000_000., step=100000., key="ws_liquidity")
        st.code(FORMULA)
        st.caption("Requires 253 sessions aligned to SPY and no zero-volume session in the latest 20. "
                   "Ties use ticker order; tied signals share a percentile. Percentile is not a probability. "
                   "Instrument classification and exchange-calendar completeness are not independently verified.")
        st.caption("The starter names are an illustrative current universe, not recommendations. "
                   "The formula is a session approximation, not a replication of the Fama–French momentum portfolios.")
    fingerprint = digest({"source": source, "universe": universe, "price": price, "liquidity": liquidity})
    if snapshot and st.session_state.get("ws_scan_controls") != fingerprint:
        st.info("Controls differ from the saved scan. The results below retain their saved settings; rescan to change them.")
    if st.button("Scan universe", type="primary", key="ws_scan", disabled=not ready):
        try:
            with st.spinner("Checking history, eligibility, and common-session coverage…"):
                if source == "Synthetic demo":
                    batch, future = demo_universe()
                    st.session_state["ws_demo_future"] = future
                else:
                    parse_universe(universe)
                    batch = cached_universe(universe, str(utc_now().astimezone(NY).date()))
                result = scan(batch, ScreenRules(price, liquidity))
                st.session_state["ws_batch"] = batch
                st.session_state["ws_batch_scan_id"] = result["id"]
                st.session_state["ws_scan_controls"] = fingerprint
                st.session_state["ws_selected"] = result["ranked"][0]["symbol"] if result["ranked"] else None
                _save(append_event(document, "scan", result))
        except ValueError as exc:
            st.error(str(exc))
    if not snapshot:
        st.info("Run a scan to populate the queue. Nothing is fetched until you request it.")
        return
    st.divider()
    if snapshot["synthetic"]:
        st.warning("SYNTHETIC DEMO. Invented prices and outcomes; no market performance evidence.")
    a, b, c = st.columns(3)
    a.metric("Eligible", len(snapshot["ranked"]))
    b.metric("Excluded", len(snapshot["excluded"]))
    c.metric("Price session", snapshot["asof"])
    older = [s for s in records(document, "scan") if s["id"] != snapshot["id"]]
    previous = older[-1] if older else None
    table = _screen_table(snapshot, previous)
    if not table.empty:
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Momentum": st.column_config.NumberColumn(format="percent"),
                                    "Volatility": st.column_config.NumberColumn(format="percent"),
                                    "Max drawdown": st.column_config.NumberColumn(format="percent"),
                                    "Reported close": st.column_config.NumberColumn(format="$%.2f"),
                                    "Avg dollar volume": st.column_config.NumberColumn(format="$%.0f"),
                                    "Percentile": st.column_config.NumberColumn(format="%.1f")})
        symbols = [r["symbol"] for r in snapshot["ranked"]]
        selected = st.session_state.get("ws_selected")
        if selected not in symbols:
            st.session_state["ws_selected"] = symbols[0]
        st.selectbox("Research candidate", symbols, key="ws_selected")
        st.button("Open candidate research", key="ws_open_research", on_click=_open_research)
    if snapshot["excluded"]:
        with st.expander("Exclusions and provider failures", expanded=not snapshot["ranked"]):
            st.dataframe(pd.DataFrame(snapshot["excluded"]), hide_index=True, width="stretch")
    st.caption(f"Saved {snapshot['created_at']} · scan {snapshot['id'][:12]} · {snapshot['status']}. "
               "Price history ends before today's New York date. Current-universe selection bias remains.")
    st.download_button("Export frozen scan", canonical(snapshot), "selection_scan.json", "application/json")


def _research(snapshot: dict | None) -> None:
    if not snapshot or not snapshot["ranked"]:
        st.info("Run a scan with eligible names first.")
        return
    symbols = [r["symbol"] for r in snapshot["ranked"]]
    ticker = st.session_state.get("ws_selected")
    if ticker not in symbols:
        ticker = symbols[0]
    ticker = st.selectbox("Company", symbols, index=symbols.index(ticker), key="ws_research_symbol")
    st.session_state["ws_selected"] = ticker
    row = next(r for r in snapshot["ranked"] if r["symbol"] == ticker)
    st.subheader(f"{ticker} · research case")
    st.caption(f"Frozen rank #{row['rank']} of {len(symbols)} · prices through {snapshot['asof']}")
    a, b, c = st.columns(3)
    a.metric("12–1 momentum", f"{row['momentum']:.1%}")
    b.metric("Recent 21-session return", f"{row['recent_return']:.1%}")
    c.metric("63-session annualized volatility", f"{row['volatility']:.1%}")
    st.write(f"**Why it appears here:** adjusted price changed {row['momentum']:+.1%} between "
             f"{row['signal_start']} and {row['signal_end']}. That places it at the "
             f"{row['percentile']:.1f} percentile of this scan's eligible universe.")
    st.write(f"**What challenges the ranking:** its latest 21-session return is {row['recent_return']:+.1%}, "
             f"and its historical drawdown in the signal window reached {row['drawdown']:.1%}. "
             "Neither the percentile nor this explanation estimates future returns.")
    batch = st.session_state.get("ws_batch")
    if batch is not None and st.session_state.get("ws_batch_scan_id") == snapshot["id"]:
        prices = batch.histories[ticker].prices.Close
        benchmark = batch.histories[BENCHMARK].prices.Close
        frame = pd.concat([prices.rename(ticker), benchmark.rename("Reference")], axis=1).dropna().tail(253)
        st.line_chart(frame / frame.iloc[0] * 100, height=290)
        st.caption("Adjusted price histories rebased to 100, not a strategy backtest.")
        peers = [r["symbol"] for r in snapshot["ranked"][:5]]
        if ticker not in peers:
            peers = [ticker, *peers[:4]]
        with st.expander("Correlation with leading candidates"):
            st.dataframe(correlation_context(batch, peers).round(2), width="stretch")
            st.caption("Latest 63 return observations; at least 40 paired observations. "
                       "Historical correlation is not a guaranteed future relationship.")
    if snapshot["synthetic"]:
        st.warning("Demo company names and price data are fictional. Live company/valuation requests are disabled here.")
        return
    if st.button("Load company fundamentals", key="ws_load_company"):
        with st.spinner("Loading current company context…"):
            try:
                context = company_context(ticker)
                if context is None:
                    raise ValueError("Company data unavailable. The price ranking is still preserved.")
                st.session_state["ws_company_context"] = context
                st.session_state["ws_company_loaded"] = utc_now().isoformat()
            except Exception as exc:
                st.error(f"Company lookup failed: {type(exc).__name__}: {str(exc)[:180]}")
    context = st.session_state.get("ws_company_context")
    if not context or context["ticker"] != ticker:
        st.caption("Fundamentals, sector, and earnings event: not loaded. No estimated date is invented.")
        return
    st.caption("Current company context, not historical rank inputs. Loaded " + st.session_state["ws_company_loaded"])
    info = context["info"]
    st.write(context["company_name"])
    metrics = ["sector", "industry", "profitMargins", "returnOnEquity", "debtToEquity", "freeCashflow"]
    st.dataframe(pd.DataFrame([{"Field": k, "Value": str(info[k]) if info.get(k) is not None else "Unavailable"}
                              for k in metrics]), hide_index=True, width="stretch")
    st.caption("Company summary provider data; filing dates and source financial periods need independent review. "
               "Earnings event: unverified. These fields do not alter the momentum ranking.")
    section = st.radio("Valuation context", ["Overview", "DCF scenarios", "Assumption uncertainty"],
                       horizontal=True, key=f"ws_valuation_{ticker}")
    if section == "DCF scenarios":
        from Utils.dcf_ui import render_dcf
        render_dcf(context)
    elif section == "Assumption uncertainty":
        from Utils.monte_carlo_ui import render_monte_carlo
        st.info("This distribution varies DCF assumptions. It is not a probability of a profitable trade.")
        render_monte_carlo(context)
    else:
        st.write("The momentum rank determines the research queue. DCF tests what business assumptions justify "
                 "the price; its sensitivity analysis challenges those assumptions. Neither is an extra buy vote.")


def _paper(document: dict, ready: bool, snapshot: dict | None) -> None:
    st.subheader("Model versus your selections")
    st.caption("Independent 21-session paper cohorts. This is not an automatically rebalanced brokerage account. "
               "Each cohort starts from its own stated virtual capital; do not add overlapping cohort returns.")
    decisions = records(document, "decision")
    if snapshot and len(snapshot["ranked"]) >= 2:
        names = [r["symbol"] for r in snapshot["ranked"]]
        slots = min(5, len(names))
        st.write("**Fixed model picks:** " + ", ".join(names[:slots]))
        st.caption(f"Each selected name gets 1/{slots} of initial virtual equity. "
                   "Unused user slots remain in cash. Costs: 5 bps commission + 10 bps adverse slippage per side.")
        chosen = st.multiselect("Your paper selections", names, default=names[:slots], max_selections=slots,
                                key="ws_user_picks_" + snapshot["id"][:16])
        capital = st.number_input("Virtual cohort capital", min_value=1., max_value=1_000_000.,
                                  value=10_000., step=100., key="ws_capital")
        existing = any(d["scan_id"] == snapshot["id"] for d in decisions)
        if st.button("Freeze model and my selections", type="primary", key="ws_freeze", disabled=existing or not ready):
            try:
                decision = freeze_selection(snapshot, chosen, capital)
                _save(append_event(document, "decision", decision))
            except ValueError as exc:
                st.error(str(exc))
        if existing:
            st.caption("This scan already has a frozen selection. It cannot be edited. A new experiment needs a new saved scan.")
    if not decisions:
        st.info("Freeze a scan to start its prospective paper record. No historical trade is backfilled into a live record.")
        return
    decision_id = st.selectbox("Saved paper cohort", [d["id"] for d in decisions],
                               format_func=lambda id_: next(
                                   ("DEMO · " if d["synthetic"] else "PAPER · ") + d["scan_asof"] + " · " + id_[:8]
                                   for d in decisions if d["id"] == id_), key="ws_cohort")
    decision = next(d for d in decisions if d["id"] == decision_id)
    if decision["synthetic"]:
        st.warning("SYNTHETIC PAPER EXAMPLE. Its historical dates and outcomes are simulated, not a forward track record.")
    st.write(f"Recorded **{decision['registered_at']}**. First eligible session is on or after "
             f"**{decision['earliest_entry_date']}**. Model and user choices are frozen together.")
    observations = [o for o in records(document, "observation") if o["decision_id"] == decision_id]
    complete = any(o["status"] == "COMPLETE" for o in observations)
    if st.button("Update paper outcomes", key="ws_update", disabled=not ready or complete):
        try:
            with st.spinner("Checking all frozen names against the same outcome sessions…"):
                if decision["synthetic"]:
                    _, batch = demo_universe()
                else:
                    start = str((pd.Timestamp(decision["decision_date"]) - pd.Timedelta(days=20)).date())
                    batch = load_universe(",".join(decision["eligible"]), start=start)
                outcome = evaluate_cohort(decision, batch)
                _save(append_event(document, "observation", outcome))
        except ValueError as exc:
            st.error(str(exc))
    if complete:
        st.caption("Completed result is frozen. New vendor downloads cannot silently rewrite it.")
    if observations:
        o = observations[-1]
        st.write(f"**{o['status']}** · {o['sessions']}/21 outcome sessions")
        if o["books"]:
            columns = st.columns(4)
            for col, name in zip(columns, ["Model", "User", "Equal universe", "Reference"]):
                col.metric(name, f"{o['books'][name]['net_return']:+.2%}")
            st.caption("Reference: SPY. User cash exposure can differ; the comparison is not a risk-adjusted alpha estimate.")
            frame = pd.DataFrame(o["curve"]).set_index("date")
            st.line_chart(frame, height=290)
            table = pd.DataFrame([{"Book": name, **{k: v for k, v in m.items() if k != "weights"}}
                                  for name, m in o["books"].items()])
            st.dataframe(table, hide_index=True, width="stretch")
            if o["rank_ic"] is not None:
                st.write(f"**21-session rank correlation:** {o['rank_ic']:+.3f}")
                st.caption("Spearman correlation between frozen momentum and subsequent open-to-close returns. "
                           "One small, correlated cross-section does not establish statistical significance.")
            st.download_button("Export cohort outcome", canonical(o), "paper_cohort_outcome.json", "application/json")
    with st.expander("Frozen choices and audit details"):
        st.json(decision)
    if len(decisions) > 1:
        st.caption(f"{len(decisions)} experiments retained. Repeated experiments and overlapping cohorts are not independent evidence.")


def render_workspace() -> None:
    st.title("Stock Selection Workspace")
    st.caption("Rank systematically. Research the disagreement. Keep the evidence.")
    st.info("Research and paper simulation only. No broker connection or real orders. Rankings are not return forecasts.")
    document, ready = sync_vault()
    if not ready:
        st.caption("Waiting for private browser storage. Scans and records unlock after storage is available.")
    st.session_state.setdefault("ws_view", "Screen")
    view = st.radio("Workspace", ["Screen", "Research", "Paper & results"], horizontal=True,
                    key="ws_view", label_visibility="collapsed")
    scans = records(document, "scan")
    snapshot = scans[-1] if scans else None
    if view == "Screen":
        _show_screen(document, ready, snapshot)
    elif view == "Research":
        _research(snapshot)
    else:
        _paper(document, ready, snapshot)
    with st.expander("Record storage and backups", key="ws_backups_open", on_change="rerun"):
        st.caption("Saved in this browser on this deployment origin, not cloud-synced. Clearing site data, "
                   "private browsing, or opening a different preview can lose access. Export backups. "
                   "The app keeps an append-only hash chain, but users can edit/re-hash browser data; "
                   "this is not an independently attested track record. Nothing is committed to GitHub.")
        st.download_button("Export private record", canonical(document), "paper_research_record.json", "application/json")
        backup = st.file_uploader("Restore JSON backup into an empty browser record", type=["json"], key="ws_restore_file")
        if st.button("Restore backup", key="ws_restore", disabled=not ready or document["revision"] != 0 or backup is None):
            try:
                _save(parse_backup(backup.getvalue()))
            except ValueError as exc:
                st.error(str(exc))
        st.caption(f"{document['revision']}/200 events. Maximum backup size: 2 MB. Existing records are never overwritten by import.")
