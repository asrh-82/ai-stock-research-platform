"""Daily research workflow. Network calls and holdout evaluation require explicit clicks."""
import json
from dataclasses import asdict
from datetime import date
from hashlib import sha256

import pandas as pd
import streamlit as st

from Utils.research_data import demo_snapshot, download_daily, parse_daily_csv
from Utils.research_protocol import DEFAULT_RULES, Execution, Protocol, Rule, develop, evaluate_holdout

st.set_page_config(page_title="Walk-forward Research", layout="wide")
st.title("Walk-forward Research")
st.caption("Declare the rules. Select on earlier data. Evaluate later data separately.")
st.info("PAPER RESEARCH ONLY. No broker connection, live orders, leverage, or automatic capital deployment.")

source = st.radio("Data source", ["Historical daily data", "Upload adjusted daily CSV", "Synthetic software demo"],
                  horizontal=True, key="wf_source")
payload = None
confirmed = False
if source == "Historical daily data":
    c1, c2, c3 = st.columns(3)
    symbol = c1.text_input("Research ticker", "SPY", key="wf_symbol").strip().upper()
    start_date = c2.date_input("Start date", date(2010, 1, 1), key="wf_start")
    end_date = c3.date_input("End date (exclusive)", date(2026, 1, 1), key="wf_end")
    request_key = json.dumps([source, symbol, str(start_date), str(end_date)])
    st.caption("SPY is an example dataset, not a recommendation. Yahoo access may fail or be rate-limited. "
               "Data is fetched only after Load data; no paid subscription is created.")
elif source == "Upload adjusted daily CSV":
    upload = st.file_uploader("Date, Open, High, Low, Close, Volume", type=["csv"], key="wf_csv")
    payload = upload.getvalue() if upload is not None else None
    confirmed = st.checkbox("I confirm all OHLC columns use a consistent adjusted-price basis.", key="wf_adjusted")
    request_key = json.dumps([source, sha256(payload).hexdigest() if payload else None, confirmed])
    st.caption("Use YYYY-MM-DD session dates, one asset, at most 8,000 daily rows / 10 MiB, and no Signal column. "
               "This is intentionally different from the intraday Validation Lab CSV format.")
else:
    request_key = "synthetic-daily-v1"
    st.warning("SYNTHETIC SOFTWARE DEMO. These prices are not market data and prove no trading edge.")

with st.expander("Predeclared research protocol"):
    c1, c2, c3, c4 = st.columns(4)
    train = c1.number_input("Initial training sessions", 300, 3000, 504, key="wf_train")
    test = c2.number_input("Walk-forward test sessions", 20, 504, 126, key="wf_test")
    holdout = c3.number_input("Reserved final sessions", 20, 1000, 252, key="wf_reserved")
    gap = c4.number_input("Gap sessions", 1, 20, 1, key="wf_gap")
    c1, c2, c3, c4 = st.columns(4)
    initial = c1.number_input("Virtual starting equity", 100.0, 1_000_000.0, 10_000.0, key="wf_cash")
    allocation = c2.number_input("Research exposure on entry (%)", 1.0, 100.0, 100.0, key="wf_allocation")
    commission = c3.number_input("Commission per side (bps)", 0.0, 1000.0, 5.0, key="wf_fee")
    slip = c4.number_input("Slippage per side (bps)", 0.0, 250.0, 10.0, key="wf_slip")
    st.caption("Generic normalized research assumptions, not a personal allocation plan. "
               "The seven candidate rules stay fixed; winner selection uses net training Sharpe with a zero risk-free rate. "
               "All candidates share the same warmup and evaluation dates. Cash is eligible.")

config = Protocol(train=int(train), test=int(test), holdout=int(holdout), gap=int(gap),
                  execution=Execution(initial, allocation / 100, commission, slip))
config_key = json.dumps(asdict(config), sort_keys=True)
if st.button("Load data", key="wf_load"):
    for state_key in ("wf_snapshot", "wf_loaded_key", "wf_development", "wf_holdout"):
        st.session_state.pop(state_key, None)
    try:
        if source == "Historical daily data":
            snapshot = download_daily(symbol, str(start_date), str(end_date))
        elif source == "Upload adjusted daily CSV":
            snapshot = parse_daily_csv(payload, confirmed)
        else:
            snapshot = demo_snapshot()
        st.session_state["wf_snapshot"] = snapshot
        st.session_state["wf_loaded_key"] = request_key
        st.session_state.pop("wf_development", None)
        st.session_state.pop("wf_holdout", None)
    except (ValueError, ArithmeticError) as exc:
        st.error(str(exc))
        st.stop()

if "wf_snapshot" not in st.session_state:
    st.write("Load a daily snapshot to begin. No historical results have been calculated.")
    st.stop()
if st.session_state.get("wf_loaded_key") != request_key:
    st.warning("Source settings changed. Load a new snapshot before viewing results.")
    st.stop()
snapshot = st.session_state["wf_snapshot"]
manifest = snapshot.manifest()
st.caption(f"Loaded {manifest['rows']:,} sessions: {manifest['first_session']} to {manifest['last_session']}. "
           f"Source: {manifest['source']}.")
if manifest["floating_point_ohlc_discrepancies"]:
    st.caption(f"OHLC floating-point discrepancies: {manifest['floating_point_ohlc_discrepancies']}. "
               "Accepted only within eight double-precision epsilons; input prices are unchanged.")
if manifest["gaps_over_seven_calendar_days"] or manifest["zero_volume_sessions"]:
    st.warning("This snapshot has long date gaps or zero-volume sessions. Review the data audit before interpreting results.")
if st.button("Run development and freeze plan", key="wf_run", type="primary"):
    try:
        st.session_state["wf_development"] = develop(snapshot, config)
        st.session_state["wf_developed_config"] = config_key
        st.session_state.pop("wf_holdout", None)
        st.session_state["wf_attempts"] = st.session_state.get("wf_attempts", 0) + 1
    except (ValueError, ArithmeticError) as exc:
        st.error(str(exc))
        st.stop()
if "wf_development" not in st.session_state:
    st.write("The final holdout is reserved automatically. Run development to create a frozen experiment plan.")
    st.stop()
if st.session_state.get("wf_developed_config") != config_key:
    st.warning("Protocol changed. Run a new, separately labeled experiment; the old results are not shown under new settings.")
    st.stop()

result = st.session_state["wf_development"]
plan = result.plan
seen = st.session_state.get("wf_seen_holdout_datasets", set())
if manifest["data_sha256"] in seen:
    st.warning("This dataset's holdout has already been opened in this session. Further tuning is exploratory, not a fresh blind test.")
st.subheader("Strategy status: NOT VALIDATED")
st.caption(f"Experiment runs in this browser session: {st.session_state.get('wf_attempts', 1)}. "
           "This counter is not persistent, shared, or tamper-proof. A hidden panel does not prove an untouched holdout.")


def show_comparison(simulation, passive, prefix):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{prefix} strategy return", f"{simulation.metrics['net_return']:.2%}")
    c2.metric("Same-asset passive return", f"{passive.metrics['net_return']:.2%}")
    c3.metric("Maximum drawdown", f"{simulation.metrics['maximum_drawdown']:.2%}")
    c4.metric("Completed round trips", str(simulation.metrics["completed_trades"]))
    st.line_chart(pd.DataFrame({"Strategy": simulation.curve.Equity,
                               "Same-asset passive": passive.curve.Equity,
                               "Cash, no interest": config.execution.initial_cash}))
    st.caption("Same exposure-on-entry and cost assumptions; no intra-position rebalancing. "
               "The passive comparator is the selected asset, not a market-alpha estimate. "
               "Open positions are marked, not silently liquidated. Fractional adjusted units are not executable historical shares.")


dev_tab, holdout_tab, audit_tab = st.tabs(
    ["Development", "Reserved holdout", "Rules and data audit"],
    key="wf_active_tab", on_change="rerun",
)
with dev_tab:
    show_comparison(result.simulation, result.passive, "Walk-forward")
    st.dataframe(result.folds, hide_index=True, width="stretch")
    st.caption("Selections are refit on expanding training prefixes. Test windows do not overlap. "
               "The final development fold can be shorter. Portfolio cash and holdings carry across folds without a free reset.")
    with st.expander("All candidate training scores, including losers"):
        st.dataframe(result.training_scores, hide_index=True, width="stretch")
    st.download_button("Export development equity", result.simulation.curve.to_csv(),
                       "walk_forward_equity.csv", "text/csv", key="wf_export_dev")
    st.download_button("Export development orders", result.simulation.orders.to_csv(index=False),
                       "walk_forward_orders.csv", "text/csv", key="wf_export_orders")

with holdout_tab:
    chosen = Rule(**plan["selected_rule"]).label
    st.write(f"Frozen rule: **{chosen}**. Reserved period: **{plan['holdout_first_session']} to {plan['holdout_last_session']}**.")
    st.caption("The rule is selected using development data only. Holdout starts a separate virtual account; "
               "it is not spliced onto the development equity curve. Reading these results consumes the holdout for future tuning.")
    st.download_button("Export frozen plan before evaluation", json.dumps(plan, indent=2, allow_nan=False),
                       "frozen_research_plan.json", "application/json", key="wf_plan")
    if st.button("Evaluate frozen holdout", key="wf_evaluate", disabled="wf_holdout" in st.session_state):
        try:
            st.session_state["wf_holdout"] = evaluate_holdout(snapshot, plan)
            st.session_state["wf_seen_holdout_datasets"] = seen | {manifest["data_sha256"]}
            st.rerun()
        except (ValueError, ArithmeticError) as exc:
            st.error(str(exc))
            st.stop()
    if "wf_holdout" in st.session_state:
        held, passive, stress = st.session_state["wf_holdout"]
        st.write("Holdout evaluated after freeze. This does not certify a trading edge.")
        show_comparison(held, passive, "Holdout")
        st.dataframe(stress, hide_index=True, width="stretch")
        st.caption("1x / 2x / 4x slippage, fixed signals and selections, unchanged commissions. No retuning to improve stressed results.")
        st.download_button("Export holdout equity", held.curve.to_csv(), "holdout_equity.csv", "text/csv", key="wf_held_equity")
        st.download_button("Export holdout orders", held.orders.to_csv(index=False), "holdout_orders.csv", "text/csv", key="wf_held_orders")
    else:
        st.write("Holdout metrics have not been calculated. Export the plan before opening them.")

with audit_tab:
    st.dataframe(pd.DataFrame([{"rule": r.label, **asdict(r)} for r in DEFAULT_RULES]), hide_index=True, width="stretch")
    st.write("SMA: long when the trailing fast average exceeds the slow average. Momentum: long when the trailing price change is positive. "
             "RSI: enter at or below 30, exit at or above 55, otherwise retain the signal. Cash: always uninvested.")
    st.caption("RSI uses exponential gain/loss averages with alpha=1/window, adjust=False, and a full-window warmup. "
               "All decisions use completed sessions and become eligible at the next session's open. "
               "These are independent daily research baselines, not a reconstruction of a discretionary minute-level strategy.")
    st.json(manifest)
    st.warning("Remaining evidence gaps: independently verified prices/calendar, historical universe and delistings, calibrated bid/ask/depth fills, "
               "multiple-testing adjustment, persistent experiment registration, and a forward paper record. "
               "No statistical significance or profitability pass is issued.")
    st.caption("Corporate actions are represented only through vendor-adjusted OHLC. Annualization assumes 252 sessions/year. "
               "No interest on cash, taxes, settlement, shorting, leverage, intraday stops, or liquidity-capacity model. "
               "The source may revise history. Review provider terms before redistributing downloaded data.")

report = {"status": "NOT_VALIDATED", "plan": plan, "development": result.simulation.metrics,
          "development_passive": result.passive.metrics, "folds": result.folds.to_dict("records"),
          "training_scores": result.training_scores.to_dict("records"),
          "session_experiment_count": st.session_state.get("wf_attempts", 1)}
if "wf_holdout" in st.session_state:
    held, passive, stress = st.session_state["wf_holdout"]
    report.update({"holdout_status": "EVALUATED_AFTER_FREEZE", "holdout": held.metrics,
                   "holdout_passive": passive.metrics, "holdout_cost_stress": stress.to_dict("records")})
    st.download_button("Export full snapshot for private reproducibility", snapshot.prices.to_csv(date_format="%Y-%m-%d", float_format="%.17g"),
                       "daily_adjusted_snapshot.csv", "text/csv", key="wf_snapshot_export")
st.download_button("Export experiment report", json.dumps(report, indent=2, allow_nan=False),
                   "walk_forward_report.json", "application/json", key="wf_report")
