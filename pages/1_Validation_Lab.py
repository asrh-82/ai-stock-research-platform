"""Network-independent replay page; no broker integration or order submission."""
import json
from dataclasses import replace

import pandas as pd
import streamlit as st

from Utils.validation_lab import (
    ReplayConfig, bootstrap_returns, parse_replay_csv, replay, synthetic_fixture,
)

st.set_page_config(page_title="Validation Lab", layout="wide")
st.title("Validation Lab")
st.caption("Execution audit / Long and cash / Research only")
st.info("Simulation only. No live orders or capital deployment.")

uploaded = st.file_uploader("OHLCV and close-of-bar signals", type=["csv"], key="validation_csv")
demo = st.toggle("Use synthetic demonstration", value=False, key="validation_demo")
left, right = st.columns(2)
with left:
    cash = st.number_input("Virtual starting cash", min_value=1.0, max_value=1_000_000.0,
                           value=10_000.0, step=100.0, key="validation_cash")
with right:
    fraction = st.number_input("Virtual allocation on each entry (%)", min_value=1.0,
                               max_value=100.0, value=25.0, step=1.0, key="validation_fraction")

with st.expander("Execution assumptions"):
    c1, c2, c3 = st.columns(3)
    with c1:
        commission = st.number_input("Commission per side (bps)", 0.0, 1000.0, 5.0,
                                    key="validation_fee")
        stop = st.number_input("Simulated stop (%) | 0 disables", 0.0, 99.0, 0.0,
                              key="validation_stop")
    with c2:
        slip = st.number_input("Adverse slippage per side (bps)", 0.0, 250.0, 10.0,
                              key="validation_slip")
        target = st.number_input("Simulated target (%) | 0 disables", 0.0, 99.0, 0.0,
                                key="validation_target")
    with c3:
        delay = st.number_input("Execution delay (bars)", 1, 100, 1, key="validation_delay")
        halt = st.number_input("Drawdown halt (%) | 0 disables", 0.0, 99.0, 15.0,
                              key="validation_halt")
    st.caption("1 bp = 0.01%. Whole-share entries are capped at 1% of previous-bar volume. "
               "These are uncalibrated simulation assumptions, not risk recommendations. "
               "Stops and halts do not guarantee a maximum loss.")

if demo:
    data = synthetic_fixture()
    source = "SYNTHETIC SOFTWARE FIXTURE: NOT MARKET DATA"
    st.warning(source)
elif uploaded is not None:
    try:
        data = parse_replay_csv(uploaded.getvalue())
        source = "USER UPLOAD: DATA AND SIGNAL PROVENANCE UNVERIFIED"
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
else:
    st.write("Upload a CSV or enable the synthetic demonstration. No market data is downloaded.")
    st.code("Timestamp,Open,High,Low,Close,Volume,Signal")
    st.caption("One asset per CSV; maximum 10 MiB and 100,000 bars. Timestamp labels the bar OPEN "
               "and needs Z or a UTC offset. Signal is 0 or 1, known only after that bar CLOSES. "
               "Use consistently adjusted prices and volume. Signals must not use future data.")
    st.download_button("Download synthetic CSV template", synthetic_fixture().to_csv(),
                       "synthetic_template.csv", "text/csv")
    st.stop()

config = ReplayConfig(initial_cash=cash, entry_fraction=fraction / 100,
                      commission_bps=commission, slippage_bps=slip,
                      execution_delay=int(delay), stop_loss=stop / 100 if stop else None,
                      profit_target=target / 100 if target else None,
                      halt_drawdown=halt / 100 if halt else None)
try:
    result = replay(data, config)
except (ValueError, ArithmeticError) as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Strategy status: NOT VALIDATED")
st.caption("The simulator does not establish signal causality, a historical edge, or real-world "
           "fill quality. Synthetic output cannot satisfy those requirements.")
m = result.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ending virtual equity", f"${m['ending_equity']:,.2f}")
c2.metric("Net simulated return", f"{m['net_return']:.2%}")
c3.metric("Bar-close maximum drawdown", f"{m['maximum_drawdown']:.2%}")
c4.metric("Completed simulated trades", str(m["completed_trades"]))

replay_tab, stress_tab, evidence_tab = st.tabs(["Replay", "Cost stress", "Evidence gaps"])
with replay_tab:
    passive_data = data.copy()
    passive_data["Signal"] = 1
    passive = replay(passive_data, replace(config, stop_loss=None, profit_target=None,
                                          halt_drawdown=None))
    chart = pd.DataFrame({"Strategy": result.equity.Equity,
                          "Same-sizing passive": passive.equity.Equity,
                          "Cash (no interest)": cash})
    # Bound chart payload without changing the replay or export precision.
    stride = max(1, (len(chart) + 1999) // 2000)
    chart_view = chart.iloc[::stride]
    if chart_view.index[-1] != chart.index[-1]:
        chart_view = pd.concat([chart_view, chart.iloc[[-1]]])
    st.line_chart(chart_view)
    st.caption("Passive uses the same asset, entry fraction, capacity and costs, without bracket "
               "or drawdown exits. It is not a diversified benchmark. Charts sample at most about "
               "2,000 points; exports retain every bar. Equity is marked at each bar close.")
    st.write(f"Estimated equity after closing the remaining position: "
             f"**${m['estimated_liquidation_equity']:,.2f}**. "
             f"Open shares: **{m['open_shares']}**. Fees: **${m['fees_paid']:,.2f}**. "
             f"Modeled slippage: **${m['modeled_slippage']:,.2f}**.")
    st.caption(f"Ambiguous bars: {m['ambiguous_bars']} | Rejected entries: {m['rejected_entries']} "
               f"| Drawdown halt triggered: {m['halted']}")
    st.dataframe(result.trades, hide_index=True, use_container_width=True)
    st.download_button("Export equity ledger", result.equity.to_csv(),
                       "simulated_equity.csv", "text/csv")
    st.download_button("Export closed-trade ledger", result.trades.to_csv(index=False),
                       "simulated_trades.csv", "text/csv")

with stress_tab:
    rows = []
    for multiplier in (1, 2, 4):
        stressed = replay(data, replace(config, slippage_bps=slip * multiplier))
        rows.append({"Slippage multiplier": multiplier,
                     "Per-side slippage (bps)": slip * multiplier,
                     "Ending virtual equity": stressed.metrics["ending_equity"],
                     "Net return": stressed.metrics["net_return"],
                     "Maximum drawdown": stressed.metrics["maximum_drawdown"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Only slippage is multiplied; commission stays fixed. Different fills can change "
               "subsequent trades. This is sensitivity analysis, not calibrated execution.")
    daily = result.equity.Equity.resample("D").last().dropna()
    daily_returns = daily.pct_change()
    daily_returns.iloc[0] = daily.iloc[0] / cash - 1
    if len(daily_returns) >= 20:
        with st.expander("Block-resampling uncertainty"):
            try:
                uncertainty = bootstrap_returns(daily_returns.to_numpy(), paths=500)
                st.json(uncertainty)
                st.caption("CSV timestamps use UTC. Resampled outcomes are conditional on this "
                           "dataset and block length, not probabilities of future returns.")
            except ValueError as exc:
                st.warning(str(exc))
    else:
        st.caption("Resampling requires at least 20 daily observations; short intraday replays are insufficient.")

with evidence_tab:
    st.dataframe(pd.DataFrame({
        "Requirement": ["Market data", "Deterministic strategy", "Causal signals",
                        "Point-in-time universe", "Execution realism", "Out-of-sample evidence",
                        "Forward paper record"],
        "Current evidence": [source, "Not established by imported signals", "Unverified",
                             "Unverified; delistings not established", "Simplified bar model only",
                             "Not established", "Not connected"],
    }), hide_index=True, use_container_width=True)
    for warning in result.warnings:
        st.caption(warning)
    st.caption("Do not test minute-level rules using daily bars. A chronological split helper "
               "exists in the engine; automated walk-forward fitting and untouched-holdout "
               "evaluation are not implemented in this phase. The existing Backtest tab is unchanged.")

report = {"source": source, "manifest": result.manifest,
          "metrics": result.metrics, "warnings": result.warnings}
st.download_button("Export reproducibility report", json.dumps(report, indent=2, allow_nan=False),
                   "research_replay_report.json", "application/json")
