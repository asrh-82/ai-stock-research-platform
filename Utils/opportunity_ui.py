"""Explicit-run opportunity workspace. No orders or automatic position sizing."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from Utils.opportunities import OpportunityConfig
from Utils.opportunity_scan import export_scan, run_scan, scan_table


def render_opportunities() -> None:
    st.title("Opportunities")
    st.caption("Learned return estimates, benchmark-relative gaps, and evidence before conviction.")
    st.write("This screen searches for conditional outperformance, not a wider range around today's price. "
             "It learns from past price signals and evaluates forecasts on later, unseen outcomes.")
    st.info("A research candidate must beat both forecasting baselines and clear cost-aware evidence checks. "
            "This is not fundamental fair value, proven mispricing, or beta-adjusted alpha.")
    key = "opportunities:completed"
    with st.form("opportunities:form"):
        left, right = st.columns([2, 1])
        text = left.text_input("Research universe (up to 12 tickers)", "PLTR, USO, SPCX")
        benchmark = right.text_input("Comparison benchmark", "SPY")
        a, b, c, d = st.columns(4)
        horizon = a.selectbox("Holding horizon (sessions)", [21, 63])
        years = b.selectbox("History request (years)", [5, 8])
        costs = c.number_input("Round-trip cost assumption (bps)", 0., 1000., 20., step=5.)
        hurdle = d.number_input("Excess-return hurdle above costs (bps)", 0., 1000., 100., step=25.)
        st.caption("100 bps = 1 percentage point. A forecast made after session t assumes entry at the next "
                   "session's close, then holds for the selected horizon. No same-close execution. "
                   "The ridge specification and signal rules are fixed, not optimized on the displayed tests.")
        submit = st.form_submit_button("Run opportunity scan", type="primary")
    if submit:
        st.session_state.pop(key, None)
        st.session_state.pop("opportunities:bundle", None)
        try:
            with st.spinner("Loading aligned histories, fitting past-only models, and checking held-out forecasts..."):
                st.session_state[key] = run_scan(text, benchmark,
                    OpportunityConfig(horizon=horizon, round_trip_bps=costs, hurdle_bps=hurdle), years)
        except Exception as exc:
            st.error(f"Scan not produced: {type(exc).__name__}: {str(exc)[:400]}")
            st.caption("Previous results were cleared. No synthetic fallback or substituted securities.")
    scan = st.session_state.get(key)
    if scan is None:
        st.caption("Run explicitly to load data. No market-data requests occur on opening this page.")
        return
    table = scan_table(scan)
    candidates = table[table.Status == "Research candidate"]
    st.subheader("Completed scan")
    st.caption(f"Captured {scan.captured} · Benchmark {scan.benchmark} · {scan.config.horizon} sessions · "
               f"{scan.config.round_trip_bps:g} bps assumed costs · {scan.config.hurdle_bps:g} bps extra hurdle. "
               "Edits above apply only after running again. Current New York calendar day is excluded.")
    a, b, c = st.columns(3)
    a.metric("Research candidates", len(candidates))
    b.metric("Assets evaluated", len(scan.results))
    c.metric("Blocked requests", len(scan.errors))
    if candidates.empty:
        st.warning("No candidate cleared all evidence gates. Estimates below are not validated opportunities.")
    else:
        st.markdown("**Candidates for further research**")
        ranked = candidates.sort_values("Gap / OOS error", ascending=False)
        st.dataframe(ranked.drop(columns=["Reason"]), hide_index=True, use_container_width=True)
        st.caption("Ranked by net predicted excess return divided by historical OOS forecast error. "
                   "Clearing exploratory gates is not proof of future performance.")
    st.markdown("**All requested assets, including failures**")
    st.dataframe(table.drop(columns=["Reason"]), hide_index=True, use_container_width=True)
    st.caption("Net excess pp = estimated asset return minus benchmark return minus assumed costs, in percentage "
               "points. Skill is reduction in squared forecast error, not prediction accuracy or win probability.")
    for symbol, error in scan.errors.items():
        st.error(f"{symbol}: {error}")
    if scan.results:
        symbol = st.selectbox("Inspect completed result", list(scan.results))
        result = scan.results[symbol]
        s, ev = result.summary, result.summary["evidence"]
        forecast_tab, evidence_tab, drivers_tab, data_tab = st.tabs(["Forecast", "Validation", "Drivers", "Data & export"])
        with forecast_tab:
            st.markdown(f"**{symbol}: {s['status']}**")
            a, b, c = st.columns(3)
            a.metric("Model total-return estimate", f"{s['forecast_total']:+.2%}")
            b.metric("Model excess estimate, net", f"{100*s['net_forecast_excess']:+.2f} pp")
            c.metric("Gap above costs + hurdle", f"{100*s['gap_to_hurdle']:+.2f} pp")
            for reason in s["reasons"]:
                st.caption(reason)
            if not result.scenarios.empty:
                a, b = st.columns(2)
                a.metric("Error-resampled net excess P5–P95",
                         f"{100*s['net_excess_p05']:+.1f} to {100*s['net_excess_p95']:+.1f} pp")
                b.metric("Scenario outperformance frequency", f"{s['outperform_after_cost_frequency']:.1%}")
                st.caption(f"{len(result.scenarios):,} draws resample only {s['residual_observations']} paired historical "
                           "out-of-sample errors around today's learned forecast. Errors are not recentered, "
                           "so their empirical bias remains. These are terminal return scenarios, not price paths "
                           "or calibrated odds. More draws do not create more independent evidence.")
            else:
                st.caption("Uncertainty distribution withheld: too few held-out errors or invalid implied returns.")
        with evidence_tab:
            st.write("Every historical forecast refits its scaler and regularized regression only on labels "
                     "fully observed before that forecast. Forward holding windows do not overlap. "
                     "Both a zero-excess forecast and a rolling historical-mean forecast are scored on the same dates.")
            a, b, c = st.columns(3)
            a.metric("Non-overlapping OOS windows", ev["oos_folds"])
            b.metric("Prior rule-selected windows", ev.get("selected_count", 0))
            skill = ev.get("skill_vs_zero")
            c.metric("Error reduction vs zero excess", "N/A" if skill is None else f"{skill:+.1%}")
            st.json(ev, expanded=False)
            st.dataframe(result.oos, hide_index=True, use_container_width=True)
            st.caption("Minimum 24 evaluation windows and 12 rule-selected windows. Positive lower diagnostic bounds "
                       "are required for improvement against both baselines and net excess returns of prior selections. "
                       "Moving-block resampling uses three folds per block; tail thresholds account for three checks "
                       "per requested ticker. This does not correct prior scans, configuration searches, survivorship, "
                       "or all model-selection bias. Non-overlap is not independence. Prospective paper validation remains necessary.")
        with drivers_tab:
            frame = result.contributions
            fig = go.Figure(go.Bar(y=frame.feature, x=frame.excess_contribution*100, orientation="h"))
            fig.update_layout(height=420, xaxis_title="Contribution to excess-return estimate (percentage points)",
                              margin=dict(l=10, r=10, t=15, b=30))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Training-standardized feature contributions plus the fitted intercept sum to the raw "
                       "excess forecast before costs. Associations, not causal explanations. No latest fundamentals "
                       "are backfilled into history, and no arbitrary bullish drift is inserted.")
            st.dataframe(frame, hide_index=True, use_container_width=True)
        with data_tab:
            st.json(result.manifest, expanded=False)
            st.caption("Adjusted price histories are provider-revised and not independently point-in-time audited. "
                       "The latest common session is required; internal date mismatches are rejected, not filled. "
                       "A price-based forecast is not evidence that the business is undervalued.")
    if st.button("Prepare complete scan evidence"):
        st.session_state["opportunities:bundle"] = export_scan(scan)
    if "opportunities:bundle" in st.session_state:
        st.download_button("Download scan evidence", st.session_state["opportunities:bundle"],
                           file_name="axion_opportunity_scan.zip", mime="application/zip")
    st.caption("Evidence includes every requested ticker, failed requests, exact aligned histories, model coefficients, "
               "settings, dates, OOS prediction ledger, and conditional terminal scenarios. No orders are placed.")
