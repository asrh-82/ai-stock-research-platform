"""Explicit-run market simulation UI; independent of all fundamental-data loaders."""
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from Utils.market_data import demo_snapshot, load_market_snapshot
from Utils.market_export import evidence_bundle
from Utils.market_monte_carlo import (
    MODELS, MarketConfig, path_bands, simulate_market, summarize_market, walk_forward,
)

MODEL_NAMES = {
    "stationary_bootstrap": "Historical blocks",
    "ewma_bootstrap": "Volatility-adaptive bootstrap",
    "gbm": "Gaussian / GBM baseline",
}
DRIFT_NAMES = {
    "zero": "Zero log drift (risk baseline)",
    "shrunk_historical": "Shrunk historical log drift",
    "manual": "Manual annual log drift",
}


@st.cache_data(ttl=900, max_entries=12, show_spinner=False)
def _load_snapshot(symbol, years, session_day):
    # Date in cache key prevents yesterday's snapshot being reused across NY midnight.
    return load_market_snapshot(symbol, years)


def _figure_layout(figure, title, ytitle):
    figure.update_layout(title=title, height=400, template="plotly_dark",
                         margin=dict(l=20, r=20, t=45, b=25),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         yaxis_title=ytitle, legend=dict(orientation="h", y=1.10))
    return figure


def _path_figure(result):
    bands = path_bands(result)
    figure = go.Figure()
    x = bands.index
    for low, high, name in (("P05", "P95", "90% pointwise band"), ("P25", "P75", "50% pointwise band")):
        figure.add_trace(go.Scatter(x=x, y=bands[low], mode="lines", line=dict(width=0),
                                   showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=x, y=bands[high], mode="lines", line=dict(width=0),
                                   fill="tonexty", name=name, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=x, y=bands.Median, mode="lines", name="Median",
                               hovertemplate="Step %{x}<br>Price-equivalent $%{y:.2f}<extra></extra>"))
    figure.add_hline(y=result.paths[0, 0], line_dash="dot", annotation_text="Reference close")
    figure.update_xaxes(title="Future trading steps (not calendar dates)")
    return _figure_layout(figure, "Simulated market paths", "USD price-equivalent")


def _terminal_figure(result):
    figure = go.Figure(go.Histogram(x=result.paths[:, -1], nbinsx=60, name="Terminal outcomes"))
    figure.add_vline(x=result.paths[0, 0], line_dash="dot", annotation_text="Reference close")
    figure.update_xaxes(title="Terminal USD price-equivalent")
    return _figure_layout(figure, "Terminal outcome distribution", "Simulated paths")


def render_market_monte_carlo(symbol: str | None = None, key_prefix: str = "market") -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    st.subheader("Market Monte Carlo")
    st.caption("Historical returns → stochastic paths → market-risk scenarios. No DCF inputs.")
    st.info("These are model-conditional market scenarios, not fair values or verified odds of making money.")
    result_key = f"{key_prefix}:result"
    previous = st.session_state.get(result_key)
    if symbol and previous and previous["requested_symbol"] != symbol:
        st.session_state.pop(result_key, None)

    with st.form(f"{key_prefix}:form"):
        left, middle, right = st.columns(3)
        with left:
            source = st.selectbox("Data source", ["Live market data", "Synthetic demo"], key=f"{key_prefix}:source")
            requested_symbol = symbol or st.text_input("Ticker", "PLTR", key=f"{key_prefix}:ticker")
            if symbol:
                st.caption(f"Company-tools ticker: {symbol}")
            years = st.selectbox("History window (years)", [1, 3, 5, 8], index=2, key=f"{key_prefix}:years")
        with middle:
            model = st.selectbox("Simulation model", MODELS, format_func=MODEL_NAMES.get, key=f"{key_prefix}:model")
            horizon = st.select_slider("Horizon (trading sessions)", [21, 63, 126, 252], value=63,
                                       key=f"{key_prefix}:horizon")
            simulations = st.select_slider("Paths", [1_000, 5_000, 10_000, 20_000], value=5_000,
                                           key=f"{key_prefix}:paths")
        with right:
            drift = st.selectbox("Drift assumption", list(DRIFT_NAMES), format_func=DRIFT_NAMES.get,
                                  key=f"{key_prefix}:drift")
            upside = st.number_input("Upper level (% above reference)", 1.0, 500.0, 20.0,
                                      key=f"{key_prefix}:upper")
            downside = st.number_input("Lower level (% below reference)", 1.0, 99.0, 20.0,
                                        key=f"{key_prefix}:lower")
        with st.expander("Advanced assumptions"):
            a, b, c = st.columns(3)
            seed = a.number_input("Random seed", 0, 2**32 - 1, 42, key=f"{key_prefix}:seed")
            block = b.number_input("Expected block length (sessions)", 1, 63, 10, key=f"{key_prefix}:block")
            decay = c.number_input("EWMA decay", 0.80, 0.995, 0.94, format="%.3f", key=f"{key_prefix}:decay")
            weight = a.number_input("Historical drift weight", 0.0, 1.0, 0.25, key=f"{key_prefix}:weight")
            manual = b.number_input("Manual annual log drift (%)", -100.0, 100.0, 0.0,
                                     key=f"{key_prefix}:manual")
            st.caption("Drift is the mean LOG return, not expected price appreciation. Zero log drift can still "
                       "produce positive mean price returns. Weight applies only to shrunk drift; the manual "
                       "value applies only to manual drift. Block length and EWMA decay apply to their respective models.")
        submitted = st.form_submit_button("Run simulation", type="primary", key=f"{key_prefix}:run")

    if submitted:
        # Never leave an old successful run displayed as the result of a failed request.
        st.session_state.pop(result_key, None)
        try:
            with st.spinner("Loading completed sessions and simulating paths..."):
                snapshot = (demo_snapshot() if source == "Synthetic demo" else
                            _load_snapshot(requested_symbol.strip().upper(), years,
                                           str(datetime.now(ZoneInfo("America/New_York")).date())))
                config = MarketConfig(model=model, horizon=horizon, simulations=simulations, seed=int(seed),
                                      drift=drift, historical_weight=weight, annual_log_drift=manual / 100,
                                      block_length=int(block), ewma_decay=decay)
                result = simulate_market(snapshot.prices, config, anchor=snapshot.reference_close)
                st.session_state[result_key] = {
                    "snapshot": snapshot, "result": result, "requested_symbol": requested_symbol,
                    "upper": snapshot.reference_close * (1 + upside / 100),
                    "lower": snapshot.reference_close * (1 - downside / 100),
                    "comparison": None, "calibration": None, "bundle": None,
                }
        except Exception as exc:
            st.error(f"No simulation produced: {type(exc).__name__}: {str(exc)[:350]}")
            st.caption("No synthetic fallback, ticker substitution, or invented prices were used.")

    saved = st.session_state.get(result_key)
    if saved is None:
        st.caption("Choose a listed US equity or ETF, then run. The synthetic demo works without a data provider "
                   "and is never a fallback for a failed live request. At least 126 daily returns are required.")
        return
    snapshot, result = saved["snapshot"], saved["result"]
    summary = summarize_market(result, saved["upper"], saved["lower"])
    if snapshot.provenance.get("synthetic"):
        st.warning("SYNTHETIC DEMO. Not a real security, current market price, or investment forecast.")
    st.markdown(f"**Last completed run: {snapshot.symbol} · {MODEL_NAMES[result.config.model]}**")
    st.caption(f"Reference close: ${snapshot.reference_close:,.2f} · As of {snapshot.provenance['as_of_session']} · "
               f"{result.config.horizon} trading sessions · {result.config.simulations:,} paths · Seed {result.config.seed}. "
               "Form changes apply only after Run simulation. Full current NY calendar day is excluded.")
    st.caption("All dollar paths below are ADJUSTED-RETURN PRICE EQUIVALENTS, rebased at that reference close. "
               "They include historical dividend effects and are not literal future exchange-quote predictions.")
    a, b, c, d = st.columns(4)
    a.metric("Median terminal equivalent", f"${summary['median']:,.2f}")
    b.metric("5th–95th percentiles", f"${summary['p05']:,.2f} – ${summary['p95']:,.2f}")
    c.metric("Mean terminal equivalent", f"${summary['mean']:,.2f}")
    d.metric("Simulated loss frequency", f"{summary['probability_loss']:.1%}")

    paths_tab, risk_tab, calibration_tab, evidence_tab = st.tabs(
        ["Paths & models", "Risk & levels", "Coverage checks", "Data & evidence"])
    with paths_tab:
        st.plotly_chart(_path_figure(result), use_container_width=True)
        st.caption("Bands are pointwise percentiles at each step, not a 90% chance that a whole path stays inside them.")
        st.plotly_chart(_terminal_figure(result), use_container_width=True)
        if st.button("Compare all three models", key=f"{key_prefix}:compare"):
            rows = []
            with st.spinner("Comparing the same snapshot and drift policy..."):
                for other_model in MODELS:
                    try:
                        other = simulate_market(snapshot.prices, replace(result.config, model=other_model),
                                                anchor=snapshot.reference_close)
                        stats = summarize_market(other)
                        rows.append({"Model": MODEL_NAMES[other_model], "Median": stats["median"],
                                     "P05": stats["p05"], "P95": stats["p95"],
                                     "Loss frequency": stats["probability_loss"], "95% ES loss": stats["loss_es95"]})
                    except ValueError as exc:
                        rows.append({"Model": MODEL_NAMES[other_model], "Error": str(exc)})
            saved["comparison"] = pd.DataFrame(rows)
        if saved["comparison"] is not None:
            st.dataframe(saved["comparison"], hide_index=True, use_container_width=True)
            st.caption("Same data, horizon, path count, seed, and drift policy. This shows model disagreement, "
                       "not an automatic ranking or validation of any model.")
    with risk_tab:
        a, b, c, d = st.columns(4)
        a.metric("95% horizon loss VaR", f"{summary['loss_var95']:.1%}")
        b.metric("95% expected shortfall", f"{summary['loss_es95']:.1%}")
        c.metric("Median maximum drawdown", f"{summary['median_max_drawdown']:.1%}")
        d.metric("95th-percentile drawdown", f"{summary['p95_max_drawdown']:.1%}")
        st.caption("VaR is the 95th percentile of terminal loss; ES averages the worst 5% of terminal losses. "
                   "A negative loss value denotes a modeled gain. Drawdown is peak-to-trough within each path, "
                   "including the starting reference. These are not guarantees or position-sizing instructions.")
        levels = pd.DataFrame([
            {"Level": f"Upper: ${saved['upper']:,.2f}", "Touch frequency": summary["probability_touch_upper"],
             "Finish beyond level": summary["probability_finish_above_upper"]},
            {"Level": f"Lower: ${saved['lower']:,.2f}", "Touch frequency": summary["probability_touch_lower"],
             "Finish beyond level": summary["probability_finish_below_lower"]},
        ])
        st.dataframe(levels, hide_index=True, use_container_width=True)
        st.caption("Touch means reaching a level at a simulated daily close, including time zero; not intraday "
                   "barrier monitoring or a tradable stop-fill estimate. Frequencies are conditional on this model.")
    with calibration_tab:
        st.markdown("**Walk-forward interval coverage**")
        st.write("Fit on 504 past returns at each origin, then evaluate the next horizon against observations "
                 "excluded from that fit. Forward evaluation windows do not overlap. At most 12 folds and "
                 "2,000 paths per fold keep this diagnostic bounded.")
        if st.button("Run coverage check", key=f"{key_prefix}:coverage"):
            saved["calibration"] = None
            saved["bundle"] = None
            try:
                with st.spinner("Evaluating held-out forward windows..."):
                    saved["calibration"] = walk_forward(snapshot.prices, result.config)
            except ValueError as exc:
                st.warning(str(exc))
        if saved["calibration"] is not None:
            frame = saved["calibration"]
            covered = int(frame.Covered.sum())
            st.metric("Observed 90% interval coverage", f"{covered}/{len(frame)} ({frame.Covered.mean():.1%})")
            st.metric("Mean interval score (lower is better)", f"{frame['Interval score'].mean():.4f}")
            st.dataframe(frame, hide_index=True, use_container_width=True)
            st.warning("A small coverage sample is not proof of accuracy or trading edge. Training windows "
                       "overlap, history may be revised, and inspecting several models creates selection bias. "
                       "Synthetic-demo coverage is only a software check, not market evidence.")
    with evidence_tab:
        for warning in (*snapshot.warnings, *result.warnings):
            st.caption(warning)
        st.json({"data": snapshot.provenance, "model": result.diagnostics}, expanded=False)
        if st.button("Prepare evidence bundle", key=f"{key_prefix}:export"):
            saved["bundle"] = evidence_bundle(snapshot, result, saved["calibration"])
        if saved["bundle"] is not None:
            st.download_button("Download run evidence", saved["bundle"],
                               file_name=f"{snapshot.symbol}_market_mc_seed_{result.config.seed}.zip",
                               mime="application/zip", key=f"{key_prefix}:download")
        st.caption("Bundle: exact history, source metadata, model settings, version and seed, all terminal "
                   "outcomes, pointwise bands, first 100 paths, and any completed coverage check.")
