import streamlit as st

from Utils.backtesting_ui import render_backtesting
from Utils.dcf_ui import render_dcf
from Utils.monte_carlo_ui import render_monte_carlo as render_dcf_uncertainty
from Utils.market_monte_carlo_ui import render_market_monte_carlo
from Utils.ui_sections import (
    apply_styles, get_active_stock_context, initialize_session_state,
    render_analysis, render_compare, render_dashboard, render_header,
    render_sidebar, render_watchlist,
)

APP_NAME = "Equity Research Platform"
st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
apply_styles()
initialize_session_state()
active_ticker, period = render_sidebar()
context = get_active_stock_context(active_ticker, period)
if context is None:
    st.error("Could not fetch company data. The independent Market Monte Carlo workspace only needs price history.")
    st.stop()
render_header(context, APP_NAME)
(dashboard_tab, analysis_tab, valuation_tab, market_tab, backtest_tab, compare_tab, watchlist_tab) = st.tabs(
    ["Dashboard", "Analysis", "DCF valuation", "Market Monte Carlo", "Backtest", "Compare", "Watchlist"])
with dashboard_tab:
    render_dashboard(context)
with analysis_tab:
    render_analysis(context)
with valuation_tab:
    if context["info"].get("quoteType") != "EQUITY":
        st.info("Corporate DCF is restricted to identified equities. Funds and other instruments need different models.")
    else:
        render_dcf(context)
        with st.expander("DCF assumption uncertainty · not a price forecast"):
            st.caption("This preserved legacy simulation varies fundamental assumptions. It uses its own automatic "
                       "baseline and scenario controls, and does not inherit manual DCF edits above.")
            if st.toggle("Enable DCF uncertainty", key=f"dcf_uncertainty_enabled_{active_ticker}"):
                render_dcf_uncertainty(context)
with market_tab:
    render_market_monte_carlo(active_ticker, key_prefix="company_market")
with backtest_tab:
    render_backtesting(context)
with compare_tab:
    render_compare(active_ticker)
with watchlist_tab:
    render_watchlist(active_ticker)
