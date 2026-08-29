import streamlit as st

from Utils.dcf_ui import render_dcf
from Utils.ui_sections import (
    apply_styles,
    get_active_stock_context,
    initialize_session_state,
    render_analysis,
    render_compare,
    render_dashboard,
    render_header,
    render_sidebar,
    render_watchlist,
)

APP_NAME = "Equity Research Platform"


st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()
initialize_session_state()

active_ticker, period = render_sidebar()
context = get_active_stock_context(active_ticker, period)

if context is None:
    st.error("Could not fetch stock data. Try another ticker or company search.")
    st.stop()

render_header(context, APP_NAME)

dashboard_tab, analysis_tab, valuation_tab, compare_tab, watchlist_tab = st.tabs(
    [
        "Dashboard",
        "Analysis",
        "Valuation",
        "Compare",
        "Watchlist",
    ]
)

with dashboard_tab:
    render_dashboard(context)

with analysis_tab:
    render_analysis(context)

with valuation_tab:
    render_dcf(context)

with compare_tab:
    render_compare(active_ticker)

with watchlist_tab:
    render_watchlist(active_ticker)
