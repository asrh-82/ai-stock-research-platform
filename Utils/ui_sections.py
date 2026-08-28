import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Utils.data_utils import (
    extract_symbol_from_label,
    format_large_number,
    get_comparison_data,
    get_earnings_dates_cached,
    get_financials_cached,
    get_live_price_from_info,
    get_price_history_cached,
    get_recommendations_cached,
    get_stock_info,
    make_lookup_label,
    search_company_symbols,
)
from Utils.watchlist_utils import (
    add_to_watchlist,
    build_watchlist_df,
    load_watchlist,
    remove_from_watchlist,
)
from Utils.scoring import (
    calculate_max_drawdown,
    calculate_revenue_growth,
    calculate_returns,
    calculate_score,
    calculate_sharpe_ratio,
    calculate_volatility,
    score_interpretation,
    score_tooltip,
)

CUSTOM_CSS = """
<style>
.main {
    background: #0b1220;
    color: #e5e7eb;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.5rem;
    max-width: 1420px;
}

[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1f2937;
}

h1, h2, h3 {
    color: #f8fafc;
    letter-spacing: -0.025em;
}

h1 {
    font-size: 1.85rem !important;
}

h2 {
    font-size: 1.35rem !important;
}

h3 {
    font-size: 1.05rem !important;
}

.section-card,
.glass-card {
    background: #111827;
    border: 1px solid #243244;
    border-radius: 10px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.9rem;
    box-shadow: none;
}

.company-header {
    background: #111827;
    border: 1px solid #243244;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}

.company-title {
    font-size: 1.65rem;
    font-weight: 750;
    color: #f8fafc;
    line-height: 1.15;
    margin: 0.2rem 0;
}

.company-meta {
    color: #9ca3af;
    font-size: 0.88rem;
}

.small-muted {
    color: #9ca3af;
    font-size: 0.86rem;
}

.pill {
    display: inline-block;
    padding: 0.16rem 0.5rem;
    border-radius: 5px;
    background: #1f2937;
    border: 1px solid #374151;
    color: #d1d5db;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #243244;
    border-radius: 10px;
    padding: 0.58rem 0.65rem;
    box-shadow: none;
}

div[data-testid="stMetric"] label {
    font-size: 0.72rem !important;
    line-height: 1.1 !important;
    color: #9ca3af !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1rem !important;
    line-height: 1.15 !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 0.72rem !important;
    line-height: 1.1 !important;
}

div[data-testid="stTabs"] button {
    font-weight: 650;
}

.stDataFrame,
.stPlotlyChart {
    border-radius: 8px;
    overflow: hidden;
}

.mini-label {
    color: #94a3b8;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}

.block-note {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-top: -0.25rem;
    margin-bottom: 0.75rem;
}

hr {
    border-color: #1f2937;
}
</style>
"""


def apply_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def initialize_session_state() -> None:
    defaults = {
        "active_ticker": "AAPL",
        "global_lookup_query": "Apple",
        "global_lookup_results": [],
        "comparison_df": pd.DataFrame(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def plot_price_chart(price_data, ticker):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2.5),
        )
    )
    fig.update_layout(
        title=f"{ticker} Price Trend",
        xaxis_title="Date",
        yaxis_title="Price",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
    )
    return fig


def style_plotly_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
    )
    return fig


def generate_research_summary(info, score, volatility, max_drawdown, sharpe_ratio, revenue_growth):
    company = info.get("longName", "This company")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    rating = score_interpretation(score)
    return f"""
### Research Snapshot

**{company}** operates in the **{sector}** sector and the **{industry}** industry.

The rule-based research score is **{score}/100**, classified as **{rating}**.

**Valuation**
- Trailing P/E: **{info.get("trailingPE", "N/A")}**
- Forward P/E: **{info.get("forwardPE", "N/A")}**

**Profitability**
- Profit Margin: **{info.get("profitMargins", "N/A")}**
- Return on Equity: **{info.get("returnOnEquity", "N/A")}**

**Growth and Risk**
- Revenue Growth: **{revenue_growth}%**
- Annualized Volatility: **{volatility}%**
- Max Drawdown: **{max_drawdown}%**
- Sharpe Ratio: **{sharpe_ratio}**

This is an educational signal, not investment advice.
"""


def render_sidebar():
    with st.sidebar:
        st.markdown("### Search")
        lookup_query = st.text_input(
            "Company or ticker",
            value=st.session_state.global_lookup_query,
            key="global_lookup_query_input",
        )
        search_col, clear_col = st.columns([2, 1])
        with search_col:
            if st.button("Search", use_container_width=True):
                st.session_state.global_lookup_query = lookup_query
                st.session_state.global_lookup_results = search_company_symbols(lookup_query)
                if st.session_state.global_lookup_results:
                    st.session_state.active_ticker = st.session_state.global_lookup_results[0]["symbol"]
                    st.rerun()
        with clear_col:
            if st.button("Clear", use_container_width=True):
                st.session_state.global_lookup_results = []
        if st.session_state.global_lookup_results:
            labels = [make_lookup_label(result) for result in st.session_state.global_lookup_results]
            selected_label = st.selectbox("Results", labels, key="global_lookup_select")
            selected_symbol = extract_symbol_from_label(selected_label)
            if selected_symbol and selected_symbol != st.session_state.active_ticker:
                st.session_state.active_ticker = selected_symbol
                st.rerun()
        period = st.selectbox("Chart Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        st.divider()
        st.caption(f"Active: {st.session_state.active_ticker}")
        if st.button("Add Active Stock to Watchlist", use_container_width=True):
            added = add_to_watchlist(st.session_state.active_ticker)
            if added:
                st.success(f"{st.session_state.active_ticker} added.")
                st.rerun()
            else:
                st.info("Already in watchlist.")
    return st.session_state.active_ticker.strip().upper(), period


def get_active_stock_context(ticker, period):
    info = get_stock_info(ticker)
    price_data = get_price_history_cached(ticker, period)
    if not info or price_data.empty:
        return None
    financials = get_financials_cached(ticker)
    earnings_dates = get_earnings_dates_cached(ticker)
    recommendations = get_recommendations_cached(ticker)
    returns = calculate_returns(price_data)
    volatility = calculate_volatility(returns)
    max_drawdown = calculate_max_drawdown(price_data)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    revenue_growth = calculate_revenue_growth(financials)
    stock_score = calculate_score(info, volatility, max_drawdown, sharpe_ratio, revenue_growth)
    return {
        "ticker": ticker,
        "info": info,
        "price_data": price_data,
        "financials": financials,
        "earnings_dates": earnings_dates,
        "recommendations": recommendations,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "revenue_growth": revenue_growth,
        "stock_score": stock_score,
        "current_price": get_live_price_from_info(ticker, info),
        "company_name": info.get("longName") or info.get("shortName") or ticker,
    }


def render_header(context, app_name="Equity Research Platform"):
    ticker = context["ticker"]
    info = context["info"]
    company_name = context["company_name"]
    current_price = context["current_price"]
    stock_score = context["stock_score"]

    st.markdown(
        f'''
        <div class="company-header">
            <span class="pill">{ticker}</span>
            <span class="pill">{info.get("sector", "N/A")}</span>
            <span class="pill">{score_interpretation(stock_score)}</span>
            <div class="company-title">{company_name}</div>
            <div class="company-meta">{info.get("industry", "N/A")}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Price", f"${current_price:,.2f}" if current_price else "N/A")
    metric_col2.metric("Market Cap", format_large_number(info.get("marketCap")))
    metric_col3.metric("Forward P/E", info.get("forwardPE") if info.get("forwardPE") else "N/A")
    metric_col4.metric("Volatility", f"{context['volatility']}%" if context["volatility"] is not None else "N/A")
    metric_col5.metric("Score", f"{stock_score}/100")


def render_dashboard(context):
    ticker, info, price_data = context["ticker"], context["info"], context["price_data"]
    left, right = st.columns([1.55, 1])

    with left:
        st.subheader("Price Chart")
        st.plotly_chart(plot_price_chart(price_data, ticker), use_container_width=True)

    with right:
        st.subheader("Snapshot")
        start_close = round(price_data["Close"].iloc[0], 2)
        latest_close = round(price_data["Close"].iloc[-1], 2)
        total_return = round(((latest_close - start_close) / start_close) * 100, 2)

        r1, r2, r3 = st.columns(3)
        r1.metric("Start", f"${start_close:,.2f}")
        r2.metric("Latest", f"${latest_close:,.2f}")
        r3.metric("Return", f"{total_return}%")

        st.markdown(
            f'''
            <div class="section-card">
                <div class="mini-label">Business</div>
                <b>Sector:</b> {info.get("sector", "N/A")}<br>
                <b>Industry:</b> {info.get("industry", "N/A")}<br>
                <b>Employees:</b> {info.get("fullTimeEmployees", "N/A")}<br>
                <b>Website:</b> {info.get("website", "N/A")}
            </div>
            ''',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'''
            <div class="section-card">
                <div class="mini-label">Signal</div>
                <b>{score_interpretation(context["stock_score"])}</b><br>
                <span class="small-muted">Rule-based score: {context["stock_score"]}/100</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.subheader("Company Description")
    st.write(info.get("longBusinessSummary", "No company summary available."))


def render_analysis(context):
    info = context["info"]
    st.subheader("Research Summary")
    score_col, summary_col = st.columns([0.75, 1.25])

    with score_col:
        st.metric("Research Score", f"{context['stock_score']}/100", help=score_tooltip())
        st.progress(context["stock_score"] / 100)
        st.caption("Educational scoring model. Not financial advice.")
        with st.expander("Score calculation details"):
            st.write(score_tooltip())
            st.markdown(
                """**Point categories**
- Valuation: trailing P/E and forward P/E
- Profitability: return on equity and profit margin
- Balance sheet: debt-to-equity
- Growth: revenue growth
- Risk: volatility, max drawdown, and Sharpe ratio"""
            )

    with summary_col:
        st.markdown(
            generate_research_summary(
                info,
                context["stock_score"],
                context["volatility"],
                context["max_drawdown"],
                context["sharpe_ratio"],
                context["revenue_growth"],
            )
        )

    st.divider()
    fundamentals = {
        "Market Cap": format_large_number(info.get("marketCap")),
        "Enterprise Value": format_large_number(info.get("enterpriseValue")),
        "Revenue": format_large_number(info.get("totalRevenue")),
        "Revenue Growth": f"{context['revenue_growth']}%" if context["revenue_growth"] is not None else "N/A",
        "Gross Margin": info.get("grossMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "Profit Margin": info.get("profitMargins"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Debt to Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "PEG Ratio": info.get("pegRatio"),
        "Price to Book": info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "Beta": info.get("beta"),
    }

    st.subheader("Fundamentals")
    st.dataframe(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]), use_container_width=True)

    st.subheader("Risk")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    risk_col1.metric("Volatility", f"{context['volatility']}%" if context["volatility"] is not None else "N/A")
    risk_col2.metric("Max Drawdown", f"{context['max_drawdown']}%" if context["max_drawdown"] is not None else "N/A")
    risk_col3.metric("Sharpe Ratio", context["sharpe_ratio"] if context["sharpe_ratio"] is not None else "N/A")
    risk_col4.metric("Beta", info.get("beta", "N/A"))

    st.subheader("Financial Statements")
    financials = context["financials"]
    if financials is not None and not financials.empty:
        st.dataframe(financials, use_container_width=True)
    else:
        st.info("No financial statement data available.")

    st.subheader("Earnings and Analyst Data")
    earnings_col, analyst_col = st.columns(2)
    with earnings_col:
        st.write("Recent / Upcoming Earnings")
        if context["earnings_dates"] is not None and not context["earnings_dates"].empty:
            st.dataframe(context["earnings_dates"].head(8), use_container_width=True)
        else:
            st.info("No earnings dates available from yfinance.")
    with analyst_col:
        st.write("Analyst Recommendations")
        if context["recommendations"] is not None and not context["recommendations"].empty:
            st.dataframe(context["recommendations"].tail(10), use_container_width=True)
        else:
            st.info("No analyst recommendation data available.")


def render_compare(active_ticker):
    st.subheader("Peer Comparison")
    default_peers = f"{active_ticker}, MSFT, GOOGL, AMZN, NVDA"
    peer_input = st.text_input("Tickers separated by comma", value=default_peers, key="peer_input")
    comparison_period = st.selectbox("Comparison Period", ["6mo", "1y", "2y", "5y"], index=1, key="comparison_period")
    if st.button("Run Comparison", use_container_width=True):
        peer_tickers = [symbol.strip().upper() for symbol in peer_input.split(",") if symbol.strip()]
        st.session_state.comparison_df = get_comparison_data(peer_tickers, comparison_period)
    comparison_df = st.session_state.get("comparison_df", pd.DataFrame())
    if comparison_df.empty:
        st.info("Run a comparison to see peer rankings.")
    else:
        comparison_df = comparison_df.sort_values(by="Research Score", ascending=False)
        st.dataframe(comparison_df, use_container_width=True)
        top_stock = comparison_df.iloc[0]
        st.success(f"Top ranked by rule-based score: {top_stock['Ticker']} with {top_stock['Research Score']}/100.")
        c1, c2 = st.columns(2)
        with c1:
            fig_score = px.bar(comparison_df, x="Ticker", y="Research Score", title="Research Score")
            st.plotly_chart(style_plotly_chart(fig_score), use_container_width=True)
        with c2:
            fig_pe = px.bar(comparison_df, x="Ticker", y="Forward P/E", title="Forward P/E")
            st.plotly_chart(style_plotly_chart(fig_pe), use_container_width=True)


def render_watchlist(active_ticker):
    st.subheader("Watchlist")
    watchlist = load_watchlist()
    add_col, filter_col = st.columns([1, 1])
    with add_col:
        if st.button(f"Add {active_ticker}", use_container_width=True):
            added = add_to_watchlist(active_ticker)
            if added:
                st.success(f"{active_ticker} added to watchlist.")
                st.rerun()
            else:
                st.info("Already in watchlist.")
    with filter_col:
        watchlist_filter = st.text_input("Filter watchlist", value="", key="watchlist_filter")
    filtered_watchlist = [symbol for symbol in watchlist if watchlist_filter.upper() in symbol.upper()]
    if not filtered_watchlist:
        st.info("No watchlist items found.")
    else:
        st.dataframe(build_watchlist_df(filtered_watchlist), use_container_width=True)
        st.write("### Remove from Watchlist")
        remove_watch_symbol = st.selectbox("Select ticker to remove", filtered_watchlist, key="watchlist_remove_select")
        if st.button("Remove Selected Watchlist Stock", use_container_width=True):
            remove_from_watchlist(remove_watch_symbol)
            st.rerun()
