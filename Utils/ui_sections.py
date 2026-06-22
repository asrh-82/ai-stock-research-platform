import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Utils.data_utils import (
    extract_symbol_from_label,
    fetch_live_price_from_ticker,
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
from Utils.portfolio_utils import (
    add_or_update_holding,
    add_to_watchlist,
    build_watchlist_df,
    calculate_portfolio_data,
    calculate_portfolio_summary,
    get_portfolio_highlights,
    load_portfolio,
    load_watchlist,
    remove_from_watchlist,
    remove_portfolio_holding,
    save_portfolio,
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
.main {background: linear-gradient(180deg, #0b1120 0%, #111827 48%, #020617 100%); color: #f8fafc;}
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stSidebar"] {background: #020617; border-right: 1px solid rgba(148, 163, 184, 0.18);}
h1, h2, h3 {color: #f8fafc; letter-spacing: -0.035em;}
.hero-card {padding: 1.45rem 1.65rem; border-radius: 26px; background: radial-gradient(circle at top left, rgba(59, 130, 246, 0.26), transparent 36%), linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.72)); border: 1px solid rgba(148, 163, 184, 0.22); box-shadow: 0 22px 60px rgba(0, 0, 0, 0.28); margin-bottom: 1.25rem;}
.hero-title {font-size: 2.15rem; font-weight: 850; margin-bottom: 0.28rem; color: #ffffff; line-height: 1.1;}
.hero-subtitle {color: #cbd5e1; font-size: 1rem; max-width: 980px;}
.glass-card {padding: 1rem 1.1rem; border-radius: 20px; background: rgba(15, 23, 42, 0.78); border: 1px solid rgba(148, 163, 184, 0.20); box-shadow: 0 16px 42px rgba(0, 0, 0, 0.19); margin-bottom: 1rem;}
.portfolio-panel {padding: 1.05rem 1.15rem; border-radius: 22px; background: linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(30, 41, 59, 0.56)); border: 1px solid rgba(148, 163, 184, 0.18); margin-bottom: 1rem;}
.small-muted {color: #94a3b8; font-size: 0.88rem;}
.pill {display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px; background: rgba(59, 130, 246, 0.16); border: 1px solid rgba(96, 165, 250, 0.34); color: #bfdbfe; font-size: 0.8rem; margin-right: 0.35rem; margin-bottom: 0.35rem;}
.pill-green {background: rgba(16, 185, 129, 0.14); border: 1px solid rgba(52, 211, 153, 0.30); color: #a7f3d0;}
div[data-testid="stMetric"] {background: rgba(15, 23, 42, 0.78); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 18px; padding: 0.9rem; box-shadow: 0 12px 34px rgba(0, 0, 0, 0.18);}
div[data-testid="stTabs"] button {font-weight: 750;}
.stDataFrame, .stPlotlyChart {border-radius: 18px; overflow: hidden;}
.mini-label {color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem;}
.position-preview {padding: 1rem; border-radius: 18px; background: rgba(14, 165, 233, 0.10); border: 1px solid rgba(56, 189, 248, 0.24); margin-top: 0.4rem; margin-bottom: 1rem;}
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
        "portfolio_lookup_query": "Apple",
        "portfolio_lookup_results": [],
        "portfolio_selected_symbol": "AAPL",
        "portfolio_manual_price": 100.0,
        "portfolio_total_price": 100.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def plot_price_chart(price_data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data["Close"], mode="lines", name="Close", line=dict(width=3)))
    fig.update_layout(title=f"{ticker} Price Trend", xaxis_title="Date", yaxis_title="Price", height=465, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)", font=dict(color="#e5e7eb"), margin=dict(l=20, r=20, t=60, b=20))
    return fig


def style_plotly_chart(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)", font=dict(color="#e5e7eb"), margin=dict(l=20, r=20, t=60, b=20))
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
        lookup_query = st.text_input("Company or ticker", value=st.session_state.global_lookup_query, key="global_lookup_query_input")
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
    return {"ticker": ticker, "info": info, "price_data": price_data, "financials": financials, "earnings_dates": earnings_dates, "recommendations": recommendations, "volatility": volatility, "max_drawdown": max_drawdown, "sharpe_ratio": sharpe_ratio, "revenue_growth": revenue_growth, "stock_score": stock_score, "current_price": get_live_price_from_info(ticker, info), "company_name": info.get("longName") or info.get("shortName") or ticker}


def render_header(context, app_name="Market Research Dashboard"):
    ticker = context["ticker"]
    info = context["info"]
    company_name = context["company_name"]
    current_price = context["current_price"]
    stock_score = context["stock_score"]
    st.markdown(f'''<div class="hero-card"><div class="hero-title">{app_name}</div><div class="hero-subtitle">Research stocks, compare peers, manage a watchlist, and track portfolio exposure from one dashboard.</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="glass-card"><span class="pill">{ticker}</span><span class="pill">{info.get("sector", "N/A")}</span><span class="pill">{score_interpretation(stock_score)}</span><h2 style="margin-top: 0.6rem;">{company_name}</h2><div class="small-muted">{info.get("industry", "N/A")}</div></div>''', unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Price", f"${current_price:,.2f}" if current_price else "N/A")
    metric_col2.metric("Market Cap", format_large_number(info.get("marketCap")))
    metric_col3.metric("Forward P/E", info.get("forwardPE") if info.get("forwardPE") else "N/A")
    metric_col4.metric("Volatility", f"{context['volatility']}%" if context["volatility"] is not None else "N/A")
    metric_col5.metric("Score", f"{stock_score}/100")


def render_dashboard(context):
    ticker, info, price_data = context["ticker"], context["info"], context["price_data"]
    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Price Chart")
        st.plotly_chart(plot_price_chart(price_data, ticker), use_container_width=True)
    with right:
        st.subheader("Business Snapshot")
        st.markdown(f'''<div class="glass-card"><div class="mini-label">Company</div><b>{context["company_name"]}</b><br><br><b>Sector:</b> {info.get("sector", "N/A")}<br><b>Industry:</b> {info.get("industry", "N/A")}<br><b>Employees:</b> {info.get("fullTimeEmployees", "N/A")}<br><b>Website:</b> {info.get("website", "N/A")}</div>''', unsafe_allow_html=True)
        start_close = round(price_data["Close"].iloc[0], 2)
        latest_close = round(price_data["Close"].iloc[-1], 2)
        total_return = round(((latest_close - start_close) / start_close) * 100, 2)
        r1, r2, r3 = st.columns(3)
        r1.metric("Start", f"${start_close:,.2f}")
        r2.metric("Latest", f"${latest_close:,.2f}")
        r3.metric("Return", f"{total_return}%")
        st.markdown(f'''<div class="glass-card"><div class="mini-label">Signal</div><h3>{score_interpretation(context["stock_score"])}</h3><div class="small-muted">Rule-based score: {context["stock_score"]}/100</div></div>''', unsafe_allow_html=True)
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
            st.markdown("""**Point categories**
- Valuation: trailing P/E and forward P/E
- Profitability: return on equity and profit margin
- Balance sheet: debt-to-equity
- Growth: revenue growth
- Risk: volatility, max drawdown, and Sharpe ratio""")
    with summary_col:
        st.markdown(generate_research_summary(info, context["stock_score"], context["volatility"], context["max_drawdown"], context["sharpe_ratio"], context["revenue_growth"]))
    st.divider()
    fundamentals = {"Market Cap": format_large_number(info.get("marketCap")), "Enterprise Value": format_large_number(info.get("enterpriseValue")), "Revenue": format_large_number(info.get("totalRevenue")), "Revenue Growth": f"{context['revenue_growth']}%" if context["revenue_growth"] is not None else "N/A", "Gross Margin": info.get("grossMargins"), "Operating Margin": info.get("operatingMargins"), "Profit Margin": info.get("profitMargins"), "ROE": info.get("returnOnEquity"), "ROA": info.get("returnOnAssets"), "Debt to Equity": info.get("debtToEquity"), "Current Ratio": info.get("currentRatio"), "Trailing P/E": info.get("trailingPE"), "Forward P/E": info.get("forwardPE"), "PEG Ratio": info.get("pegRatio"), "Price to Book": info.get("priceToBook"), "Dividend Yield": info.get("dividendYield"), "Beta": info.get("beta")}
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


def render_portfolio():
    st.subheader("Portfolio")
    portfolio = load_portfolio()
    portfolio_df = calculate_portfolio_data(portfolio)
    if not portfolio_df.empty:
        total_value, total_gain_loss, total_gain_loss_pct = calculate_portfolio_summary(portfolio_df)
        highlights = get_portfolio_highlights(portfolio_df)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Portfolio Value", f"${total_value:,.2f}")
        p2.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}")
        p3.metric("Gain/Loss %", f"{total_gain_loss_pct}%")
        p4.metric("Holdings", len(portfolio_df))
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Best Position", highlights["best"])
        h2.metric("Worst Position", highlights["worst"])
        h3.metric("Largest Holding", highlights["largest"])
        h4.metric("Largest Sector", highlights["largest_sector"])
        st.markdown('<div class="portfolio-panel">', unsafe_allow_html=True)
        st.write("### Current Holdings")
        st.dataframe(portfolio_df.copy(), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_allocation = px.pie(portfolio_df, names="Ticker", values="Current Value", title="Allocation by Ticker")
            st.plotly_chart(style_plotly_chart(fig_allocation), use_container_width=True)
        with chart_col2:
            sector_df = portfolio_df.groupby("Sector")["Current Value"].sum().reset_index()
            fig_sector = px.pie(sector_df, names="Sector", values="Current Value", title="Allocation by Sector")
            st.plotly_chart(style_plotly_chart(fig_sector), use_container_width=True)
        fig_gain_loss = px.bar(portfolio_df, x="Ticker", y="Gain/Loss", title="Gain/Loss by Holding")
        st.plotly_chart(style_plotly_chart(fig_gain_loss), use_container_width=True)
        st.write("### Remove Holding")
        remove_portfolio_symbol = st.selectbox("Select ticker to remove", portfolio_df["Ticker"].tolist(), key="portfolio_remove_select")
        if st.button("Remove Selected Holding", use_container_width=True):
            remove_portfolio_holding(remove_portfolio_symbol)
            st.rerun()
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Portfolio Value", "$0.00")
        p2.metric("Total Gain/Loss", "$0.00")
        p3.metric("Gain/Loss %", "0%")
        p4.metric("Holdings", "0")
        st.info("No portfolio holdings yet.")
    st.divider()
    with st.expander("Add Position", expanded=portfolio_df.empty):
        lookup_left, lookup_right = st.columns([1.25, 0.75])
        with lookup_left:
            portfolio_lookup_query = st.text_input("Search company or ticker", value=st.session_state.portfolio_lookup_query, key="portfolio_lookup_query_input")
        with lookup_right:
            st.write("")
            st.write("")
            if st.button("Search Position", use_container_width=True):
                st.session_state.portfolio_lookup_query = portfolio_lookup_query
                st.session_state.portfolio_lookup_results = search_company_symbols(portfolio_lookup_query)
                if st.session_state.portfolio_lookup_results:
                    selected_portfolio_symbol = st.session_state.portfolio_lookup_results[0]["symbol"]
                    st.session_state.portfolio_selected_symbol = selected_portfolio_symbol
                    live_price_for_selected = fetch_live_price_from_ticker(selected_portfolio_symbol)
                    st.session_state.portfolio_manual_price = live_price_for_selected if live_price_for_selected > 0 else 100.0
                    st.session_state.portfolio_total_price = live_price_for_selected if live_price_for_selected > 0 else 100.0
                    st.rerun()
        if st.session_state.portfolio_lookup_results:
            portfolio_options = [make_lookup_label(result) for result in st.session_state.portfolio_lookup_results]
            selected_portfolio_label = st.selectbox("Search Results", portfolio_options, key="portfolio_lookup_select")
            selected_portfolio_symbol = extract_symbol_from_label(selected_portfolio_label)
            if selected_portfolio_symbol and selected_portfolio_symbol != st.session_state.portfolio_selected_symbol:
                st.session_state.portfolio_selected_symbol = selected_portfolio_symbol
                live_price_for_selected = fetch_live_price_from_ticker(selected_portfolio_symbol)
                st.session_state.portfolio_manual_price = live_price_for_selected if live_price_for_selected > 0 else 100.0
                st.session_state.portfolio_total_price = live_price_for_selected if live_price_for_selected > 0 else 100.0
                st.rerun()
        portfolio_ticker = st.session_state.portfolio_selected_symbol.strip().upper()
        live_price = fetch_live_price_from_ticker(portfolio_ticker)
        st.markdown(f'''<div class="position-preview"><span class="pill">{portfolio_ticker}</span><span class="pill pill-green">Live Price: ${live_price:,.2f}</span></div>''', unsafe_allow_html=True)
        price_col, amount_col, shares_col = st.columns(3)
        with price_col:
            manual_price = st.number_input("Buy Price Per Share", min_value=0.01, value=float(st.session_state.portfolio_manual_price), step=1.0, key="portfolio_manual_price")
        with amount_col:
            total_price = st.number_input("Total Amount to Invest", min_value=0.0, value=float(st.session_state.portfolio_total_price), step=10.0, key="portfolio_total_price")
        shares = total_price / manual_price if manual_price > 0 else 0.0
        with shares_col:
            st.metric("Shares Calculated", f"{shares:.8f}")
            st.caption("Calculated from amount ÷ buy price.")
        st.markdown(f'''<div class="glass-card"><b>Order Preview</b><br>{portfolio_ticker}: ${total_price:,.2f} ÷ ${manual_price:,.2f} = <b>{shares:.8f} shares</b></div>''', unsafe_allow_html=True)
        if live_price > 0 and manual_price != live_price:
            st.info(f"Live price is ${live_price:,.2f}, but your buy price is ${manual_price:,.2f}. Shares use your buy price. Portfolio value uses live price.")
        if st.button("Add to Portfolio", use_container_width=True):
            if not portfolio_ticker:
                st.warning("Search for a company first.")
            elif manual_price <= 0:
                st.warning("Enter a valid buy price.")
            elif total_price <= 0:
                st.warning("Enter an amount above 0.")
            else:
                portfolio = load_portfolio()
                portfolio, did_save = add_or_update_holding(portfolio, portfolio_ticker, float(shares), float(manual_price))
                if did_save:
                    save_portfolio(portfolio)
                    st.success(f"Added {shares:.8f} shares of {portfolio_ticker} at ${manual_price:,.2f}/share.")
                    st.rerun()
                else:
                    st.warning("Could not add holding.")
