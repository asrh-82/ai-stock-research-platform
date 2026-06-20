
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE SETUP
# ============================================================
APP_NAME = "Market Research Dashboard"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main {
        background: linear-gradient(180deg, #0b1120 0%, #111827 48%, #020617 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    h1, h2, h3 {
        color: #f8fafc;
        letter-spacing: -0.035em;
    }

    .hero-card {
        padding: 1.45rem 1.65rem;
        border-radius: 26px;
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.26), transparent 36%),
            linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.72));
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.28);
        margin-bottom: 1.25rem;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 850;
        margin-bottom: 0.28rem;
        color: #ffffff;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        max-width: 980px;
    }

    .glass-card {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.20);
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.19);
        margin-bottom: 1rem;
    }

    .portfolio-panel {
        padding: 1.05rem 1.15rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(30, 41, 59, 0.56));
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.88rem;
    }

    .pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.16);
        border: 1px solid rgba(96, 165, 250, 0.34);
        color: #bfdbfe;
        font-size: 0.8rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .pill-green {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(52, 211, 153, 0.30);
        color: #a7f3d0;
    }

    .pill-yellow {
        background: rgba(245, 158, 11, 0.13);
        border: 1px solid rgba(251, 191, 36, 0.28);
        color: #fde68a;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 0.9rem;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="stTabs"] button {
        font-weight: 750;
    }

    .stDataFrame, .stPlotlyChart {
        border-radius: 18px;
        overflow: hidden;
    }

    .mini-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
    }

    .position-preview {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(14, 165, 233, 0.10);
        border: 1px solid rgba(56, 189, 248, 0.24);
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }

    .soft-divider {
        height: 1px;
        background: rgba(148, 163, 184, 0.18);
        margin: 1rem 0;
    }

    button[kind="secondary"] {
        border-radius: 12px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# FILE STORAGE
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# Your project currently uses uppercase .JSON files, so this app writes there.
WATCHLIST_FILE = BASE_DIR / "watchlist.JSON"
PORTFOLIO_FILE = BASE_DIR / "portfolio.JSON"

# Mirrored lowercase copies are kept too, so old code and new code both see updates.
WATCHLIST_MIRROR_FILE = BASE_DIR / "watchlist.json"
PORTFOLIO_MIRROR_FILE = BASE_DIR / "portfolio.json"


def save_json_file(file_path, data, mirror_path=None):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
        file.write("\n")

    if mirror_path is not None:
        mirror_path = Path(mirror_path)
        mirror_path.parent.mkdir(parents=True, exist_ok=True)

        with open(mirror_path, "w") as mirror_file:
            json.dump(data, mirror_file, indent=4)
            mirror_file.write("\n")


def load_json_file(primary_path, default_value, mirror_path=None):
    primary_path = Path(primary_path)
    mirror_path = Path(mirror_path) if mirror_path is not None else None

    if primary_path.exists():
        file_to_load = primary_path
    elif mirror_path is not None and mirror_path.exists():
        file_to_load = mirror_path
    else:
        save_json_file(primary_path, default_value, mirror_path)
        return default_value

    try:
        content = file_to_load.read_text().strip()

        if not content:
            save_json_file(primary_path, default_value, mirror_path)
            return default_value

        data = json.loads(content)

        # Sync both files after loading.
        save_json_file(primary_path, data, mirror_path)
        return data

    except json.JSONDecodeError:
        broken_file = file_to_load.with_suffix(file_to_load.suffix + ".broken")
        file_to_load.rename(broken_file)
        save_json_file(primary_path, default_value, mirror_path)
        return default_value


def load_watchlist():
    return load_json_file(WATCHLIST_FILE, [], WATCHLIST_MIRROR_FILE)


def save_watchlist(watchlist):
    save_json_file(WATCHLIST_FILE, watchlist, WATCHLIST_MIRROR_FILE)


def load_portfolio():
    return load_json_file(PORTFOLIO_FILE, [], PORTFOLIO_MIRROR_FILE)


def save_portfolio(portfolio):
    save_json_file(PORTFOLIO_FILE, portfolio, PORTFOLIO_MIRROR_FILE)


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
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


init_state()


# ============================================================
# DATA HELPERS
# ============================================================
@st.cache_data(ttl=900)
def search_company_symbols(query, max_results=10):
    query = query.strip()

    if not query:
        return []

    results = []

    try:
        search = yf.Search(
            query,
            max_results=max_results,
            news_count=0,
            lists_count=0,
            include_research=False,
            enable_fuzzy_query=True,
            raise_errors=False,
        )

        quotes = getattr(search, "quotes", []) or []

        for quote in quotes:
            symbol = quote.get("symbol")
            name = quote.get("shortname") or quote.get("longname") or quote.get("name")
            quote_type = quote.get("quoteType", "N/A")
            exchange = quote.get("exchange", "N/A")

            if symbol and name:
                results.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "quote_type": quote_type,
                        "exchange": exchange,
                    }
                )
    except Exception:
        pass

    unique_results = []
    seen = set()

    for item in results:
        if item["symbol"] not in seen:
            unique_results.append(item)
            seen.add(item["symbol"])

    return unique_results[:max_results]


def make_lookup_label(result):
    return f"{result['symbol']} | {result['name']} | {result['quote_type']} | {result['exchange']}"


def extract_symbol_from_label(label):
    if not label:
        return ""

    return label.split("|")[0].strip().upper()


@st.cache_data(ttl=300)
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info or {}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_price_history_cached(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_financials_cached(ticker):
    try:
        return yf.Ticker(ticker).financials
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_earnings_dates_cached(ticker):
    try:
        return yf.Ticker(ticker).earnings_dates
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_recommendations_cached(ticker):
    try:
        return yf.Ticker(ticker).recommendations
    except Exception:
        return pd.DataFrame()


def get_live_price_from_info(ticker, info=None):
    if info is None:
        info = get_stock_info(ticker)

    for key in [
        "currentPrice",
        "regularMarketPrice",
        "postMarketPrice",
        "preMarketPrice",
        "previousClose",
    ]:
        price = info.get(key)

        if price is not None:
            try:
                return float(price)
            except Exception:
                pass

    history = get_price_history_cached(ticker, "5d")

    if not history.empty:
        try:
            return float(history["Close"].dropna().iloc[-1])
        except Exception:
            pass

    return None


def fetch_live_price_from_ticker(ticker):
    ticker = ticker.strip().upper()

    if not ticker:
        return 0.0

    price = get_live_price_from_info(ticker)

    if price is None:
        return 0.0

    return round(float(price), 2)


def calculate_returns(price_data):
    if price_data.empty or "Close" not in price_data:
        return pd.Series(dtype=float)

    return price_data["Close"].pct_change().dropna()


def calculate_volatility(returns):
    if returns.empty:
        return None

    return round(returns.std() * np.sqrt(252) * 100, 2)


def calculate_max_drawdown(price_data):
    if price_data.empty or "Close" not in price_data:
        return None

    prices = price_data["Close"]
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max

    return round(drawdown.min() * 100, 2)


def calculate_sharpe_ratio(returns, risk_free_rate=0.04):
    if returns.empty or returns.std() == 0:
        return None

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    return round((excess_returns.mean() / returns.std()) * np.sqrt(252), 2)


def calculate_revenue_growth(financials):
    try:
        revenue = financials.loc["Total Revenue"]

        if len(revenue) < 2:
            return None

        latest_revenue = revenue.iloc[0]
        previous_revenue = revenue.iloc[1]

        if previous_revenue == 0:
            return None

        growth = ((latest_revenue - previous_revenue) / previous_revenue) * 100

        return round(growth, 2)

    except Exception:
        return None


def calculate_score(info, volatility, max_drawdown, sharpe_ratio, revenue_growth):
    score = 0

    pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    roe = info.get("returnOnEquity")
    profit_margin = info.get("profitMargins")
    debt_to_equity = info.get("debtToEquity")

    if pe and pe < 25:
        score += 15
    elif pe and pe < 40:
        score += 8

    if forward_pe and forward_pe < 25:
        score += 15
    elif forward_pe and forward_pe < 40:
        score += 8

    if roe and roe > 0.15:
        score += 15
    elif roe and roe > 0.08:
        score += 8

    if profit_margin and profit_margin > 0.15:
        score += 15
    elif profit_margin and profit_margin > 0.05:
        score += 8

    if debt_to_equity and debt_to_equity < 100:
        score += 10
    elif debt_to_equity and debt_to_equity < 200:
        score += 5

    if revenue_growth and revenue_growth > 10:
        score += 15
    elif revenue_growth and revenue_growth > 0:
        score += 8

    if volatility is not None:
        if volatility < 25:
            score += 10
        elif volatility < 40:
            score += 5

    if max_drawdown is not None:
        if max_drawdown > -25:
            score += 10
        elif max_drawdown > -45:
            score += 5

    if sharpe_ratio is not None:
        if sharpe_ratio > 1:
            score += 10
        elif sharpe_ratio > 0:
            score += 5

    return min(score, 100)


def score_tooltip():
    return (
        "Score is rule-based out of 100. "
        "It adds points for lower trailing and forward P/E, stronger ROE, better profit margin, lower debt-to-equity, positive revenue growth, lower volatility, smaller max drawdown, and positive Sharpe ratio. "
        "It is educational only and not investment advice."
    )


def score_interpretation(score):
    if score >= 75:
        return "Strong Watch"
    if score >= 55:
        return "Neutral"
    return "Needs Research"


def format_large_number(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)

        if value >= 1_000_000_000_000:
            return f"${value / 1_000_000_000_000:.2f}T"
        if value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        if value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"

        return f"${value:,.0f}"

    except Exception:
        return "N/A"


def plot_price_chart(price_data, ticker):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["Close"],
            mode="lines",
            name="Close",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title=f"{ticker} Price Trend",
        xaxis_title="Date",
        yaxis_title="Price",
        height=465,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def style_plotly_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=60, b=20),
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


def get_comparison_data(tickers, period="1y"):
    rows = []

    for ticker in tickers:
        ticker = ticker.strip().upper()

        if not ticker:
            continue

        try:
            info = get_stock_info(ticker)
            price_data = get_price_history_cached(ticker, period)
            financials = get_financials_cached(ticker)

            if not info or price_data.empty:
                continue

            returns = calculate_returns(price_data)
            volatility = calculate_volatility(returns)
            max_drawdown = calculate_max_drawdown(price_data)
            sharpe_ratio = calculate_sharpe_ratio(returns)
            revenue_growth = calculate_revenue_growth(financials)
            score = calculate_score(
                info,
                volatility,
                max_drawdown,
                sharpe_ratio,
                revenue_growth,
            )

            rows.append(
                {
                    "Ticker": ticker,
                    "Company": info.get("longName") or ticker,
                    "Sector": info.get("sector"),
                    "Price": round(get_live_price_from_info(ticker, info) or 0, 2),
                    "Market Cap": info.get("marketCap"),
                    "Revenue Growth %": revenue_growth,
                    "Trailing P/E": info.get("trailingPE"),
                    "Forward P/E": info.get("forwardPE"),
                    "Profit Margin": info.get("profitMargins"),
                    "ROE": info.get("returnOnEquity"),
                    "Debt to Equity": info.get("debtToEquity"),
                    "Beta": info.get("beta"),
                    "Volatility %": volatility,
                    "Max Drawdown %": max_drawdown,
                    "Sharpe Ratio": sharpe_ratio,
                    "Research Score": score,
                    "Signal": score_interpretation(score),
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)


# ============================================================
# WATCHLIST HELPERS
# ============================================================
def add_to_watchlist(ticker):
    ticker = ticker.strip().upper()
    watchlist = load_watchlist()

    if ticker and ticker not in watchlist:
        watchlist.append(ticker)
        save_watchlist(watchlist)
        return True

    return False


def remove_from_watchlist(ticker):
    ticker = ticker.strip().upper()
    watchlist = load_watchlist()
    watchlist = [symbol for symbol in watchlist if symbol.strip().upper() != ticker]
    save_watchlist(watchlist)


def build_watchlist_df(watchlist):
    rows = []

    for symbol in watchlist:
        info = get_stock_info(symbol)
        price = get_live_price_from_info(symbol, info)
        history = get_price_history_cached(symbol, "1y")
        returns = calculate_returns(history)

        rows.append(
            {
                "Ticker": symbol,
                "Company": info.get("shortName") or info.get("longName") or symbol,
                "Price": round(price, 2) if price else None,
                "Sector": info.get("sector", "N/A"),
                "Market Cap": format_large_number(info.get("marketCap")),
                "Signal": score_interpretation(
                    calculate_score(
                        info,
                        calculate_volatility(returns),
                        calculate_max_drawdown(history),
                        calculate_sharpe_ratio(returns),
                        calculate_revenue_growth(get_financials_cached(symbol)),
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PORTFOLIO HELPERS
# ============================================================
def add_or_update_holding(portfolio, ticker, shares, avg_cost):
    ticker = ticker.strip().upper()

    if not ticker or shares <= 0 or avg_cost <= 0:
        return portfolio, False

    for holding in portfolio:
        if holding["ticker"].strip().upper() == ticker:
            old_shares = float(holding["shares"])
            old_avg_cost = float(holding["avg_cost"])
            total_shares = old_shares + shares
            weighted_avg_cost = ((old_shares * old_avg_cost) + (shares * avg_cost)) / total_shares

            holding["ticker"] = ticker
            holding["shares"] = round(total_shares, 8)
            holding["avg_cost"] = round(weighted_avg_cost, 8)

            return portfolio, True

    portfolio.append(
        {
            "ticker": ticker,
            "shares": round(float(shares), 8),
            "avg_cost": round(float(avg_cost), 8),
        }
    )

    return portfolio, True


def remove_portfolio_holding(ticker_to_remove):
    ticker_to_remove = ticker_to_remove.strip().upper()
    portfolio = load_portfolio()

    portfolio = [
        holding
        for holding in portfolio
        if holding.get("ticker", "").strip().upper() != ticker_to_remove
    ]

    save_portfolio(portfolio)


def calculate_portfolio_data(portfolio):
    rows = []

    for holding in portfolio:
        ticker = holding["ticker"].strip().upper()
        shares = float(holding["shares"])
        avg_cost = float(holding["avg_cost"])

        try:
            info = get_stock_info(ticker)
            current_price = get_live_price_from_info(ticker, info) or 0.0

            invested_value = shares * avg_cost
            current_value = shares * current_price
            gain_loss = current_value - invested_value
            gain_loss_pct = (gain_loss / invested_value) * 100 if invested_value else 0

            rows.append(
                {
                    "Ticker": ticker,
                    "Company": info.get("shortName") or info.get("longName") or ticker,
                    "Shares": shares,
                    "Avg Cost": round(avg_cost, 2),
                    "Current Price": round(current_price, 2),
                    "Invested Value": round(invested_value, 2),
                    "Current Value": round(current_value, 2),
                    "Gain/Loss": round(gain_loss, 2),
                    "Gain/Loss %": round(gain_loss_pct, 2),
                    "Sector": info.get("sector", "N/A"),
                }
            )

        except Exception:
            rows.append(
                {
                    "Ticker": ticker,
                    "Company": ticker,
                    "Shares": shares,
                    "Avg Cost": round(avg_cost, 2),
                    "Current Price": 0.0,
                    "Invested Value": round(shares * avg_cost, 2),
                    "Current Value": 0.0,
                    "Gain/Loss": round(-(shares * avg_cost), 2),
                    "Gain/Loss %": -100,
                    "Sector": "N/A",
                }
            )

    return pd.DataFrame(rows)


def calculate_portfolio_summary(portfolio_df):
    if portfolio_df.empty:
        return 0, 0, 0

    total_value = portfolio_df["Current Value"].sum()
    total_invested = portfolio_df["Invested Value"].sum()
    total_gain_loss = total_value - total_invested
    total_gain_loss_pct = (total_gain_loss / total_invested) * 100 if total_invested else 0

    return round(total_value, 2), round(total_gain_loss, 2), round(total_gain_loss_pct, 2)


def get_portfolio_highlights(portfolio_df):
    if portfolio_df.empty:
        return {
            "best": "N/A",
            "worst": "N/A",
            "largest": "N/A",
            "largest_sector": "N/A",
        }

    best_row = portfolio_df.sort_values("Gain/Loss %", ascending=False).iloc[0]
    worst_row = portfolio_df.sort_values("Gain/Loss %", ascending=True).iloc[0]
    largest_row = portfolio_df.sort_values("Current Value", ascending=False).iloc[0]

    sector_df = portfolio_df.groupby("Sector")["Current Value"].sum().reset_index()

    if sector_df.empty:
        largest_sector = "N/A"
    else:
        largest_sector_row = sector_df.sort_values("Current Value", ascending=False).iloc[0]
        largest_sector = f"{largest_sector_row['Sector']} (${largest_sector_row['Current Value']:,.2f})"

    return {
        "best": f"{best_row['Ticker']} ({best_row['Gain/Loss %']}%)",
        "worst": f"{worst_row['Ticker']} ({worst_row['Gain/Loss %']}%)",
        "largest": f"{largest_row['Ticker']} (${largest_row['Current Value']:,.2f})",
        "largest_sector": largest_sector,
    }


# ============================================================
# SIDEBAR
# ============================================================
watchlist = load_watchlist()
portfolio = load_portfolio()

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
                first_symbol = st.session_state.global_lookup_results[0]["symbol"]
                st.session_state.active_ticker = first_symbol
                st.rerun()

    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state.global_lookup_results = []

    if st.session_state.global_lookup_results:
        labels = [make_lookup_label(result) for result in st.session_state.global_lookup_results]

        selected_label = st.selectbox(
            "Results",
            labels,
            key="global_lookup_select",
        )

        selected_symbol = extract_symbol_from_label(selected_label)

        if selected_symbol and selected_symbol != st.session_state.active_ticker:
            st.session_state.active_ticker = selected_symbol
            st.rerun()

    period = st.selectbox(
        "Chart Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
    )

    st.divider()

    st.caption(f"Active: {st.session_state.active_ticker}")

    if st.button("Add Active Stock to Watchlist", use_container_width=True):
        added = add_to_watchlist(st.session_state.active_ticker)

        if added:
            st.success(f"{st.session_state.active_ticker} added.")
            st.rerun()
        else:
            st.info("Already in watchlist.")


# ============================================================
# LOAD ACTIVE STOCK
# ============================================================
ticker = st.session_state.active_ticker.strip().upper()

if not ticker:
    st.warning("Search for a company or ticker.")
    st.stop()

info = get_stock_info(ticker)
price_data = get_price_history_cached(ticker, period)

if not info or price_data.empty:
    st.error("Could not fetch stock data. Try another ticker or company search.")
    st.stop()

company_name = info.get("longName") or info.get("shortName") or ticker
current_price = get_live_price_from_info(ticker, info)
market_cap = info.get("marketCap")
pe_ratio = info.get("trailingPE")
forward_pe = info.get("forwardPE")

financials = get_financials_cached(ticker)
earnings_dates = get_earnings_dates_cached(ticker)
recommendations = get_recommendations_cached(ticker)

returns = calculate_returns(price_data)
volatility = calculate_volatility(returns)
max_drawdown = calculate_max_drawdown(price_data)
sharpe_ratio = calculate_sharpe_ratio(returns)
revenue_growth = calculate_revenue_growth(financials)
stock_score = calculate_score(info, volatility, max_drawdown, sharpe_ratio, revenue_growth)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">{APP_NAME}</div>
        <div class="hero-subtitle">
            Research stocks, compare peers, manage a watchlist, and track portfolio exposure from one dashboard.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="glass-card">
        <span class="pill">{ticker}</span>
        <span class="pill">{info.get("sector", "N/A")}</span>
        <span class="pill">{score_interpretation(stock_score)}</span>
        <h2 style="margin-top: 0.6rem;">{company_name}</h2>
        <div class="small-muted">{info.get("industry", "N/A")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

metric_col1.metric("Price", f"${current_price:,.2f}" if current_price else "N/A")
metric_col2.metric("Market Cap", format_large_number(market_cap))
metric_col3.metric("Forward P/E", forward_pe if forward_pe else "N/A")
metric_col4.metric("Volatility", f"{volatility}%" if volatility is not None else "N/A")
metric_col5.metric("Score", f"{stock_score}/100")


# ============================================================
# TABS
# ============================================================
dashboard_tab, analysis_tab, compare_tab, watchlist_tab, portfolio_tab = st.tabs(
    [
        "Dashboard",
        "Analysis",
        "Compare",
        "Watchlist",
        "Portfolio",
    ]
)


# ============================================================
# DASHBOARD TAB
# ============================================================
with dashboard_tab:
    left, right = st.columns([1.45, 1])

    with left:
        st.subheader("Price Chart")
        st.plotly_chart(plot_price_chart(price_data, ticker), use_container_width=True)

    with right:
        st.subheader("Business Snapshot")

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="mini-label">Company</div>
                <b>{company_name}</b><br><br>
                <b>Sector:</b> {info.get("sector", "N/A")}<br>
                <b>Industry:</b> {info.get("industry", "N/A")}<br>
                <b>Employees:</b> {info.get("fullTimeEmployees", "N/A")}<br>
                <b>Website:</b> {info.get("website", "N/A")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        start_close = round(price_data["Close"].iloc[0], 2)
        latest_close = round(price_data["Close"].iloc[-1], 2)
        total_return = round(((latest_close - start_close) / start_close) * 100, 2)

        r1, r2, r3 = st.columns(3)
        r1.metric("Start", f"${start_close:,.2f}")
        r2.metric("Latest", f"${latest_close:,.2f}")
        r3.metric("Return", f"{total_return}%")

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="mini-label">Signal</div>
                <h3>{score_interpretation(stock_score)}</h3>
                <div class="small-muted">Rule-based score: {stock_score}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Company Description")
    st.write(info.get("longBusinessSummary", "No company summary available."))


# ============================================================
# ANALYSIS TAB
# ============================================================
with analysis_tab:
    st.subheader("Research Summary")

    score_col, summary_col = st.columns([0.75, 1.25])

    with score_col:
        st.metric("Research Score", f"{stock_score}/100", help=score_tooltip())
        st.progress(stock_score / 100)
        st.caption("Educational scoring model. Not financial advice.")

        with st.expander("Score calculation details"):
            st.write(score_tooltip())
            st.markdown(
                """
                **Point categories**
                - Valuation: trailing P/E and forward P/E
                - Profitability: return on equity and profit margin
                - Balance sheet: debt-to-equity
                - Growth: revenue growth
                - Risk: volatility, max drawdown, and Sharpe ratio
                """
            )

    with summary_col:
        st.markdown(generate_research_summary(info, stock_score, volatility, max_drawdown, sharpe_ratio, revenue_growth))

    st.divider()

    fundamentals = {
        "Market Cap": format_large_number(info.get("marketCap")),
        "Enterprise Value": format_large_number(info.get("enterpriseValue")),
        "Revenue": format_large_number(info.get("totalRevenue")),
        "Revenue Growth": f"{revenue_growth}%" if revenue_growth is not None else "N/A",
        "Gross Margin": info.get("grossMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "Profit Margin": info.get("profitMargins"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Debt to Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Trailing P/E": pe_ratio,
        "Forward P/E": forward_pe,
        "PEG Ratio": info.get("pegRatio"),
        "Price to Book": info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "Beta": info.get("beta"),
    }

    st.subheader("Fundamentals")
    st.dataframe(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]), use_container_width=True)

    st.subheader("Risk")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    risk_col1.metric("Volatility", f"{volatility}%" if volatility is not None else "N/A")
    risk_col2.metric("Max Drawdown", f"{max_drawdown}%" if max_drawdown is not None else "N/A")
    risk_col3.metric("Sharpe Ratio", sharpe_ratio if sharpe_ratio is not None else "N/A")
    risk_col4.metric("Beta", info.get("beta", "N/A"))

    st.subheader("Financial Statements")

    if financials is not None and not financials.empty:
        st.dataframe(financials, use_container_width=True)
    else:
        st.info("No financial statement data available.")

    st.subheader("Earnings and Analyst Data")

    earnings_col, analyst_col = st.columns(2)

    with earnings_col:
        st.write("Recent / Upcoming Earnings")

        if earnings_dates is not None and not earnings_dates.empty:
            st.dataframe(earnings_dates.head(8), use_container_width=True)
        else:
            st.info("No earnings dates available from yfinance.")

    with analyst_col:
        st.write("Analyst Recommendations")

        if recommendations is not None and not recommendations.empty:
            st.dataframe(recommendations.tail(10), use_container_width=True)
        else:
            st.info("No analyst recommendation data available.")


# ============================================================
# COMPARE TAB
# ============================================================
with compare_tab:
    st.subheader("Peer Comparison")

    default_peers = f"{ticker}, MSFT, GOOGL, AMZN, NVDA"

    peer_input = st.text_input(
        "Tickers separated by comma",
        value=default_peers,
        key="peer_input",
    )

    comparison_period = st.selectbox(
        "Comparison Period",
        ["6mo", "1y", "2y", "5y"],
        index=1,
        key="comparison_period",
    )

    if st.button("Run Comparison", use_container_width=True):
        peer_tickers = [symbol.strip().upper() for symbol in peer_input.split(",") if symbol.strip()]
        comparison_df = get_comparison_data(peer_tickers, comparison_period)
        st.session_state.comparison_df = comparison_df

    comparison_df = st.session_state.get("comparison_df", pd.DataFrame())

    if comparison_df.empty:
        st.info("Run a comparison to see peer rankings.")
    else:
        comparison_df = comparison_df.sort_values(by="Research Score", ascending=False)

        st.dataframe(comparison_df, use_container_width=True)

        top_stock = comparison_df.iloc[0]

        st.success(
            f"Top ranked by rule-based score: {top_stock['Ticker']} "
            f"with {top_stock['Research Score']}/100."
        )

        c1, c2 = st.columns(2)

        with c1:
            fig_score = px.bar(
                comparison_df,
                x="Ticker",
                y="Research Score",
                title="Research Score",
            )
            st.plotly_chart(style_plotly_chart(fig_score), use_container_width=True)

        with c2:
            fig_pe = px.bar(
                comparison_df,
                x="Ticker",
                y="Forward P/E",
                title="Forward P/E",
            )
            st.plotly_chart(style_plotly_chart(fig_pe), use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig_margin = px.bar(
                comparison_df,
                x="Ticker",
                y="Profit Margin",
                title="Profit Margin",
            )
            st.plotly_chart(style_plotly_chart(fig_margin), use_container_width=True)

        with c4:
            fig_volatility = px.bar(
                comparison_df,
                x="Ticker",
                y="Volatility %",
                title="Volatility",
            )
            st.plotly_chart(style_plotly_chart(fig_volatility), use_container_width=True)


# ============================================================
# WATCHLIST TAB
# ============================================================
with watchlist_tab:
    st.subheader("Watchlist")

    watchlist = load_watchlist()

    add_col, filter_col = st.columns([1, 1])

    with add_col:
        if st.button(f"Add {ticker}", use_container_width=True):
            added = add_to_watchlist(ticker)

            if added:
                st.success(f"{ticker} added to watchlist.")
                st.rerun()
            else:
                st.info("Already in watchlist.")

    with filter_col:
        watchlist_filter = st.text_input("Filter watchlist", value="", key="watchlist_filter")

    filtered_watchlist = [
        symbol for symbol in watchlist
        if watchlist_filter.upper() in symbol.upper()
    ]

    if not filtered_watchlist:
        st.info("No watchlist items found.")
    else:
        watch_df = build_watchlist_df(filtered_watchlist)

        st.dataframe(watch_df, use_container_width=True)

        st.write("### Remove from Watchlist")

        remove_watch_symbol = st.selectbox(
            "Select ticker to remove",
            filtered_watchlist,
            key="watchlist_remove_select",
        )

        if st.button("Remove Selected Watchlist Stock", use_container_width=True):
            remove_from_watchlist(remove_watch_symbol)
            st.rerun()


# ============================================================
# PORTFOLIO TAB
# ============================================================
with portfolio_tab:
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
        display_df = portfolio_df.copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_allocation = px.pie(
                portfolio_df,
                names="Ticker",
                values="Current Value",
                title="Allocation by Ticker",
            )
            st.plotly_chart(style_plotly_chart(fig_allocation), use_container_width=True)

        with chart_col2:
            sector_df = portfolio_df.groupby("Sector")["Current Value"].sum().reset_index()

            fig_sector = px.pie(
                sector_df,
                names="Sector",
                values="Current Value",
                title="Allocation by Sector",
            )
            st.plotly_chart(style_plotly_chart(fig_sector), use_container_width=True)

        fig_gain_loss = px.bar(
            portfolio_df,
            x="Ticker",
            y="Gain/Loss",
            title="Gain/Loss by Holding",
        )
        st.plotly_chart(style_plotly_chart(fig_gain_loss), use_container_width=True)

        st.write("### Remove Holding")

        remove_portfolio_symbol = st.selectbox(
            "Select ticker to remove",
            portfolio_df["Ticker"].tolist(),
            key="portfolio_remove_select",
        )

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
            portfolio_lookup_query = st.text_input(
                "Search company or ticker",
                value=st.session_state.portfolio_lookup_query,
                key="portfolio_lookup_query_input",
            )

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

            selected_portfolio_label = st.selectbox(
                "Search Results",
                portfolio_options,
                key="portfolio_lookup_select",
            )

            selected_portfolio_symbol = extract_symbol_from_label(selected_portfolio_label)

            if selected_portfolio_symbol and selected_portfolio_symbol != st.session_state.portfolio_selected_symbol:
                st.session_state.portfolio_selected_symbol = selected_portfolio_symbol

                live_price_for_selected = fetch_live_price_from_ticker(selected_portfolio_symbol)
                st.session_state.portfolio_manual_price = live_price_for_selected if live_price_for_selected > 0 else 100.0
                st.session_state.portfolio_total_price = live_price_for_selected if live_price_for_selected > 0 else 100.0

                st.rerun()

        portfolio_ticker = st.session_state.portfolio_selected_symbol.strip().upper()
        live_price = fetch_live_price_from_ticker(portfolio_ticker)

        st.markdown(
            f"""
            <div class="position-preview">
                <span class="pill">{portfolio_ticker}</span>
                <span class="pill pill-green">Live Price: ${live_price:,.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        price_col, amount_col, shares_col = st.columns(3)

        with price_col:
            manual_price = st.number_input(
                "Buy Price Per Share",
                min_value=0.01,
                value=float(st.session_state.portfolio_manual_price),
                step=1.0,
                key="portfolio_manual_price",
            )

        with amount_col:
            total_price = st.number_input(
                "Total Amount to Invest",
                min_value=0.0,
                value=float(st.session_state.portfolio_total_price),
                step=10.0,
                key="portfolio_total_price",
            )

        shares = total_price / manual_price if manual_price > 0 else 0.0

        with shares_col:
            st.metric("Shares Calculated", f"{shares:.8f}")
            st.caption("Calculated from amount ÷ buy price.")

        st.markdown(
            f"""
            <div class="glass-card">
                <b>Order Preview</b><br>
                {portfolio_ticker}: ${total_price:,.2f} ÷ ${manual_price:,.2f} = <b>{shares:.8f} shares</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if live_price > 0 and manual_price != live_price:
            st.info(
                f"Live price is ${live_price:,.2f}, but your buy price is ${manual_price:,.2f}. "
                f"Shares use your buy price. Portfolio value uses live price."
            )

        if st.button("Add to Portfolio", use_container_width=True):
            if not portfolio_ticker:
                st.warning("Search for a company first.")
            elif manual_price <= 0:
                st.warning("Enter a valid buy price.")
            elif total_price <= 0:
                st.warning("Enter an amount above 0.")
            else:
                portfolio = load_portfolio()

                portfolio, did_save = add_or_update_holding(
                    portfolio,
                    portfolio_ticker,
                    float(shares),
                    float(manual_price),
                )

                if did_save:
                    save_portfolio(portfolio)
                    st.success(
                        f"Added {shares:.8f} shares of {portfolio_ticker} at ${manual_price:,.2f}/share."
                    )
                    st.rerun()
                else:
                    st.warning("Could not add holding.")
