import pandas as pd
import streamlit as st
import yfinance as yf

from Utils.scoring import (
    calculate_max_drawdown,
    calculate_revenue_growth,
    calculate_returns,
    calculate_score,
    calculate_sharpe_ratio,
    calculate_volatility,
    score_interpretation,
)


@st.cache_data(ttl=900)
def search_company_symbols(query: str, max_results: int = 10):
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

    unique = []
    seen = set()

    for item in results:
        if item["symbol"] not in seen:
            unique.append(item)
            seen.add(item["symbol"])

    return unique[:max_results]


def make_lookup_label(result) -> str:
    return f"{result['symbol']} | {result['name']} | {result['quote_type']} | {result['exchange']}"


def extract_symbol_from_label(label: str) -> str:
    if not label:
        return ""

    return label.split("|")[0].strip().upper()


@st.cache_data(ttl=300)
def get_stock_info(ticker: str):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_price_history_cached(ticker: str, period: str = "1y"):
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_financials_cached(ticker: str):
    try:
        return yf.Ticker(ticker).financials
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_earnings_dates_cached(ticker: str):
    try:
        return yf.Ticker(ticker).earnings_dates
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_recommendations_cached(ticker: str):
    try:
        return yf.Ticker(ticker).recommendations
    except Exception:
        return pd.DataFrame()


def get_live_price_from_info(ticker: str, info=None):
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


def fetch_live_price_from_ticker(ticker: str) -> float:
    ticker = ticker.strip().upper()

    if not ticker:
        return 0.0

    price = get_live_price_from_info(ticker)

    return round(float(price), 2) if price is not None else 0.0


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


def get_comparison_data(tickers, period: str = "1y") -> pd.DataFrame:
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
            score = calculate_score(info, volatility, max_drawdown, sharpe_ratio, revenue_growth)

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