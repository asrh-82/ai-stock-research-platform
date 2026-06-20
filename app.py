import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
 
 
WATCHLIST_FILE = "watchlist.json"
 
 
st.set_page_config(
    page_title="AI Stock Research Platform",
    page_icon="📈",
    layout="wide"
)
 
 
def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return []
 
    with open(WATCHLIST_FILE, "r") as file:
        return json.load(file)
 
 
def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as file:
        json.dump(watchlist, file, indent=4)
 
 
def get_stock(ticker):
    return yf.Ticker(ticker)
 
 
def get_company_info(stock):
    try:
        return stock.info
    except Exception:
        return {}
 
 
def get_price_history(stock, period="1y"):
    try:
        return stock.history(period=period)
    except Exception:
        return pd.DataFrame()
 
 
def calculate_returns(price_data):
    return price_data["Close"].pct_change().dropna()
 
 
def calculate_volatility(returns):
    return round(returns.std() * np.sqrt(252) * 100, 2)
 
 
def calculate_max_drawdown(price_data):
    prices = price_data["Close"]
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return round(drawdown.min() * 100, 2)
 
 
def calculate_sharpe_ratio(returns, risk_free_rate=0.04):
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
 
    if returns.std() == 0:
        return 0
 
    return round((excess_returns.mean() / returns.std()) * np.sqrt(252), 2)
 
 
def plot_price_chart(price_data, ticker):
    fig = go.Figure()
 
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["Close"],
            mode="lines",
            name="Close Price"
        )
    )
 
    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=450
    )
 
    return fig
 
 
st.title("📈 AI Stock Research Platform")
st.caption("Mini Bloomberg-style stock research dashboard using Python and Streamlit")
 
watchlist = load_watchlist()
 
with st.sidebar:
    st.header("Search Stock")
 
    ticker = st.text_input("Enter ticker", value="AAPL").upper()
    period = st.selectbox(
        "Price history",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
 
    if st.button("Add to Watchlist"):
        if ticker and ticker not in watchlist:
            watchlist.append(ticker)
            save_watchlist(watchlist)
            st.success(f"{ticker} added to watchlist")
 
    st.divider()
    st.header("Watchlist")
 
    if watchlist:
        for item in watchlist:
            st.write(f"⭐ {item}")
    else:
        st.write("No stocks added yet.")
 
 
if ticker:
    stock = get_stock(ticker)
    info = get_company_info(stock)
    price_data = get_price_history(stock, period)
 
    if not info or price_data.empty:
        st.error("Could not fetch stock data. Try another ticker.")
    else:
        company_name = info.get("longName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        website = info.get("website", "N/A")
        summary = info.get("longBusinessSummary", "No company summary available.")
 
        st.header(company_name)
 
        col1, col2, col3, col4 = st.columns(4)
 
        current_price = info.get("currentPrice", "N/A")
        market_cap = info.get("marketCap", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
 
        col1.metric("Current Price", current_price)
        col2.metric("Market Cap", market_cap)
        col3.metric("Trailing P/E", pe_ratio)
        col4.metric("Forward P/E", forward_pe)
 
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Company Analysis",
                "Price Chart",
                "Fundamentals",
                "Risk Analysis",
                "Watchlist"
            ]
        )
 
        with tab1:
            st.subheader("Business Overview")
 
            st.write(f"**Sector:** {sector}")
            st.write(f"**Industry:** {industry}")
            st.write(f"**Website:** {website}")
 
            st.write(summary)
 
        with tab2:
            st.subheader("Price Performance")
            fig = plot_price_chart(price_data, ticker)
            st.plotly_chart(fig, use_container_width=True)
 
            latest_close = round(price_data["Close"].iloc[-1], 2)
            first_close = round(price_data["Close"].iloc[0], 2)
            total_return = round(((latest_close - first_close) / first_close) * 100, 2)
 
            col1, col2, col3 = st.columns(3)
            col1.metric("Start Price", first_close)
            col2.metric("Latest Price", latest_close)
            col3.metric("Return", f"{total_return}%")
 
        with tab3:
            st.subheader("Fundamental Metrics")
 
            metrics = {
                "Market Cap": info.get("marketCap"),
                "Enterprise Value": info.get("enterpriseValue"),
                "Revenue": info.get("totalRevenue"),
                "Gross Margins": info.get("grossMargins"),
                "Operating Margins": info.get("operatingMargins"),
                "Profit Margins": info.get("profitMargins"),
                "Return on Equity": info.get("returnOnEquity"),
                "Return on Assets": info.get("returnOnAssets"),
                "Debt to Equity": info.get("debtToEquity"),
                "Current Ratio": info.get("currentRatio"),
                "Trailing P/E": info.get("trailingPE"),
                "Forward P/E": info.get("forwardPE"),
                "PEG Ratio": info.get("pegRatio"),
                "Price to Book": info.get("priceToBook"),
                "Dividend Yield": info.get("dividendYield"),
                "Beta": info.get("beta")
            }
 
            df = pd.DataFrame(
                metrics.items(),
                columns=["Metric", "Value"]
            )
 
            st.dataframe(df, use_container_width=True)
 
        with tab4:
            st.subheader("Risk Analysis")
 
            returns = calculate_returns(price_data)
 
            volatility = calculate_volatility(returns)
            max_drawdown = calculate_max_drawdown(price_data)
            sharpe_ratio = calculate_sharpe_ratio(returns)
            beta = info.get("beta", "N/A")
 
            col1, col2, col3, col4 = st.columns(4)
 
            col1.metric("Annualized Volatility", f"{volatility}%")
            col2.metric("Max Drawdown", f"{max_drawdown}%")
            col3.metric("Sharpe Ratio", sharpe_ratio)
            col4.metric("Beta", beta)
 
            st.write("### Risk Interpretation")
 
            if volatility > 35:
                st.warning("High volatility stock.")
            elif volatility > 20:
                st.info("Moderate volatility stock.")
            else:
                st.success("Relatively low volatility stock.")
 
            if max_drawdown < -40:
                st.warning("Large historical drawdown risk.")
            elif max_drawdown < -20:
                st.info("Moderate drawdown risk.")
            else:
                st.success("Lower drawdown risk.")
 
        with tab5:
            st.subheader("Your Watchlist")
 
            if watchlist:
                watchlist_data = []
 
                for symbol in watchlist:
                    try:
                        s = yf.Ticker(symbol)
                        i = s.info
 
                        watchlist_data.append(
                            {
                                "Ticker": symbol,
                                "Company": i.get("longName"),
                                "Price": i.get("currentPrice"),
                                "Market Cap": i.get("marketCap"),
                                "P/E": i.get("trailingPE"),
                                "Sector": i.get("sector")
                            }
                        )
                    except Exception:
                        pass
 
                st.dataframe(
                    pd.DataFrame(watchlist_data),
                    use_container_width=True
                )
            else:
                st.write("No stocks in watchlist.")