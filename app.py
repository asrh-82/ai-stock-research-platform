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
 
 
def safe_get_info(stock):

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
 
 
def get_financials(stock):

    try:

        return stock.financials

    except Exception:

        return pd.DataFrame()
 
 
def get_earnings_dates(stock):

    try:

        return stock.earnings_dates

    except Exception:

        return pd.DataFrame()
 
 
def get_recommendations(stock):

    try:

        return stock.recommendations

    except Exception:

        return pd.DataFrame()
 
 
def calculate_revenue_growth(financials):

    try:

        revenue = financials.loc["Total Revenue"]
 
        if len(revenue) < 2:

            return None
 
        latest_revenue = revenue.iloc[0]

        previous_revenue = revenue.iloc[1]
 
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
 
    if volatility < 25:

        score += 10

    elif volatility < 40:

        score += 5
 
    if max_drawdown > -25:

        score += 10

    elif max_drawdown > -45:

        score += 5
 
    if sharpe_ratio > 1:

        score += 10

    elif sharpe_ratio > 0:

        score += 5
 
    return min(score, 100)
 
 
def score_interpretation(score):

    if score >= 75:

        return "Strong Watch / Attractive"

    elif score >= 55:

        return "Hold / Neutral"

    else:

        return "Avoid / Needs More Research"
 
 
def generate_research_summary(info, score, volatility, max_drawdown, sharpe_ratio, revenue_growth):

    company = info.get("longName", "This company")

    sector = info.get("sector", "N/A")

    industry = info.get("industry", "N/A")

    pe = info.get("trailingPE", "N/A")

    forward_pe = info.get("forwardPE", "N/A")

    profit_margin = info.get("profitMargins", "N/A")

    roe = info.get("returnOnEquity", "N/A")

    debt_to_equity = info.get("debtToEquity", "N/A")
 
    rating = score_interpretation(score)
 
    summary = f"""

### AI-Style Research Summary
 
**{company}** operates in the **{sector}** sector and the **{industry}** industry.
 
The stock receives a current research score of **{score}/100**, which suggests: **{rating}**.
 
#### Valuation View

- Trailing P/E: **{pe}**

- Forward P/E: **{forward_pe}**
 
A lower P/E may suggest cheaper valuation, while a higher P/E may indicate stronger growth expectations or possible overvaluation.
 
#### Profitability View

- Profit Margin: **{profit_margin}**

- Return on Equity: **{roe}**
 
Higher margins and ROE generally indicate better business quality and management efficiency.
 
#### Growth View

- Latest Revenue Growth: **{revenue_growth}%**
 
Positive revenue growth suggests business expansion. Negative growth may require deeper investigation.
 
#### Balance Sheet View

- Debt-to-Equity: **{debt_to_equity}**
 
Lower debt-to-equity usually means lower financial risk.
 
#### Risk View

- Annualized Volatility: **{volatility}%**

- Maximum Drawdown: **{max_drawdown}%**

- Sharpe Ratio: **{sharpe_ratio}**
 
Lower volatility, smaller drawdowns, and higher Sharpe ratios are generally preferred.
 
#### Final View

This stock should be researched further before making any investment decision. The score is only a rule-based educational signal, not financial advice.

"""
 
    return summary
 
 
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

    info = safe_get_info(stock)

    price_data = get_price_history(stock, period)
 
    if not info or price_data.empty:

        st.error("Could not fetch stock data. Try another ticker.")

    else:

        company_name = info.get("longName", ticker)
 
        st.header(company_name)
 
        current_price = info.get("currentPrice", "N/A")

        market_cap = info.get("marketCap", "N/A")

        pe_ratio = info.get("trailingPE", "N/A")

        forward_pe = info.get("forwardPE", "N/A")
 
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Current Price", current_price)

        col2.metric("Market Cap", market_cap)

        col3.metric("Trailing P/E", pe_ratio)

        col4.metric("Forward P/E", forward_pe)
 
        returns = calculate_returns(price_data)

        volatility = calculate_volatility(returns)

        max_drawdown = calculate_max_drawdown(price_data)

        sharpe_ratio = calculate_sharpe_ratio(returns)
 
        financials = get_financials(stock)

        earnings_dates = get_earnings_dates(stock)

        recommendations = get_recommendations(stock)

        revenue_growth = calculate_revenue_growth(financials)
 
        stock_score = calculate_score(

            info,

            volatility,

            max_drawdown,

            sharpe_ratio,

            revenue_growth

        )
 
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(

            [

                "Company Analysis",

                "Price Chart",

                "Fundamentals",

                "Risk Analysis",

                "Earnings & Analysts",

                "AI Research Summary"

            ]

        )
 
        with tab1:

            st.subheader("Business Overview")
 
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")

            st.write(f"**Industry:** {info.get('industry', 'N/A')}")

            st.write(f"**Website:** {info.get('website', 'N/A')}")
 
            st.write(info.get("longBusinessSummary", "No company summary available."))
 
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

                "Revenue Growth": f"{revenue_growth}%" if revenue_growth else "N/A",

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
 
            df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

            st.dataframe(df, use_container_width=True)
 
        with tab4:

            st.subheader("Risk Analysis")
 
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

            st.subheader("Earnings Summary")
 
            if earnings_dates is not None and not earnings_dates.empty:

                st.write("Recent / Upcoming Earnings Dates")

                st.dataframe(earnings_dates.head(8), use_container_width=True)

            else:

                st.info("No earnings dates available from yfinance.")
 
            st.divider()
 
            st.subheader("Financial Statements")
 
            if financials is not None and not financials.empty:

                st.dataframe(financials, use_container_width=True)

            else:

                st.info("No financial statement data available.")
 
            st.divider()
 
            st.subheader("Analyst Recommendations")
 
            if recommendations is not None and not recommendations.empty:

                st.dataframe(recommendations.tail(10), use_container_width=True)

            else:

                st.info("No analyst recommendation data available.")
 
        with tab6:

            st.subheader("AI-Style Stock Research Score")
 
            col1, col2 = st.columns(2)
 
            col1.metric("Research Score", f"{stock_score}/100")

            col2.metric("Interpretation", score_interpretation(stock_score))
 
            st.progress(stock_score / 100)
 
            summary = generate_research_summary(

                info,

                stock_score,

                volatility,

                max_drawdown,

                sharpe_ratio,

                revenue_growth

            )
 
            st.markdown(summary)
 
            st.warning(

                "Educational project only. This is not financial advice."

            )
 