import numpy as np
import pandas as pd


def calculate_returns(price_data: pd.DataFrame) -> pd.Series:
    if price_data.empty or "Close" not in price_data:
        return pd.Series(dtype=float)

    return price_data["Close"].pct_change().dropna()


def calculate_volatility(returns: pd.Series):
    if returns.empty:
        return None

    return round(returns.std() * np.sqrt(252) * 100, 2)


def calculate_max_drawdown(price_data: pd.DataFrame):
    if price_data.empty or "Close" not in price_data:
        return None

    prices = price_data["Close"]
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max

    return round(drawdown.min() * 100, 2)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04):
    if returns.empty or returns.std() == 0:
        return None

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    return round((excess_returns.mean() / returns.std()) * np.sqrt(252), 2)


def calculate_revenue_growth(financials: pd.DataFrame):
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


def score_interpretation(score: int) -> str:
    if score >= 75:
        return "Strong Watch"
    if score >= 55:
        return "Neutral"
    return "Needs Research"


def score_tooltip() -> str:
    return (
        "Score is rule-based out of 100. "
        "It adds points for lower trailing and forward P/E, stronger ROE, better profit margin, "
        "lower debt-to-equity, positive revenue growth, lower volatility, smaller max drawdown, "
        "and positive Sharpe ratio. It is educational only and not investment advice."
    )