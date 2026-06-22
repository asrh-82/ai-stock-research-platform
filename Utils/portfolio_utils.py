import json
from pathlib import Path

import pandas as pd

from Utils.data_utils import (
    format_large_number,
    get_financials_cached,
    get_live_price_from_info,
    get_price_history_cached,
    get_stock_info,
)

from Utils.scoring import (
    calculate_max_drawdown,
    calculate_revenue_growth,
    calculate_returns,
    calculate_score,
    calculate_sharpe_ratio,
    calculate_volatility,
    score_interpretation,
)


BASE_DIR = Path(__file__).resolve().parent.parent

if (BASE_DIR / "Data").exists():
    DATA_DIR = BASE_DIR / "Data"
else:
    DATA_DIR = BASE_DIR / "data"

WATCHLIST_FILE = DATA_DIR / "watchlist.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"


def save_json_file(file_path, data):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
        file.write("\n")


def load_json_file(file_path, default_value):
    file_path = Path(file_path)

    if not file_path.exists():
        save_json_file(file_path, default_value)
        return default_value

    try:
        content = file_path.read_text().strip()

        if not content:
            save_json_file(file_path, default_value)
            return default_value

        return json.loads(content)

    except json.JSONDecodeError:
        broken_file = file_path.with_suffix(file_path.suffix + ".broken")
        file_path.rename(broken_file)
        save_json_file(file_path, default_value)
        return default_value


def load_watchlist():
    return load_json_file(WATCHLIST_FILE, [])


def save_watchlist(watchlist):
    save_json_file(WATCHLIST_FILE, watchlist)


def load_portfolio():
    return load_json_file(PORTFOLIO_FILE, [])


def save_portfolio(portfolio):
    save_json_file(PORTFOLIO_FILE, portfolio)


def add_to_watchlist(ticker: str) -> bool:
    ticker = ticker.strip().upper()
    watchlist = load_watchlist()

    if ticker and ticker not in watchlist:
        watchlist.append(ticker)
        save_watchlist(watchlist)
        return True

    return False


def remove_from_watchlist(ticker: str) -> None:
    ticker = ticker.strip().upper()
    watchlist = load_watchlist()
    watchlist = [
        symbol for symbol in watchlist
        if symbol.strip().upper() != ticker
    ]
    save_watchlist(watchlist)


def build_watchlist_df(watchlist) -> pd.DataFrame:
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


def add_or_update_holding(portfolio, ticker, shares, avg_cost):
    ticker = ticker.strip().upper()

    if not ticker or shares <= 0 or avg_cost <= 0:
        return portfolio, False

    for holding in portfolio:
        if holding["ticker"].strip().upper() == ticker:
            old_shares = float(holding["shares"])
            old_avg_cost = float(holding["avg_cost"])
            total_shares = old_shares + shares

            weighted_avg_cost = (
                (old_shares * old_avg_cost) + (shares * avg_cost)
            ) / total_shares

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


def remove_portfolio_holding(ticker_to_remove: str) -> None:
    ticker_to_remove = ticker_to_remove.strip().upper()
    portfolio = load_portfolio()

    portfolio = [
        holding
        for holding in portfolio
        if holding.get("ticker", "").strip().upper() != ticker_to_remove
    ]

    save_portfolio(portfolio)


def calculate_portfolio_data(portfolio) -> pd.DataFrame:
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


def calculate_portfolio_summary(portfolio_df: pd.DataFrame):
    if portfolio_df.empty:
        return 0, 0, 0

    total_value = portfolio_df["Current Value"].sum()
    total_invested = portfolio_df["Invested Value"].sum()
    total_gain_loss = total_value - total_invested
    total_gain_loss_pct = (
        (total_gain_loss / total_invested) * 100
        if total_invested
        else 0
    )

    return round(total_value, 2), round(total_gain_loss, 2), round(total_gain_loss_pct, 2)


def get_portfolio_highlights(portfolio_df: pd.DataFrame):
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
        largest_sector = (
            f"{largest_sector_row['Sector']} "
            f"(${largest_sector_row['Current Value']:,.2f})"
        )

    return {
        "best": f"{best_row['Ticker']} ({best_row['Gain/Loss %']}%)",
        "worst": f"{worst_row['Ticker']} ({worst_row['Gain/Loss %']}%)",
        "largest": f"{largest_row['Ticker']} (${largest_row['Current Value']:,.2f})",
        "largest_sector": largest_sector,
    }