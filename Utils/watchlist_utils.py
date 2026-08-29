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
    calculate_returns,
    calculate_revenue_growth,
    calculate_score,
    calculate_sharpe_ratio,
    calculate_volatility,
    score_interpretation,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"


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
