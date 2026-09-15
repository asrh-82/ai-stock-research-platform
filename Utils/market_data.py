"""Bounded, explicit daily-data access for US equity/ETF market simulations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Utils.market_monte_carlo import price_digest, validate_prices

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    prices: pd.Series
    reference_close: float
    provenance: dict
    warnings: tuple[str, ...] = ()


def normalize_symbol(symbol: str) -> str:
    result = str(symbol).strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,14}", result):
        raise ValueError("Enter one equity or ETF ticker, such as PLTR or USO.")
    return result


def snapshot_from_history(symbol: str, raw: pd.DataFrame, metadata: dict,
                          captured: datetime, start_date: str, provider_version: str) -> MarketSnapshot:
    """Validate the provider response; never fabricate, fill, repair, or substitute prices."""
    symbol = normalize_symbol(symbol)
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("Retrieval timestamp must be timezone aware.")
    if str(metadata.get("symbol", "")).upper() != symbol:
        raise ValueError("Provider security identity did not match the requested ticker.")
    if metadata.get("instrumentType") not in {"EQUITY", "ETF"}:
        raise ValueError("Only listed equities and ETFs are supported by this daily market model.")
    if metadata.get("currency") != "USD" or metadata.get("exchangeTimezoneName") != "America/New_York":
        raise ValueError("This version requires USD equities/ETFs with US Eastern trading sessions.")
    if raw is None or raw.empty or not {"Close", "Adj Close"} <= set(raw.columns):
        raise ValueError("Both reported Close and Adj Close are required. No fallback price series was used.")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("Provider did not return daily session dates.")
    frame = raw.copy()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(NY).tz_localize(None)
    end_date = captured.astimezone(NY).date()
    if frame.index.min() < pd.Timestamp(start_date) or frame.index.max() >= pd.Timestamp(end_date):
        raise ValueError("Provider returned dates outside the request; the entire current NY day is excluded.")
    prices = validate_prices(frame["Adj Close"])
    reported = frame["Close"].to_numpy(dtype=float)
    if not np.isfinite(reported).all() or (reported <= 0).any():
        raise ValueError("Reported closes contain invalid prices.")
    age_days = (end_date - prices.index[-1].date()).days
    if age_days > 7:
        raise ValueError(f"Stale history: last session was {prices.index[-1].date()}. No current forecast generated.")
    warnings = [
        "Provider-adjusted history is not independently audited or point-in-time certified.",
        "No prices were filled or repaired. Exchange-calendar completeness is not independently verified.",
    ]
    if age_days > 4:
        warnings.append("The reference close is more than four calendar days old; verify the last trading session.")
    return MarketSnapshot(symbol, prices, float(reported[-1]), {
        "source": "Yahoo Finance via yfinance", "provider_version": provider_version,
        "symbol": symbol, "instrument_type": metadata["instrumentType"], "currency": "USD",
        "exchange_timezone": "America/New_York", "retrieved_at": captured.isoformat(),
        "requested_start": start_date, "requested_end_exclusive": str(end_date),
        "as_of_session": str(prices.index[-1].date()), "reference_close": float(reported[-1]),
        "return_basis": "Yahoo Adj Close log returns (split and dividend adjusted)",
        "path_basis": "Adjusted-return index rebased at the last reported Close, not a literal quote forecast",
        "history_sha256": price_digest(prices), "provenance_verified": False, "synthetic": False,
    }, tuple(warnings))


def load_market_snapshot(symbol: str, lookback_years: int = 5,
                         now: datetime | None = None) -> MarketSnapshot:
    symbol = normalize_symbol(symbol)
    if isinstance(lookback_years, bool) or lookback_years not in {1, 3, 5, 8}:
        raise ValueError("History window must be 1, 3, 5, or 8 years.")
    captured = now or datetime.now(timezone.utc)
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("Timestamp must be timezone aware.")
    end = captured.astimezone(NY).date()
    start = end - timedelta(days=round(365.25 * lookback_years))
    import yfinance as yf
    security = yf.Ticker(symbol)
    raw = security.history(start=str(start), end=str(end), interval="1d", auto_adjust=False,
                           back_adjust=False, actions=True, repair=False, keepna=True,
                           timeout=15, raise_errors=True)
    # The chart metadata comes from the same history request, not an assumed ticker mapping.
    metadata = security.get_history_metadata()
    return snapshot_from_history(symbol, raw, metadata, captured, str(start), yf.__version__)


def demo_snapshot() -> MarketSnapshot:
    """Explicitly synthetic, offline UI fixture. Never used as a live-data fallback."""
    rng = np.random.Generator(np.random.PCG64(20260915))
    count = 1_261
    index = pd.bdate_range("2021-01-04", periods=count)
    volatility = np.where(np.arange(count - 1) % 240 < 160, 0.012, 0.025)
    returns = volatility * rng.standard_normal(count - 1)
    prices = pd.Series(100 * np.exp(np.r_[0, np.cumsum(returns)]), index=index)
    return MarketSnapshot("SYNTHETIC", prices, float(prices.iloc[-1]), {
        "source": "Seeded synthetic test fixture", "synthetic": True,
        "as_of_session": str(index[-1].date()), "history_sha256": price_digest(prices),
        "return_basis": "Synthetic log returns; no actual security or exchange-calendar claim",
        "provenance_verified": False,
    }, ("SYNTHETIC DEMO: these prices do not represent PLTR, USO, or any real security.",))
