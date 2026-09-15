"""Bounded daily adjusted-price research snapshots. No live trading interface."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MAX_ROWS = 8000


def validate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not 3 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"Provide 3 to {MAX_ROWS} daily observations.")
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.hasnans or idx.tz is not None:
        raise ValueError("Use timezone-free daily session dates, not intraday timestamps.")
    if idx.has_duplicates or not idx.is_monotonic_increasing or not idx.equals(idx.normalize()):
        raise ValueError("Session dates must be unique, ordered, and date-only.")
    if float(idx.to_series().diff().dt.days.median()) > 4:
        raise ValueError("This page requires daily data, not weekly or monthly observations.")
    if frame.columns.has_duplicates or not set(COLUMNS).issubset(frame.columns):
        raise ValueError("Provide unique Open, High, Low, Close, Volume columns.")
    if "Signal" in frame:
        raise ValueError("Research generates its own causal signals; remove the Signal column.")
    for name in ("Symbol", "Ticker"):
        if name in frame and frame[name].nunique(dropna=False) != 1:
            raise ValueError("Only one asset per snapshot is supported.")
    try:
        data = frame[COLUMNS].astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("OHLCV values must be numeric.") from exc
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Missing or infinite observations must be investigated, not silently filled.")
    if (data[COLUMNS[:4]] <= 0).any().any() or (data.Volume < 0).any():
        raise ValueError("Prices must be positive; volume must be nonnegative.")
    high_error = (data[["Open", "Low", "Close"]].max(axis=1) - data.High).clip(lower=0)
    low_error = (data.Low - data[["Open", "High", "Close"]].min(axis=1)).clip(lower=0)
    invalid = (high_error > 0) | (low_error > 0)
    if invalid.any():
        relative_error = np.maximum(high_error, low_error) / data[COLUMNS[:4]].max(axis=1)
        first_bad_date = data.index[invalid][0].date()
        raise ValueError(
            f"Inconsistent OHLC bounds: {int(invalid.sum())} rows; "
            f"first session {first_bad_date}; maximum relative violation "
            f"{float(relative_error.max()):.6g}. No rows were repaired or dropped."
        )
    data.index.name = "Date"
    return data


def data_hash(data: pd.DataFrame) -> str:
    return sha256(data.to_csv(float_format="%.17g").encode()).hexdigest()


@dataclass
class DailySnapshot:
    prices: pd.DataFrame
    metadata: dict

    def manifest(self) -> dict:
        data = validate_daily(self.prices)
        gaps = data.index.to_series().diff().dt.days
        return {**self.metadata, "data_sha256": data_hash(data), "rows": len(data),
                "first_session": str(data.index[0].date()),
                "last_session": str(data.index[-1].date()),
                "zero_volume_sessions": int((data.Volume == 0).sum()),
                "gaps_over_seven_calendar_days": int((gaps > 7).sum()),
                "calendar_completeness": "not independently verified",
                "units": "fractional adjusted research units, not executable historical shares"}


def parse_daily_csv(payload: bytes, adjusted_confirmed: bool = False) -> DailySnapshot:
    if adjusted_confirmed is not True:
        raise ValueError("Confirm consistent adjusted OHLC before using this research engine.")
    if not isinstance(payload, bytes) or len(payload) > 10 * 1024 * 1024:
        raise ValueError("Provide a UTF-8 CSV no larger than 10 MiB.")
    try:
        text = payload.decode("utf-8-sig")
        headers = next(csv.reader(io.StringIO(text)))
        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate CSV headers are not accepted.")
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1, float_precision="round_trip")
    except (UnicodeError, StopIteration, csv.Error, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise ValueError("Could not parse the UTF-8 CSV.") from exc
    if "Date" not in frame:
        raise ValueError("Include a Date column formatted YYYY-MM-DD.")
    dates = frame.pop("Date").astype(str)
    if not dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Date must contain daily YYYY-MM-DD session labels.")
    frame.index = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    return DailySnapshot(validate_daily(frame), {
        "source": "user_upload", "adjustment": "user asserts consistently adjusted OHLC",
        "interval": "1d", "provenance_verified": False})


def download_daily(symbol: str, start: str, end: str) -> DailySnapshot:
    """An explicit request only; never fetch on import. End date is exclusive."""
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=-]{0,14}", symbol):
        raise ValueError("Provide one valid ticker, not a ticker list or URL.")
    try:
        first, last = (datetime.strptime(s, "%Y-%m-%d").date() for s in (start, end))
    except (ValueError, TypeError) as exc:
        raise ValueError("Dates must be YYYY-MM-DD.") from exc
    if first >= last or (last - first).days > 10957:
        raise ValueError("Choose an ordered range no longer than 30 years.")
    if last > datetime.now(timezone.utc).date():
        raise ValueError("Exclude incomplete/future sessions; end must be today or earlier.")
    import yfinance as yf
    try:
        raw = yf.Ticker(symbol).history(start=start, end=end, interval="1d",
                                       auto_adjust=True, back_adjust=False, repair=False,
                                       actions=False, keepna=True, timeout=15,
                                       raise_errors=True)
    except Exception as exc:
        raise ValueError(f"Market-data request failed ({type(exc).__name__}); no synthetic fallback.") from exc
    if raw is None or raw.empty:
        raise ValueError("Provider returned no history; no synthetic fallback.")
    provider_tz = str(raw.index.tz)
    frame = raw[COLUMNS].copy()
    # Provider daily labels are session dates, not actual midnight executions.
    frame.index = raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    prices = validate_daily(frame)
    if prices.index[0].date() < first or prices.index[-1].date() >= last:
        raise ValueError("Provider data falls outside the requested date range.")
    return DailySnapshot(prices, {
        "source": "Yahoo Finance via yfinance", "symbol": symbol,
        "provider_version": yf.__version__, "interval": "1d",
        "requested_start": start, "requested_end_exclusive": end,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_timezone": provider_tz,
        "adjustment": "vendor auto_adjust=True on OHLC; volume as returned",
        "start_coverage_gap_days": (prices.index[0].date() - first).days,
        "end_coverage_gap_days": (last - prices.index[-1].date()).days,
        "provenance_verified": False,
        "license_note": "Review Yahoo terms; this research interface grants no redistribution rights."})


def demo_snapshot(n: int = 1260, seed: int = 19) -> DailySnapshot:
    """Synthetic software fixture, never historical strategy evidence."""
    if not isinstance(n, int) or not 3 <= n <= MAX_ROWS:
        raise ValueError("Invalid fixture length.")
    rng = np.random.default_rng(seed)
    opening = 100 * np.exp(np.cumsum(rng.normal(.0001, .011, n)))
    close = opening * np.exp(rng.normal(0, .006, n))
    data = pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) * 1.01,
                         "Low": np.minimum(opening, close) * .99, "Close": close,
                         "Volume": np.full(n, 1_000_000)},
                        index=pd.date_range("2010-01-01", periods=n, freq="B", name="Date"))
    return DailySnapshot(data, {"source": "SYNTHETIC SOFTWARE FIXTURE", "seed": seed,
                                "interval": "1d", "provenance_verified": False})
