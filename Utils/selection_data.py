"""Explicit, bounded provider access. No data downloads occur on import."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from Utils.research_data import DailySnapshot, validate_daily
from Utils.selection import BENCHMARK, NY, UniverseData, aware, parse_universe, utc_now


def fetch_history(symbol: str, start: str, end: str) -> tuple[DailySnapshot, pd.Series]:
    import yfinance as yf
    raw = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=False,
                                   back_adjust=False, actions=False, repair=False,
                                   keepna=True, timeout=15, raise_errors=True)
    if raw is None or raw.empty or "Adj Close" not in raw:
        raise ValueError("Adjusted and reported closes were not both supplied.")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("Provider did not return daily session labels.")
    source_tz = str(raw.index.tz)
    if raw.index.tz is not None:
        raw = raw.copy()
        raw.index = raw.index.tz_localize(None)
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Provider dates are duplicated or unsorted.")
    if raw.index[0] < pd.Timestamp(start) or raw.index[-1] >= pd.Timestamp(end):
        raise ValueError("Provider returned data outside the requested interval.")
    required = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    vals = raw[required].astype(float)
    if not np.isfinite(vals.to_numpy()).all() or (vals.Close <= 0).any() or (vals["Adj Close"] <= 0).any():
        raise ValueError("Nonfinite or invalid provider prices; no rows repaired or filled.")
    ratio = vals["Adj Close"] / vals.Close
    adjusted = vals[["Open", "High", "Low", "Close"]].mul(ratio, axis=0)
    adjusted["Volume"] = vals.Volume
    return DailySnapshot(validate_daily(adjusted), {
        "source": "Yahoo Finance via yfinance", "symbol": symbol, "provider_version": yf.__version__,
        "retrieved_at": utc_now().isoformat(), "source_timezone": source_tz,
        "adjustment": "OHLC multiplied by Adj Close / reported Close; volume as supplied",
        "requested_start": start, "requested_end_exclusive": end,
        "provenance_verified": False}), vals.Close.rename("Reported Close")


def load_universe(text: str, now: datetime | None = None,
                  start: str | None = None) -> UniverseData:
    symbols = parse_universe(text)
    captured = aware(now or utc_now())
    # Always exclude the entire current NY calendar day, even after market close.
    end_date = captured.astimezone(NY).date()
    start_date = end_date - timedelta(days=600) if start is None else pd.Timestamp(start).date()
    if not 1 <= (end_date - start_date).days <= 3650:
        raise ValueError("History request must span 1 day to 10 years.")
    histories, raw, errors = {}, {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(fetch_history, s, str(start_date), str(end_date)): s
                for s in sorted(set(symbols) | {BENCHMARK})}
        for job in as_completed(jobs):
            s = jobs[job]
            try:
                histories[s], raw[s] = job.result()
            except Exception as exc:
                errors[s] = f"Provider/validation failure ({type(exc).__name__}): {str(exc)[:200]}"
    if BENCHMARK not in histories:
        raise ValueError("Reference data unavailable. " + errors.get(BENCHMARK, "No response."))
    return UniverseData(symbols, histories, raw, errors, False, captured.isoformat())
