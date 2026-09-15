"""Transparent cross-sectional screening. This module neither predicts nor trades."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from numbers import Real
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Utils.research_data import DailySnapshot, data_hash, validate_daily

VERSION = "selection-1.0"
NY = ZoneInfo("America/New_York")
MAX_SYMBOLS = 25
BENCHMARK = "SPY"
DEFAULT_UNIVERSE = "AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, JNJ, PG, XOM, HD, KO"
FORMULA = "AdjustedClose[t-21] / AdjustedClose[t-252] - 1"


def _portable(value: object):
    # JavaScript serializes 1.0 as 1. Normalize numeric types before hashing so
    # browser storage/JSON backups do not invalidate otherwise identical records.
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        return int(value)
    return value


def canonical(value: object) -> str:
    return json.dumps(_portable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return sha256(canonical(value).encode()).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Use a timezone-aware timestamp.")
    return value.astimezone(timezone.utc)


def parse_universe(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or len(text) > 1500:
        raise ValueError("Provide a comma/space-separated ticker list, not a URL.")
    tokens = [s for s in re.split(r"[,;\s]+", text.strip().upper()) if s]
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", s) for s in tokens):
        raise ValueError("Use ordinary ticker symbols (for example AAPL or BRK-B).")
    symbols = tuple(sorted(set(tokens)))
    if not 2 <= len(symbols) <= MAX_SYMBOLS:
        raise ValueError(f"Choose 2 to {MAX_SYMBOLS} distinct research tickers.")
    return symbols


@dataclass(frozen=True)
class ScreenRules:
    minimum_price: float = 5.0
    minimum_dollar_volume: float = 2_000_000.0

    def validate(self) -> None:
        for x in asdict(self).values():
            if not isinstance(x, Real) or isinstance(x, bool) or not np.isfinite(x) or x < 0:
                raise ValueError("Screening thresholds must be finite nonnegative numbers.")
        if self.minimum_price > 100_000 or self.minimum_dollar_volume > 1e12:
            raise ValueError("Screening threshold exceeds this release's bounds.")


@dataclass
class UniverseData:
    symbols: tuple[str, ...]
    histories: dict[str, DailySnapshot]
    raw_close: dict[str, pd.Series]
    errors: dict[str, str] = field(default_factory=dict)
    synthetic: bool = False
    retrieved_at: str = ""


def scan(batch: UniverseData, rules: ScreenRules | None = None,
         now: datetime | None = None) -> dict:
    """Rank only aligned, complete histories; retain every exclusion and failure."""
    rules = rules or ScreenRules()
    rules.validate()
    created = aware(now or utc_now())
    symbols = parse_universe(",".join(batch.symbols))
    if BENCHMARK not in batch.histories:
        raise ValueError("Reference history is unavailable; scan cannot establish a common session.")
    reference = validate_daily(batch.histories[BENCHMARK].prices)
    asof = reference.index[-1]
    if len(reference) < 253:
        raise ValueError("Reference needs at least 253 complete daily sessions.")
    if not batch.synthetic:
        age = (created.astimezone(NY).date() - asof.date()).days
        if not 1 <= age <= 7:
            raise ValueError("Reference must end before today's New York date and be at most 7 calendar days old.")
    calendar = reference.index[-253:]
    rows, excluded, manifests = [], [], {}
    for symbol in symbols:
        try:
            if symbol in batch.errors:
                raise ValueError(batch.errors[symbol])
            if symbol not in batch.histories:
                raise ValueError("Provider returned no history.")
            prices = validate_daily(batch.histories[symbol].prices)
            prices = prices.loc[prices.index <= asof]
            if len(prices) < 253 or not prices.index[-253:].equals(calendar):
                raise ValueError("Missing, stale, or misaligned sessions in the 253-session signal window.")
            prices = prices.loc[calendar]
            raw = batch.raw_close.get(symbol)
            if raw is None or raw.index.has_duplicates or not calendar.isin(raw.index).all():
                raise ValueError("Reported close missing; price/liquidity screen cannot be computed.")
            reported = pd.to_numeric(raw.reindex(calendar), errors="coerce")
            if not np.isfinite(reported.to_numpy()).all() or (reported <= 0).any():
                raise ValueError("Reported closing prices must be positive and finite.")
            dollar_volume = float((reported.iloc[-20:] * prices.Volume.iloc[-20:]).mean())
            if float(reported.iloc[-1]) < rules.minimum_price:
                raise ValueError("Below the declared reported-price threshold.")
            if dollar_volume < rules.minimum_dollar_volume:
                raise ValueError("Below the declared 20-session average dollar-volume threshold.")
            if (prices.Volume.iloc[-20:] <= 0).any():
                raise ValueError("Zero-volume session in the latest 20 sessions.")
            close = prices.Close
            momentum = float(close.iloc[-22] / close.iloc[-253] - 1)
            returns = close.pct_change(fill_method=None).iloc[-63:]
            volatility = float(returns.std(ddof=1) * np.sqrt(252))
            drawdown = float((close / close.cummax() - 1).min())
            rows.append({"symbol": symbol, "momentum": momentum,
                         "reported_close": float(reported.iloc[-1]),
                         "volatility": volatility, "drawdown": drawdown,
                         "dollar_volume": dollar_volume,
                         "recent_return": float(close.iloc[-1] / close.iloc[-22] - 1),
                         "sector": "Not loaded", "signal_start": str(calendar[0].date()),
                         "signal_end": str(calendar[-22].date())})
            manifests[symbol] = {"prices_sha256": data_hash(prices),
                                 "reported_close_sha256": data_hash(reported.to_frame()),
                                 "source": batch.histories[symbol].metadata.get("source", "Unverified")}
        except (ValueError, TypeError, KeyError) as exc:
            excluded.append({"symbol": symbol, "reason": str(exc)[:350]})
    rows.sort(key=lambda r: (-r["momentum"], r["symbol"]))
    values = pd.Series([r["momentum"] for r in rows], dtype=float)
    percentiles = ((values.rank(method="average") - 1) / (len(values) - 1) * 100
                   if len(values) > 1 else pd.Series([50.0] * len(values)))
    for i, row in enumerate(rows):
        row["rank"] = i + 1
        row["percentile"] = float(percentiles.iloc[i])
    result = {"version": VERSION, "created_at": created.isoformat(),
              "asof": str(asof.date()), "universe": list(symbols),
              "rules": asdict(rules), "formula": FORMULA,
              "synthetic": bool(batch.synthetic), "benchmark": BENCHMARK,
              "ranked": rows, "excluded": excluded, "sources": manifests,
              "retrieved_at": batch.retrieved_at,
              "status": "DEMO_ONLY" if batch.synthetic else "UNVALIDATED_RANKING",
              "universe_basis": "user-defined current universe; historical membership not verified"}
    result["id"] = digest(result)
    return result


def rank_changes(current: dict, previous: dict | None) -> dict[str, int]:
    if previous is None or any(current[k] != previous[k] for k in
                               ("version", "universe", "rules", "synthetic")):
        return {}
    old = {r["symbol"]: r["rank"] for r in previous["ranked"]}
    return {r["symbol"]: old[r["symbol"]] - r["rank"] for r in current["ranked"]
            if r["symbol"] in old}


def correlation_context(batch: UniverseData, symbols: list[str]) -> pd.DataFrame:
    if len(symbols) < 2:
        return pd.DataFrame()
    frames = {s: batch.histories[s].prices.Close.pct_change(fill_method=None).tail(63)
              for s in symbols if s in batch.histories}
    returns = pd.DataFrame(frames)
    return returns.corr(min_periods=40)


def demo_universe() -> tuple[UniverseData, UniverseData]:
    """First object has NO future rows; second supplies synthetic outcome examples."""
    symbols = ("ALFA", "BRAV", "CHAR", "DELT", "ECHO", "FOXT")
    dates = pd.bdate_range("2023-01-03", periods=420, name="Date")
    histories, raw = {}, {}
    for i, symbol in enumerate((*symbols, BENCHMARK)):
        rng = np.random.default_rng(100 + i)
        close = (55 + i * 9) * np.exp(np.cumsum(rng.normal((i - 2) * .0005, .014, len(dates))))
        opening = close * np.exp(rng.normal(0, .003, len(dates)))
        frame = pd.DataFrame({"Open": opening, "High": np.maximum(close, opening) * 1.01,
                              "Low": np.minimum(close, opening) * .99, "Close": close,
                              "Volume": np.full(len(dates), 1_000_000.)}, index=dates)
        histories[symbol] = DailySnapshot(frame, {"source": "SYNTHETIC SOFTWARE FIXTURE"})
        raw[symbol] = frame.Close.copy()
    full = UniverseData(symbols, histories, raw, synthetic=True, retrieved_at="synthetic fixture")
    prefix = UniverseData(symbols,
                          {s: DailySnapshot(h.prices.iloc[:-42].copy(), dict(h.metadata))
                           for s, h in histories.items()},
                          {s: r.iloc[:-42].copy() for s, r in raw.items()},
                          synthetic=True, retrieved_at="synthetic fixture")
    return prefix, full
