"""Research-only OHLC replay. No broker, live orders, or edge certification."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from io import StringIO
from hashlib import sha256
from math import floor, isfinite
from numbers import Real

import numpy as np
import pandas as pd

ENGINE_VERSION = "0.1.0"
BAR_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "Signal")
TRADE_COLUMNS = ("Signal Time", "Entry Time", "Exit Time", "Shares", "Entry Price",
                 "Exit Price", "Net PnL", "Fees", "Reason")


def _finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


@dataclass(frozen=True)
class ReplayConfig:
    initial_cash: float = 10_000.0
    entry_fraction: float = 0.25
    commission_bps: float = 5.0
    slippage_bps: float = 10.0
    prior_volume_fraction: float = 0.01
    execution_delay: int = 1
    stop_loss: float | None = None
    profit_target: float | None = None
    halt_drawdown: float | None = 0.15

    def validate(self) -> None:
        values = (self.initial_cash, self.entry_fraction, self.commission_bps,
                  self.slippage_bps, self.prior_volume_fraction)
        if not all(_finite(x) for x in values):
            raise ValueError("Configuration must contain finite numbers.")
        if self.initial_cash <= 0 or not 0 < self.entry_fraction <= 1:
            raise ValueError("Cash must be positive; entry fraction must be in (0, 1].")
        if not 0 <= self.commission_bps <= 1000 or not 0 <= self.slippage_bps <= 1000:
            raise ValueError("Per-side costs must be between 0 and 1000 basis points.")
        if not 0 < self.prior_volume_fraction <= 1:
            raise ValueError("Prior-volume fraction must be in (0, 1].")
        if isinstance(self.execution_delay, bool) or not isinstance(self.execution_delay, int):
            raise ValueError("Execution delay must be an integer.")
        if not 1 <= self.execution_delay <= 100:
            raise ValueError("Execution delay must be between 1 and 100 bars.")
        for name, value in (("stop_loss", self.stop_loss),
                            ("profit_target", self.profit_target),
                            ("halt_drawdown", self.halt_drawdown)):
            if value is not None and (not _finite(value) or not 0 < value < 1):
                raise ValueError(f"{name} must be None or a finite fraction in (0, 1).")


@dataclass
class ReplayResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    manifest: dict
    warnings: tuple[str, ...]


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Fail closed rather than silently sort, deduplicate, or fill prices."""
    if not isinstance(bars, pd.DataFrame) or len(bars) < 3:
        raise ValueError("Provide a DataFrame with at least three bars.")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.hasnans:
        raise ValueError("Use a valid DatetimeIndex of bar-open timestamps.")
    if bars.index.tz is None:
        raise ValueError("Timestamps must specify a timezone.")
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and strictly increasing.")
    if bars.columns.has_duplicates:
        raise ValueError("Duplicate columns are not allowed.")
    for column in ("Symbol", "Ticker"):
        if column in bars and bars[column].nunique(dropna=False) != 1:
            raise ValueError("Replay accepts exactly one asset at a time.")
    missing = set(BAR_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}.")
    try:
        clean = bars.loc[:, list(BAR_COLUMNS)].astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Bar values and signals must be numeric.") from exc
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Missing and infinite values are not accepted.")
    if (clean[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive.")
    if (clean.Volume < 0).any() or not clean.Signal.isin([0, 1]).all():
        raise ValueError("Volume must be nonnegative; Signal must be 0 (cash) or 1 (long).")
    if ((clean.High < clean[["Open", "Close", "Low"]].max(axis=1)).any()
            or (clean.Low > clean[["Open", "Close", "High"]].min(axis=1)).any()):
        raise ValueError("Inconsistent OHLC bounds.")
    return clean


def parse_replay_csv(payload: bytes) -> pd.DataFrame:
    """Validate a bounded UTF-8 CSV without repairing or publishing its contents."""
    if not isinstance(payload, bytes) or len(payload) > 10 * 1024 * 1024:
        raise ValueError("Provide a UTF-8 CSV no larger than 10 MiB.")
    try:
        text = payload.decode("utf-8-sig")
        header = next(csv.reader(StringIO(text)))
    except (UnicodeError, StopIteration, csv.Error) as exc:
        raise ValueError("The CSV is empty or is not valid UTF-8.") from exc
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV headers are not allowed.")
    try:
        data = pd.read_csv(StringIO(text), nrows=100_001)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("The CSV could not be parsed.") from exc
    if "Timestamp" not in data or len(data) > 100_000:
        raise ValueError("Include Timestamp and no more than 100,000 bars.")
    timestamps = data.pop("Timestamp").astype(str).str.strip()
    if not timestamps.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
        raise ValueError("Every timestamp needs Z or an explicit UTC offset.")
    try:
        data.index = pd.to_datetime(timestamps, utc=True, format="ISO8601", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Use valid ISO 8601 timestamps with UTC offsets.") from exc
    validate_bars(data)
    return data


def replay(bars: pd.DataFrame, config: ReplayConfig | None = None) -> ReplayResult:
    """Replay long/cash target signals known ONLY after their bar closes.

    Timestamps label bar opens. Signal[t] can first fill at Open[t+delay].
    Brackets are price-triggered market-fill proxies, not exact limit orders.
    Dual-touch bars resolve stop-first; opening gaps take precedence.
    Entry capacity uses previous-bar volume, not current/future total volume.
    All exits assume full fills on nonzero-volume bars. Constant long signals
    permit re-entry on a later bar after a bracket exit. No same-bar re-entry.
    """
    config = config or ReplayConfig()
    config.validate()
    data = validate_bars(bars)
    fee_rate, slip = config.commission_bps / 10_000, config.slippage_bps / 10_000
    cash, peak = float(config.initial_cash), float(config.initial_cash)
    shares = 0
    entry_price = entry_cost = entry_fee = 0.0
    entry_time = signal_time = None
    halted = False
    fees = slippage = 0.0
    ambiguous = rejected = 0
    records, ledger = [], []
    array = data.to_numpy()

    def sell(time, reference: float, reason: str) -> None:
        nonlocal cash, shares, fees, slippage
        price = reference * (1 - slip)
        charge = shares * price * fee_rate
        proceeds = shares * price - charge
        ledger.append((signal_time, entry_time, time, shares, entry_price, price,
                       proceeds - entry_cost, entry_fee + charge, reason))
        cash += proceeds
        fees += charge
        slippage += shares * reference * slip
        shares = 0

    for i, (time, row) in enumerate(zip(data.index, array)):
        opening, high, low, close, volume, _ = row
        desired = int(array[i - config.execution_delay, 5]) if i >= config.execution_delay else 0
        exited = False
        # A zero-volume bar cannot generate any simulated fill, including a stop.
        if volume > 0:
            if shares and (halted or desired == 0):
                sell(time, opening, "drawdown halt" if halted else "signal")
                exited = True
            if not shares and desired and not halted and not exited:
                price = opening * (1 + slip)
                budget = min(cash, cash * config.entry_fraction)
                capacity = floor(array[i - 1, 4] * config.prior_volume_fraction) if i else 0
                quantity = min(floor(budget / (price * (1 + fee_rate))), capacity)
                if quantity > 0:
                    shares = quantity
                    entry_price = price
                    entry_fee = shares * price * fee_rate
                    entry_cost = shares * price + entry_fee
                    cash -= entry_cost
                    fees += entry_fee
                    slippage += shares * opening * slip
                    entry_time = time
                    signal_time = data.index[i - config.execution_delay]
                else:
                    rejected += 1
            if shares:
                stop = entry_price * (1 - config.stop_loss) if config.stop_loss else None
                target = entry_price * (1 + config.profit_target) if config.profit_target else None
                if stop is not None and opening <= stop:
                    sell(time, opening, "stop gap")
                elif target is not None and opening >= target:
                    sell(time, opening, "target gap")
                else:
                    hit_stop = stop is not None and low <= stop
                    hit_target = target is not None and high >= target
                    if hit_stop and hit_target:
                        ambiguous += 1
                    if hit_stop:
                        sell(time, stop, "stop-first ambiguous" if hit_target else "stop")
                    elif hit_target:
                        sell(time, target, "target")
        equity = cash + shares * close
        if not isfinite(cash) or not isfinite(equity):
            raise ArithmeticError("Numerical overflow; inspect price and volume scale.")
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        if config.halt_drawdown is not None and drawdown <= -config.halt_drawdown:
            halted = True  # Exit at NEXT tradable open; further losses remain possible.
        if cash < -1e-7:
            raise ArithmeticError("Replay violated the no-borrowing invariant.")
        records.append((cash, shares, equity, drawdown, halted))

    curve = pd.DataFrame(records, index=data.index,
                         columns=["Cash", "Shares", "Equity", "Drawdown", "Halted"])
    trades = pd.DataFrame(ledger, columns=TRADE_COLUMNS)
    final = float(curve.Equity.iloc[-1])
    liquidation = cash + shares * array[-1, 3] * (1 - slip) * (1 - fee_rate)
    losses = -float(trades.loc[trades["Net PnL"] < 0, "Net PnL"].sum())
    profits = float(trades.loc[trades["Net PnL"] > 0, "Net PnL"].sum())
    metrics = {
        "ending_equity": final, "net_return": final / config.initial_cash - 1,
        "estimated_liquidation_equity": float(liquidation),
        "maximum_drawdown": float(curve.Drawdown.min()),
        "fees_paid": float(fees), "modeled_slippage": float(slippage),
        "completed_trades": len(trades), "open_shares": int(shares),
        "open_position_pnl": float(shares * array[-1, 3] - entry_cost) if shares else 0.0,
        "win_rate": float((trades["Net PnL"] > 0).mean()) if len(trades) else None,
        "profit_factor": profits / losses if losses > 0 else None,
        "ambiguous_bars": ambiguous, "rejected_entries": rejected, "halted": halted,
    }
    manifest = {
        "engine_version": ENGINE_VERSION, "status": "NOT_VALIDATED",
        "data_sha256": sha256(data.to_csv(float_format="%.17g").encode()).hexdigest(),
        "config": asdict(config), "observations": len(data),
        "start": data.index[0].isoformat(), "end": data.index[-1].isoformat(),
        "median_spacing_seconds": float(data.index.to_series().diff().dt.total_seconds().median()),
        "timestamp_semantics": "bar-open; signal known and equity marked after bar close",
        "signal_provenance": "caller-supplied; causality not established by replay",
    }
    warnings = (
        "Simulation only. Profitable output and passing software tests do not establish an edge.",
        "Costs are assumptions. No quotes, order book, queue, latency, settlement, or broker model.",
        "Prior-volume caps apply to entries only; full exits on tradable bars are assumed.",
        "Brackets activate immediately after entry; ambiguous bars resolve stop-first.",
        "Open positions are marked, not liquidated. Estimated liquidation is hypothetical.",
        "Drawdown halt is a next-bar rule, not a maximum-loss guarantee. Intrabar drawdown is not measured.",
        "Use consistently adjusted OHLC and volume; corporate actions and dividends are not separately modeled.",
        "Historical universe, delistings, signal causality, and out-of-sample status need independent evidence.",
    )
    return ReplayResult(curve, trades, metrics, manifest, warnings)


def arithmetic_break_even(profit: float, loss: float, round_trip_cost: float = 0.0) -> float:
    """Fixed two-outcome arithmetic, not a forecast. All inputs are fractions.

    A result >= 1 means no positive expectancy even with a 100% win rate.
    Actual outcomes, gaps, costs, and geometric growth can differ substantially.
    """
    if not all(_finite(x) for x in (profit, loss, round_trip_cost)):
        raise ValueError("Payoff inputs must be finite.")
    if profit <= 0 or loss <= 0 or round_trip_cost < 0:
        raise ValueError("Profit/loss must be positive and costs nonnegative.")
    return (loss + round_trip_cost) / (profit + loss)


def expanding_splits(n: int, train: int, test: int, gap: int = 1) -> list[dict]:
    """Half-open chronological folds, not automated walk-forward model fitting.

    Fit every preprocessing/model step on training rows only. The gap must cover
    the full label horizon when labels overlap. Trailing partial folds are unused.
    """
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (n, train, test, gap)):
        raise ValueError("Split lengths must be integers.")
    if min(n, train, test) < 1 or gap < 1 or train + gap + test > n:
        raise ValueError("Insufficient rows or invalid train/test/gap lengths.")
    return [{"train_start": 0, "train_stop": end,
             "test_start": end + gap, "test_stop": end + gap + test}
            for end in range(train, n - gap - test + 1, test)]


def bootstrap_returns(returns, block: int = 5, paths: int = 2000, seed: int = 42) -> dict:
    """Circular block resampling of supplied daily NET returns; not a forecast."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("Provide at least 20 finite daily returns.")
    if (values <= -1).any():
        raise ValueError("Returns must be greater than -100%.")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (block, paths, seed)):
        raise ValueError("Block, paths, and seed must be integers.")
    if not 1 <= block <= len(values) or not 100 <= paths <= 20_000 or seed < 0:
        raise ValueError("Invalid block, paths, or seed.")
    if paths * (len(values) + block) > 5_000_000:
        raise ValueError("Reduce paths or sample length; resampling memory limit exceeded.")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(paths, (len(values) + block - 1) // block))
    indices = (starts[:, :, None] + np.arange(block)) % len(values)
    sampled = values[indices.reshape(paths, -1)[:, :len(values)]]
    log_equity = np.cumsum(np.log1p(sampled), axis=1)
    peaks = np.maximum.accumulate(np.maximum(log_equity, 0), axis=1)
    with np.errstate(over="raise"):
        try:
            total = np.expm1(log_equity[:, -1])
        except FloatingPointError as exc:
            raise ValueError("Return scale produces numerical overflow.") from exc
    drawdowns = np.expm1(log_equity - peaks).min(axis=1)
    return {
        "method": "circular block bootstrap of supplied daily net returns",
        "interpretation": "conditional resampling uncertainty; NOT future-return probabilities",
        "seed": seed, "paths": paths, "block_days": block, "sample_days": len(values),
        "return_p05": float(np.quantile(total, .05)),
        "return_median": float(np.median(total)), "return_p95": float(np.quantile(total, .95)),
        "drawdown_p05": float(np.quantile(drawdowns, .05)),
        "loss_fraction_in_resamples": float((total < 0).mean()),
    }


def synthetic_fixture(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Deterministic software fixture: NOT real prices and NOT strategy evidence."""
    if isinstance(n, bool) or not isinstance(n, int) or not 3 <= n <= 100_000:
        raise ValueError("Fixture length must be an integer between 3 and 100,000.")
    rng = np.random.default_rng(seed)
    opening = 100 * np.exp(np.cumsum(rng.normal(0, .012, n)))
    close = opening * np.exp(rng.normal(0, .007, n))
    return pd.DataFrame({
        "Open": opening, "High": np.maximum(opening, close) * 1.005,
        "Low": np.minimum(opening, close) * .995, "Close": close,
        "Volume": np.full(n, 100_000), "Signal": ((np.arange(n) // 8) % 2).astype(float),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC", name="Timestamp"))
