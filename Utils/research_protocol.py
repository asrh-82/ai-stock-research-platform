"""Chronological daily research with frozen choices and a separately opened holdout.

This is normalized adjusted-unit research, NOT the whole-share execution lab.
No trained ML model, minute strategy, broker orders, or profitability certification.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from math import isfinite, sqrt
from numbers import Real

import numpy as np
import pandas as pd

from Utils.research_data import DailySnapshot, data_hash, validate_daily

VERSION = "phase10.1"


def _number(x) -> bool:
    return isinstance(x, Real) and not isinstance(x, (bool, np.bool_)) and isfinite(x)


def _integer(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


@dataclass(frozen=True)
class Rule:
    kind: str
    lookback: int = 1
    fast: int = 0
    entry: float = 30.0
    exit: float = 55.0

    @property
    def label(self) -> str:
        if self.kind == "cash":
            return "CASH"
        if self.kind == "sma":
            return f"SMA_{self.fast}_{self.lookback}"
        if self.kind == "momentum":
            return f"MOM_{self.lookback}"
        return f"RSI_{self.lookback}_{self.entry:g}_{self.exit:g}"

    def validate(self) -> None:
        if self.kind not in ("cash", "sma", "momentum", "rsi"):
            raise ValueError("Unsupported strategy family.")
        if not _integer(self.lookback) or not 1 <= self.lookback <= 504:
            raise ValueError("Lookback must be an integer from 1 to 504.")
        if not _integer(self.fast) or not 0 <= self.fast <= 503:
            raise ValueError("Fast window must be an integer from 0 to 503.")
        if self.kind != "cash" and self.lookback < 2:
            raise ValueError("Indicator lookback must be at least two sessions.")
        if self.kind == "sma" and not 2 <= self.fast < self.lookback:
            raise ValueError("SMA requires 2 <= fast < slow.")
        if not all(_number(x) for x in (self.entry, self.exit)) or not 0 <= self.entry < self.exit <= 100:
            raise ValueError("RSI thresholds must satisfy 0 <= entry < exit <= 100.")


DEFAULT_RULES = (Rule("cash"), Rule("sma", 100, 20), Rule("sma", 200, 50),
                 Rule("momentum", 63), Rule("momentum", 126), Rule("momentum", 252),
                 Rule("rsi", 14))


@dataclass(frozen=True)
class Execution:
    initial_cash: float = 10_000.0
    allocation: float = 1.0
    commission_bps: float = 5.0
    slippage_bps: float = 10.0

    def validate(self) -> None:
        if not all(_number(x) for x in asdict(self).values()):
            raise ValueError("Execution assumptions must be finite numbers.")
        if not 0 < self.initial_cash <= 1e9 or not 0 < self.allocation <= 1:
            raise ValueError("Use positive virtual capital and an unlevered allocation in (0, 1].")
        if not 0 <= self.commission_bps <= 1000 or not 0 <= self.slippage_bps <= 1000:
            raise ValueError("Per-side costs must be between 0 and 1000 bps.")


@dataclass(frozen=True)
class Protocol:
    train: int = 504
    test: int = 126
    holdout: int = 252
    gap: int = 1
    rules: tuple[Rule, ...] = DEFAULT_RULES
    execution: Execution = field(default_factory=Execution)

    @property
    def warmup(self) -> int:
        return max(r.lookback for r in self.rules) + 1

    def validate(self, n: int) -> None:
        if not isinstance(self.rules, tuple) or not 1 <= len(self.rules) <= 12:
            raise ValueError("Declare a tuple of 1 to 12 rules before testing.")
        for rule in self.rules:
            if not isinstance(rule, Rule):
                raise ValueError("Each candidate must be a Rule.")
            rule.validate()
        if len({r.label for r in self.rules}) != len(self.rules):
            raise ValueError("Candidate labels must be unique.")
        if sum(r.kind == "cash" for r in self.rules) != 1:
            raise ValueError("Include exactly one cash candidate as a selection floor.")
        if not all(_integer(x) for x in (self.train, self.test, self.holdout, self.gap)):
            raise ValueError("Window sizes must be integers.")
        if self.train < self.warmup + 40 or min(self.test, self.holdout) < 20 or not 1 <= self.gap <= 20:
            raise ValueError("Need 40 training returns after common warmup, 20 test/holdout sessions, and a 1-20 session gap.")
        required = self.train + self.test + self.holdout + 2 * self.gap
        if n < required:
            raise ValueError(f"Protocol needs at least {required} sessions; received {n}.")
        folds = (n - self.holdout - 2 * self.gap - self.train + self.test - 1) // self.test
        if folds > 48:
            raise ValueError("Limit this release to 48 folds; increase the test window.")
        self.execution.validate()
        if self.execution.slippage_bps > 250:
            raise ValueError("Base slippage must not exceed 250 bps to allow the 4x stress.")


def signal(close: pd.Series, rule: Rule) -> pd.Series:
    """Close-of-session signal. Prefix invariance is covered by regression tests."""
    rule.validate()
    if not isinstance(close, pd.Series) or not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
        raise ValueError("Signals require finite positive closes.")
    if rule.kind == "cash":
        return pd.Series(0.0, index=close.index)
    if rule.kind == "sma":
        return (close.rolling(rule.fast).mean() > close.rolling(rule.lookback).mean()).astype(float)
    if rule.kind == "momentum":
        return (close.pct_change(rule.lookback, fill_method=None) > 0).astype(float)
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / rule.lookback, adjust=False, min_periods=rule.lookback).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / rule.lookback, adjust=False, min_periods=rule.lookback).mean()
    rsi = 100 - 100 / (1 + up / down.replace(0, np.nan))
    rsi = rsi.mask((down == 0) & (up > 0), 100).mask((up == 0) & (down > 0), 0)
    rsi = rsi.mask((up == 0) & (down == 0), 50)
    decisions = pd.Series(np.nan, index=close.index)
    decisions.loc[rsi <= rule.entry] = 1.0
    decisions.loc[rsi >= rule.exit] = 0.0
    return decisions.ffill().fillna(0.0)


@dataclass
class Simulation:
    curve: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict


def _metrics_unchecked(equity: pd.Series, initial: float) -> dict:
    values = equity.to_numpy(dtype=float)
    previous = np.r_[initial, values[:-1]]
    returns = values / previous - 1
    peaks = np.maximum.accumulate(np.r_[initial, values])[1:]
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return {"ending_equity": float(values[-1]), "net_return": float(values[-1] / initial - 1),
            "cagr_252_session_basis": float((values[-1] / initial) ** (252 / len(values)) - 1),
            "sharpe_rf_zero": float(returns.mean() / std * sqrt(252)) if std > 1e-12 else None,
            "maximum_drawdown": float(np.min(values / peaks - 1))}


def _metrics(equity: pd.Series, initial: float) -> dict:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = _metrics_unchecked(equity, initial)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Input scale produced unstable research metrics.") from exc
    if any(value is not None and not _number(value) for value in result.values()):
        raise ValueError("Research metrics must remain finite.")
    return result


def _simulate(prices: pd.DataFrame, targets: pd.Series, execution: Execution,
              start: int, stop: int) -> Simulation:
    """Internal: prices already validated; targets are desired exposure AT the open.

    Signals must be lagged by the caller. Fractional adjusted units remove the
    anachronistic whole-share rounding of retrospectively adjusted prices.
    No intraday brackets, spread/depth, volume capacity, settlement, or taxes.
    """
    execution.validate()
    if not 1 <= start < stop <= len(prices) or not targets.index.equals(prices.index):
        raise ValueError("Invalid simulation interval or target alignment.")
    if not targets.isin([0.0, 1.0]).all():
        raise ValueError("Open targets must be exactly cash or long.")
    cash = float(execution.initial_cash)
    units = entry_cost = fees = slippage = 0.0
    slip, rate = execution.slippage_bps / 10_000, execution.commission_bps / 10_000
    rows, orders = [], []
    arr, desired = prices[["Open", "Close", "Volume"]].to_numpy(), targets.to_numpy()
    for i in range(start, stop):
        opening, close, volume = arr[i]
        date, previous_date = prices.index[i], prices.index[i - 1]
        if volume > 0 and units > 0 and desired[i] == 0:
            fill = opening * (1 - slip)
            fee = units * fill * rate
            proceeds = units * fill - fee
            orders.append((date, previous_date, "SELL", units, opening, fill, fee, proceeds, proceeds - entry_cost))
            cash += proceeds
            fees += fee
            slippage += units * opening * slip
            units = entry_cost = 0.0
        elif volume > 0 and units == 0 and desired[i] == 1:
            fill = opening * (1 + slip)
            units = cash * execution.allocation / (fill * (1 + rate))
            fee = units * fill * rate
            entry_cost = units * fill + fee
            cash -= entry_cost
            fees += fee
            slippage += units * opening * slip
            orders.append((date, previous_date, "BUY", units, opening, fill, fee, -entry_cost, 0.0))
        equity = cash + units * close
        if not isfinite(equity) or equity <= 0 or cash < -1e-6:
            raise ArithmeticError("Nonfinite equity or violation of the no-borrowing invariant.")
        rows.append((cash, units, equity, float(desired[i])))
    curve = pd.DataFrame(rows, index=prices.index[start:stop], columns=["Cash", "Adjusted units", "Equity", "Open target"])
    curve["Net return"] = curve.Equity.pct_change(fill_method=None)
    curve.iloc[0, curve.columns.get_loc("Net return")] = curve.Equity.iloc[0] / execution.initial_cash - 1
    peak = np.maximum.accumulate(np.r_[execution.initial_cash, curve.Equity.to_numpy()])[1:]
    curve["Drawdown"] = curve.Equity.to_numpy() / peak - 1
    ledger = pd.DataFrame(orders, columns=["Session", "Signal session", "Side", "Adjusted units",
                                           "Reference open", "Fill", "Fee", "Cash flow", "Realized PnL"])
    metrics = _metrics(curve.Equity, execution.initial_cash)
    liquidation = cash + units * arr[stop - 1, 1] * (1 - slip) * (1 - rate)
    metrics.update({"estimated_liquidation_equity": float(liquidation), "fees": float(fees),
                    "modeled_slippage": float(slippage), "completed_trades": int((ledger.Side == "SELL").sum()),
                    "executions": len(ledger), "open_adjusted_units": float(units),
                    "open_position_pnl": float(units * arr[stop - 1, 1] - entry_cost),
                    "realized_pnl": float(ledger["Realized PnL"].sum()),
                    "fraction_sessions_in_market": float((curve["Adjusted units"] > 0).mean())})
    return Simulation(curve, ledger, metrics)


def _rank(training_prices: pd.DataFrame, protocol: Protocol) -> tuple[Rule, list[dict]]:
    """This function receives ONLY the training prefix, never future prices."""
    scores = []
    for rule in protocol.rules:
        targets = signal(training_prices.Close, rule).shift(1).fillna(0.0)
        result = _simulate(training_prices, targets, protocol.execution, protocol.warmup, len(training_prices))
        sharpe = result.metrics["sharpe_rf_zero"]
        score = 0.0 if rule.kind == "cash" else (sharpe if sharpe is not None else -1e9)
        scores.append({"rule": rule.label, "selection_score": score,
                       "training_net_return": result.metrics["net_return"],
                       "training_maximum_drawdown": result.metrics["maximum_drawdown"],
                       "training_executions": result.metrics["executions"]})
    scores.sort(key=lambda x: (-x["selection_score"], x["rule"]))
    winner = next(r for r in protocol.rules if r.label == scores[0]["rule"])
    return winner, scores


def _digest(payload: dict) -> str:
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _protocol_from_dict(payload: dict) -> Protocol:
    params = dict(payload)
    params["rules"] = tuple(Rule(**r) for r in params["rules"])
    params["execution"] = Execution(**params["execution"])
    return Protocol(**params)


@dataclass
class DevelopmentResult:
    simulation: Simulation
    passive: Simulation
    folds: pd.DataFrame
    training_scores: pd.DataFrame
    open_targets: pd.Series
    plan: dict


def develop(snapshot: DailySnapshot, protocol: Protocol | None = None) -> DevelopmentResult:
    protocol = protocol or Protocol()
    prices = validate_daily(snapshot.prices)
    protocol.validate(len(prices))
    holdout_start = len(prices) - protocol.holdout
    development_stop = holdout_start - protocol.gap
    first_test = protocol.train + protocol.gap
    # No holdout bars reach ranking, indicator fitting, or development simulations.
    development_prices = prices.iloc[:development_stop].copy()
    targets = pd.Series(0.0, index=development_prices.index)
    fold_rows, score_rows = [], []
    for fold, start in enumerate(range(first_test, development_stop, protocol.test), 1):
        stop = min(start + protocol.test, development_stop)
        train_stop = start - protocol.gap
        winner, scores = _rank(development_prices.iloc[:train_stop].copy(), protocol)
        prefix = development_prices.iloc[:stop]
        generated = signal(prefix.Close, winner).shift(1).fillna(0.0)
        targets.iloc[start:stop] = generated.iloc[start:stop].to_numpy()
        fold_rows.append({"fold": fold, "rule": winner.label, "train_stop_exclusive": train_stop,
                          "last_training_session": str(prices.index[train_stop - 1].date()),
                          "test_start": start, "test_stop_exclusive": stop,
                          "first_test_session": str(prices.index[start].date()),
                          "last_test_session": str(prices.index[stop - 1].date()),
                          "test_sessions": stop - start})
        score_rows.extend({"fold": fold, **score} for score in scores)
    simulation = _simulate(development_prices, targets, protocol.execution, first_test, development_stop)
    passive = _simulate(development_prices, pd.Series(1.0, index=development_prices.index),
                        protocol.execution, first_test, development_stop)
    prior, prior_passive = protocol.execution.initial_cash, protocol.execution.initial_cash
    for row in fold_rows:
        last = prices.index[row["test_stop_exclusive"] - 1]
        ending, passive_ending = simulation.curve.loc[last, "Equity"], passive.curve.loc[last, "Equity"]
        row["oos_net_return"] = float(ending / prior - 1)
        row["passive_net_return"] = float(passive_ending / prior_passive - 1)
        prior, prior_passive = ending, passive_ending
    winner, final_scores = _rank(development_prices, protocol)
    plan = {"version": VERSION, "status": "FROZEN_NOT_EVALUATED", "source": snapshot.manifest(),
            "protocol": asdict(protocol), "selected_rule": asdict(winner),
            "selection_metric": "training net daily Sharpe; zero risk-free rate; cash score 0; label tie-break",
            "common_warmup_sessions": protocol.warmup,
            "development_stop_exclusive": development_stop, "holdout_start": holdout_start,
            "holdout_stop_exclusive": len(prices), "final_training_scores": final_scores,
            "holdout_first_session": str(prices.index[holdout_start].date()),
            "holdout_last_session": str(prices.index[-1].date()),
            "certification": "NONE; a hash is reproducibility metadata, not an access-control or preregistration service"}
    plan["plan_sha256"] = _digest(plan)
    return DevelopmentResult(simulation, passive, pd.DataFrame(fold_rows), pd.DataFrame(score_rows), targets, plan)


def evaluate_holdout(snapshot: DailySnapshot, plan: dict) -> tuple[Simulation, Simulation, pd.DataFrame]:
    """Explicit second stage. No parameter selection or refitting occurs here."""
    copied = dict(plan)
    supplied_hash = copied.pop("plan_sha256", None)
    if _digest(copied) != supplied_hash or copied.get("version") != VERSION:
        raise ValueError("The frozen plan was modified or uses a different engine version.")
    prices = validate_daily(snapshot.prices)
    if data_hash(prices) != copied["source"]["data_sha256"]:
        raise ValueError("Snapshot changed after the plan was frozen. Create a separately labeled new experiment.")
    protocol = _protocol_from_dict(copied["protocol"])
    protocol.validate(len(prices))
    winner = Rule(**copied["selected_rule"])
    if winner not in protocol.rules:
        raise ValueError("Frozen rule is not in the declared candidate set.")
    start, stop = copied["holdout_start"], copied["holdout_stop_exclusive"]
    if start != len(prices) - protocol.holdout or stop != len(prices):
        raise ValueError("Frozen holdout boundaries do not match the protocol.")
    targets = signal(prices.Close, winner).shift(1).fillna(0.0)
    simulation = _simulate(prices, targets, protocol.execution, start, stop)
    passive = _simulate(prices, pd.Series(1.0, index=prices.index), protocol.execution, start, stop)
    stress = stress_test(prices, targets, protocol.execution, start, stop)
    return simulation, passive, stress


def stress_test(prices: pd.DataFrame, targets: pd.Series, execution: Execution,
                start: int, stop: int) -> pd.DataFrame:
    """Keep the selected rules and targets fixed; never re-optimize for each cost."""
    rows = []
    for multiplier in (1, 2, 4):
        result = _simulate(prices, targets, replace(execution, slippage_bps=execution.slippage_bps * multiplier), start, stop)
        rows.append({"slippage_multiplier": multiplier, "slippage_bps_per_side": execution.slippage_bps * multiplier,
                     "net_return": result.metrics["net_return"],
                     "maximum_drawdown": result.metrics["maximum_drawdown"],
                     "ending_equity": result.metrics["ending_equity"]})
    return pd.DataFrame(rows)
