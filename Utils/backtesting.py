from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
SUPPORTED_STRATEGIES = (
    "SMA Crossover",
    "Price Momentum",
    "RSI Mean Reversion",
    "Buy and Hold",
)
SUPPORTED_REBALANCING = ("Daily", "Weekly", "Monthly")
SUPPORTED_POSITION_MODES = ("Long / Cash", "Long / Short")
SUPPORTED_EVALUATION_MODES = ("Full History", "Holdout")


@dataclass(frozen=True)
class BacktestConfig:
    strategy: str = "SMA Crossover"
    fast_window: int = 50
    slow_window: int = 200
    momentum_lookback: int = 126
    momentum_threshold: float = 0.0
    rsi_window: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 55.0
    rebalancing: str = "Daily"
    position_mode: str = "Long / Cash"
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    risk_free_rate: float = 0.0
    evaluation_mode: str = "Full History"
    holdout_fraction: float = 0.30


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    maximum_drawdown: float


@dataclass(frozen=True)
class BacktestResult:
    performance: pd.DataFrame
    trades: pd.DataFrame
    strategy_metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    annualized_turnover: float
    market_exposure: float
    trade_count: int
    win_rate: float | None
    best_trade: float | None
    worst_trade: float | None
    evaluation_start: pd.Timestamp
    warnings: tuple[str, ...]


def validate_backtest_config(config: BacktestConfig) -> None:
    if config.strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {config.strategy}.")
    if config.rebalancing not in SUPPORTED_REBALANCING:
        raise ValueError(f"Unsupported rebalancing frequency: {config.rebalancing}.")
    if config.position_mode not in SUPPORTED_POSITION_MODES:
        raise ValueError(f"Unsupported position mode: {config.position_mode}.")
    if config.evaluation_mode not in SUPPORTED_EVALUATION_MODES:
        raise ValueError(f"Unsupported evaluation mode: {config.evaluation_mode}.")
    if not 2 <= config.fast_window < config.slow_window:
        raise ValueError("The fast moving average must be shorter than the slow moving average.")
    if config.momentum_lookback < 2:
        raise ValueError("Momentum lookback must be at least two trading days.")
    if config.rsi_window < 2:
        raise ValueError("RSI window must be at least two trading days.")
    if not 0 <= config.rsi_entry < config.rsi_exit <= 100:
        raise ValueError("RSI entry must be below RSI exit, within 0 to 100.")
    if not 0 <= config.transaction_cost_bps <= 1_000:
        raise ValueError("Transaction cost must be between 0 and 1,000 basis points.")
    if not 0 <= config.slippage_bps <= 1_000:
        raise ValueError("Slippage must be between 0 and 1,000 basis points.")
    if not -0.50 <= config.risk_free_rate <= 1:
        raise ValueError("Risk-free rate must be between -50% and 100%.")
    if not 0.10 <= config.holdout_fraction <= 0.80:
        raise ValueError("Holdout fraction must be between 10% and 80%.")


def clean_close_series(series: pd.Series, label: str = "Price") -> pd.Series:
    if series is None or len(series) == 0:
        raise ValueError(f"{label} history is empty.")

    cleaned = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()]
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    cleaned = cleaned[cleaned > 0]

    if len(cleaned) < 3:
        raise ValueError(f"{label} history has too few valid observations.")
    return cleaned


def _relative_strength_index(close: pd.Series, window: int) -> pd.Series:
    changes = close.diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50.0)


def build_strategy_signal(close: pd.Series, config: BacktestConfig) -> pd.Series:
    if config.strategy == "Buy and Hold":
        return pd.Series(1.0, index=close.index, name="Signal")

    if config.strategy == "SMA Crossover":
        fast_average = close.rolling(
            config.fast_window,
            min_periods=config.fast_window,
        ).mean()
        slow_average = close.rolling(
            config.slow_window,
            min_periods=config.slow_window,
        ).mean()
        signal = pd.Series(
            np.where(fast_average > slow_average, 1.0, -1.0),
            index=close.index,
            name="Signal",
        )
        return signal.where(slow_average.notna(), 0.0)

    if config.strategy == "Price Momentum":
        momentum = close.pct_change(config.momentum_lookback, fill_method=None)
        signal = pd.Series(
            np.where(momentum > config.momentum_threshold, 1.0, -1.0),
            index=close.index,
            name="Signal",
        )
        return signal.where(momentum.notna(), 0.0)

    rsi = _relative_strength_index(close, config.rsi_window)
    decisions = pd.Series(np.nan, index=close.index, dtype=float)
    decisions.loc[rsi <= config.rsi_entry] = 1.0
    decisions.loc[rsi >= config.rsi_exit] = -1.0
    return decisions.ffill().fillna(0.0).rename("Signal")


def _apply_rebalancing(signal: pd.Series, frequency: str) -> pd.Series:
    if frequency == "Daily":
        return signal

    period_frequency = "W-FRI" if frequency == "Weekly" else "M"
    periods = signal.index.to_period(period_frequency)
    period_series = pd.Series(periods, index=signal.index)
    rebalance_dates = ~period_series.duplicated(keep="last")
    return signal.where(rebalance_dates).ffill().fillna(0.0)


def _performance_metrics(
    returns: pd.Series,
    risk_free_rate: float,
) -> PerformanceMetrics:
    returns = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if returns.empty:
        raise ValueError("No valid returns were available for performance metrics.")

    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    elapsed_years = max(
        (returns.index[-1] - returns.index[0]).days / 365.25,
        len(returns) / TRADING_DAYS_PER_YEAR,
        1 / TRADING_DAYS_PER_YEAR,
    )
    ending_value = float(equity.iloc[-1])
    cagr = ending_value ** (1 / elapsed_years) - 1 if ending_value > 0 else -1.0

    volatility = float(returns.std(ddof=1) * sqrt(TRADING_DAYS_PER_YEAR))
    daily_risk_free = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = returns - daily_risk_free
    excess_std = float(excess_returns.std(ddof=1))
    sharpe = (
        float(excess_returns.mean() / excess_std * sqrt(TRADING_DAYS_PER_YEAR))
        if excess_std > 0 and isfinite(excess_std)
        else None
    )

    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside_returns))))
        if not downside_returns.empty
        else 0.0
    )
    sortino = (
        float(excess_returns.mean() / downside_deviation * sqrt(TRADING_DAYS_PER_YEAR))
        if downside_deviation > 0 and isfinite(downside_deviation)
        else None
    )

    drawdown = equity / equity.cummax() - 1
    return PerformanceMetrics(
        total_return=total_return,
        cagr=float(cagr),
        annualized_volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        maximum_drawdown=float(drawdown.min()),
    )


def _build_trade_ledger(
    frame: pd.DataFrame,
    cost_rate: float,
) -> pd.DataFrame:
    ledger = []
    position = frame["Position"].astype(float)
    close = frame["Asset Close"].astype(float)
    execution_price = close.shift(1).fillna(close)
    prior_position = 0.0
    entry_date = None
    entry_price = None
    entry_side = None

    for date, current_position in position.items():
        if current_position == prior_position:
            continue

        transition_price = float(execution_price.loc[date])
        if prior_position != 0 and entry_date is not None and entry_price is not None:
            gross_return = (
                transition_price / entry_price - 1
                if prior_position > 0
                else entry_price / transition_price - 1
            )
            net_return = gross_return - (2 * cost_rate)
            ledger.append(
                {
                    "Entry Date": entry_date,
                    "Exit Date": date,
                    "Side": entry_side,
                    "Entry Price": entry_price,
                    "Exit Price": transition_price,
                    "Net Return": net_return,
                    "Bars": int(frame.loc[entry_date:date].shape[0] - 1),
                    "Status": "Closed",
                }
            )

        if current_position != 0:
            entry_date = date
            entry_price = transition_price
            entry_side = "Long" if current_position > 0 else "Short"
        else:
            entry_date = None
            entry_price = None
            entry_side = None

        prior_position = current_position

    if prior_position != 0 and entry_date is not None and entry_price is not None:
        final_date = frame.index[-1]
        final_price = float(close.iloc[-1])
        gross_return = (
            final_price / entry_price - 1
            if prior_position > 0
            else entry_price / final_price - 1
        )
        ledger.append(
            {
                "Entry Date": entry_date,
                "Exit Date": pd.NaT,
                "Side": entry_side,
                "Entry Price": entry_price,
                "Exit Price": final_price,
                "Net Return": gross_return - cost_rate,
                "Bars": int(frame.loc[entry_date:final_date].shape[0] - 1),
                "Status": "Open",
            }
        )

    return pd.DataFrame(ledger)


def run_backtest(
    asset_close: pd.Series,
    benchmark_close: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    validate_backtest_config(config)

    asset = clean_close_series(asset_close, "Asset price")
    benchmark = clean_close_series(benchmark_close, "Benchmark price")
    prices = pd.concat(
        [asset.rename("Asset Close"), benchmark.rename("Benchmark Close")],
        axis=1,
        join="inner",
    ).dropna()

    minimum_history = 30
    warmup_observations = 1
    if config.strategy == "SMA Crossover":
        minimum_history = config.slow_window + 20
        warmup_observations = config.slow_window
    elif config.strategy == "Price Momentum":
        minimum_history = config.momentum_lookback + 20
        warmup_observations = config.momentum_lookback + 1
    elif config.strategy == "RSI Mean Reversion":
        minimum_history = config.rsi_window + 20
        warmup_observations = config.rsi_window + 1
    if len(prices) < minimum_history:
        raise ValueError(
            f"This strategy needs at least {minimum_history} aligned price observations; "
            f"only {len(prices)} were available."
        )

    signal = build_strategy_signal(prices["Asset Close"], config)
    if config.position_mode == "Long / Cash":
        target_position = signal.clip(lower=0, upper=1)
    else:
        target_position = signal.clip(lower=-1, upper=1)
    target_position = _apply_rebalancing(target_position, config.rebalancing)

    position = target_position.shift(1).fillna(0.0)
    asset_returns = prices["Asset Close"].pct_change(fill_method=None).fillna(0.0)
    benchmark_returns = prices["Benchmark Close"].pct_change(fill_method=None).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000
    strategy_returns = (position * asset_returns) - (turnover * cost_rate)

    full_performance = prices.copy()
    full_performance["Signal"] = signal
    full_performance["Target Position"] = target_position
    full_performance["Position"] = position
    full_performance["Turnover"] = turnover
    full_performance["Strategy Return"] = strategy_returns
    full_performance["Benchmark Return"] = benchmark_returns

    evaluation_start_index = warmup_observations
    warnings = []
    if config.evaluation_mode == "Holdout":
        evaluation_start_index = max(
            warmup_observations,
            int(len(full_performance) * (1 - config.holdout_fraction)),
        )
        warnings.append(
            "Metrics use only the holdout window; earlier prices are used solely to warm up the selected signal."
        )

    performance = full_performance.iloc[evaluation_start_index:].copy()
    performance["Strategy Equity"] = (1 + performance["Strategy Return"]).cumprod()
    performance["Benchmark Equity"] = (1 + performance["Benchmark Return"]).cumprod()
    performance["Strategy Drawdown"] = (
        performance["Strategy Equity"]
        / performance["Strategy Equity"].cummax()
        - 1
    )

    strategy_metrics = _performance_metrics(
        performance["Strategy Return"],
        config.risk_free_rate,
    )
    benchmark_metrics = _performance_metrics(
        performance["Benchmark Return"],
        config.risk_free_rate,
    )
    elapsed_years = max(
        (performance.index[-1] - performance.index[0]).days / 365.25,
        len(performance) / TRADING_DAYS_PER_YEAR,
        1 / TRADING_DAYS_PER_YEAR,
    )
    annualized_turnover = float(performance["Turnover"].sum() / elapsed_years)
    market_exposure = float((performance["Position"] != 0).mean())
    trades = _build_trade_ledger(performance, cost_rate)
    closed_trades = (
        trades.loc[trades["Status"] == "Closed"]
        if not trades.empty
        else trades
    )

    if trades.empty:
        warnings.append("The selected rules produced no trades in the evaluation window.")
        trade_count = 0
        win_rate = None
        best_trade = None
        worst_trade = None
    else:
        trade_count = len(trades)
        if closed_trades.empty:
            win_rate = None
        else:
            win_rate = float((closed_trades["Net Return"] > 0).mean())
        best_trade = float(trades["Net Return"].max())
        worst_trade = float(trades["Net Return"].min())

    return BacktestResult(
        performance=performance,
        trades=trades,
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        annualized_turnover=annualized_turnover,
        market_exposure=market_exposure,
        trade_count=trade_count,
        win_rate=win_rate,
        best_trade=best_trade,
        worst_trade=worst_trade,
        evaluation_start=performance.index[0],
        warnings=tuple(warnings),
    )


def build_stability_table(
    result: BacktestResult,
    segments: int = 5,
) -> pd.DataFrame:
    frame = result.performance
    if segments < 2 or len(frame) < segments * 10:
        return pd.DataFrame()

    rows = []
    for segment_number, indices in enumerate(np.array_split(np.arange(len(frame)), segments), 1):
        if len(indices) == 0:
            continue
        segment = frame.iloc[indices]
        rows.append(
            {
                "Period": segment_number,
                "Start": segment.index[0],
                "End": segment.index[-1],
                "Strategy Return": float(
                    (1 + segment["Strategy Return"]).prod() - 1
                ),
                "Benchmark Return": float(
                    (1 + segment["Benchmark Return"]).prod() - 1
                ),
                "Active Return": float(
                    (1 + segment["Strategy Return"]).prod()
                    - (1 + segment["Benchmark Return"]).prod()
                ),
            }
        )
    return pd.DataFrame(rows)
