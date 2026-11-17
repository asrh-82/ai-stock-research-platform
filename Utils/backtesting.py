"""Daily long/cash simulator. Adjusted OHLC units, next-open execution.

Reinvested distributions are included in adjusted prices, never credited twice.
The ledger is an economic research model, not literal historical broker fills.
"""
from dataclasses import dataclass
from math import isfinite, sqrt
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
SUPPORTED_STRATEGIES = ('SMA Crossover', 'Price Momentum', 'RSI Mean Reversion', 'Donchian Breakout', 'Buy and Hold', 'Cash')
SUPPORTED_REBALANCING = ('Daily', 'Weekly', 'Monthly')
SUPPORTED_POSITION_MODES = ('Long / Cash',)
SUPPORTED_EVALUATION_MODES = ('Full History', 'Holdout')

@dataclass(frozen=True)
class BacktestConfig:
    strategy: str = 'SMA Crossover'
    fast_window: int = 50
    slow_window: int = 200
    momentum_lookback: int = 126
    momentum_threshold: float = 0.0
    rsi_window: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 55.0
    breakout_entry: int = 55
    breakout_exit: int = 20
    rebalancing: str = 'Daily'
    position_mode: str = 'Long / Cash'
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 10.0
    risk_free_rate: float = 0.0
    cash_rate: float = 0.0
    evaluation_mode: str = 'Full History'
    holdout_fraction: float = 0.30
    initial_capital: float = 10_000.0
    max_participation: float = 0.01
    liquidate_at_end: bool = True

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
    orders: pd.DataFrame
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


def validate_backtest_config(config):
    for field, choices in [('strategy', SUPPORTED_STRATEGIES), ('rebalancing', SUPPORTED_REBALANCING),
                           ('position_mode', SUPPORTED_POSITION_MODES), ('evaluation_mode', SUPPORTED_EVALUATION_MODES)]:
        if getattr(config, field) not in choices:
            raise ValueError(f'Unsupported {field}: {getattr(config, field)}. Long/short requires a borrow and margin model.')
    for field in ['fast_window', 'slow_window', 'momentum_lookback', 'rsi_window', 'breakout_entry', 'breakout_exit']:
        v = getattr(config, field)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 2:
            raise ValueError(f'{field} must be an integer of at least two.')
    if not config.fast_window < config.slow_window:
        raise ValueError('The fast moving average must be shorter than the slow moving average.')
    if not config.breakout_exit < config.breakout_entry:
        raise ValueError('Breakout exit window must be shorter than entry window.')
    bounds = {'transaction_cost_bps': (0, 1000), 'slippage_bps': (0, 1000),
              'risk_free_rate': (-0.5, 1), 'cash_rate': (-0.5, 1), 'holdout_fraction': (0.1, 0.8),
              'max_participation': (0.00001, 1), 'initial_capital': (1, 1e12),
              'momentum_threshold': (-1, 10), 'rsi_entry': (0, 100), 'rsi_exit': (0, 100)}
    for field, (lo, hi) in bounds.items():
        v = getattr(config, field)
        if not isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'{field} must be finite and between {lo} and {hi}.')
    if not config.rsi_entry < config.rsi_exit:
        raise ValueError('RSI entry must be below RSI exit.')


def prepare_prices(history, label='Asset'):
    """Validate raw Yahoo OHLCV + Adj Close; adjust ALL prices by same factor."""
    required = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if not isinstance(history, pd.DataFrame) or history.empty or any(c not in history for c in required):
        raise ValueError(f'{label} requires daily Open, High, Low, Close, Adj Close and Volume. Close-only execution is not supported.')
    out = history[required].copy()
    try:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out.index = out.index.normalize()
        out = out.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} has invalid dates or nonnumeric prices.') from exc
    if out.index.hasnans or out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError(f'{label} dates must be unique and increasing.')
    if not np.isfinite(out.to_numpy()).all() or (out[required[:-1]] <= 0).any().any() or (out.Volume < 0).any():
        raise ValueError(f'{label} has missing, infinite, nonpositive prices or negative volume.')
    if ((out.High < out[['Open', 'Close', 'Low']].max(axis=1) - 1e-7) |
        (out.Low > out[['Open', 'Close', 'High']].min(axis=1) + 1e-7)).any():
        raise ValueError(f'{label} contains inconsistent OHLC bars.')
    out['Dollar Volume'] = out.Close * out.Volume
    factor = out['Adj Close'] / out.Close
    out[['Open', 'High', 'Low', 'Close']] = out[['Open', 'High', 'Low', 'Close']].mul(factor, axis=0)
    return out


def _relative_strength_index(close, window):
    """Wilder RSI seeded with mean of the first window changes."""
    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) <= window:
        return rsi
    delta = close.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = float(gains.iloc[1:window + 1].mean()), float(losses.iloc[1:window + 1].mean())
    for i in range(window, len(close)):
        if i > window:
            avg_gain = (avg_gain * (window - 1) + gains.iloc[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses.iloc[i]) / window
        rsi.iloc[i] = 50 if avg_gain == avg_loss == 0 else 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi


def warmup_bars(config):
    return {'SMA Crossover': config.slow_window, 'Price Momentum': config.momentum_lookback + 1,
            'RSI Mean Reversion': config.rsi_window + 1, 'Donchian Breakout': config.breakout_entry + 1,
            'Buy and Hold': 1, 'Cash': 1}[config.strategy]


def build_strategy_signal(close, config, high=None, low=None):
    if config.strategy == 'Cash':
        return pd.Series(0.0, index=close.index, name='Signal')
    if config.strategy == 'Buy and Hold':
        return pd.Series(1.0, index=close.index, name='Signal')
    if config.strategy == 'SMA Crossover':
        return (close.rolling(config.fast_window).mean() > close.rolling(config.slow_window).mean()).astype(float).rename('Signal')
    if config.strategy == 'Price Momentum':
        return (close.pct_change(config.momentum_lookback, fill_method=None) > config.momentum_threshold).astype(float).rename('Signal')
    decisions = pd.Series(np.nan, index=close.index, dtype=float)
    if config.strategy == 'RSI Mean Reversion':
        rsi = _relative_strength_index(close, config.rsi_window)
        decisions.loc[rsi <= config.rsi_entry] = 1.0
        decisions.loc[rsi >= config.rsi_exit] = 0.0
    elif config.strategy == 'Donchian Breakout':
        if high is None or low is None:
            raise ValueError('Donchian requires high and low prices.')
        decisions.loc[close > high.rolling(config.breakout_entry).max().shift(1)] = 1.0
        decisions.loc[close < low.rolling(config.breakout_exit).min().shift(1)] = 0.0
    else:
        raise ValueError(f'Unknown strategy {config.strategy}')
    return decisions.ffill().fillna(0.0).rename('Signal')


def _execution_targets(signal, frequency):
    prior = signal.shift(1).fillna(0.0)
    if frequency == 'Daily':
        return prior
    periods = signal.index.to_period('W-FRI' if frequency == 'Weekly' else 'M')
    period_series = pd.Series(periods, index=signal.index)
    changed = period_series.ne(period_series.shift())
    changed.iloc[0] = False
    return prior.where(changed).ffill().fillna(0.0)


def _performance_metrics(returns, risk_free_rate, start_date=None):
    if returns.empty or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError('Returns must be finite, nonempty and greater than -100%.')
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1
    days = (returns.index[-1] - (start_date if start_date is not None else returns.index[0])).days
    years = max((days + 1) / 365.25, 1 / TRADING_DAYS_PER_YEAR)
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = returns - daily_rf
    std = float(excess.std(ddof=1)) if len(returns) > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    return PerformanceMetrics(float(equity.iloc[-1] - 1), float(equity.iloc[-1] ** (1 / years) - 1),
                              float(returns.std(ddof=1) * sqrt(252)) if len(returns) > 1 else 0.0,
                              float(excess.mean() / std * sqrt(252)) if std > 1e-15 else None,
                              float(excess.mean() / downside * sqrt(252)) if downside > 1e-15 else None,
                              float(drawdown.min()))


def _simulate(bars, targets, config):
    """Cash + fixed adjusted units. All-or-none fills, no hidden daily resizing."""
    cash, units = config.initial_capital, 0.0
    fee, slip = config.transaction_cost_bps / 1e4, config.slippage_bps / 1e4
    records, orders, trades = [], [], []
    entry = None
    previous_equity, previous_date = cash, None
    for date, bar in bars.iterrows():
        interest = 0.0
        if previous_date is not None:
            interest = cash * ((1 + config.cash_rate) ** ((date - previous_date).days / 365.25) - 1)
            cash += interest
        day_cost = day_turnover = 0.0
        target = float(targets.loc[date])

        def execute(side, price, terminal=False):
            nonlocal cash, units, entry, day_cost, day_turnover
            fill = price * (1 + slip if side == 'Buy' else 1 - slip)
            quantity = cash / (fill * (1 + fee)) if side == 'Buy' else units
            notional = quantity * fill
            capacity = bar['Prior ADV'] * config.max_participation
            if not np.isfinite(capacity) or capacity <= 0 or notional > capacity or bar.Volume == 0:
                orders.append({'Date': date, 'Side': side, 'Status': 'Rejected', 'Reason': 'Prior 20-session dollar-volume limit or zero session volume', 'Notional': notional, 'Terminal': terminal})
                return
            commission = notional * fee
            slippage = quantity * abs(fill - price)
            day_cost += commission + slippage
            day_turnover += quantity * price / previous_equity
            orders.append({'Date': date, 'Side': side, 'Status': 'Filled', 'Reason': 'Scheduled final close' if terminal else 'Previous completed close signal', 'Notional': notional, 'Fill Price': fill, 'Units': quantity, 'Commission': commission, 'Slippage': slippage, 'Terminal': terminal})
            if side == 'Buy':
                outlay = notional + commission
                cash -= outlay
                units = quantity
                entry = {'Entry Date': date, 'Entry Price': fill, 'Units': quantity, 'Entry Outlay': outlay, 'Entry Commission': commission}
            else:
                proceeds = notional - commission
                cash += proceeds
                trades.append({**entry, 'Exit Date': date, 'Exit Price': fill, 'Exit Proceeds': proceeds, 'Exit Commission': commission,
                               'PnL': proceeds - entry['Entry Outlay'], 'Net Return': proceeds / entry['Entry Outlay'] - 1,
                               'Side': 'Long', 'Status': 'Closed', 'Bars': int(bars.loc[entry['Entry Date']:date].shape[0] - 1)})
                units, entry = 0.0, None

        if target == 1 and units == 0:
            execute('Buy', bar.Open)
        elif target == 0 and units > 0:
            execute('Sell', bar.Open)
        intraday_position = float(units > 0)
        if config.liquidate_at_end and date == bars.index[-1] and units > 0:
            execute('Sell', bar.Close, terminal=True)
        equity = cash + units * bar.Close
        if not np.isfinite(equity) or equity <= 0 or cash < -1e-6:
            raise ValueError('Accounting invariant failed: invalid equity or negative unlevered cash.')
        records.append({'Date': date, 'Asset Open': bar.Open, 'Asset Close': bar.Close, 'Position': intraday_position,
                        'Target Position': target, 'Cash': cash, 'Units': units, 'Equity': equity, 'Costs': day_cost,
                        'Cash Interest': interest, 'Turnover': day_turnover, 'Strategy Return': equity / previous_equity - 1})
        previous_equity, previous_date = equity, date
    if entry is not None:
        value = units * bars.Close.iloc[-1]
        trades.append({**entry, 'Exit Date': pd.NaT, 'Exit Price': bars.Close.iloc[-1], 'Exit Proceeds': value,
                       'Exit Commission': 0.0, 'PnL': value - entry['Entry Outlay'], 'Net Return': value / entry['Entry Outlay'] - 1,
                       'Side': 'Long', 'Status': 'Open', 'Bars': len(bars.loc[entry['Entry Date']:]) - 1})
    columns = ['Entry Date', 'Entry Price', 'Units', 'Entry Outlay', 'Entry Commission', 'Exit Date', 'Exit Price', 'Exit Proceeds', 'Exit Commission', 'PnL', 'Net Return', 'Side', 'Status', 'Bars']
    return pd.DataFrame(records).set_index('Date'), pd.DataFrame(trades, columns=columns), pd.DataFrame(orders)


def run_backtest(asset_history, benchmark_history, config=None, *, evaluation_start=None, evaluation_end=None,
                 asset_currency=None, benchmark_currency=None):
    config = config or BacktestConfig()
    validate_backtest_config(config)
    if not asset_currency or not benchmark_currency or asset_currency != benchmark_currency:
        raise ValueError('Supply matching asset and benchmark currencies; explicit FX conversion is required for different currencies.')
    asset, benchmark = prepare_prices(asset_history), prepare_prices(benchmark_history, 'Benchmark')
    if not asset.index.equals(benchmark.index):
        raise ValueError('Asset and benchmark session dates differ. Audit missing bars/calendars; no silent inner join.')
    if len(asset) < max(21, warmup_bars(config)) + 2:
        raise ValueError('Too few observations for strategy and liquidity warm-up.')
    for bars in (asset, benchmark):
        bars['Prior ADV'] = bars['Dollar Volume'].rolling(20, min_periods=20).mean().shift(1)
    signal = build_strategy_signal(asset.Close, config, asset.High, asset.Low)
    targets = _execution_targets(signal, config.rebalancing)
    start = max(20, warmup_bars(config))
    warnings = ['Adjusted OHLC research units include vendor distribution adjustments. Ledger prices are not historical broker fills; do not add dividends again.',
                'All-or-none fills use a prior 20-session dollar-volume limit, constant adverse slippage and proportional commission. Intraday depth, taxes and payment-date dividend cash flows are not modeled.']
    if config.evaluation_mode == 'Holdout' and evaluation_start is None:
        start = max(start, int(len(asset) * (1 - config.holdout_fraction)))
        warnings.append('Holdout starts with fresh cash and pays its own entry costs. Manually revisiting it contaminates the holdout.')
    if evaluation_start is not None:
        requested = pd.Timestamp(evaluation_start)
        start = max(start, int(asset.index.searchsorted(requested)))
        if start >= len(asset) or asset.index[start] != requested:
            raise ValueError('Evaluation start must be an observed session after warm-up.')
    end = pd.Timestamp(evaluation_end) if evaluation_end is not None else asset.index[-1]
    bars = asset.iloc[start:].loc[:end]
    if len(bars) < 2:
        raise ValueError('Evaluation needs at least two sessions.')
    performance, trades, orders = _simulate(bars, targets.loc[bars.index], config)
    benchmark_frame, _, benchmark_orders = _simulate(benchmark.loc[bars.index], pd.Series(1.0, index=bars.index), config)
    performance['Signal'] = signal.loc[bars.index]
    performance['Benchmark Return'] = benchmark_frame['Strategy Return']
    performance['Benchmark Equity'] = benchmark_frame.Equity / config.initial_capital
    performance['Strategy Equity'] = performance.Equity / config.initial_capital
    performance['Strategy Drawdown'] = performance['Strategy Equity'] / performance['Strategy Equity'].cummax().clip(lower=1) - 1
    closed = trades.loc[trades.Status == 'Closed']
    for label, ledger in [('Strategy', orders), ('Benchmark', benchmark_orders)]:
        if not ledger.empty and (ledger.Status == 'Rejected').any():
            warnings.append(f"{label}: {int((ledger.Status == 'Rejected').sum())} order(s) rejected by the liquidity model.")
    if not trades.empty and (trades.Status == 'Open').any():
        warnings.append('An open position is marked to final close without an exit fee; inspect open trades.')
    if trades.empty:
        warnings.append('No trades filled in the evaluation window.')
    years = max(((bars.index[-1] - bars.index[0]).days + 1) / 365.25, 1 / 252)
    return BacktestResult(performance, trades, orders,
                          _performance_metrics(performance['Strategy Return'], config.risk_free_rate),
                          _performance_metrics(performance['Benchmark Return'], config.risk_free_rate),
                          float(performance.Turnover.sum() / years), float(performance.Position.mean()), len(trades),
                          float((closed.PnL > 0).mean()) if len(closed) else None,
                          float(closed['Net Return'].max()) if len(closed) else None,
                          float(closed['Net Return'].min()) if len(closed) else None,
                          bars.index[0], tuple(warnings))


def build_stability_table(result, segments=5):
    frame = result.performance
    if segments < 2 or len(frame) < segments * 10:
        return pd.DataFrame()
    rows = []
    for i, indices in enumerate(np.array_split(np.arange(len(frame)), segments), 1):
        segment = frame.iloc[indices]
        strategy = float((1 + segment['Strategy Return']).prod() - 1)
        benchmark = float((1 + segment['Benchmark Return']).prod() - 1)
        rows.append({'Period': i, 'Start': segment.index[0], 'End': segment.index[-1],
                     'Strategy Return': strategy, 'Benchmark Return': benchmark, 'Active Return': strategy - benchmark})
    return pd.DataFrame(rows)
