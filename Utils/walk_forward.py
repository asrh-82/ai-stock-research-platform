"""Chronological selection. The final holdout is never used to pick a rule."""
from dataclasses import asdict, dataclass, replace
import numpy as np
import pandas as pd
from Utils.backtesting import BacktestConfig, run_backtest, warmup_bars, _performance_metrics, prepare_prices, _relative_strength_index


def research_candidates(base=None):
    base = base or BacktestConfig()
    base = replace(base, evaluation_mode='Full History')
    return {
        'SMA 20/100': replace(base, strategy='SMA Crossover', fast_window=20, slow_window=100),
        'SMA 50/200': replace(base, strategy='SMA Crossover', fast_window=50, slow_window=200),
        'Momentum 63': replace(base, strategy='Price Momentum', momentum_lookback=63),
        'Momentum 126': replace(base, strategy='Price Momentum', momentum_lookback=126),
        'RSI 14 30/55': replace(base, strategy='RSI Mean Reversion', rsi_window=14, rsi_entry=30, rsi_exit=55),
        'Breakout 55/20': replace(base, strategy='Donchian Breakout', breakout_entry=55, breakout_exit=20),
    }


def _select(asset, benchmark, candidates, start, end, currency):
    scores = []
    # Physical truncation makes future bars unavailable to training calculations.
    a, b = asset.loc[:end], benchmark.loc[:end]
    for name, cfg in candidates.items():
        result = run_backtest(a, b, cfg, evaluation_start=start, evaluation_end=end,
                              asset_currency=currency, benchmark_currency=currency)
        sharpe = result.strategy_metrics.sharpe_ratio
        completed = int((result.trades.Status == 'Closed').sum())
        scores.append({'Candidate': name, 'Train Start': start, 'Train End': end,
                       'Sharpe': sharpe, 'Return': result.strategy_metrics.total_return,
                       'Closed Trades': completed, 'Eligible': completed >= 2 and sharpe is not None})
    eligible = [s for s in scores if s['Eligible']]
    # No strategy is advertised as an edge just because it won this comparison.
    # If every rule lacks two completed trades, keep cash rather than fabricate a winner.
    best = max(eligible, key=lambda s: s['Sharpe']) if eligible else None
    return (best['Candidate'] if best and best['Sharpe'] > 0 else 'Cash'), scores


def _cash_result(asset, benchmark, base, start, end, currency):
    cfg = replace(base, strategy='Cash', evaluation_mode='Full History')
    return run_backtest(asset, benchmark, cfg, evaluation_start=start, evaluation_end=end,
                        asset_currency=currency, benchmark_currency=currency)


def run_walk_forward(asset, benchmark, *, currency, candidates=None, train_bars=504, test_bars=126, holdout_bars=252):
    candidates = candidates or research_candidates()
    if not candidates or min(train_bars, test_bars, holdout_bars) < 20:
        raise ValueError('Nonempty candidates and windows of at least 20 sessions are required.')
    a, b = asset.copy(), benchmark.copy()
    # Normalize and validate without replacing raw prices.
    ai, bi = prepare_prices(a).index, prepare_prices(b).index
    a.index, b.index = ai, bi
    if not ai.equals(bi):
        raise ValueError('Asset and benchmark calendars must match.')
    warmup = max(20, max(warmup_bars(c) for c in candidates.values()))
    development_end = len(a) - holdout_bars
    if development_end < warmup + train_bars + test_bars:
        raise ValueError('Insufficient history for warm-up, training, validation and untouched holdout.')
    base = next(iter(candidates.values()))
    if any((c.transaction_cost_bps, c.slippage_bps, c.initial_capital, c.max_participation, c.cash_rate) !=
           (base.transaction_cost_bps, base.slippage_bps, base.initial_capital, base.max_participation, base.cash_rate)
           for c in candidates.values()):
        raise ValueError('Candidate execution assumptions must match.')
    folds, trials, returns = [], [], []
    cursor = warmup + train_bars
    while cursor < development_end:
        train_start, train_end = a.index[cursor - train_bars], a.index[cursor - 1]
        chosen, scores = _select(a, b, candidates, train_start, train_end, currency)
        stop = min(cursor + test_bars, development_end)
        test_start, test_end = a.index[cursor], a.index[stop - 1]
        if stop - cursor < 2:
            break
        if chosen == 'Cash':
            result = _cash_result(a.loc[:test_end], b.loc[:test_end], base, test_start, test_end, currency)
        else:
            result = run_backtest(a.loc[:test_end], b.loc[:test_end], candidates[chosen], evaluation_start=test_start,
                                  evaluation_end=test_end, asset_currency=currency, benchmark_currency=currency)
        folds.append({'Fold': len(folds)+1, 'Train Start': train_start, 'Train End': train_end,
                      'Test Start': test_start, 'Test End': test_end, 'Selected': chosen,
                      'Return': result.strategy_metrics.total_return, 'Benchmark Return': result.benchmark_metrics.total_return,
                      'Sharpe': result.strategy_metrics.sharpe_ratio, 'Closed Trades': int((result.trades.Status == 'Closed').sum())})
        trials.extend([dict(s, Fold=len(folds)) for s in scores])
        returns.append(result.performance[['Strategy Return', 'Benchmark Return']])
        cursor = stop
    final_train_start = a.index[development_end - train_bars]
    final_train_end = a.index[development_end - 1]
    selected, final_scores = _select(a, b, candidates, final_train_start, final_train_end, currency)
    trials.extend([dict(s, Fold='Final selection') for s in final_scores])
    final_config = candidates[selected] if selected != 'Cash' else replace(base, strategy='Cash')
    final = run_backtest(a, b, final_config, evaluation_start=a.index[development_end],
                         asset_currency=currency, benchmark_currency=currency)
    same_stock = run_backtest(a, a, replace(base, strategy='Buy and Hold'), evaluation_start=a.index[development_end],
                              asset_currency=currency, benchmark_currency=currency)
    validation = pd.concat(returns)
    if validation.index.has_duplicates:
        raise AssertionError('Overlapping validation windows.')
    return {'folds': pd.DataFrame(folds), 'trials': pd.DataFrame(trials), 'validation': validation,
            'validation_metrics': _performance_metrics(validation['Strategy Return'], base.risk_free_rate),
            'selected': selected, 'selected_config': asdict(final_config), 'holdout': final,
            'holdout_buy_hold': same_stock, 'holdout_start': a.index[development_end],
            'selection_rule': 'Highest net training Sharpe with at least two closed trades; cash if no positive eligible Sharpe',
            'candidate_count': len(candidates), 'train_bars': train_bars, 'test_bars': test_bars,
            'holdout_bars': holdout_bars,
            'limitations': 'Validation folds reset to fresh capital and liquidate at each boundary. Historical holdout was withheld from code selection, but is not a prospective live test. Re-running after changing rules invalidates its untouched status.'}


def block_bootstrap_active_mean(strategy_returns, benchmark_returns, *, seed=42, draws=2000, block=20):
    """Descriptive paired circular-block interval; no multiple-testing claim."""
    delta = (strategy_returns - benchmark_returns).to_numpy()
    if len(delta) < 2 * block or not np.isfinite(delta).all():
        raise ValueError('Insufficient finite paired returns for block bootstrap.')
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, len(delta), size=int(np.ceil(len(delta)/block)))
        indices = ((starts[:, None] + np.arange(block)) % len(delta)).ravel()[:len(delta)]
        means[i] = delta[indices].mean() * 252
    return {'annualized_arithmetic_active_mean': float(delta.mean() * 252),
            'lower_95': float(np.quantile(means, .025)), 'upper_95': float(np.quantile(means, .975)),
            'block_sessions': block, 'draws': draws, 'seed': seed,
            'interpretation': 'Descriptive uncertainty, not a probability of profit; not adjusted for repeated research or all tried ideas.'}


def pattern_diagnostics(history):
    """Development-only conditional forward returns. Overlapping, not independent."""
    h = prepare_prices(history)
    rsi = _relative_strength_index(h.Close, 14)
    patterns = {'Above 200-day SMA': h.Close > h.Close.rolling(200).mean(),
                'RSI below 30': rsi < 30,
                '55-day high breakout': h.Close > h.High.rolling(55).max().shift(1)}
    rows = []
    for horizon in [1, 5, 20]:
        forward = h.Close.shift(-horizon) / h.Open.shift(-1) - 1
        for name, mask in patterns.items():
            sample = forward.loc[mask & forward.notna()]
            rows.append({'Pattern': name, 'Holding Sessions': horizon, 'Observations': len(sample),
                         'Mean Forward Return': float(sample.mean()) if len(sample) else None,
                         'Median Forward Return': float(sample.median()) if len(sample) else None,
                         'Positive Fraction': float((sample > 0).mean()) if len(sample) else None})
    return pd.DataFrame(rows)
