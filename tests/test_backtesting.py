import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from Utils.backtesting import (BacktestConfig, run_backtest, prepare_prices, _relative_strength_index,
                               _performance_metrics, build_strategy_signal, _execution_targets)


def make_prices(length=500, flat=False):
    index = pd.bdate_range('2020-01-02', periods=length)
    close = np.full(length, 100.0) if flat else 100 * np.cumprod(1 + .0004 + np.sin(np.arange(length)/18)*.006)
    op = np.r_[close[0], close[:-1]] * (1 if flat else 1.001)
    return pd.DataFrame({'Open': op, 'High': np.maximum(op, close)*1.001, 'Low': np.minimum(op, close)*.999,
                         'Close': close, 'Adj Close': close, 'Volume': 1_000_000.0}, index=index)


def run(a, cfg=None, b=None, **kwargs):
    return run_backtest(a, a if b is None else b, cfg or BacktestConfig(),
                        asset_currency='CAD', benchmark_currency='CAD', **kwargs)


class BacktestingTests(unittest.TestCase):
    def test_next_open_excludes_unowned_overnight_gap(self):
        a = make_prices(40, flat=True)
        a.iloc[20:, a.columns.get_indexer(['Open','High','Low','Close','Adj Close'])] *= 2
        result = run(a, BacktestConfig(strategy='Buy and Hold', transaction_cost_bps=0, slippage_bps=0))
        self.assertEqual(result.trades.iloc[0]['Entry Price'], 200)
        self.assertAlmostEqual(result.strategy_metrics.total_return, 0)

    def test_hand_calculated_entry_and_exit_costs(self):
        a = make_prices(40, flat=True)
        cfg = BacktestConfig(strategy='Buy and Hold', transaction_cost_bps=100, slippage_bps=100)
        result = run(a, cfg)
        units = 10000/(101*1.01)
        expected_cash = units*99*.99
        self.assertAlmostEqual(result.performance.Equity.iloc[-1], expected_cash)
        self.assertAlmostEqual(result.trades.PnL.sum(), expected_cash-10000)
        self.assertAlmostEqual(result.strategy_metrics.total_return, expected_cash/10000-1)
        self.assertEqual(len(result.orders), 2)

    def test_closed_ledger_reconciles_equity(self):
        result = run(make_prices(), BacktestConfig(fast_window=20, slow_window=60))
        self.assertGreater(len(result.trades), 2)
        self.assertAlmostEqual(result.trades.PnL.sum()+result.performance['Cash Interest'].sum(),
                               result.performance.Equity.iloc[-1]-10000, places=7)
        np.testing.assert_allclose(result.performance.Equity, result.performance.Cash + result.performance.Units*result.performance['Asset Close'])

    def test_costs_reduce_value(self):
        a=make_prices()
        free=run(a,BacktestConfig(fast_window=20,slow_window=60,transaction_cost_bps=0,slippage_bps=0))
        costly=run(a,BacktestConfig(fast_window=20,slow_window=60,transaction_cost_bps=20,slippage_bps=20))
        self.assertLess(costly.strategy_metrics.total_return,free.strategy_metrics.total_return)

    def test_holdout_starts_with_cash_and_charges_entry(self):
        a=make_prices(flat=True)
        cfg=BacktestConfig(strategy='Buy and Hold',evaluation_mode='Holdout',holdout_fraction=.4,liquidate_at_end=False)
        r=run(a,cfg)
        self.assertEqual(r.evaluation_start,a.index[300])
        self.assertEqual(r.trades.iloc[0]['Entry Date'],a.index[300])
        self.assertGreater(r.performance.Costs.iloc[0],0)
        self.assertLess(r.performance.Equity.iloc[0],10000)

    def test_first_loss_counts_in_drawdown(self):
        returns=pd.Series([-.1,0,0],index=pd.bdate_range('2020-01-01',periods=3))
        self.assertAlmostEqual(_performance_metrics(returns,0).maximum_drawdown,-.1)

    def test_sortino_uses_all_observations(self):
        ret=pd.Series([.1,-.1,0,0],index=pd.bdate_range('2020-01-01',periods=4))
        # mean=0; test nonzero mean separately
        ret.iloc[0]=.2
        expected=ret.mean()/np.sqrt(.01/4)*np.sqrt(252)
        self.assertAlmostEqual(_performance_metrics(ret,0).sortino_ratio,expected)

    def test_rsi_monotonic_and_flat(self):
        for close, expected in [(np.arange(1.,41.),100),(np.arange(40.,0.,-1),0),(np.ones(40),50)]:
            out=_relative_strength_index(pd.Series(close),14)
            self.assertTrue(out.iloc[:14].isna().all())
            np.testing.assert_allclose(out.iloc[14:],expected)

    def test_adjusts_open_and_close_without_double_dividend(self):
        a=make_prices(40,flat=True)
        a.iloc[:25,a.columns.get_indexer(['Adj Close'])]=99
        a.loc[a.index[25]:,['Open','Close','Adj Close']]=99
        a.loc[a.index[25]:,'Low']=98.9
        a['Dividends']=0.0
        a.loc[a.index[25],'Dividends']=1
        r=run(a,BacktestConfig(strategy='Buy and Hold',transaction_cost_bps=0,slippage_bps=0))
        self.assertAlmostEqual(r.strategy_metrics.total_return,0)
        self.assertAlmostEqual(r.trades.iloc[0]['Entry Price'],99)

    def test_future_changes_cannot_change_past_signals_or_equity(self):
        a=make_prices()
        changed=a.copy()
        changed.iloc[300:,changed.columns.get_indexer(['Open','High','Low','Close','Adj Close'])]*=3
        for strategy in ['SMA Crossover','Price Momentum','RSI Mean Reversion','Donchian Breakout']:
            cfg=BacktestConfig(strategy=strategy,fast_window=20,slow_window=60,liquidate_at_end=False)
            x,y=run(a,cfg),run(changed,cfg)
            pd.testing.assert_frame_equal(x.performance.loc[:a.index[299]],y.performance.loc[:a.index[299]])

    def test_weekly_and_monthly_targets_are_prefix_invariant(self):
        index=pd.bdate_range('2020-01-01',periods=90)
        signal=pd.Series(np.arange(90)%2,index=index,dtype=float)
        for frequency in ['Weekly','Monthly']:
            full=_execution_targets(signal,frequency)
            short=_execution_targets(signal.iloc[:37],frequency)
            pd.testing.assert_series_equal(full.iloc[:37],short)

    def test_breakout_excludes_current_high(self):
        close=pd.Series(np.arange(1.,31.))
        cfg=BacktestConfig(strategy='Donchian Breakout',breakout_entry=5,breakout_exit=2)
        s=build_strategy_signal(close,cfg,high=close,low=close)
        self.assertEqual(s.iloc[4],0)
        self.assertEqual(s.iloc[5],1)

    def test_liquidity_uses_prior_volume_not_future_spike(self):
        a=make_prices(40,flat=True)
        a['Volume']=1.0
        a.iloc[20,a.columns.get_loc('Volume')]=1e9
        r=run(a,BacktestConfig(strategy='Buy and Hold'))
        self.assertEqual(r.orders.iloc[0].Status,'Rejected')
        self.assertEqual(r.performance.Units.iloc[0],0)

    def test_zero_volume_never_fills(self):
        a=make_prices(40,flat=True)
        a.loc[a.index[20]:,'Volume']=0
        r=run(a,BacktestConfig(strategy='Buy and Hold'))
        self.assertEqual(len(r.trades),0)

    def test_benchmark_has_identical_cost_convention(self):
        r=run(make_prices(),BacktestConfig(strategy='Buy and Hold'))
        pd.testing.assert_series_equal(r.performance['Strategy Return'],r.performance['Benchmark Return'],check_names=False)

    def test_cash_interest_and_risk_free_are_distinct(self):
        a=make_prices()
        r=run(a,BacktestConfig(strategy='Cash',cash_rate=.05,risk_free_rate=.04))
        days=(r.performance.index[-1]-r.performance.index[0]).days
        self.assertAlmostEqual(r.performance.Equity.iloc[-1],10000*1.05**(days/365.25))
        self.assertEqual(len(r.trades),0)

    def test_rejects_currency_mismatch_and_unknown_currency(self):
        a=make_prices()
        for b in ['USD',None]:
            with self.assertRaisesRegex(ValueError,'currenc'):
                run_backtest(a,a,asset_currency='CAD',benchmark_currency=b)

    def test_rejects_invalid_data_instead_of_silently_cleaning(self):
        for value in [np.nan,np.inf,0,-1]:
            a=make_prices();a.iloc[50,a.columns.get_loc('Close')]=value
            with self.assertRaises(ValueError): run(a)
        a=make_prices()
        with self.assertRaises(ValueError):run(pd.concat([a.iloc[:1],a]))
        with self.assertRaises(ValueError):run(a,b=a.drop(a.index[50]))
        with self.assertRaises(ValueError):run(a.Close)

    def test_rejects_nonfinite_config_and_shorting(self):
        for cfg in [BacktestConfig(momentum_threshold=float('nan')), BacktestConfig(slippage_bps=float('inf')),
                    BacktestConfig(position_mode='Long / Short'),BacktestConfig(fast_window=1.5)]:
            with self.assertRaises(ValueError):run(make_prices(),cfg)
