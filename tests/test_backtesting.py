import unittest

import numpy as np
import pandas as pd

from Utils.backtesting import BacktestConfig, run_backtest


def make_prices(length=500):
    index = pd.bdate_range("2022-01-03", periods=length)
    cycle = np.sin(np.arange(length) / 18) * 0.006
    asset_returns = 0.0004 + cycle
    benchmark_returns = np.full(length, 0.00025)
    asset = pd.Series(100 * np.cumprod(1 + asset_returns), index=index)
    benchmark = pd.Series(100 * np.cumprod(1 + benchmark_returns), index=index)
    return asset, benchmark


class BacktestingTests(unittest.TestCase):
    def test_signal_is_executed_on_the_next_period(self):
        asset, benchmark = make_prices()
        result = run_backtest(
            asset,
            benchmark,
            BacktestConfig(fast_window=20, slow_window=60),
        )

        expected_position = result.performance["Target Position"].shift(1)
        pd.testing.assert_series_equal(
            result.performance["Position"].iloc[1:],
            expected_position.iloc[1:],
            check_names=False,
        )
        self.assertGreater(result.trade_count, 0)

    def test_transaction_costs_reduce_strategy_value(self):
        asset, benchmark = make_prices()
        free = run_backtest(
            asset,
            benchmark,
            BacktestConfig(
                fast_window=20,
                slow_window=60,
                transaction_cost_bps=0,
                slippage_bps=0,
            ),
        )
        costly = run_backtest(
            asset,
            benchmark,
            BacktestConfig(
                fast_window=20,
                slow_window=60,
                transaction_cost_bps=20,
                slippage_bps=10,
            ),
        )

        self.assertLess(
            costly.strategy_metrics.total_return,
            free.strategy_metrics.total_return,
        )

    def test_holdout_uses_only_the_evaluation_window_for_metrics(self):
        asset, benchmark = make_prices()
        full = run_backtest(
            asset,
            benchmark,
            BacktestConfig(fast_window=20, slow_window=60),
        )
        holdout = run_backtest(
            asset,
            benchmark,
            BacktestConfig(
                fast_window=20,
                slow_window=60,
                evaluation_mode="Holdout",
                holdout_fraction=0.40,
            ),
        )

        self.assertLess(len(holdout.performance), len(full.performance))
        self.assertGreater(holdout.evaluation_start, full.evaluation_start)
        pd.testing.assert_series_equal(
            holdout.performance["Position"].iloc[1:],
            holdout.performance["Target Position"].shift(1).iloc[1:],
            check_names=False,
        )
        self.assertTrue(any("holdout" in warning.lower() for warning in holdout.warnings))

    def test_rejects_invalid_moving_average_windows(self):
        asset, benchmark = make_prices()
        with self.assertRaisesRegex(ValueError, "fast moving average"):
            run_backtest(
                asset,
                benchmark,
                BacktestConfig(fast_window=100, slow_window=50),
            )


if __name__ == "__main__":
    unittest.main()
