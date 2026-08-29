import unittest

import pandas as pd

from Utils.dcf import DCFInputs
from Utils.monte_carlo import (
    MonteCarloConfig,
    run_monte_carlo,
    summarize_monte_carlo,
)


def make_inputs(**overrides):
    values = {
        "base_revenue": 1_000,
        "revenue_growth": (0.08, 0.07, 0.06, 0.05, 0.04),
        "ebit_margin": (0.20, 0.20, 0.21, 0.21, 0.22),
        "tax_rate": 0.21,
        "depreciation_pct_revenue": 0.03,
        "capex_pct_revenue": 0.04,
        "nwc_pct_incremental_revenue": 0.05,
        "wacc": 0.09,
        "terminal_growth": 0.03,
        "cash": 100,
        "debt": 50,
        "diluted_shares": 10,
    }
    values.update(overrides)
    return DCFInputs(**values)


class MonteCarloTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_complete(self):
        config = MonteCarloConfig(simulation_count=300, seed=17)

        first = run_monte_carlo(make_inputs(), config)
        second = run_monte_carlo(make_inputs(), config)

        self.assertEqual(len(first.samples), 300)
        pd.testing.assert_frame_equal(first.samples, second.samples)
        self.assertTrue(
            (first.samples["WACC"] > first.samples["Terminal Growth"]).all()
        )

    def test_summary_reports_percentiles_and_market_probability(self):
        result = run_monte_carlo(
            make_inputs(),
            MonteCarloConfig(simulation_count=300, seed=5),
        )
        summary = summarize_monte_carlo(result, current_price=100)

        self.assertLessEqual(summary["p05"], summary["median"])
        self.assertLessEqual(summary["median"], summary["p95"])
        self.assertGreaterEqual(summary["probability_above_market"], 0)
        self.assertLessEqual(summary["probability_above_market"], 1)

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "between 100 and 50,000"):
            run_monte_carlo(
                make_inputs(),
                MonteCarloConfig(simulation_count=50),
            )


if __name__ == "__main__":
    unittest.main()
