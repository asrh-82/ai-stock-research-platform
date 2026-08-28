import unittest

from Utils.dcf import (
    DCFInputs,
    apply_scenario_adjustments,
    calculate_dcf,
    calculate_sensitivity,
)


def make_inputs(**overrides):
    values = {
        "base_revenue": 1_000,
        "revenue_growth": (0.10, 0.08, 0.06, 0.04, 0.03),
        "ebit_margin": (0.20, 0.20, 0.20, 0.20, 0.20),
        "tax_rate": 0.25,
        "depreciation_pct_revenue": 0.03,
        "capex_pct_revenue": 0.04,
        "nwc_pct_incremental_revenue": 0.10,
        "wacc": 0.10,
        "terminal_growth": 0.03,
        "cash": 100,
        "debt": 50,
        "diluted_shares": 100,
    }
    values.update(overrides)
    return DCFInputs(**values)


class DCFCalculationTests(unittest.TestCase):
    def test_calculates_forecast_and_bridge_to_equity_value(self):
        result = calculate_dcf(make_inputs())

        self.assertEqual(len(result.forecast), 5)
        self.assertAlmostEqual(result.forecast[0].revenue, 1_100)
        self.assertAlmostEqual(result.forecast[0].ebit, 220)
        self.assertAlmostEqual(result.forecast[0].nopat, 165)
        self.assertAlmostEqual(result.forecast[0].depreciation, 33)
        self.assertAlmostEqual(result.forecast[0].capex, 44)
        self.assertAlmostEqual(result.forecast[0].change_in_nwc, 10)
        self.assertAlmostEqual(result.forecast[0].unlevered_fcf, 144)
        self.assertAlmostEqual(
            result.equity_value,
            result.enterprise_value + 50,
        )
        self.assertAlmostEqual(
            result.implied_value_per_share,
            result.equity_value / 100,
        )

    def test_rejects_terminal_growth_at_or_above_wacc(self):
        with self.assertRaisesRegex(ValueError, "WACC must be greater"):
            calculate_dcf(make_inputs(wacc=0.03, terminal_growth=0.03))

    def test_rejects_mismatched_forecast_lengths(self):
        with self.assertRaisesRegex(ValueError, "same number of years"):
            calculate_dcf(make_inputs(ebit_margin=(0.20,)))

    def test_scenario_adjustments_do_not_mutate_base_inputs(self):
        base = make_inputs()
        bull = apply_scenario_adjustments(
            base,
            growth_delta=0.02,
            margin_delta=0.01,
            wacc_delta=-0.005,
        )

        self.assertEqual(base.revenue_growth[0], 0.10)
        self.assertAlmostEqual(bull.revenue_growth[0], 0.12)
        self.assertAlmostEqual(bull.ebit_margin[0], 0.21)
        self.assertAlmostEqual(bull.wacc, 0.095)

    def test_sensitivity_returns_none_for_invalid_pairs(self):
        table = calculate_sensitivity(
            make_inputs(),
            wacc_values=(0.08, 0.09),
            terminal_growth_values=(0.03, 0.08),
        )

        self.assertIsNone(table[0.08][0.08])
        self.assertGreater(table[0.08][0.03], table[0.09][0.03])


if __name__ == "__main__":
    unittest.main()
