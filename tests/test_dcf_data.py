import unittest

import pandas as pd

from Utils.dcf_data import build_dcf_defaults


class DCFDefaultExtractionTests(unittest.TestCase):
    def setUp(self):
        columns = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")]
        self.income_statement = pd.DataFrame(
            {
                columns[0]: [1_000, 200, 180, 36, 100],
                columns[1]: [900, 180, 160, 32, 95],
            },
            index=[
                "Total Revenue",
                "Operating Income",
                "Pretax Income",
                "Tax Provision",
                "Diluted Average Shares",
            ],
        )
        self.cashflow_statement = pd.DataFrame(
            {
                columns[0]: [30, -40, -10],
                columns[1]: [27, -35, -8],
            },
            index=[
                "Depreciation And Amortization",
                "Capital Expenditure",
                "Change In Working Capital",
            ],
        )
        self.balance_sheet = pd.DataFrame(
            {columns[0]: [110, 50]},
            index=[
                "Cash Cash Equivalents And Short Term Investments",
                "Total Debt",
            ],
        )

    def test_extracts_and_normalizes_model_defaults(self):
        defaults = build_dcf_defaults(
            info={"totalCash": 120, "totalDebt": 60, "sharesOutstanding": 105},
            income_statement=self.income_statement,
            cashflow_statement=self.cashflow_statement,
            balance_sheet=self.balance_sheet,
        )

        self.assertEqual(defaults.base_revenue, 1_000)
        self.assertAlmostEqual(defaults.revenue_growth[0], 1_000 / 900 - 1)
        self.assertAlmostEqual(defaults.revenue_growth[-1], 0.04)
        self.assertEqual(defaults.ebit_margin, (0.20,) * 5)
        self.assertAlmostEqual(defaults.tax_rate, 0.20)
        self.assertAlmostEqual(defaults.depreciation_pct_revenue, 0.03)
        self.assertAlmostEqual(defaults.capex_pct_revenue, 0.04)
        self.assertAlmostEqual(defaults.nwc_pct_incremental_revenue, 0.10)
        self.assertEqual(defaults.cash, 120)
        self.assertEqual(defaults.debt, 60)
        self.assertEqual(defaults.diluted_shares, 100)

    def test_missing_required_fields_produce_explicit_warnings(self):
        defaults = build_dcf_defaults(
            info={},
            income_statement=pd.DataFrame(),
            cashflow_statement=pd.DataFrame(),
            balance_sheet=pd.DataFrame(),
        )

        self.assertEqual(defaults.base_revenue, 0)
        self.assertEqual(defaults.diluted_shares, 0)
        self.assertTrue(any("Revenue was unavailable" in warning for warning in defaults.warnings))
        self.assertTrue(any("Diluted shares were unavailable" in warning for warning in defaults.warnings))


if __name__ == "__main__":
    unittest.main()
