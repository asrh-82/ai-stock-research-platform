import re
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from Utils.dcf_data import DCFDefaults


def _render_test_page():
    import pandas as pd

    from Utils.dcf_ui import render_dcf

    render_dcf(
        {
            "ticker": "TEST",
            "info": {"sector": "Technology", "industry": "Software"},
            "current_price": 75.0,
            "financials": pd.DataFrame(),
        }
    )


def _implied_price(app: AppTest) -> float:
    panel = next(
        element.value
        for element in app.markdown
        if 'class="dcf-value-panel' in element.value
    )
    match = re.search(r'class="dcf-value-price">\$([\d,.]+)', panel)
    if match is None:
        raise AssertionError("The implied-value panel did not contain a price.")
    return float(match.group(1).replace(",", ""))


class DCFInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.defaults = DCFDefaults(
            base_revenue=1_000_000_000,
            revenue_growth=(0.08, 0.07, 0.06, 0.05, 0.04),
            ebit_margin=(0.20, 0.20, 0.21, 0.21, 0.22),
            tax_rate=0.21,
            depreciation_pct_revenue=0.03,
            capex_pct_revenue=0.04,
            nwc_pct_incremental_revenue=0.05,
            wacc=0.09,
            terminal_growth=0.03,
            cash=100_000_000,
            debt=50_000_000,
            diluted_shares=10_000_000,
            sources={"base_revenue": "Test fixture"},
            warnings=(),
        )

    def test_scenario_switch_recalculates_full_valuation(self):
        with (
            patch("Utils.dcf_ui.get_cashflow_cached", return_value=pd.DataFrame()),
            patch("Utils.dcf_ui.get_balance_sheet_cached", return_value=pd.DataFrame()),
            patch("Utils.dcf_ui.build_dcf_defaults", return_value=self.defaults),
        ):
            app = AppTest.from_function(_render_test_page).run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.segmented_control[0].value, "Neutral")
            neutral_price = _implied_price(app)

            app.segmented_control[0].set_value("Bear").run()
            self.assertEqual(list(app.exception), [])
            bear_price = _implied_price(app)

            app.segmented_control[0].set_value("Bull").run()
            self.assertEqual(list(app.exception), [])
            bull_price = _implied_price(app)

        self.assertLess(bear_price, neutral_price)
        self.assertLess(neutral_price, bull_price)

    def test_research_layout_stays_flat_and_assumptions_optional(self):
        with (
            patch("Utils.dcf_ui.get_cashflow_cached", return_value=pd.DataFrame()),
            patch("Utils.dcf_ui.get_balance_sheet_cached", return_value=pd.DataFrame()),
            patch("Utils.dcf_ui.build_dcf_defaults", return_value=self.defaults),
        ):
            app = AppTest.from_function(_render_test_page).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.expander[0].label, "Model assumptions · optional")
        self.assertFalse(app.expander[0].proto.expanded)
        styles = next(
            element.value
            for element in app.markdown
            if ".dcf-value-panel" in element.value
        )
        self.assertNotIn("gradient", styles.lower())
        shadow_rules = re.findall(r"box-shadow:[^;]+;", styles.lower())
        self.assertEqual(shadow_rules, ["box-shadow: none !important;"])


if __name__ == "__main__":
    unittest.main()
