import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from Utils.dcf import DCFInputs
from Utils.dcf_data import DCFDefaults


def _render_monte_carlo_page():
    import pandas as pd

    from Utils.monte_carlo_ui import render_monte_carlo

    render_monte_carlo(
        {
            "ticker": "TEST",
            "info": {"sector": "Technology", "industry": "Software"},
            "current_price": 75.0,
            "financials": pd.DataFrame(),
        }
    )


def _render_backtest_page():
    from Utils.backtesting_ui import render_backtesting

    render_backtesting({"ticker": "TEST"})


def _price_history(ticker, _period):
    length = 600
    index = pd.bdate_range("2022-01-03", periods=length)
    base_return = 0.00025 if ticker == "SPY" else 0.0004
    cycle = 0 if ticker == "SPY" else np.sin(np.arange(length) / 18) * 0.006
    returns = base_return + cycle
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + returns)}, index=index)


class PhaseEightInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.inputs = DCFInputs(
            base_revenue=1_000,
            revenue_growth=(0.08, 0.07, 0.06, 0.05, 0.04),
            ebit_margin=(0.20, 0.20, 0.21, 0.21, 0.22),
            tax_rate=0.21,
            depreciation_pct_revenue=0.03,
            capex_pct_revenue=0.04,
            nwc_pct_incremental_revenue=0.05,
            wacc=0.09,
            terminal_growth=0.03,
            cash=100,
            debt=50,
            diluted_shares=10,
        )
        self.defaults = DCFDefaults(
            base_revenue=1_000_000_000,
            revenue_growth=self.inputs.revenue_growth,
            ebit_margin=self.inputs.ebit_margin,
            tax_rate=self.inputs.tax_rate,
            depreciation_pct_revenue=self.inputs.depreciation_pct_revenue,
            capex_pct_revenue=self.inputs.capex_pct_revenue,
            nwc_pct_incremental_revenue=self.inputs.nwc_pct_incremental_revenue,
            wacc=self.inputs.wacc,
            terminal_growth=self.inputs.terminal_growth,
            cash=100_000_000,
            debt=50_000_000,
            diluted_shares=10_000_000,
            sources={"base_revenue": "Test fixture"},
            warnings=(),
        )

    def test_monte_carlo_renders_automatic_result(self):
        with patch(
            "Utils.monte_carlo_ui._automatic_inputs",
            return_value=(self.inputs, self.defaults),
        ):
            app = AppTest.from_function(_render_monte_carlo_page).run(timeout=15)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.expander[0].label, "Simulation settings · optional")
        self.assertFalse(app.expander[0].proto.expanded)
        self.assertTrue(
            any("median simulated value" in element.value for element in app.markdown)
        )

    def test_backtest_renders_and_strategy_control_recalculates(self):
        with patch(
            "Utils.backtesting_ui.get_price_history_cached",
            side_effect=_price_history,
        ):
            app = AppTest.from_function(_render_backtest_page).run(timeout=15)
            self.assertEqual(list(app.exception), [])
            strategy = next(
                element for element in app.selectbox if element.label == "Strategy"
            )
            strategy.set_value("Price Momentum").run(timeout=15)

        self.assertEqual(list(app.exception), [])
        self.assertTrue(
            any('class="bt-summary"' in element.value for element in app.markdown)
        )
        settings = next(
            element
            for element in app.expander
            if element.label == "Strategy and execution settings · optional"
        )
        self.assertFalse(settings.proto.expanded)


if __name__ == "__main__":
    unittest.main()
