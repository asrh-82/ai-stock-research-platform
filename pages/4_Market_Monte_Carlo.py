"""Independent market-risk workspace; does not load company fundamentals."""
from Utils.market_monte_carlo_ui import render_market_monte_carlo
from Utils.ui_sections import apply_styles

apply_styles()
render_market_monte_carlo()
