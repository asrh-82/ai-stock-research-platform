"""Bad-bar diagnostics must not silently relax input validation."""
import pandas as pd
import pytest

from Utils.research_data import demo_snapshot, validate_daily


@pytest.mark.parametrize("violation", [1e-12, 1.0])
def test_ohlc_diagnostics_preserve_strict_rejection(violation):
    data = demo_snapshot(30).prices.copy()
    data.iloc[0, :4] = [100.0, 101.0, 99.0, 101.0 + violation]
    before = data.copy(deep=True)
    with pytest.raises(ValueError, match=r"Inconsistent OHLC bounds: 1 rows; first session 2010-01-01; maximum relative violation"):
        validate_daily(data)
    pd.testing.assert_frame_equal(data, before)
