"""Reject substantive bad bars; audit floating-point-only discrepancies."""
import numpy as np
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


def test_one_ulp_discrepancy_is_retained_without_repair():
    data = demo_snapshot(30).prices.copy()
    data.iloc[0, :4] = [100.0, 101.0, 99.0, np.nextafter(101.0, np.inf)]
    before = data.copy(deep=True)
    result = validate_daily(data)
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(result, before, check_dtype=False)
    assert result.Close.iloc[0] > result.High.iloc[0]


def test_roundoff_acceptance_is_explicit_in_manifest():
    snapshot = demo_snapshot(30)
    snapshot.prices.iloc[0, :4] = [100.0, 101.0, 99.0, np.nextafter(101.0, np.inf)]
    manifest = snapshot.manifest()
    assert manifest["floating_point_ohlc_discrepancies"] == 1
    assert manifest["ohlc_relative_tolerance"] == 8 * np.finfo(float).eps
