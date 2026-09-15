from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from Utils.market_data import normalize_symbol, snapshot_from_history


def sample():
    dates = pd.bdate_range(end="2026-09-14", periods=300)
    raw = pd.DataFrame({"Close": np.linspace(90, 120, 300), "Adj Close": np.linspace(85, 119, 300)}, index=dates)
    metadata = {"symbol": "USO", "instrumentType": "ETF", "currency": "USD", "exchangeTimezoneName": "America/New_York"}
    now = datetime(2026, 9, 15, 19, tzinfo=timezone.utc)
    return raw, metadata, now


def make(raw=None, metadata=None, now=None):
    default_raw, default_meta, default_now = sample()
    return snapshot_from_history("USO", default_raw if raw is None else raw,
                                 default_meta if metadata is None else metadata,
                                 default_now if now is None else now, "2024-01-01", "test-provider")


def test_etf_does_not_require_company_cashflows():
    result = make()
    assert result.reference_close == 120
    assert result.prices.iloc[-1] == 119
    assert result.provenance["instrument_type"] == "ETF"
    assert result.provenance["requested_end_exclusive"] == "2026-09-15"
    assert result.provenance["provenance_verified"] is False


def test_timezone_aware_daily_labels():
    raw, metadata, now = sample()
    raw.index = raw.index.tz_localize("America/New_York")
    assert make(raw, metadata, now).prices.index.tz is None


@pytest.mark.parametrize("field,value", [
    ("symbol", "OTHER"), ("instrumentType", "FUTURE"), ("instrumentType", "CRYPTOCURRENCY"),
    ("instrumentType", ""), ("currency", "EUR"), ("exchangeTimezoneName", "Asia/Tokyo"),
])
def test_security_identity_and_asset_class_gate(field, value):
    raw, metadata, now = sample()
    metadata[field] = value
    with pytest.raises(ValueError):
        make(raw, metadata, now)


def test_no_current_day_even_after_market_close():
    raw, _, _ = sample()
    raw.loc[pd.Timestamp("2026-09-15")] = [121, 120]
    with pytest.raises(ValueError, match="current NY day"):
        make(raw=raw, now=datetime(2026, 9, 15, 23, tzinfo=timezone.utc))


def test_no_stale_forecast():
    raw, _, _ = sample()
    with pytest.raises(ValueError, match="Stale"):
        make(raw=raw.iloc[:-10])


def test_missing_adjusted_prices_no_fallback():
    raw, _, _ = sample()
    with pytest.raises(ValueError, match="Both"):
        make(raw=raw.drop(columns="Adj Close"))


def test_missing_price_is_not_dropped():
    raw, _, _ = sample()
    raw.iloc[50, 1] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        make(raw=raw)


def test_reported_close_must_be_valid():
    raw, _, _ = sample()
    raw.iloc[-1, 0] = 0
    with pytest.raises(ValueError, match="Reported"):
        make(raw=raw)


@pytest.mark.parametrize("symbol", ["SPY,PLTR", "../../a", "", "<script>", "^GSPC"])
def test_symbol_validation(symbol):
    with pytest.raises(ValueError):
        normalize_symbol(symbol)


def test_symbol_normalization():
    assert normalize_symbol(" brk-b ") == "BRK-B"


def test_timestamp_cannot_be_naive():
    with pytest.raises(ValueError, match="timezone aware"):
        make(now=datetime(2026, 9, 15))
