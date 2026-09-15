"""Provider behavior is tested using mocked responses, never invented market history."""
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from Utils.research_data import demo_snapshot, download_daily


def fake_provider(monkeypatch, raw=None, error=None):
    calls = []
    class Ticker:
        def __init__(self, symbol):
            calls.append({"symbol": symbol})
        def history(self, **kwargs):
            calls.append(kwargs)
            if error:
                raise error
            return raw
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=Ticker, __version__="mock-only"))
    return calls


def test_explicit_adjustment_dates_and_provenance(monkeypatch):
    raw = demo_snapshot(100).prices
    raw.index = raw.index.tz_localize("America/New_York")
    calls = fake_provider(monkeypatch, raw)
    result = download_daily("spy", "2010-01-01", "2011-01-01")
    assert calls[0]["symbol"] == "SPY"
    assert calls[1]["auto_adjust"] is True
    assert calls[1]["keepna"] is True
    assert calls[1]["repair"] is False
    assert calls[1]["interval"] == "1d"
    assert calls[1]["end"] == "2011-01-01"
    assert result.prices.index.tz is None
    assert result.metadata["provider_version"] == "mock-only"
    assert result.metadata["requested_end_exclusive"] == "2011-01-01"


@pytest.mark.parametrize("raw", [None, pd.DataFrame()])
def test_no_data_means_no_synthetic_fallback(monkeypatch, raw):
    fake_provider(monkeypatch, raw)
    with pytest.raises(ValueError, match="no history"):
        download_daily("SPY", "2010-01-01", "2011-01-01")


def test_provider_failure_is_explicit(monkeypatch):
    fake_provider(monkeypatch, error=RuntimeError("network failure"))
    with pytest.raises(ValueError, match="no synthetic fallback"):
        download_daily("SPY", "2010-01-01", "2011-01-01")


@pytest.mark.parametrize("symbol", ["SPY,QQQ", "https://example.com", "", "A B"])
def test_invalid_tickers_do_not_trigger_download(monkeypatch, symbol):
    calls = fake_provider(monkeypatch)
    with pytest.raises(ValueError):
        download_daily(symbol, "2010-01-01", "2011-01-01")
    assert calls == []


def test_provider_does_not_silently_fix_missing_rows(monkeypatch):
    raw = demo_snapshot(100).prices
    raw.iloc[5, 0] = float("nan")
    fake_provider(monkeypatch, raw)
    with pytest.raises(ValueError, match="Missing"):
        download_daily("SPY", "2010-01-01", "2011-01-01")
