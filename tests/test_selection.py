import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from Utils.selection import (BENCHMARK, ScreenRules, demo_universe, digest,
                             parse_universe, rank_changes, scan)

NOW = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)


def fixture():
    return demo_universe()[0]


@pytest.mark.parametrize("text", ["", "AAPL", "https://example.com", "AAPL,../x", "AAPL,$BAD", "a"*1501])
def test_invalid_universe(text):
    with pytest.raises(ValueError):
        parse_universe(text)


def test_normalization_deduplication_and_bound():
    assert parse_universe(" msft,AAPL\nmsft ; brk-b ") == ("AAPL", "BRK-B", "MSFT")
    with pytest.raises(ValueError):
        parse_universe(",".join("T"+str(i) for i in range(26)))


def test_formula_is_exact_and_recent_month_excluded():
    b = fixture()
    r = scan(b, now=NOW)
    for row in r["ranked"]:
        close = b.histories[row["symbol"]].prices.Close
        assert row["momentum"] == pytest.approx(close.iloc[-22]/close.iloc[-253]-1)
    sym = b.symbols[0]
    b.histories[sym].prices.loc[b.histories[sym].prices.index[-10]:, ["Open", "High", "Low", "Close"]] *= 2
    second = scan(b, now=NOW)
    assert {r['symbol']: r['momentum'] for r in r['ranked']} == {r['symbol']: r['momentum'] for r in second['ranked']}


def test_future_rows_never_affect_signal():
    b, full = demo_universe()
    original = scan(b, now=NOW)
    full.histories[BENCHMARK] = b.histories[BENCHMARK]
    extended = scan(full, now=NOW)
    assert original["ranked"] == extended["ranked"]
    assert original["sources"] == extended["sources"]


def test_hash_reproducible_and_json_finite():
    one = scan(fixture(), now=NOW)
    two = scan(fixture(), now=NOW)
    assert one == two
    assert one["id"] == digest({k: v for k, v in one.items() if k != 'id'})


@pytest.mark.parametrize("mutation,reason", [
    ("missing", "Missing"), ("stale", "Missing"), ("nan", "Missing"),
    ("bad_ohlc", "Inconsistent"), ("zero_volume", "Zero-volume"),
    ("low_price", "reported-price"), ("low_liquidity", "dollar-volume"),
    ("missing_raw", "Reported close missing"), ("provider_error", "failed"),
])
def test_exclusions_never_silently_disappear(mutation, reason):
    b = fixture(); s = b.symbols[0]; p = b.histories[s].prices
    if mutation == "missing":
        b.histories[s].prices = p.drop(p.index[-30])
    elif mutation == "stale":
        b.histories[s].prices = p.iloc[:-1]
    elif mutation == "nan":
        p.loc[p.index[-1], "Close"] = np.nan
    elif mutation == "bad_ohlc":
        p.loc[p.index[-1], "High"] = 1
    elif mutation == "zero_volume":
        p.loc[p.index[-2], "Volume"] = 0
    elif mutation == "low_price":
        b.raw_close[s].iloc[-1] = 1
    elif mutation == "low_liquidity":
        p.loc[p.index[-20]:, "Volume"] = 1
    elif mutation == "missing_raw":
        del b.raw_close[s]
    else:
        b.errors[s] = "request failed"
    result = scan(b, now=NOW)
    assert len(result["ranked"]) == 5
    assert result["excluded"][0]["symbol"] == s
    assert reason in result["excluded"][0]["reason"]


def test_no_reference_fails_closed():
    b = fixture(); del b.histories[BENCHMARK]
    with pytest.raises(ValueError, match="Reference"):
        scan(b)


@pytest.mark.parametrize("field,value", [("minimum_price", -1), ("minimum_price", True),
                                         ("minimum_price", float('nan')), ("minimum_dollar_volume", float('inf'))])
def test_invalid_rules(field,value):
    with pytest.raises(ValueError):
        scan(fixture(), ScreenRules(**{field:value}))


def test_ties_have_equal_percentile_and_stable_ticker_order():
    b = fixture()
    for s in b.symbols:
        b.histories[s] = copy.deepcopy(b.histories[BENCHMARK])
    result = scan(b)
    assert [r['symbol'] for r in result['ranked']] == sorted(b.symbols)
    assert {r['percentile'] for r in result['ranked']} == {50.}


def test_rank_changes_require_same_experiment():
    r = scan(fixture(), now=NOW)
    assert set(rank_changes(r,r).values()) == {0}
    changed = copy.deepcopy(r); changed["rules"]["minimum_price"] += 1
    assert rank_changes(changed,r) == {}
    assert rank_changes(r,None) == {}


def test_live_scans_cannot_use_incomplete_today_or_stale_reference():
    b = fixture(); b.synthetic=False
    asof = b.histories[BENCHMARK].prices.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    for now in [asof+timedelta(hours=20), asof+timedelta(days=10)]:
        with pytest.raises(ValueError,match="Reference must"):
            scan(b,now=now)
    assert len(scan(b,now=asof+timedelta(days=1,hours=12))["ranked"]) == 6
