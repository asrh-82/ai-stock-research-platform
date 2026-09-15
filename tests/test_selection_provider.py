from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from Utils.selection import BENCHMARK, demo_universe
from Utils.selection_data import load_universe


def test_provider_called_only_explicitly_and_all_failures_retained():
    _,b=demo_universe()
    def fake(symbol,start,end):
        assert end=='2026-09-15'
        if symbol=='FAIL':raise RuntimeError('unavailable')
        return b.histories[BENCHMARK], b.raw_close[BENCHMARK]
    with patch('Utils.selection_data.fetch_history',side_effect=fake):
        result=load_universe('ALFA,FAIL',now=datetime(2026,9,15,20,tzinfo=timezone.utc))
    assert not result.synthetic
    assert 'FAIL' in result.errors and 'FAIL' not in result.histories
    assert result.symbols==('ALFA','FAIL')


def test_reference_failure_never_becomes_demo():
    with patch('Utils.selection_data.fetch_history',side_effect=ValueError('provider down')):
        with pytest.raises(ValueError,match='Reference data unavailable'):
            load_universe('ALFA,BRAV')


def test_history_request_excludes_current_new_york_day():
    _,b=demo_universe(); calls=[]
    def fake(s,start,end):
        calls.append((start,end));return b.histories[BENCHMARK],b.raw_close[BENCHMARK]
    with patch('Utils.selection_data.fetch_history',side_effect=fake):
        load_universe('ALFA,BRAV',now=datetime(2026,9,15,2,tzinfo=timezone.utc))
    assert all(end=='2026-09-14' for _,end in calls)


def test_invalid_start():
    with pytest.raises(ValueError):load_universe('ALFA,BRAV',start='2050-01-01')
