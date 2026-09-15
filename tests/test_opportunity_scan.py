from dataclasses import replace
from datetime import datetime,timezone
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from Utils.opportunities import OpportunityConfig
from Utils.opportunity_scan import align_snapshots,export_scan,parse_symbols,run_scan,scan_table


def snapshot(symbol, periods=1500):
    rng=np.random.default_rng(sum(map(ord,symbol)))
    prices=pd.Series(100*np.exp(np.cumsum(rng.normal(.0001,.014,periods))),
                     index=pd.bdate_range('2020-01-02',periods=periods))
    return SimpleNamespace(symbol=symbol,prices=prices,provenance={'synthetic':True},warnings=())


def test_parse_and_deduplicate():
    assert parse_symbols('pltr, USO pltr\nAAPL')==('PLTR','USO','AAPL')


@pytest.mark.parametrize('text',['',' ', 'TSLA; rm', '../../etc', 'AAPL $MSFT', ','.join(f'A{i}' for i in range(13))])
def test_bad_symbols(text):
    with pytest.raises(ValueError): parse_symbols(text)


def test_alignment_missing_internal_day_rejected():
    a,b=snapshot('A'),snapshot('B')
    b.prices=b.prices.drop(b.prices.index[30])
    with pytest.raises(ValueError,match='Session dates'): align_snapshots(a,b)


def test_alignment_same_latest_date_required():
    a,b=snapshot('A'),snapshot('B')
    b.prices=b.prices.iloc[:-1]
    with pytest.raises(ValueError,match='last sessions'): align_snapshots(a,b)


def test_alignment_only_trims_before_common_start():
    a,b=snapshot('A'),snapshot('B')
    a.prices=a.prices.iloc[20:]
    h=align_snapshots(a,b)
    assert len(h)==1480
    assert h.index.equals(a.prices.index)


def test_each_failure_retained_and_family_fixed():
    def load(s,years,now):
        if s=='SPCX': raise ValueError('Not enough history')
        return snapshot(s)
    scan=run_scan('PLTR USO SPCX',loader=load)
    assert set(scan.results)=={'PLTR','USO'}
    assert set(scan.errors)=={'SPCX'}
    assert all(r.manifest['family_size']==3 for r in scan.results.values())
    assert list(scan_table(scan).Symbol)==['PLTR','USO','SPCX']
    assert all(r.summary['status']!='Research candidate' for r in scan.results.values())


def test_benchmark_fetched_once_with_same_cutoff():
    calls=[]
    now=datetime(2026,9,15,tzinfo=timezone.utc)
    def load(s,years,now):
        calls.append((s,years,now));return snapshot(s)
    run_scan('A B',now=now,loader=load)
    assert [s for s,_,_ in calls].count('SPY')==1
    assert all(t==now for _,_,t in calls)


def test_benchmark_failure_aborts_scan():
    def load(*args,**kwargs): raise ValueError('Provider down')
    with pytest.raises(ValueError,match='Provider down'): run_scan('PLTR',loader=load)


def test_wrong_benchmark_identity_aborts():
    with pytest.raises(ValueError,match='identity'):
        run_scan('PLTR',loader=lambda *a,**k:snapshot('WRONG'))


def test_asset_identity_mismatch_is_visible():
    def load(s,*a,**k): return snapshot(s if s=='SPY' else 'WRONG')
    scan=run_scan('PLTR',loader=load)
    assert 'identity' in scan.errors['PLTR']
    assert not scan.results


@pytest.mark.parametrize('kw',[{'benchmark':'SPY QQQ'}, {'benchmark':'PLTR'}, {'years':1}, {'now':datetime(2026,9,15)}])
def test_invalid_scan_rejected_before_network(kw):
    def load(*a,**k): raise AssertionError('Network must not run')
    with pytest.raises(ValueError): run_scan('PLTR',loader=load,**kw)


def test_batch_export_retains_failure():
    def load(s,*a,**k):
        if s=='BAD': raise ValueError('Missing')
        return snapshot(s)
    scan=run_scan('PLTR BAD',loader=load)
    with ZipFile(BytesIO(export_scan(scan))) as z:
        assert 'PLTR_evidence.zip' in z.namelist()
        assert 'BAD' in z.read('scan.json').decode()
        assert 'Data/model blocked' in z.read('scan.csv').decode()
