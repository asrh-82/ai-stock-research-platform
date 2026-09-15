import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from Utils.selection import BENCHMARK, demo_universe, scan, digest, canonical
from Utils.paper_record import (append_event, empty_ledger, evaluate_cohort, freeze_selection,
                               parse_backup, validate_ledger, verify_decision)

NOW = datetime(2026,9,15,12,tzinfo=timezone.utc)


def example():
    b,f=demo_universe(); s=scan(b,now=NOW); d=freeze_selection(s,[r["symbol"] for r in s["ranked"][:5]],now=NOW)
    return b,f,s,d


@pytest.mark.parametrize('cash',[0,-1,True,float('nan'),float('inf'),'100',1e12])
def test_bad_capital(cash):
    _,_,s,_=example()
    with pytest.raises(ValueError): freeze_selection(s,[],cash,now=NOW)


@pytest.mark.parametrize('selected', [['BOGUS'],['ALFA','ALFA'],['ALFA','BRAV','CHAR','DELT','ECHO','FOXT']])
def test_bad_selections(selected):
    _,_,s,_=example()
    with pytest.raises(ValueError): freeze_selection(s,selected,now=NOW)


def test_choice_and_universe_freeze_copy():
    _,_,s,d=example(); original=copy.deepcopy(d)
    s['ranked'][0]['momentum']=999
    assert d == original
    assert d['model']==d['eligible'][:5]
    verify_decision(d)


def test_live_date_not_backfilled_before_registration():
    _,_,s,_=example(); s['synthetic']=False; s['asof']='2026-09-14'
    s['created_at']=NOW.isoformat(); s['id']=digest({k:v for k,v in s.items() if k!='id'})
    d=freeze_selection(s,[],now=NOW)
    assert d['earliest_entry_date']=='2026-09-16'
    with pytest.raises(ValueError): freeze_selection(s,[],now=NOW-timedelta(days=2))
    with pytest.raises(ValueError): freeze_selection(s,[],now=NOW+timedelta(days=10))


def test_same_choices_produce_same_returns():
    _,f,_,d=example(); o=evaluate_cohort(d,f,now=NOW)
    assert o['status']=='COMPLETE'
    assert o['books']['Model']['net_return']==pytest.approx(o['books']['User']['net_return'])
    assert len(o['curve'])==21
    assert o['first_session'] >= d['earliest_entry_date']
    assert -1 <= o['rank_ic'] <= 1


def test_empty_user_portfolio_is_cash():
    _,f,s,_=example();d=freeze_selection(s,[],now=NOW);o=evaluate_cohort(d,f,now=NOW)
    assert o['books']['User']['net_return']==0
    assert o['books']['User']['fees']==0
    assert o['books']['User']['maximum_drawdown']==0


def test_no_next_session_has_no_returns():
    b,_,_,d=example();o=evaluate_cohort(d,b,now=NOW)
    assert o['status']=='AWAITING_NEXT_SESSION'
    assert not o['books'] and o['rank_ic'] is None


def test_partial_does_not_claim_completion_or_ranking_evidence():
    b,f,_,d=example()
    for h in f.histories.values(): h.prices=h.prices.iloc[:len(b.histories[BENCHMARK].prices)+5]
    o=evaluate_cohort(d,f,now=NOW)
    assert o['status']=='IN_PROGRESS' and o['sessions']==5
    assert o['rank_ic'] is None
    assert o['books']['Model']['estimated_liquidation_equity'] < o['books']['Model']['ending_equity']


@pytest.mark.parametrize('mutation',['missing_name','missing_first','missing_middle','zero_volume','source_mismatch'])
def test_no_survivor_only_results(mutation):
    b,f,_,d=example();s=d['eligible'][-1];start=len(b.histories[s].prices)
    if mutation=='missing_name':del f.histories[s]
    elif mutation=='missing_first':f.histories[s].prices=f.histories[s].prices.drop(f.histories[s].prices.index[start])
    elif mutation=='missing_middle':f.histories[s].prices=f.histories[s].prices.drop(f.histories[s].prices.index[start+4])
    elif mutation=='zero_volume':f.histories[s].prices.iloc[start,4]=0
    else:f.synthetic=False
    with pytest.raises(ValueError):evaluate_cohort(d,f,now=NOW)


def test_rebased_adjusted_prices_do_not_change_returns():
    _,f,_,d=example();o=evaluate_cohort(d,f,now=NOW)
    for h in f.histories.values():h.prices.loc[:,['Open','High','Low','Close']]*=7
    second=evaluate_cohort(d,f,now=NOW)
    for k in o['books']:
        assert o['books'][k]['net_return']==pytest.approx(second['books'][k]['net_return'])


def test_constant_prices_pay_both_sides_and_initial_drawdown():
    _,f,_,d=example()
    for h in f.histories.values():h.prices.loc[:,['Open','High','Low','Close']]=100.
    o=evaluate_cohort(d,f,now=NOW)
    expected=(1-.001)*(1-.0005)/((1+.001)*(1+.0005))-1
    assert o['books']['Model']['net_return']==pytest.approx(expected)
    assert o['books']['Model']['maximum_drawdown']==pytest.approx(expected)
    assert o['rank_ic'] is None


def ledger():
    _,f,s,d=example();o=evaluate_cohort(d,f,now=NOW)
    doc=append_event(empty_ledger(),'scan',s,now=NOW)
    doc=append_event(doc,'decision',d,now=NOW)
    return append_event(doc,'observation',o,now=NOW)


def test_backup_roundtrip():
    doc=ledger()
    assert parse_backup(canonical(doc).encode())==doc
    assert doc['revision']==3


@pytest.mark.parametrize('mutation',['payload','previous','revision','schema','unknown','duplicate'])
def test_corrupt_records_rejected(mutation):
    doc=ledger()
    if mutation=='payload':doc['events'][0]['payload']['ranked'][0]['momentum']=9
    elif mutation=='previous':doc['events'][1]['previous']='wrong'
    elif mutation=='revision':doc['revision']=100
    elif mutation=='schema':doc['schema']=100
    elif mutation=='unknown':doc['events'][0]['kind']='order'
    else:doc['events'].append(doc['events'][0]);doc['revision']+=1
    with pytest.raises(ValueError):validate_ledger(doc)


def test_orphan_decision_and_duplicate_scan_rejected():
    _,_,s,d=example()
    with pytest.raises(ValueError):append_event(empty_ledger(),'decision',d)
    doc=append_event(empty_ledger(),'scan',s)
    with pytest.raises(ValueError):append_event(doc,'scan',s)


def test_same_scan_cannot_be_retuned():
    _,_,s,d=example(); doc=append_event(append_event(empty_ledger(),'scan',s),'decision',d)
    changed=freeze_selection(s,[],now=NOW+timedelta(seconds=1))
    with pytest.raises(ValueError):append_event(doc,'decision',changed)


@pytest.mark.parametrize('payload',[b'garbage',b'[]',b'\xff',b'x'*2000001])
def test_invalid_backup(payload):
    with pytest.raises(ValueError):parse_backup(payload)


def test_browser_json_number_roundtrip_preserves_hashes():
    import json
    import subprocess
    doc=ledger()
    browser=subprocess.run(['node','-e','let s="";process.stdin.on("data",d=>s+=d);process.stdin.on("end",()=>process.stdout.write(JSON.stringify(JSON.parse(s))));'],
                           input=canonical(doc),text=True,capture_output=True,check=True)
    assert validate_ledger(json.loads(browser.stdout)) == doc


def test_recorded_outcome_is_not_overwritten():
    doc=ledger()
    with pytest.raises(ValueError,match='advance'):
        append_event(doc,'observation',doc['events'][-1]['payload'])
