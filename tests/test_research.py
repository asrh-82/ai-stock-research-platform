from copy import deepcopy
import pytest
from Utils.research import load_evidence, validate_evidence, calculated_metrics, resolve_research_subject, render_report


def test_kahala_resolves_to_explicit_parent():
    result=resolve_research_subject('Kahala Brands')
    assert result['symbol']=='MTY.TO'
    assert not result['standalone_public_stock']
    assert 'parent' in result['name']
    assert resolve_research_subject('KHL') is None


def test_report_keeps_missing_kahala_financials_unknown():
    text=render_report()
    assert 'not separately established' in text
    assert 'MTY consolidated' in text
    assert 'not a live' in text
    assert 'CAD43 is neither an agreed' in text


def test_report_arithmetic_is_reproducible():
    m=calculated_metrics(load_evidence())
    assert m['trailing_eps_approx']==pytest.approx(4.79)
    assert m['trailing_fcf_approx_m']==pytest.approx(127.305)
    assert m['bank_debt_minus_cash_m']==pytest.approx(515.127)
    assert m['lease_liabilities_m']==pytest.approx(469.214)


def test_historical_report_cannot_use_future_verification():
    with pytest.raises(ValueError,match='historical'):
        render_report(as_of='2025-12-31')


@pytest.mark.parametrize('mutation', ['scope','currency','source','value','future'])
def test_evidence_guardrails(mutation):
    e=deepcopy(load_evidence())
    if mutation=='scope':e['facts'][0]['scope']='Kahala standalone'
    if mutation=='currency':e['quote']['currency']='USD'
    if mutation=='source':e['facts'][0]['source']='invented'
    if mutation=='value':e['facts'][0]['value']=float('nan')
    if mutation=='future':e['sources']['ceo']['published_on']='2027-01-01'
    with pytest.raises(ValueError):validate_evidence(e,'2026-11-17')


def test_future_run_marks_stale_snapshot():
    assert 'Stale snapshot' in render_report(as_of='2027-01-01')
