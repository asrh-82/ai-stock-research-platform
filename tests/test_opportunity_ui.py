from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from Utils.opportunities import OpportunityConfig,forecast_opportunity
from Utils.opportunity_scan import ScanResult


def page():
    from Utils.opportunity_ui import render_opportunities
    render_opportunities()


def fixture():
    rng=np.random.default_rng(12)
    h=pd.DataFrame(100*np.exp(rng.normal(0,.01,(1500,2)).cumsum(0)),
                    columns=['asset','benchmark'],index=pd.bdate_range('2020-01-02',periods=1500))
    r=forecast_opportunity(h,'PLTR',provenance={'synthetic':True})
    return ScanResult(('PLTR','SPCX'),'SPY','2026-09-15T20:00:00+00:00',OpportunityConfig(),
                       {'PLTR':r},{'SPCX':'Insufficient history'},5)


def button(app,label): return next(b for b in app.button if b.label==label)


def test_initial_render_has_no_data_request():
    with patch('Utils.opportunity_ui.run_scan') as run:
        app=AppTest.from_function(page).run(timeout=20)
        assert not list(app.exception)
        run.assert_not_called()


def test_scan_render_and_export():
    with patch('Utils.opportunity_ui.run_scan',return_value=fixture()) as run:
        app=AppTest.from_function(page).run(timeout=20)
        button(app,'Run opportunity scan').click().run(timeout=20)
        assert not list(app.exception)
        assert any(m.label=='Research candidates' for m in app.metric)
        assert any('SPCX' in e.value for e in app.error)
        button(app,'Prepare complete scan evidence').click().run(timeout=20)
        assert not list(app.exception)
        assert run.call_count==1


def test_failure_clears_previous_results():
    with patch('Utils.opportunity_ui.run_scan',side_effect=[fixture(),ValueError('Provider failed')]):
        app=AppTest.from_function(page).run(timeout=20)
        button(app,'Run opportunity scan').click().run(timeout=20)
        button(app,'Run opportunity scan').click().run(timeout=20)
        assert not list(app.exception)
        assert any('Provider failed' in e.value for e in app.error)
        assert not any(m.label=='Research candidates' for m in app.metric)


def test_blocked_only_run_is_displayed():
    scan=ScanResult(('SPCX',),'SPY','2026-09-15',OpportunityConfig(),{}, {'SPCX':'Not enough history'},5)
    with patch('Utils.opportunity_ui.run_scan',return_value=scan):
        app=AppTest.from_function(page).run(timeout=20)
        button(app,'Run opportunity scan').click().run(timeout=20)
        assert not list(app.exception)
        assert any('Not enough history' in e.value for e in app.error)
