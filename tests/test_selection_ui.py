"""Component storage is mocked here; actual persistence is covered in the browser."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

st_testing=pytest.importorskip('streamlit.testing.v1')
from Utils.paper_record import empty_ledger, records
from Utils.selection import scan, demo_universe

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def vault():
    state={'document':empty_ledger()}
    def component(**kwargs):
        cmd=kwargs.get('command')
        if cmd:
            assert cmd['base_revision']==state['document']['revision']
            state['document']=copy.deepcopy(cmd['document'])
        return {'document':copy.deepcopy(state['document']), 'ready':True, 'error':None,
                'ack':cmd['id'] if cmd else None}
    with patch('Utils.paper_vault.vault_component',side_effect=component):
        yield state


def page():
    return st_testing.AppTest.from_file(str(ROOT/'app.py'),default_timeout=45).run()


def demo(at):
    at.radio(key='ws_source').set_value('Synthetic demo').run()
    at.button(key='ws_scan').click().run()
    at.run()
    assert not at.exception
    return at


def test_start_is_network_independent(vault):
    with patch('Utils.selection_ui.cached_universe', side_effect=AssertionError('network on startup')):
        at=page()
    assert not at.exception
    assert at.title[0].value=='Stock Selection Workspace'
    assert not records(vault['document'],'scan')


def test_scan_records_all_names_and_routes_to_research(vault):
    at=demo(page())
    assert len(records(vault['document'],'scan'))==1
    assert any('6'==str(m.value) for m in at.metric)
    at.selectbox(key='ws_selected').set_value('ALFA').run()
    at.button(key='ws_open_research').click().run()
    assert not at.exception
    assert at.radio(key='ws_view').value=='Research'
    assert at.selectbox(key='ws_research_symbol').value=='ALFA'
    assert any('ALFA' in x.value for x in at.subheader)


def test_freeze_and_evaluate_demo(vault):
    at=demo(page())
    at.radio(key='ws_view').set_value('Paper & results').run()
    at.button(key='ws_freeze').click().run(); at.run()
    assert not at.exception
    assert len(records(vault['document'],'decision'))==1
    assert at.button(key='ws_freeze').disabled
    at.button(key='ws_update').click().run();at.run()
    assert not at.exception
    assert len(at.metric)==4
    assert at.button(key='ws_update').disabled
    assert records(vault['document'],'observation')[-1]['status']=='COMPLETE'


def test_reopen_app_recovers_record(vault):
    at=demo(page())
    restored=page()
    assert not restored.exception
    assert len(records(vault['document'],'scan'))==1
    assert restored.selectbox(key='ws_selected').value


def test_provider_failure_is_visible_not_a_fake_scan(vault):
    at=page()
    with patch('Utils.selection_ui.cached_universe',side_effect=ValueError('Provider refused the request')):
        at.button(key='ws_scan').click().run()
    assert not at.exception
    assert any('Provider refused' in e.value for e in at.error)
    assert not records(vault['document'],'scan')


def test_advanced_page_navigation_preserved(vault):
    at=page();at.switch_page('pages/2_Walk_Forward.py').run()
    assert not at.exception and at.title[0].value=='Walk-forward Research'


@pytest.mark.parametrize("section,target", [
    ("DCF scenarios", "Utils.dcf_ui.render_dcf"),
    ("Assumption uncertainty", "Utils.monte_carlo_ui.render_monte_carlo"),
])
def test_research_routes_selected_company_to_existing_valuation(vault, section, target):
    import pandas as pd
    from Utils.paper_record import append_event
    batch, _ = demo_universe()
    batch.synthetic = False
    recorded = (batch.histories['SPY'].prices.index[-1].tz_localize('America/New_York')
                + pd.Timedelta(days=1, hours=12)).to_pydatetime()
    frozen = scan(batch, now=recorded)
    vault['document'] = append_event(empty_ledger(), 'scan', frozen)
    at = page()
    at.selectbox(key='ws_selected').set_value('ALFA').run()
    at.button(key='ws_open_research').click().run()
    context = {'ticker': 'ALFA', 'company_name': 'Mock company',
               'info': {'sector': 'Technology'}, 'current_price': 50.,
               'financials': pd.DataFrame()}
    with patch('Utils.selection_ui.company_context', return_value=context):
        at.button(key='ws_load_company').click().run()
    with patch(target) as renderer:
        at.radio(key='ws_valuation_ALFA').set_value(section).run()
        assert renderer.call_count == 1
        assert renderer.call_args.args[0]['ticker'] == 'ALFA'
    assert not at.exception
    assert records(vault['document'], 'scan')[-1] == frozen
