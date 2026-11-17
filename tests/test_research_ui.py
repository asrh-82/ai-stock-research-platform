from unittest.mock import patch
from streamlit.testing.v1 import AppTest


def report_page():
    from Utils.research_ui import render_research
    render_research({'ticker':'MTY.TO'})


def unsupported_page():
    from Utils.research_ui import render_research
    render_research({'ticker':'AAPL'})


def test_report_renders_evidence_and_dated_results_without_network():
    app=AppTest.from_function(report_page).run(timeout=15)
    assert not app.exception
    assert any('wholly owned' in m.value for m in app.markdown)
    assert len(app.dataframe[0].value)==3
    assert any('three completed' in w.value for w in app.warning)


def test_unverified_company_does_not_get_kahala_report():
    app=AppTest.from_function(unsupported_page).run(timeout=15)
    assert not app.exception
    assert not app.dataframe
    assert 'currently covers' in app.info[0].value


def test_report_remains_available_when_market_lookup_fails():
    app=AppTest.from_file('app.py')
    app.session_state['active_ticker']='MTY.TO'
    with patch('Utils.ui_sections.get_active_stock_context',return_value=None):
        app.run(timeout=15)
    assert not app.exception
    assert any('wholly owned' in m.value for m in app.markdown)
