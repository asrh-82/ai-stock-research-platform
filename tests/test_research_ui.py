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
    assert any(x.label == "Generate report" for x in app.button)
    assert not any("Kahala Brands and its public parent" in x.value for x in app.markdown)


def test_report_remains_available_when_market_lookup_fails():
    app=AppTest.from_file('app.py')
    app.session_state['active_ticker']='MTY.TO'
    with patch('Utils.ui_sections.get_active_stock_context',return_value=None):
        app.run(timeout=15)
    assert not app.exception
    assert any('wholly owned' in m.value for m in app.markdown)


def test_report_form_runs_pipeline_and_clears_old_result_on_failure():
    from Utils.filings.report import FilingReport
    from Utils.filings.sources import ResearchError
    report = FilingReport("Example Inc.", "Example Inc.", "EXM", "10-Q", "2026-11-01",
                          "2026-09-30", "2026-11-17", "https://www.sec.gov/filing.htm")
    app = AppTest.from_function(unsupported_page).run()
    with patch("Utils.research_ui.research_company", return_value=report) as fetch:
        app.text_input[0].set_value("EXM")
        app.button[0].click().run()
        fetch.assert_called_once_with("EXM")
    assert not app.exception and any("Example Inc." in x.value for x in app.markdown)
    with patch("Utils.research_ui.research_company", side_effect=ResearchError("Source unavailable")):
        app.text_input[0].set_value("MISSING")
        app.button[0].click().run()
    assert app.error[0].value == "Source unavailable"
    assert not any("Example Inc." in x.value for x in app.markdown)


def test_direct_report_route_skips_dashboard_provider():
    app = AppTest.from_file("app.py")
    app.query_params["research"] = "Kahala Brands"
    with patch("Utils.ui_sections.get_active_stock_context", side_effect=AssertionError("Market lookup invoked")):
        app.run(timeout=15)
    assert not app.exception
    assert app.text_input[0].value == "Kahala Brands"
