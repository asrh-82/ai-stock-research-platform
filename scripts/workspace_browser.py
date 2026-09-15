"""Real Chromium interaction and browser-storage reload checks. No market calls."""
from pathlib import Path
import json

from playwright.sync_api import sync_playwright, expect

OUT = Path('artifacts')
OUT.mkdir(exist_ok=True)
URL = 'http://127.0.0.1:8501'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1050})
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    try:
        page.goto(URL)
        page.get_by_role('heading', name='Stock Selection Workspace', exact=True).wait_for()
        page.get_by_role('button', name='Scan universe', exact=True).wait_for(state='visible')
        page.wait_for_function('''() => Array.from(document.querySelectorAll('button')).some(b => b.textContent.trim()==='Scan universe' && !b.disabled)''')
        page.get_by_text('Synthetic demo', exact=True).click()
        page.get_by_role('button', name='Scan universe', exact=True).click()
        page.get_by_role('button', name='Open candidate research', exact=True).wait_for()
        page.get_by_text('SYNTHETIC DEMO. Invented prices and outcomes; no market performance evidence.', exact=True).wait_for()
        # Wait for the end of this view, then capture its ranking rather than a
        # loading skeleton or the previous view's stale elements.
        page.get_by_role('button', name='Export frozen scan', exact=True).wait_for()
        page.get_by_role('button', name='Open candidate research', exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/'workspace_screen.png'), full_page=True)
        page.get_by_role('button', name='Open candidate research', exact=True).click()
        page.get_by_role('heading', name='FOXT · research case', exact=True).wait_for()
        page.get_by_text('Demo company names and price data are fictional. Live company/valuation requests are disabled here.', exact=True).wait_for()
        expect(page.get_by_role('button', name='Open candidate research', exact=True)).to_have_count(0)
        page.get_by_role('heading', name='FOXT · research case', exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/'workspace_research.png'), full_page=True)
        page.get_by_text('Paper & results', exact=True).click()
        page.get_by_role('button', name='Freeze model and my selections', exact=True).click()
        page.get_by_role('button', name='Update paper outcomes', exact=True).wait_for()
        page.wait_for_function('''() => Array.from(document.querySelectorAll('button')).some(b => b.textContent.trim()==='Update paper outcomes' && !b.disabled)''')
        page.get_by_role('button', name='Update paper outcomes', exact=True).click()
        page.get_by_text('Completed result is frozen. New vendor downloads cannot silently rewrite it.', exact=True).wait_for()
        page.get_by_role('button', name='Export cohort outcome', exact=True).wait_for()
        page.get_by_role('button', name='Export cohort outcome', exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/'workspace_paper.png'), full_page=True)
        # A reload replaces the Streamlit session; only browser storage can restore this record.
        page.reload()
        page.get_by_role('heading', name='Stock Selection Workspace', exact=True).wait_for()
        page.get_by_role('button', name='Open candidate research', exact=True).wait_for()
        page.get_by_text('Paper & results', exact=True).click()
        page.get_by_text('Completed result is frozen. New vendor downloads cannot silently rewrite it.', exact=True).wait_for()
        assert page.get_by_role('button', name='Update paper outcomes', exact=True).is_disabled()
        # Portable backup must include all saved events and pass strict Python validation in CI.
        page.get_by_text('Record storage and backups', exact=True).click()
        with page.expect_download() as download_info:
            page.get_by_role('button', name='Export private record', exact=True).click()
        download_info.value.save_as(str(OUT/'browser_record.json'))
        ledger = json.loads((OUT/'browser_record.json').read_text())
        assert ledger['revision'] == 3
        assert [e['kind'] for e in ledger['events']] == ['scan','decision','observation']
        assert page.locator('[data-testid="stException"]').count() == 0
        # A second browser context must not see another user's paper record.
        isolated = browser.new_context(viewport={"width": 1200,"height": 900})
        other = isolated.new_page(); other.goto(URL)
        other.get_by_role('heading',name='Stock Selection Workspace',exact=True).wait_for()
        other.get_by_text('Run a scan to populate the queue. Nothing is fetched until you request it.',exact=True).wait_for()
        assert other.get_by_role('button',name='Open candidate research',exact=True).count()==0
        # Import through the real file picker into an empty browser. No state injection.
        other.get_by_text('Record storage and backups', exact=True).click()
        other.locator('input[type="file"]').set_input_files(str(OUT/'browser_record.json'))
        expect(other.get_by_role('button', name='Restore backup', exact=True)).to_be_enabled()
        other.get_by_role('button', name='Restore backup', exact=True).click()
        other.get_by_role('button', name='Open candidate research', exact=True).wait_for()
        other.get_by_text('Paper & results', exact=True).click()
        other.get_by_text('Completed result is frozen. New vendor downloads cannot silently rewrite it.', exact=True).wait_for()
        assert other.get_by_role('button',name='Update paper outcomes',exact=True).is_disabled()
        assert other.locator('[data-testid="stException"]').count()==0
        isolated.close()
        assert not errors, errors
        (OUT/'workspace_browser.json').write_text(json.dumps({
            'status':'PASSED', 'environment':'CI-local real Chromium; synthetic data only',
            'checks':['scan','linked research','freeze paired choices','evaluate outcomes',
                      'reload restores 3 events','backup export','separate-browser isolation',
                      'backup import into an empty browser','settled-view screenshots','no page exceptions'],
            'page_errors':errors},indent=2))
        print('WORKSPACE_BROWSER=PASSED')
    except Exception:
        page.screenshot(path=str(OUT/'workspace_browser_failure.png'), full_page=True)
        (OUT/'workspace_browser_failure.txt').write_text(page.locator('body').inner_text())
        raise
    finally:
        context.close(); browser.close()
