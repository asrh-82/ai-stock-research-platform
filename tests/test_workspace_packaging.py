from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_container_includes_pages_and_browser_component():
    docker=(ROOT/'Dockerfile.vercel').read_text()
    assert 'COPY pages ./pages' in docker
    assert 'COPY components ./components' in docker
    assert (ROOT/'components/paper_vault/index.html').is_file()


def test_old_math_modules_are_not_replaced_by_the_workspace():
    router=(ROOT/'app.py').read_text()
    for page in ['0_Stock_Selection.py','1_Validation_Lab.py','2_Walk_Forward.py','3_Company_Tools.py']:
        assert page in router and (ROOT/'pages'/page).is_file()


def test_vault_does_not_need_cdn_or_embed_secrets():
    html=(ROOT/'components/paper_vault/index.html').read_text()
    assert 'localStorage.setItem' in html and 'navigator.locks.request' in html
    assert '<script src=' not in html and 'API_KEY' not in html
    assert 'event.source!==window.parent' in html
