import json
from decimal import Decimal
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from Utils.filings import mty, sec
from Utils.filings.pipeline import research_company
from Utils.filings.sources import ResearchError, SourceClient

ACC = "0000000042-26-000003"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/42/000000004226000003/issuer.htm"
SUBMISSIONS = sec.DATA + "/submissions/CIK0000000042.json"
FACTS = sec.DATA + "/api/xbrl/companyfacts/CIK0000000042.json"


def inline_example():
    return """
    <ix:nonnumeric name="dei:EntityCentralIndexKey">0000000042</ix:nonnumeric>
    <ix:nonnumeric name="dei:DocumentType">10-Q</ix:nonnumeric>
    <ix:nonnumeric name="dei:DocumentPeriodEndDate">September 30, 2026</ix:nonnumeric>
    <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
    <xbrli:context id="quarter"><xbrli:entity><xbrli:identifier>0000000042</xbrli:identifier></xbrli:entity>
      <xbrli:period><xbrli:startdate>2026-07-01</xbrli:startdate><xbrli:enddate>2026-09-30</xbrli:enddate></xbrli:period></xbrli:context>
    <ix:nonfraction id="rev" name="us-gaap:Revenues" contextref="quarter" unitref="usd" scale="6" format="ixt:num-dot-decimal">12.5</ix:nonfraction>
    """


class Documents:
    started_at = "2026-11-17T12:00:00+00:00"

    def __init__(self):
        self.sources = {}
        self.html = inline_example()
        self.row = {
            "start": "2026-07-01",
            "end": "2026-09-30",
            "val": 12_500_000,
            "form": "10-Q",
            "filed": "2026-11-03",
            "accn": ACC,
        }
        self.directory = {"0": {"title": "Example Company Inc.", "ticker": "EXPL", "cik_str": 42}}

    def get(self, url):
        assert url == ARCHIVE
        return self.html.encode()

    def json(self, url):
        if url == sec.DIRECTORY:
            return self.directory
        if url == FACTS:
            return {"cik": 42, "facts": {"us-gaap": {"Revenues": {"units": {"USD": [self.row]}}}}}
        assert url == SUBMISSIONS
        return {
            "name": "Example Company Inc.",
            "cik": "0000000042",
            "tickers": ["EXPL"],
            "filings": {
                "recent": {
                    "accessionNumber": [ACC],
                    "primaryDocument": ["issuer.htm"],
                    "form": ["10-Q"],
                    "reportDate": ["2026-09-30"],
                    "filingDate": ["2026-11-03"],
                }
            },
        }


def test_unseen_issuer_reconciles_tag_units_scale_and_context():
    sources = Documents()
    r = research_company("Example Company", sources)
    assert len(r.facts) == 1
    f = r.facts[0]
    assert (r.ticker, f.value, f.start, f.end, f.unit) == (
        "EXPL",
        Decimal(12500000),
        "2026-07-01",
        "2026-09-30",
        "USD",
    )
    assert f.sources[0].url == ARCHIVE + "#rev"
    assert "12.500" in r.markdown() and "Kahala" not in r.markdown()
    assert Decimal(json.loads(r.audit())["facts"][0]["value"]) == f.value


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("12.5</ix:nonfraction>", "12.6</ix:nonfraction>"),
        ('scale="6"', 'scale="3"'),
        ("iso4217:USD", "iso4217:CAD"),
        ("2026-07-01", "2026-01-01"),
        ("</xbrli:entity>", "<xbrli:segment>Europe</xbrli:segment></xbrli:entity>"),
        ("<xbrli:identifier>0000000042", "<xbrli:identifier>0000000017"),
        ("September 30, 2026", "June 30, 2026"),
        (">0000000042</ix:nonnumeric>", ">0000000043</ix:nonnumeric>"),
    ],
)
def test_conflicting_or_wrong_scope_facts_cannot_enter_report(old, new):
    sources = Documents()
    sources.html = sources.html.replace(old, new)
    with pytest.raises(ResearchError):
        research_company("EXPL", sources)


def test_identical_period_in_different_filing_cannot_fill_gap():
    sources = Documents()
    sources.row["accn"] = "0000000042-26-000004"
    with pytest.raises(ResearchError):
        research_company("EXPL", sources)


def test_report_preserves_verified_zero():
    sources = Documents()
    sources.row["val"] = 0
    sources.html = sources.html.replace("12.5</ix:nonfraction>", "0</ix:nonfraction>")
    assert research_company("EXPL", sources).facts[0].value == 0


def test_restatement_requires_review_and_future_filing_is_excluded():
    index = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q/A", "10-Q", "10-Q/A"],
                "accessionNumber": ["x", "b", "a", "c"],
                "filingDate": ["2027-02-03", "2026-11-16", "2026-11-03", "2026-11-17"],
                "reportDate": ["2026-12-31", "2026-09-30", "2026-09-30", "2025-06-30"],
                "primaryDocument": ["x.htm", "b.htm", "a.htm", "c.htm"],
            }
        }
    }
    assert sec.select_filing(index, "2026-11-12")["accessionNumber"] == "a"
    with pytest.raises(ResearchError, match="amendment"):
        sec.select_filing(index, "2026-11-17")


def test_ambiguous_listings_require_ticker():
    d = Documents().directory
    d["1"] = {"title": "Example Company Inc.", "ticker": "EXPL-A", "cik_str": 42}
    with pytest.raises(ResearchError, match="EXPL-A"):
        sec.resolve_company("Example Company", d)


def test_quarter_does_not_take_nine_month_operating_results():
    row = Documents().row
    ytd = {**row, "start": "2026-01-01", "val": 38_000_000}
    assert sec.choose_current([row, ytd], "income", row["end"], "10-Q") == ([row], "Quarter")
    assert sec.choose_current([ytd], "cfo", row["end"], "10-Q") == ([ytd], "Year to date")
    assert sec.choose_current([ytd], "income", row["end"], "10-Q")[0] == []


def test_new_quarter_unpaired_on_issuer_index_cannot_reuse_old_document():
    data = b"""<h2>Current year 2026</h2><p>Quarterly Financial Statements</p>
    <a href="/s1.pdf">Quarter 1</a><a href="/s2.pdf">Quarter 2</a>
    <a href="/r1.pdf">MTY REPORTS FIRST QUARTER RESULTS FOR FISCAL 2026</a>
    <h2>Year 2025</h2><p>Annual Report</p><p><a href="/a.pdf">here</a> for the 2025 Annual Report</p>"""
    with pytest.raises(ResearchError, match="results release"):
        mty.discover(data)


def test_lease_reconciliation_uses_dated_liability_note_and_rejects_wrong_total():
    class Doc:
        pages = (object(), object())

    url = "https://mtygroup.com/financial-statements.pdf"
    sheets = {
        1: [
            "MTY Food Group Inc.",
            "Consolidated statements of financial position",
            "In thousands of Canadian dollars",
            "August 30, November 30,",
            "2026 2025",
            "Cash 70,552 51,989",
            "Long-term debt 8 585,679 632,172",
            "Current portion of lease liabilities 7 105,350 106,472",
            "Lease liabilities 7 369,974 392,365",
        ],
        2: [
            "Finance lease receivables",
            "Balance as at August 30, 2026 324,618",
            "Lease liabilities",
            "Balance as at August 30, 2026 475,324",
        ],
    }

    def run():
        r = mty.FilingReport(
            "Kahala Brands", mty.PARENT, "MTY.TO", "Q3", "2026-10-09", "2026-08-30", "2026-11-17", url
        )
        mty.add_balance(r, sheets, Doc(), url)
        return r

    assert next(f for f in run().facts if f.key == "leases").value == 475_324_000
    sheets[1][-2] = "Current portion of lease liabilities 7 99,240 106,472"
    r = run()
    assert "leases" not in {f.key for f in r.facts}
    assert any("lease liabilities withheld" in x for x in r.gaps)


def test_replay_verifies_bytes_and_never_substitutes_network(tmp_path):
    import hashlib

    url = "https://data.sec.gov/fixture"
    content = b'{"ok":true}'
    digest = hashlib.sha256(content).hexdigest()
    meta = {
        "started_at": Documents.started_at,
        "sources": [
            {"url": url, "sha256": digest, "retrieved_at": Documents.started_at, "size": len(content)}
        ],
    }

    def pack(path, data):
        with ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(meta))
            archive.writestr("documents/" + digest, data)

    p = tmp_path / "test.zip"
    pack(p, content)
    with patch("requests.get", side_effect=AssertionError("Network invoked during replay")):
        c = SourceClient(p)
        assert c.json(url) == {"ok": True}
        with pytest.raises(ResearchError, match="no source"):
            c.get(url + "/missing")
    pack(p, content[:-1] + b"]")
    with pytest.raises(ResearchError, match="checksum"):
        SourceClient(p)


def test_quarter_and_year_to_date_with_same_end_date_remain_distinct():
    assert mty.quarter_first(["Financial Highlights 13-week Three-month 39-week Nine-month"])
    assert not mty.quarter_first(["Financial Highlights 39-week Nine-month 13-week Three-month"])
    assert mty.column_dates(["Years ended November 30, 2025 and 2024", "2025 2024"]) == [
        "2025-11-30",
        "2024-11-30",
    ]
