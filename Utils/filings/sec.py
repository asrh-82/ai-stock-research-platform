"""Discover an EDGAR filing and reconcile Company Facts to the filed inline XBRL."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

from lxml import html

from .report import Citation, Fact, FilingReport
from .sources import ResearchError, SourceClient

DIRECTORY = "https://www.sec.gov/files/company_tickers.json"
DATA = "https://data.sec.gov"
# Each fallback retains the accounting meaning of its own concept.
METRICS = {
    "revenue": [
        ("Revenues", "Revenue"),
        ("SalesRevenueNet", "Net sales"),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "Contract revenue excluding assessed tax"),
    ],
    "income": [
        ("NetIncomeLoss", "Net income attributable to parent"),
        ("ProfitLoss", "Consolidated net income"),
    ],
    "eps": [("EarningsPerShareDiluted", "Diluted earnings per share")],
    "cfo": [("NetCashProvidedByUsedInOperatingActivities", "Operating cash flow")],
    "cash": [("CashAndCashEquivalentsAtCarryingValue", "Cash and cash equivalents")],
    "assets": [("Assets", "Total assets")],
    "liabilities": [("Liabilities", "Total liabilities")],
}
INSTANT = {"cash", "assets", "liabilities"}
TAGS = {"us-gaap:" + name for options in METRICS.values() for name, _ in options}


def company_name(value):
    name = re.sub(r"[^A-Z0-9 ]", " ", value.upper())
    return re.sub(r"\s+(INCORPORATED|INC|CORPORATION|CORP|LTD|LLC|PLC|CO)$", "", " ".join(name.split()))


def resolve_company(query, directory):
    symbol = query.strip().upper().replace(".", "-")
    rows = list(directory.values())
    exact = [row for row in rows if row["ticker"].upper() == symbol]
    matches = exact or [row for row in rows if company_name(row["title"]) == company_name(query)]
    if len(matches) != 1:
        choices = ", ".join(sorted({row["ticker"] for row in matches})[:8])
        if choices:
            raise ResearchError(f"Several SEC listings match this name. Choose a ticker from {choices}.")
        raise ResearchError(
            "No exact SEC company or ticker match. Use the listed ticker or full issuer name. Coverage also includes Kahala Brands through MTY."
        )
    return matches[0]


def select_filing(submissions, cutoff):
    fields = submissions["filings"]["recent"]
    keys = ("form", "accessionNumber", "filingDate", "reportDate", "primaryDocument")
    if len({len(fields[k]) for k in keys}) != 1:
        raise ResearchError("SEC filing index columns have inconsistent lengths.")
    filings = [dict(zip(keys, values)) for values in zip(*(fields[k] for k in keys))]
    supported = [
        f
        for f in filings
        if f["form"] in {"10-K", "10-Q", "10-K/A", "10-Q/A"}
        and f["reportDate"]
        and f["reportDate"] <= f["filingDate"] <= cutoff
    ]
    if not supported:
        raise ResearchError("No supported 10-K or 10-Q in this issuer’s recent EDGAR index.")
    latest = max(supported, key=lambda f: (f["reportDate"], f["filingDate"], f["accessionNumber"]))
    if latest["form"].endswith("/A"):
        raise ResearchError(
            "An amendment applies to the latest reporting period. Reconcile it with the original filing before using these figures."
        )
    return latest


def local_name(element):
    return element.tag.rsplit(":", 1)[-1] if isinstance(element.tag, str) else ""


def parse_inline(content, cik, end, form):
    document = html.fromstring(content)
    # Prefixes can vary between issuers. HTML parsing lowercases element and attribute names.
    contexts, units, identity = {}, {}, defaultdict(set)
    facts = defaultdict(dict)
    for node in document.iter():
        tag = local_name(node)
        if tag == "context":
            children = {local_name(e): (e.text or "").strip() for e in node.iter()}
            if {"segment", "scenario"} & children.keys() or children.get("identifier", "").lstrip("0") != str(
                cik
            ):
                continue
            dates = (children.get("startdate"), children.get("enddate") or children.get("instant"))
            contexts[node.get("id")] = dates
        elif tag == "unit":
            measures = [e.text for e in node.iter() if local_name(e) == "measure"]
            if measures == ["iso4217:USD"]:
                units[node.get("id")] = "USD"
            elif measures == ["iso4217:USD", "xbrli:shares"] and any(local_name(e) == "divide" for e in node):
                units[node.get("id")] = "USD/shares"
        elif tag == "nonnumeric" and node.get("name") in {
            "dei:EntityCentralIndexKey",
            "dei:DocumentPeriodEndDate",
            "dei:DocumentType",
        }:
            identity[node.get("name")].add(" ".join(node.text_content().split()))
    if {v.lstrip("0") for v in identity["dei:EntityCentralIndexKey"]} != {str(cik)} or identity[
        "dei:DocumentType"
    ] != {form}:
        raise ResearchError("The filing’s identity or form disagrees with the SEC index.")

    def parse_date(value):
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%B %d %Y"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    if {parse_date(v) for v in identity["dei:DocumentPeriodEndDate"]} != {end}:
        raise ResearchError("Could not confirm the reporting date in the filed document.")
    for node in document.iter():
        concept = node.get("name")
        if local_name(node) != "nonfraction" or concept not in TAGS:
            continue
        span = contexts.get(node.get("contextref"))
        unit = units.get(node.get("unitref"))
        if not span or not unit or node.get("xsi:nil") == "true":
            continue
        raw = node.text_content().strip().replace(",", "").replace("\u00a0", "")
        fmt = (node.get("format") or "").split(":")[-1].lower()
        if fmt in {"zerodash", "numdash", "fixed-zero"} and raw in {"—", "–", "-", ""}:
            raw = "0"
        elif fmt not in {"", "numdotdecimal", "num-dot-decimal"}:
            continue
        try:
            value = Decimal(raw) * Decimal(10) ** int(node.get("scale", 0))
        except (InvalidOperation, ValueError):
            continue
        if value.is_finite():
            if node.get("sign") == "-":
                value = -value
            facts[(concept.split(":")[1], unit, *span)][value] = node.get("id") or ""
    return facts


def duration(row):
    return (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days + 1


def choose_current(rows, key, end, form):
    matching = [row for row in rows if row["end"] == end]
    if key in INSTANT:
        matching = [row for row in matching if not row.get("start")]
        label = "Balance sheet"
    else:
        matching = [row for row in matching if row.get("start")]
        if form == "10-K":
            matching = [row for row in matching if 345 <= duration(row) <= 380]
            label = "Fiscal year"
        else:
            quarter = [row for row in matching if 75 <= duration(row) <= 105]
            if key == "cfo" and not quarter:
                matching = [row for row in matching if 155 <= duration(row) <= 290]
                label = "Year to date"
            else:
                matching = quarter
                label = "Quarter"
    periods = {(row.get("start"), row["end"]) for row in matching}
    if len(periods) > 1:
        raise ResearchError("Competing periods for the same reported figure.")
    return matching, label


def comparable(rows, current, key):
    candidates = []
    for row in rows:
        gap = (date.fromisoformat(current["end"]) - date.fromisoformat(row["end"])).days
        if not 345 <= gap <= 380:
            continue
        if (
            key in INSTANT
            and not row.get("start")
            or key not in INSTANT
            and row.get("start")
            and abs(duration(row) - duration(current)) <= 8
        ):
            candidates.append(row)
    return candidates


def reconcile(rows, tag, unit, inline):
    observations = {(r.get("start"), r["end"], Decimal(str(r["val"]))) for r in rows}
    if len(observations) != 1:
        raise ResearchError("Conflicting periods or values in the SEC facts feed.")
    start, end, value = observations.pop()
    reported = inline.get((tag, unit, start, end), {})
    if set(reported) != {value}:
        raise ResearchError(
            "The amount, unit or reporting context could not be reconciled to the filed statement."
        )
    return value, reported[value]


def build_sec_report(query: str, client: SourceClient) -> FilingReport:
    company = resolve_company(query, client.json(DIRECTORY))
    cik = int(company["cik_str"])
    cik_id = f"CIK{cik:010}"
    filings_url = f"{DATA}/submissions/{cik_id}.json"
    submissions = client.json(filings_url)
    ticker = company["ticker"]
    if int(submissions["cik"]) != cik or ticker not in submissions["tickers"]:
        raise ResearchError("Ticker and issuer disagree between SEC directories.")
    f = select_filing(submissions, client.started_at[:10])
    acc = f["accessionNumber"]
    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", acc) or not re.fullmatch(
        r"[\w.-]+\.html?", f["primaryDocument"], re.IGNORECASE
    ):
        raise ResearchError("The SEC index does not identify a supported inline HTML filing.")
    filed_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{f['primaryDocument']}"
    facts_url = f"{DATA}/api/xbrl/companyfacts/{cik_id}.json"
    facts_api = client.json(facts_url)
    if int(facts_api["cik"]) != cik:
        raise ResearchError("The fact feed belongs to a different SEC registrant.")
    inline = parse_inline(client.get(filed_url), cik, f["reportDate"], f["form"])
    if not inline:
        raise ResearchError("The filing contains no supported consolidated USD figures.")
    issuer = submissions["name"]
    report = FilingReport(
        issuer, issuer, ticker, f["form"], f["filingDate"], f["reportDate"], client.started_at, filed_url
    )
    report.notes.append(
        f"SEC industry classification: {submissions.get('sicDescription') or 'unavailable'}. "
        f"[Issuer filing history]({filings_url})."
    )
    index = submissions["filings"]["recent"]
    disclosures = []
    for form, filed, accession in zip(index["form"], index["filingDate"], index["accessionNumber"]):
        if (
            form in {"8-K", "8-K/A"}
            and report.published <= filed <= client.started_at[:10]
            and re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
        ):
            link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-index.htm"
            disclosures.append((filed, f"[{form} filed {filed}]({link})"))
    if disclosures:
        report.notes.append(
            "Recent current reports filed on or after the financial filing: "
            + " · ".join(link for _, link in sorted(disclosures, reverse=True)[:3])
            + "."
        )
    report.gaps.append(
        "Coverage includes supported US GAAP tags in the latest 10-K or 10-Q. Both representations originate with the issuer."
    )
    taxonomy = facts_api.get("facts", {}).get("us-gaap", {})
    for key, concepts in METRICS.items():
        unit = "USD/shares" if key == "eps" else "USD"
        for tag, label in concepts:
            rows = [
                r
                for r in taxonomy.get(tag, {}).get("units", {}).get(unit, [])
                if r.get("accn") == acc and r.get("form") == f["form"] and r.get("filed") == f["filingDate"]
            ]
            try:
                current, period = choose_current(rows, key, report.period_end, report.document)
                if not current:
                    continue
                value, fragment = reconcile(current, tag, unit, inline)
            except ResearchError as exc:
                report.gaps.append(f"{label} withheld. {exc}")
                break
            previous = comparable(rows, current[0], key)
            prior_value = prior_end = prior_start = None
            if previous:
                try:
                    prior_value, _ = reconcile(previous, tag, unit, inline)
                    prior_end, prior_start = previous[0]["end"], previous[0].get("start")
                except ResearchError:
                    report.gaps.append(f"{label} comparison was ambiguous or could not be cross-checked.")
            target = filed_url + ("#" + quote(fragment) if fragment else "")
            report.facts.append(
                Fact(
                    key,
                    label,
                    value,
                    "USD/share" if key == "eps" else "USD",
                    period,
                    current[0].get("start"),
                    report.period_end,
                    (Citation(target, tag), Citation(facts_url, "SEC facts")),
                    "Exact agreement between the filed inline XBRL and SEC Company Facts for the same accession, dates, unit and entity.",
                    prior_value,
                    prior_end,
                    prior_start,
                )
            )
            break
        else:
            report.gaps.append(f"{concepts[0][1]} unavailable under the supported tags for this filing.")
    if not report.facts:
        raise ResearchError("No figures passed both source checks for this filing.")
    return report
