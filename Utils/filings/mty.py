"""Discover MTY financials on its investor site and read labeled rows from the PDFs."""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from io import BytesIO
from urllib.parse import urljoin, urlsplit

from lxml import html
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .report import Citation, Fact, FilingReport
from .sources import ResearchError, SourceClient

INDEX = "https://mtygroup.com/en/financial-reports/"
QUARTERS = ["first", "second", "third", "fourth"]
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DAY = rf"({MONTH})\s+(\d{{1,2}})\s*,"
AMOUNT = r"(?:\(?-?\d[\d,]*(?:\.\d+)?\)?|[—–-])"
PARENT = "MTY Food Group Inc."


def compact(text):
    return " ".join(text.split())


def discover(content):
    document = html.fromstring(content)
    statements, releases, annuals = {}, {}, {}
    year = group = None
    announcements = []
    for el in document.iter():
        if el.tag not in {"p", "h2", "h3", "a"}:
            continue
        text = compact(el.text_content())
        match = re.fullmatch(r"(?:Current )?Year\s*(20\d\d)", text, re.IGNORECASE)
        if match:
            year, group = int(match[1]), None
        if text.lower() in {
            "annual report",
            "annual reports",
            "quarterly financial statements",
            "management discussion & analysis",
            "press releases",
            "investor fact sheets",
            "annual information form",
        }:
            group = text.lower()
        if el.tag != "a" or not el.get("href"):
            continue
        url = urljoin(INDEX, el.get("href"))
        if urlsplit(url).hostname not in {"mtygroup.com", "www.mtygroup.com"} or not urlsplit(
            url
        ).path.lower().endswith(".pdf"):
            continue
        quarter = re.fullmatch(r"Quarter ([1-4])", text, re.IGNORECASE)
        if year and group == "quarterly financial statements" and quarter:
            statements.setdefault((year, int(quarter[1])), set()).add(url)
        if year and group in {"annual report", "annual reports"}:
            surrounding = compact(el.getparent().text_content())
            if re.search(rf"{year}\s+Annual Report", surrounding, re.IGNORECASE):
                annuals.setdefault(year, set()).add(url)
        release = re.search(
            r"\b(?:reports|releases)\b.*?\b(first|second|third|fourth)\s+quarter\b.*?\b(20\d\d)\b",
            text,
            re.IGNORECASE,
        )
        if release:
            releases.setdefault((int(release[2]), QUARTERS.index(release[1].lower()) + 1), set()).add(url)
        elif year and group == "press releases" and text.upper().startswith("MTY") and text:
            announcements.append((year, text, url))
    if not statements or not releases or not annuals:
        raise ResearchError(
            "The MTY index did not expose the expected statements, results releases and annual report."
        )
    # Do not silently fall back to last quarter when one document for a new quarter is missing.
    latest = max(set(statements) | set(releases))

    def unique(values, label):
        if len(values) != 1:
            raise ResearchError(
                f"Could not establish a unique {label} for the newest indexed period {latest}."
            )
        return next(iter(values))

    return (
        latest,
        unique(statements.get(latest, set()), "financial statement"),
        unique(releases.get(latest, set()), "results release"),
        max(annuals),
        unique(annuals[max(annuals)], "annual report"),
        announcements,
    )


def pdf_reader(content):
    if not content.startswith(b"%PDF"):
        raise ResearchError("MTY returned a document that could not be identified as a PDF.")
    try:
        document = PdfReader(BytesIO(content))
        if document.is_encrypted or not 1 <= len(document.pages) <= 400:
            raise ResearchError("Unsupported encrypted or unusually sized issuer document.")
        return document
    except PdfReadError as exc:
        raise ResearchError("Could not read the issuer PDF.") from exc


def page_lines(page):
    text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False) or ""
    return [compact(s) for s in text.splitlines() if s.strip()]


def table_row(lines, label, width):
    # A notes column can precede the amounts. Accept only explicit note syntax.
    pattern = re.compile(
        rf"^{label}(?:\s+\(\d+\)|\(\d+\))?(?:\s+\d+(?:\s*&\s*\d+)?)?\s+"
        + r"\s+".join([f"({AMOUNT})"] * width)
        + "$",
        re.IGNORECASE,
    )
    matches = {m.groups() for line in lines if (m := pattern.fullmatch(line))}
    if len(matches) != 1:
        raise ResearchError("A unique labeled table row could not be established.")

    def amount(raw):
        if raw in {"—", "–", "-"}:
            return Decimal(0)
        return Decimal(raw.replace(",", "").replace("(", "-").replace(")", ""))

    return tuple(amount(cell) for cell in matches.pop())


def column_dates(lines):
    for i in range(1, len(lines)):
        years = re.fullmatch(r"(20\d\d(?:\s+20\d\d){1,3})", lines[i])
        if years:
            days = re.findall(DAY, lines[i - 1], re.IGNORECASE)
            if len(days) == len(years[1].split()):
                return [
                    datetime.strptime(f"{m} {d} {y}", "%B %d %Y").date().isoformat()
                    for (m, d), y in zip(days, years[1].split())
                ]
    # Annual statements sometimes print just the years above the two columns.
    annual = re.search(
        rf"(?:Years ended|As at)\s+{DAY}\s*(20\d\d) and (20\d\d)\b", " ".join(lines[:5]), re.IGNORECASE
    )
    if annual and f"{annual[3]} {annual[4]}" in lines:
        return [
            datetime.strptime(f"{annual[1]} {annual[2]} {y}", "%B %d %Y").date().isoformat()
            for y in (annual[3], annual[4])
        ]
    raise ResearchError("Could not establish the date for each amount column.")


def quarter_first(lines):
    """Period end repeats for the quarter and year to date. Also check the duration header."""
    for line in lines:
        tokens = re.findall(
            r"\b(?:13|26|39|52)(?=\s*[- ]\s*week)|\b(?:3|6|9|12)(?=\s*[- ]\s*month)"
            r"|\b(?:three|six|nine|twelve)\b",
            line,
            re.IGNORECASE,
        )
        if tokens and "periods ended" not in line.lower() and not line.lower().startswith("for the "):
            return tokens[0].lower() in {"three", "3", "13"}
    return False


def find_page(pages, heading, unit_phrase):
    for n, lines in pages.items():
        headers = " ".join(lines[:9]).lower()
        if heading in headers and unit_phrase in headers:
            return n, lines
    raise ResearchError(f"Could not locate the issuer table for {heading} in the expected units.")


def ownership(annual, year, url):
    # Scan the filing for the ownership table. Page numbers and ownership percentages can change.
    for n, page in enumerate(annual.pages, 1):
        plain = page.extract_text() or ""
        if "principal subsidiaries" not in plain.lower() or "kahala" not in plain.lower():
            continue
        lines = page_lines(page)
        joined = " ".join(lines)
        if "MTY Food Group Inc." not in joined or "Percentage of equity interest" not in joined:
            continue
        if not any(
            re.search(rf"Principal subsidiaries\s+{year}\s+{year - 1}\b", x, re.IGNORECASE) for x in lines
        ):
            continue
        for line in lines:
            row = re.fullmatch(
                r"Kahala Brands(?:,? (?:Inc|Ltd)\.?)?\s+(\d+(?:\.\d+)?)\s+\d+(?:\.\d+)?\s+.+",
                line,
                re.IGNORECASE,
            )
            if row:
                value = Decimal(row[1])
                if not 50 < value <= 100:
                    raise ResearchError(
                        "The latest ownership table does not establish MTY as the controlling parent of Kahala."
                    )
                return (
                    f"MTY reported a {value}% equity interest in Kahala Brands in fiscal {year}. "
                    f"[Annual report, PDF page {n}]({url}#page={n})."
                )
    raise ResearchError("Kahala’s ownership could not be verified in the latest available MTY annual report.")


def build_mty_report(client: SourceClient, kahala=True):
    latest, statement_url, release_url, annual_year, annual_url, announcements = discover(client.get(INDEX))
    year, quarter = latest
    release = pdf_reader(client.get(release_url))
    statement = pdf_reader(client.get(statement_url))
    rp = {n + 1: page_lines(p) for n, p in enumerate(release.pages[:5])}
    sp = {n + 1: page_lines(p) for n, p in enumerate(statement.pages[:16])}
    cover = compact(release.pages[0].extract_text() or "")
    if not re.search(r"MTY Food Group Inc", cover, re.IGNORECASE) or not re.search(
        r"TSX\s*:\s*MTY\b", cover, re.IGNORECASE
    ):
        raise ResearchError("Issuer or exchange ticker missing from the results release.")
    if not re.search(rf"{QUARTERS[quarter - 1]} quarter\b.{{0,55}}\b{year}\b", cover, re.IGNORECASE):
        raise ResearchError("The results release does not match the indexed fiscal quarter.")
    publication = re.search(
        rf"Montr[eé]al[^a-zA-Z]{{0,12}}({MONTH}\s+\d{{1,2}},\s*20\d\d)", cover, re.IGNORECASE
    )
    if not publication:
        raise ResearchError("The results release has no recognized publication date.")
    published = datetime.strptime(publication[1], "%B %d, %Y").date().isoformat()
    matches = [
        (n, lines)
        for n, lines in rp.items()
        if any("financial highlights" in x.lower() for x in lines)
        and any(re.search(r"in thousands of (?:\$|canadian dollars)", x, re.IGNORECASE) for x in lines)
    ]
    if len(matches) != 1:
        raise ResearchError("Could not identify a single dated financial highlights table in the release.")
    hn, highlights = matches[0]
    start = next(i for i, line in enumerate(highlights) if "financial highlights" in line.lower())
    highlights = highlights[start:]
    if quarter == 4:
        _, annual_lines = find_page(sp, "consolidated statements of income", "thousands of canadian dollars")
        annual_dates = column_dates(annual_lines[:21])
        year_header = " ".join(highlights[:5])
        if not re.search(
            rf"Q4\s*-\s*{year}\s+Q4\s*-\s*{year - 1}\s+{year}\s+{year - 1}", year_header, re.IGNORECASE
        ):
            raise ResearchError(
                "Could not match the release’s annual and quarterly columns to their fiscal years."
            )
        if [int(x[:4]) for x in annual_dates] != [year, year - 1]:
            raise ResearchError("Annual statements and release identify different fiscal years.")
        dates = annual_dates + annual_dates
    else:
        dates = column_dates(highlights[:14])
    ncols = len(dates)
    if quarter == 4:
        # Use the annual columns of the release and audited statements at the year-end.
        if ncols != 4 or not re.search(
            r"\b(?:year|twelve|12.months?|52.week)", " ".join(highlights[:8]), re.IGNORECASE
        ):
            raise ResearchError("Could not establish the financial-year columns in the year-end release.")
        cols = (2, 3)
        period = "Fiscal year"
    else:
        cols = (0, 1)
        if not quarter_first(highlights[:8]):
            raise ResearchError("Could not establish the quarter columns in the results release.")
        period = "Quarter"
    end, prior_end = (dates[i] for i in cols)
    if (
        not (end <= published <= client.started_at[:10])
        or not 345 <= (date.fromisoformat(end) - date.fromisoformat(prior_end)).days <= 380
    ):
        raise ResearchError("Invalid or future reporting or publication dates.")
    subject = "Kahala Brands" if kahala else PARENT
    report = FilingReport(
        subject,
        PARENT,
        "MTY.TO",
        f"Fiscal {year} Q{quarter} results",
        published,
        end,
        client.started_at,
        statement_url,
    )
    annual = pdf_reader(client.get(annual_url))
    if kahala:
        report.notes.append(ownership(annual, annual_year, annual_url))
        report.gaps.append(
            "These documents supply consolidated MTY figures. Standalone Kahala revenue, earnings and cash flow remain unavailable."
        )
    report.notes.append(
        f"Document links were discovered on [MTY’s investor index]({INDEX}). The issuer’s results announcement supplies the publication date."
    )
    if quarter != 4:
        report.notes.append(
            "The comparison uses the quarter columns as presented. Fiscal weeks and calendar-month periods can differ."
        )
    incoming = (
        ("revenue", "Revenue", "Revenue", "Revenue", "income"),
        (
            "income",
            "Net income attributable to owners",
            "Net income attributable to owners",
            "Owners",
            "income",
        ),
        (
            "eps",
            "Diluted earnings per share",
            "Earnings per share, basic and diluted",
            "Basic and diluted",
            "income",
        ),
        (
            "cfo",
            "Operating cash flow",
            "Cash flows from operations",
            "Cash provided by operating activities",
            "cash flows",
        ),
    )
    for key, label, release_label, fs_label, heading in incoming:
        try:
            n, lines = find_page(sp, "consolidated statements of " + heading, "thousands of canadian dollars")
            if not any("mty food group inc." in x.lower() for x in lines[:8]):
                raise ResearchError("The statement table is missing the parent issuer name.")
            if quarter != 4 and not quarter_first(lines[:16]):
                raise ResearchError(
                    "The statement column order does not establish a quarter-first comparison."
                )
            fs_dates = column_dates(lines[:21])
            if fs_dates[:2] != [end, prior_end]:
                raise ResearchError("The statement and release columns refer to different periods.")
            values = table_row(lines, fs_label, len(fs_dates))
            released = table_row(highlights, release_label, ncols)
            if values[:2] != tuple(released[i] for i in cols):
                raise ResearchError("The financial statement and earnings release disagree.")
            scale = 1 if key == "eps" else 1000
            unit = "CAD/share" if key == "eps" else "CAD"
            report.facts.append(
                Fact(
                    key,
                    label,
                    values[0] * scale,
                    unit,
                    period,
                    None,
                    end,
                    (
                        Citation(f"{statement_url}#page={n}", f"Statement p. {n}"),
                        Citation(f"{release_url}#page={hn}", f"Release p. {hn}"),
                    ),
                    "Current and comparative amounts match across the two issuer documents.",
                    values[1] * scale,
                    prior_end,
                )
            )
            if (
                key == "eps"
                and quarter != 4
                and (
                    (ncols == 4 and fs_dates == dates and values == released) or (quarter == 1 and ncols == 2)
                )
            ):
                # In Q1, year-to-date and quarterly spans are the same.
                report.facts.append(
                    Fact(
                        "ytd_eps",
                        "Year-to-date diluted EPS",
                        values[-2],
                        unit,
                        "Year to date",
                        None,
                        end,
                        report.facts[-1].sources,
                        "Current and comparative year-to-date EPS agree across both documents.",
                        values[-1],
                        prior_end,
                    )
                )
        except ResearchError as exc:
            report.gaps.append(f"{label} withheld. {exc}")
    for key, label, row in [
        (
            "fcf",
            "Free cash flow after net lease payments (issuer definition)",
            "Free cash flows net of lease payments",
        ),
        ("ebitda", "Normalized adjusted EBITDA (issuer definition)", "Normalized adjusted EBITDA"),
    ]:
        try:
            values = table_row(highlights, row, ncols)
            report.facts.append(
                Fact(
                    key,
                    label,
                    values[cols[0]] * 1000,
                    "CAD",
                    period,
                    None,
                    end,
                    (Citation(f"{release_url}#page={hn}", f"Release p. {hn}"),),
                    "Single issuer disclosure of a management-defined measure.",
                    values[cols[1]] * 1000,
                    prior_end,
                )
            )
        except ResearchError as exc:
            report.gaps.append(f"{label} unavailable. {exc}")
    if quarter != 4 and annual_year == year - 1:
        add_annual_eps(report, annual, annual_url, annual_year)
    add_dividend(report, release, release_url)
    add_balance(report, sp, statement, statement_url)
    if not any(len(f.sources) == 2 and f.key in {"income", "eps", "revenue", "cfo"} for f in report.facts):
        raise ResearchError("No operating figures agreed across the statement and release.")
    current_news = dict.fromkeys((title, url) for y, title, url in announcements if y == year)
    if current_news:
        links = [f"[{title}]({url})" for title, url in list(current_news)[:3]]
        report.notes.append(
            "Other issuer announcements on the same fiscal-year index: " + " · ".join(links) + "."
        )
    return report


def add_balance(report, sp, statement, url):
    try:
        page, lines = find_page(
            sp, "consolidated statements of financial position", "thousands of canadian dollars"
        )
        head_dates = column_dates(lines[:20])
        if head_dates[0] != report.period_end or len(head_dates) != 2:
            raise ResearchError("The balance sheet dates do not match the results period.")
    except ResearchError as exc:
        report.gaps.append(f"Balance sheet withheld. {exc}")
        return
    stmt = Citation(f"{url}#page={page}", f"Statement p. {page}")
    for key, label, row in [
        ("cash", "Cash", "Cash"),
        ("bank_debt", "Long-term debt carrying value", "Long-term debt"),
    ]:
        try:
            now, prior = table_row(lines, row, 2)
            report.facts.append(
                Fact(
                    key,
                    label,
                    now * 1000,
                    "CAD",
                    "Balance sheet",
                    None,
                    head_dates[0],
                    (stmt,),
                    "Read from the dated balance sheet.",
                    prior * 1000,
                    head_dates[1],
                )
            )
        except ResearchError as exc:
            report.gaps.append(f"{label} withheld. {exc}")
    try:
        current, _ = table_row(lines, "Current portion of lease liabilities", 2)
        noncurrent, _ = table_row(lines, "Lease liabilities", 2)
        total = current + noncurrent
        stamp = date.fromisoformat(report.period_end)
        display = f"{stamp:%B} {stamp.day}, {stamp.year}"
        for n, p in enumerate(statement.pages, 1):
            note = sp.get(n)
            if note is None:
                if "lease liabilities" not in (p.extract_text() or "").lower():
                    continue
                note = page_lines(p)
            # A single page can contain BOTH lease receivable and liability reconciliations.
            # Check only the liability section, preserving its own date and row heading.
            headings = [i for i, line in enumerate(note) if line.lower() == "lease liabilities"]
            if not headings:
                continue
            liabilities = note[headings[0] + 1 :]
            amounts = {
                Decimal(m[1].replace(",", ""))
                for line in liabilities
                if (m := re.fullmatch(rf"Balance as at {re.escape(display)}\s+([\d,]+)", line, re.IGNORECASE))
            }
            if amounts == {total}:
                report.facts.append(
                    Fact(
                        "leases",
                        "Total lease liabilities",
                        total * 1000,
                        "CAD",
                        "Balance sheet",
                        None,
                        report.period_end,
                        (stmt, Citation(f"{url}#page={n}", f"Lease note p. {n}")),
                        f"Current CAD {current / 1000:,.3f}m plus noncurrent CAD {noncurrent / 1000:,.3f}m agrees with the lease note.",
                    )
                )
                return
        raise ResearchError("Current plus noncurrent leases could not be reconciled to the dated lease note.")
    except ResearchError as exc:
        report.gaps.append(f"Total lease liabilities withheld. {exc}")


def add_annual_eps(report, annual, url, fiscal_year):
    for n, page in enumerate(annual.pages, 1):
        plain = compact(page.extract_text() or "").lower()
        if "consolidated statements of income" not in plain or "basic and diluted" not in plain:
            continue
        lines = page_lines(page)
        head = " ".join(lines[:8]).lower()
        if (
            "consolidated statements of income" not in head
            or "canadian dollars, except per share" not in head
        ):
            continue
        try:
            ends = column_dates(lines[:18])
            value, prior = table_row(lines, "Basic and diluted", 2)
        except ResearchError:
            continue
        if int(ends[0][:4]) != fiscal_year or ends[0] >= report.period_end:
            continue
        report.facts.append(
            Fact(
                "annual_eps",
                "Prior fiscal-year diluted EPS",
                value,
                "CAD/share",
                "Fiscal year",
                None,
                ends[0],
                (Citation(f"{url}#page={n}", f"Annual report p. {n}"),),
                "Read from the annual consolidated income statement.",
                prior,
                ends[1],
            )
        )
        return
    report.gaps.append("Prior annual diluted EPS could not be established for trailing arithmetic.")


def add_dividend(report, release, url):
    amounts = {}
    for n, page in enumerate(release.pages[:4], 1):
        text = compact(page.extract_text() or "")
        for match in re.finditer(
            r"quarterly dividend (?:of|to)\s+(\$?)(\d+(?:\.\d+)?)\s*(¢|cents)? per share", text, re.IGNORECASE
        ):
            if bool(match[1]) == bool(match[3]):
                continue
            amount = Decimal(match[2]) / (100 if match[3] else 1)
            amounts[amount] = n
    if len(amounts) == 1:
        value, page = amounts.popitem()
        report.facts.append(
            Fact(
                "dividend",
                "Quarterly dividend in results announcement",
                value,
                "CAD/share",
                "Announcement",
                None,
                report.published,
                (Citation(f"{url}#page={page}", f"Release p. {page}"),),
                "Amount reported by MTY in this results release. Later dividend announcements may supersede it.",
            )
        )
