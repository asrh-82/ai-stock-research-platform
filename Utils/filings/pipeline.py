"""Single entry point used by Streamlit and the CLI."""

from __future__ import annotations

from datetime import date

from lxml.etree import ParserError
from pypdf.errors import PdfReadError

from Utils.research import resolve_research_subject

from .mty import build_mty_report
from .sec import build_sec_report
from .sources import ResearchError, SourceClient


def research_company(query, client=None):
    query = query.strip()
    if not query or len(query) > 160:
        raise ResearchError("Enter a company name or ticker of up to 160 characters.")
    client = client or SourceClient()
    try:
        if resolve_research_subject(query):
            report = build_mty_report(client, kahala="kahala" in query.lower())
        else:
            report = build_sec_report(query, client)
    except ResearchError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, ParserError, PdfReadError) as exc:
        raise ResearchError(
            "The source format changed. The report could not confirm the required fields."
        ) from exc
    report.sources = list(client.sources.values())
    age = (date.fromisoformat(report.retrieved[:10]) - date.fromisoformat(report.period_end)).days
    if age > 185:
        report.gaps.append(
            f"The selected reporting period ended {age} days before retrieval. Check for newer disclosures."
        )
    return report
