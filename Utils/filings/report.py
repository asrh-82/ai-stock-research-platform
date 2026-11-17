"""Report values have explicit scope, periods, units, and a location in a primary source."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal

from .sources import Source


@dataclass(frozen=True)
class Citation:
    url: str
    locator: str


@dataclass(frozen=True)
class Fact:
    key: str
    label: str
    value: Decimal
    unit: str
    period: str
    start: str | None
    end: str
    sources: tuple[Citation, ...]
    check: str
    prior: Decimal | None = None
    prior_end: str | None = None
    prior_start: str | None = None

    def __post_init__(self):
        if not self.sources or not self.value.is_finite():
            raise ValueError("A reported value needs a finite number and a primary source.")
        if self.prior is not None and (not self.prior.is_finite() or not self.prior_end):
            raise ValueError("A comparison needs a finite value and a dated period.")

    def number(self, value: Decimal) -> str:
        if self.unit.endswith("/share"):
            return f"{value:,.2f}"
        return f"{value / 1_000_000:,.3f}"

    @property
    def display_unit(self):
        return self.unit if self.unit.endswith("/share") else self.unit + " million"


@dataclass
class FilingReport:
    subject: str
    issuer: str
    ticker: str
    document: str
    published: str
    period_end: str
    retrieved: str
    filing_url: str
    facts: list[Fact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)

    def audit(self) -> str:
        return json.dumps({"schema_version": 1, **asdict(self)}, default=str, indent=2, ensure_ascii=False)

    def markdown(self) -> str:
        lines = [
            f"# {self.subject}",
            f"Reporting company: **{self.issuer}** ({self.ticker}).",
            (
                f"[{self.document}]({self.filing_url}) · Period ended {self.period_end} · "
                f"Filing or results publication {self.published} · Retrieved {self.retrieved[:10]} UTC."
            ),
            *self.notes,
            "## Reported figures",
            "Every amount below belongs to the consolidated reporting company. Periods and comparison dates are shown on each row.",
            "| Measure | Period ending | Unit | Reported | Comparison | Evidence |",
            "|---|---|---|---:|---:|---|",
        ]
        for f in self.facts:
            prior = "—" if f.prior is None else f"{f.number(f.prior)} ({f.prior_end})"
            refs = " · ".join(f"[{s.locator}]({s.url})" for s in f.sources)
            lines.append(
                f"| {f.label} | {f.period}, {f.end} | {f.display_unit} | {f.number(f.value)} | {prior} | {refs} |"
            )
        by_key = {f.key: f for f in self.facts}
        observations = []
        for key in ("revenue", "income", "cfo"):
            f = by_key.get(key)
            if f and f.prior is not None and f.prior > 0:
                change = f.value / f.prior - 1
                observations.append(
                    f"{f.label} changed {change:+.1%} from the comparable period ending {f.prior_end}."
                )
        revenue = by_key.get("revenue")
        earnings = by_key.get("income")
        if (
            revenue
            and earnings
            and revenue.value > 0
            and (revenue.start, revenue.end, revenue.period)
            == (earnings.start, earnings.end, earnings.period)
        ):
            observations.append(
                f"{earnings.label} was {earnings.value / revenue.value:.1%} of the reported revenue line for the {earnings.period.lower()}."
            )
        if observations:
            lines += ["## Calculations", *observations]
        lines += ["## Verification"]
        for f in self.facts:
            lines.append(f"- {f.label}: {f.check}")
        if self.gaps:
            lines += ["## Coverage limits", *["- " + gap for gap in self.gaps]]
        return "\n\n".join(lines).replace("|\n\n|", "|\n|") + "\n"
