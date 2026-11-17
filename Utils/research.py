"""Deterministic, source-tagged reporting. Unknown inputs never become invented facts."""
import json
from datetime import date
from math import isfinite
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'Data' / 'research' / 'kahala.json'


def resolve_research_subject(query):
    normalized=' '.join(query.lower().strip().replace('.', '').split())
    if normalized in {'kahala','kahala brands','kahala brands ltd','kahala brands inc','mty','mtyto','mty food group','mty food group inc'}:
        return {'subject':'Kahala Brands', 'symbol':'MTY.TO', 'name':'MTY Food Group Inc. (public parent of Kahala Brands)',
                'quote_type':'EQUITY', 'exchange':'TSX', 'standalone_public_stock':False}
    return None


def load_evidence(path=DATA):
    return json.loads(Path(path).read_text())


def validate_evidence(evidence, as_of):
    if evidence.get('schema_version') != 1:
        raise ValueError('Unsupported evidence schema')
    cutoff=date.fromisoformat(str(as_of))
    verified=date.fromisoformat(evidence['verified_on'])
    if cutoff<verified:
        raise ValueError('This evidence snapshot was not verified at that historical date; cannot generate a point-in-time report.')
    if evidence['currency']!='CAD' or evidence['ticker']!='MTY.TO' or evidence['standalone_public_stock']:
        raise ValueError('Kahala identity or currency mismatch.')
    expected_units={'q3_eps':'CAD per share','fy_eps':'CAD per share','ytd_eps':'CAD per share','quarter_dividend':'CAD per share','shares':'million shares'}
    if not isfinite(evidence['quote']['value']) or evidence['quote']['value']<=0:
        raise ValueError('Quote must be positive and finite')
    seen=set()
    for fact in evidence['facts']:
        if fact['id'] in seen:raise ValueError('Duplicate fact ID')
        seen.add(fact['id'])
        if fact['source'] not in evidence['sources']:raise ValueError('Fact without a source')
        if fact['scope']!='MTY consolidated':raise ValueError('Parent data cannot be labeled standalone Kahala.')
        if fact['unit'] != expected_units.get(fact['id'],'CAD million'):raise ValueError('Incorrect unit for financial fact')
        if not isfinite(fact['value']) or ('prior' in fact and not isfinite(fact['prior'])):raise ValueError('Nonfinite fact')
        if date.fromisoformat(fact['period_end'])>cutoff:raise ValueError('Future-period fact')
    for src in evidence['sources'].values():
        if not src['url'].startswith('https://'):raise ValueError('Source must have an HTTPS URL')
        if src['published_on'] and date.fromisoformat(src['published_on'])>cutoff:raise ValueError('Future source')
    if evidence['quote']['currency']!=evidence['currency']:raise ValueError('Quote currency mismatch')
    if date.fromisoformat(evidence['quote']['session'])>cutoff:raise ValueError('Future quote')
    return cutoff


def calculated_metrics(evidence):
    f={v['id']:v for v in evidence['facts']}
    price=evidence['quote']['value']
    # Fiscal-calendar transition makes these trailing-period approximations.
    eps=f['fy_eps']['value']+f['ytd_eps']['value']-f['ytd_eps']['prior']
    fcf=f['fy_fcf']['value']+f['ytd_fcf']['value']-f['ytd_fcf']['prior']
    cap=price*f['shares']['value']
    return {'illustrative_equity_value_m':cap,'trailing_eps_approx':eps,'reported_pe_approx':price/eps if eps>0 else None,
            'trailing_fcf_approx_m':fcf,'management_fcf_yield_approx':fcf/cap,
            'indicated_dividend_yield':4*f['quarter_dividend']['value']/price,
            'bank_debt_minus_cash_m':f['debt']['value']-f['cash']['value'],
            'lease_liabilities_m':f['lease_current']['value']+f['lease_noncurrent']['value']}


def render_report(evidence=None, *, as_of='2026-11-17'):
    e=evidence or load_evidence();cutoff=validate_evidence(e,as_of);m=calculated_metrics(e)
    sources=e['sources']
    def cite(key):return f"[{key}]({sources[key]['url']})"
    lines=['# Kahala Brands and its public parent',f'Research snapshot verified {e["verified_on"]}. Requested as of {cutoff}. All financial amounts are CAD unless specified.',
           '## What can actually be analyzed',
           f"Kahala is wholly owned by MTY. The 2016 acquisition covered all Kahala shares for approximately US$310 million. MTY trades on the Toronto Stock Exchange as MTY, represented as MTY.TO by Yahoo. Its returns reflect the entire group. {cite('acquisition')} {cite('annual')}",
           f"Kahala operates a franchise brand portfolio including Cold Stone Creamery and Blimpie. Brand-level business research is possible, but MTY consolidated accounts cannot be relabeled as Kahala accounts. {cite('brands')}",
           '## Reported operating evidence',
           'The table compares the 13 weeks ended August 30, 2026 with the three months ended August 31, 2025. The different calendar definitions limit exact like-for-like comparison. Amounts are millions except EPS.',
           '| MTY consolidated metric | Q3 2026 | Q3 2025 | Change | Source |','|---|---:|---:|---:|---|']
    for fact in e['facts']:
        if fact['id'].startswith('q3_'):
            change=fact['value']/fact['prior']-1
            lines.append(f"| {fact['label']} | {fact['value']:,.3f} | {fact['prior']:,.3f} | {change:+.1%} | {cite(fact['source'])} |")
    lines += ['## What the evidence suggests',
              f"Inference: weaker restaurant demand and corporate-store economics deserve more attention than headline cash yield. System-wide same-store sales fell 1.9%; the US decline was 2.7%. Franchising normalized EBITDA was CAD55.4m while corporate operations contributed CAD0.9m. Those are MTY segment figures, not Kahala figures. {cite('release')}",
              f"Cash flow quality needs scrutiny. The quarter's management FCF calculation includes asset-disposal proceeds and lease payments. Deferred tax recoveries also support reported earnings. Treat these reported measures as starting points for normalization. {cite('quarter')}",
              '## Dated valuation arithmetic',
              f"Reference quote: CAD{e['quote']['value']:.2f} on {e['quote']['session']}; not a live execution quote. The issuer quote and downloaded Yahoo close agree to the cent. {cite('quote')}",
              '| Calculation | Result | Interpretation |','|---|---:|---|',
              f"| Price x 22.841361m weighted-average diluted shares | CAD{m['illustrative_equity_value_m']:.1f}m | Illustrative equity value; uses the quarter’s weighted average |",
              f"| 5.18 + 3.38 - 3.77 | CAD{m['trailing_eps_approx']:.2f} | Approximate trailing reported EPS, not normalized EPS |",
              f"| 36.38 / approximate EPS | {m['reported_pe_approx']:.2f}x | Backward-looking reported P/E |",
              f"| 130.618 + 89.655 - 92.968 | CAD{m['trailing_fcf_approx_m']:.3f}m | Approximate trailing management FCF after leases |",
              f"| Management FCF / illustrative equity value | {m['management_fcf_yield_approx']:.1%} | Includes nonrecurring effects; not a promised yield |",
              f"| 4 x CAD0.50 / CAD36.38 | {m['indicated_dividend_yield']:.2%} | Annualized declared rate, not a guaranteed payout |",
              f"| Bank debt carrying value minus cash | CAD{m['bank_debt_minus_cash_m']:.3f}m | Excludes leases and other obligations |",
              f"| Balance sheet lease liabilities (note 7 reconciliation) | CAD{m['lease_liabilities_m']:.3f}m | Shown separately; do not ignore in valuation |",
              f"Inputs: {cite('annual')}, {cite('quarter')}, {cite('release')}. Trailing calculations splice fiscal 2025 with the 39-week 2026 comparison. Reclassifications and the fiscal calendar change mean they are approximations. No EV/EBITDA multiple is offered without reconciling lease treatment.",
              '## Recent events and competing interpretations',
              f"On November 3, 2026, Victor E. Mandel replaced Eric Lefebvre as CEO. Leadership execution is now a fresh variable. {cite('ceo')}",
              f"MTY said it rejected Foodtastic's conditional, non-binding CAD43 proposals and ended discussions. There was no agreement. CAD43 is neither an agreed takeover price nor a guaranteed price floor. The proposal dates in October are not public-information dates for a historical trading model. {cite('proposal')}",
              f"Foodtastic renewed public pressure for engagement on November 16 and reported favorable feedback. This is the bidder's position, not evidence of a signed deal or independently verified shareholder consensus. {cite('bidder_update')}",
              'Bull interpretation: stronger franchise economics, disciplined capital returns and reduced corporate-store exposure could support cash generation. Bear interpretation: weak demand, closures, debt and execution costs could keep pressuring earnings. These are analytical possibilities, not forecasts.',
              '## What remains unknown']
    lines += ['- '+x for x in e['unknowns']]
    lines += ['## Next evidence to collect',
              'Reconcile recurring cash flow excluding disposals, establish current diluted shares and buybacks, inspect franchise openings and closures by brand, verify lease economics, and obtain point-in-time source publication timestamps before testing fundamental signals.',
              'No standalone Kahala price target or probability of future profit is supported by the present evidence.',
              '## Source register']
    for key,src in sources.items():
        lines.append(f"- [{src['title']}]({src['url']}) — {src['locator']}. Published {src['published_on'] or 'date not established in this snapshot'}; verified {e['verified_on']}.")
    if cutoff>date.fromisoformat(e['verified_on']):
        lines.insert(2,'**Stale snapshot:** This is a dated evidence pack. No later facts or quotes have been refreshed.')
    return '\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n'
