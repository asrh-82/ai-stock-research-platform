"""Attach a fixed historical price study to the company resolved by the filing pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from io import BytesIO
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile
from zoneinfo import ZoneInfo

import pandas as pd
from yfinance.exceptions import YFException

from Utils.backtesting import BacktestConfig
from Utils.market_data import fetch_daily_history
from Utils.walk_forward import (
    block_bootstrap_active_mean,
    pattern_diagnostics,
    research_candidates,
    run_walk_forward,
)

from .sources import ResearchError

MARKET_ERRORS = (ValueError, OSError, RuntimeError, KeyError, BadZipFile, YFException)


@dataclass
class PriceStudy:
    markdown: str
    audit: bytes


def _snapshot(symbol, benchmark, currency, start, end, recorded):
    data, raw, manifests = {}, {}, {}
    for ticker in (symbol, benchmark):
        path = "prices/" + ticker + ".csv"
        if recorded is None:
            frame, meta = fetch_daily_history(ticker, start, end)
            payload = frame.to_csv(index_label="Date").encode()
        else:
            meta = json.loads(recorded.read("market_sources.json"))[ticker]
            payload = recorded.read(path)
            if hashlib.sha256(payload).hexdigest() != meta["snapshot_sha256"]:
                raise ResearchError("Market snapshot checksum mismatch.")
            frame = pd.read_csv(BytesIO(payload), index_col=0, parse_dates=True, float_precision="round_trip")
        if (meta["ticker"], meta["currency"], meta["start_requested"], meta["end_exclusive"]) != (
            ticker,
            currency,
            start,
            end,
        ):
            raise ResearchError("Market data identity, currency or requested dates disagree.")
        first = frame.index[0].date().isoformat()
        last = frame.index[-1].date().isoformat()
        if first < start or last >= end or (pd.Timestamp(end) - frame.index[-1]).days > 5:
            raise ResearchError("Market snapshot dates fall outside the requested range or are stale.")
        meta["snapshot_sha256"] = hashlib.sha256(payload).hexdigest()
        data[ticker], raw[path], manifests[ticker] = frame, payload, meta
    return data, raw, manifests


def _benchmark_checks(history, benchmark):
    if benchmark != "XIU.TO":
        return []
    # Annual market and NAV series can differ modestly. Preserve the original MTY study screen.
    issuer_returns = {2021: 0.2790, 2022: -0.0636, 2023: 0.1182, 2024: 0.2077, 2025: 0.2889}
    checks = []
    for year, expected in issuer_returns.items():
        series = history["Adj Close"]
        if series.index[0] > pd.Timestamp(f"{year - 1}-12-01") or series.index[-1] < pd.Timestamp(
            f"{year}-12-31"
        ):
            continue
        observed = float(series.loc[str(year)].iloc[-1] / series.loc[str(year - 1)].iloc[-1] - 1)
        if abs(observed - expected) > 0.0025:
            raise ResearchError(f"XIU failed the issuer annual-return check for {year}.")
        checks.append(
            {
                "year": year,
                "market_return": observed,
                "issuer_NAV_return": expected,
                "source": "https://www.blackrock.com/ca/investors/en/products/239832/ishares-sp-tsx-60-index-etf",
            }
        )
    return checks


def _same_split_basis(history, earnings_end):
    actions = history.get("Stock Splits")
    return actions is not None and not actions.loc[actions.index > pd.Timestamp(earnings_end)].ne(0).any()


def run_price_study(report, recorded=None):
    ticker = report.ticker
    currency, benchmark = ("CAD", "XIU.TO") if ticker == "MTY.TO" else ("USD", "SPY")
    day = datetime.fromisoformat(report.retrieved).astimezone(ZoneInfo("America/New_York")).date()
    start = "2016-07-27" if ticker == "MTY.TO" else (day - timedelta(days=3652)).isoformat()
    end = day.isoformat()
    if recorded is None:
        data, files, manifests = _snapshot(ticker, benchmark, currency, start, end, None)
    else:
        with ZipFile(recorded) as archive:
            data, files, manifests = _snapshot(ticker, benchmark, currency, start, end, archive)
    asset, ref = data[ticker], data[benchmark]
    checks = _benchmark_checks(ref, benchmark)
    base = BacktestConfig()
    candidates = research_candidates(base)
    study = run_walk_forward(asset, ref, currency=currency, candidates=candidates)
    sim = study["holdout"]
    stock = study["holdout_buy_hold"]
    beginning = study["holdout_start"]
    diagnostics = pattern_diagnostics(asset.loc[asset.index < beginning])
    interval = block_bootstrap_active_mean(
        sim.performance["Strategy Return"], sim.performance["Benchmark Return"]
    )
    completed = int((sim.trades.Status == "Closed").sum())
    gap = float(
        sim.performance.Equity.iloc[-1]
        - base.initial_capital
        - sim.trades.PnL.sum()
        - sim.performance["Cash Interest"].sum()
    )
    if abs(gap) > 0.00001:
        raise ResearchError("The trade ledger does not reconcile to ending equity.")
    portfolio_rows = [
        (study["selected"], sim.strategy_metrics),
        (ticker + " buy and hold", stock.strategy_metrics),
        (benchmark + " buy and hold", sim.benchmark_metrics),
    ]
    lines = [
        "## Historical price study",
        (
            f"The price series belongs to {report.issuer} ({ticker}). The latest included close was {currency} "
            f"{asset.Close.iloc[-1]:,.2f} on {asset.index[-1].date()}."
        ),
        (
            f"Earlier prices selected {study['selected']} from six rules with a cash fallback. The final {study['holdout_bars']} sessions "
            f"ran from {beginning.date()} to {asset.index[-1].date()}. {completed} strategy trades closed."
        ),
        "| Portfolio | Net return | Maximum drawdown |",
        "|---|---:|---:|",
        *[f"| {label} | {m.total_return:+.2%} | {m.maximum_drawdown:.2%} |" for label, m in portfolio_rows],
        (
            f"Earlier walk-forward validation returned {study['validation_metrics'].total_return:+.2%} across "
            f"{len(study['folds'])} folds. Each fold starts a fresh account. The 95% descriptive block-bootstrap "
            f"interval for annualized arithmetic active return versus {benchmark} was {interval['lower_95']:+.1%} to {interval['upper_95']:+.1%}."
        ),
        (
            f"The simulation started with {currency} {base.initial_capital:,.0f}. Orders used the next adjusted open, "
            "5 basis points of commission and 10 basis points of slippage per side. The order limit was 1% of prior average daily dollar volume."
        ),
        "Patterns in the development period are descriptive. The forward returns in this table overlap and exclude trading costs.",
        "| Pattern at close | Forward holding period | Observations | Mean forward return |",
        "|---|---|---:|---:|",
    ]
    for _, row in diagnostics.loc[diagnostics["Holding Sessions"] == 5].iterrows():
        mean = row["Mean Forward Return"]
        formatted = "—" if pd.isna(mean) else f"{mean:+.2%}"
        lines.append(f"| {row['Pattern']} | 5 sessions | {row['Observations']} | {formatted} |")
    lines += [
        (
            "The strategy uses only prices available at each signal. Today’s filing figures supply context. "
            "Daily quotes come from one vendor, execution is simulated, and the results do not establish future profitability."
        )
    ]
    eps = next(
        (f for f in report.facts if f.key == "eps" and f.period == "Fiscal year" and f.value > 0), None
    )
    if eps and eps.unit == currency + "/share" and _same_split_basis(asset, eps.end):
        lines.append(
            f"The dated close divided by diluted EPS for the fiscal year ended {eps.end} was "
            f"{float(asset.Close.iloc[-1]) / float(eps.value):.2f}×."
        )
    by_key = {f.key: f for f in report.facts}
    annual, ytd = by_key.get("annual_eps"), by_key.get("ytd_eps")
    if (
        annual
        and ytd
        and ytd.prior is not None
        and ytd.prior_end < annual.end < ytd.end
        and _same_split_basis(asset, ytd.prior_end)
    ):
        bridge = annual.value + ytd.value - ytd.prior
        if bridge > 0:
            lines.append(
                f"Approximate trailing diluted EPS was {currency} {bridge:.2f} "
                f"({annual.value:.2f} + {ytd.value:.2f} − {ytd.prior:.2f}). The dated close divided by "
                f"that EPS was {float(asset.Close.iloc[-1]) / float(bridge):.2f}×. Fiscal week and month definitions can differ."
            )
    dividend = by_key.get("dividend")
    if dividend and _same_split_basis(asset, dividend.end):
        yield_value = 4 * float(dividend.value) / float(asset.Close.iloc[-1])
        lines.append(
            f"Repeating the {currency} {dividend.value:.2f} quarterly announcement four times would give "
            f"an indicated {yield_value:.2%} yield at the dated close. Check subsequent dividend declarations."
        )
    protocol = {
        "asset": ticker,
        "benchmark": benchmark,
        "currency": currency,
        "start": start,
        "end_exclusive": end,
        "candidates": {label: asdict(cfg) for label, cfg in candidates.items()},
        "selection": study["selection_rule"],
        "train_sessions": 504,
        "validation_sessions": 126,
        "reserved_sessions": 252,
        "first_test_session": str(beginning.date()),
        "selected": study["selected"],
        "closed_trades": completed,
        "ledger_error": gap,
        "portfolio_metrics": {label: asdict(m) for label, m in portfolio_rows},
        "bootstrap": interval,
        "benchmark_annual_checks": checks,
        "warnings": sim.warnings,
        "limitations": study["limitations"],
    }
    files["market_sources.json"] = json.dumps(manifests, indent=2)
    files["protocol_and_results.json"] = json.dumps(protocol, indent=2, allow_nan=False)
    for name in ("trials", "folds", "validation"):
        files[name + ".csv"] = study[name].to_csv(index=(name == "validation"))
    files["development_patterns.csv"] = diagnostics.to_csv(index=False)
    for name, frame in (("trades", sim.trades), ("orders", sim.orders), ("daily", sim.performance)):
        files[name + ".csv"] = frame.to_csv(index=(name == "daily"))
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return PriceStudy("\n\n".join(lines).replace("|\n\n|", "|\n|") + "\n", output.getvalue())
