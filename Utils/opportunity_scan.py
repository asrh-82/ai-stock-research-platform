"""Bounded opportunity scans. Explicit provider execution; no calls on import."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import re
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from Utils.opportunities import (OpportunityConfig, OpportunityResult, export_opportunity,
                                 forecast_opportunity, validate_config, validate_history)


@dataclass(frozen=True)
class ScanResult:
    symbols: tuple[str, ...]
    benchmark: str
    captured: str
    config: OpportunityConfig
    results: dict[str, OpportunityResult]
    errors: dict[str, str]
    lookback_years: int


def parse_symbols(text: str) -> tuple[str, ...]:
    symbols = tuple(dict.fromkeys(s.upper() for s in re.split(r"[,\s]+", text.strip()) if s))
    if not 1 <= len(symbols) <= 12:
        raise ValueError("Enter 1 to 12 distinct equity/ETF tickers, separated by commas or spaces.")
    if any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,14}", s) for s in symbols):
        raise ValueError("Invalid ticker. No company-name guessing or symbol substitution.")
    return symbols


def align_snapshots(asset, benchmark) -> pd.DataFrame:
    if asset.symbol == benchmark.symbol:
        raise ValueError("The research asset cannot be its own benchmark.")
    if asset.prices.index[-1] != benchmark.prices.index[-1]:
        raise ValueError("Asset and benchmark have different last sessions; refusing stale alignment.")
    start = max(asset.prices.index[0], benchmark.prices.index[0])
    a, b = asset.prices.loc[start:], benchmark.prices.loc[start:]
    if not a.index.equals(b.index):
        raise ValueError("Session dates differ inside the common history; no intersection/drop/fill was used.")
    return validate_history(pd.DataFrame({"asset": a, "benchmark": b}))


def run_scan(text: str, benchmark: str = "SPY", config: OpportunityConfig | None = None,
             years: int = 5, now: datetime | None = None, loader=None) -> ScanResult:
    c = config or OpportunityConfig()
    validate_config(c)
    symbols = parse_symbols(text)
    bench_symbols = parse_symbols(benchmark)
    if len(bench_symbols) != 1 or bench_symbols[0] in symbols:
        raise ValueError("Choose one separate benchmark, not one of the scanned assets.")
    benchmark = bench_symbols[0]
    if isinstance(years, bool) or years not in (5, 8):
        raise ValueError("Opportunity research requests five or eight years of history.")
    captured = now or datetime.now(timezone.utc)
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("The scan timestamp must be timezone aware.")
    if loader is None:
        from Utils.market_data import load_market_snapshot
        loader = load_market_snapshot
    # One reference and one cutoff across the complete prespecified scan family.
    bench = loader(benchmark, years, now=captured)
    if bench.symbol != benchmark:
        raise ValueError("Benchmark identity mismatch.")
    results, errors = {}, {}

    def process(symbol):
        snapshot = loader(symbol, years, now=captured)
        if snapshot.symbol != symbol:
            raise ValueError("Asset identity mismatch.")
        history = align_snapshots(snapshot, bench)
        provenance = {"asset": snapshot.provenance, "benchmark": bench.provenance,
                      "captured": captured.isoformat(), "requested_symbols": symbols,
                      "common_start": str(history.index[0].date()),
                      "asset_rows_before_common_start": int((snapshot.prices.index < history.index[0]).sum()),
                      "benchmark_rows_before_common_start": int((bench.prices.index < history.index[0]).sum()),
                      "synthetic": bool(snapshot.provenance.get("synthetic") or bench.provenance.get("synthetic")),
                      "warnings": [*snapshot.warnings, *bench.warnings]}
        return forecast_opportunity(history, symbol, benchmark, c, len(symbols), provenance)

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(process, s): s for s in symbols}
        for future in as_completed(jobs):
            s = jobs[future]
            try:
                results[s] = future.result()
            except Exception as exc:
                errors[s] = f"{type(exc).__name__}: {str(exc)[:400]}"
    # Stable ordering is independent of thread completion order.
    return ScanResult(symbols, benchmark, captured.isoformat(), c,
                      {s: results[s] for s in symbols if s in results},
                      {s: errors[s] for s in symbols if s in errors}, years)


def scan_table(scan: ScanResult) -> pd.DataFrame:
    rows = []
    for symbol in scan.symbols:
        if symbol in scan.errors:
            rows.append({"Symbol": symbol, "Status": "Data/model blocked", "Reason": scan.errors[symbol]})
            continue
        s = scan.results[symbol].summary
        ev = s["evidence"]
        rows.append({"Symbol": symbol, "Status": s["status"],
                     "Forecast total %": 100*s["forecast_total"],
                     "Net excess pp": 100*s["net_forecast_excess"],
                     "Gap / OOS error": s["gap_per_oos_error"],
                     "OOS windows": ev["oos_folds"],
                     "Skill vs zero %": None if ev.get("skill_vs_zero") is None else 100*ev["skill_vs_zero"],
                     "Skill vs mean %": None if ev.get("skill_vs_historical_mean") is None else 100*ev["skill_vs_historical_mean"],
                     "As of": s["as_of"], "Reason": "; ".join(s["reasons"]) or "Exploratory gates cleared; prospective validation still required."})
    return pd.DataFrame(rows)


def export_scan(scan: ScanResult) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as z:
        z.writestr("scan.json", json.dumps({"symbols": scan.symbols, "benchmark": scan.benchmark,
                    "captured": scan.captured, "config": asdict(scan.config), "lookback_years": scan.lookback_years,
                    "errors": scan.errors, "note": "All attempted assets retained; no hidden failed candidates."}, indent=2))
        z.writestr("scan.csv", scan_table(scan).to_csv(index=False))
        for symbol, result in scan.results.items():
            z.writestr(f"{symbol}_evidence.zip", export_opportunity(result))
    return buffer.getvalue()
