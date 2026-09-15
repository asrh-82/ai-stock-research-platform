"""Portable evidence bundle for one explicitly generated simulation."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from Utils.market_data import MarketSnapshot
from Utils.market_monte_carlo import MarketResult, path_bands, price_digest, summarize_market, terminal_samples


def evidence_bundle(snapshot: MarketSnapshot, result: MarketResult,
                    calibration: pd.DataFrame | None = None) -> bytes:
    if price_digest(snapshot.prices) != result.diagnostics["history_sha256"]:
        raise ValueError("The result and history snapshot do not match.")
    if snapshot.reference_close != result.diagnostics["reference_close"]:
        raise ValueError("The result and snapshot have different reference closes.")
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(), "symbol": snapshot.symbol,
        "data": snapshot.provenance, "model": result.diagnostics,
        "warnings": [*snapshot.warnings, *result.warnings],
        "summary": summarize_market(result), "path_sample_count": min(100, len(result.paths)),
        "calibration_note": "Diagnostic only. Not evidence of trading profitability or independently verified data.",
    }
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        archive.writestr("history.csv", snapshot.prices.rename("Adjusted Close").to_csv(index_label="Date", float_format="%.17g"))
        archive.writestr("terminal_outcomes.csv", terminal_samples(result).to_csv(index=False))
        archive.writestr("pointwise_bands.csv", path_bands(result).to_csv())
        archive.writestr("sample_paths_first_100.csv", pd.DataFrame(result.paths[:100].T).to_csv(index_label="Trading step"))
        if calibration is not None:
            archive.writestr("walk_forward.csv", calibration.to_csv(index=False))
        archive.writestr("REPRODUCE.txt", "Use the engine version and dependency versions in manifest.json.\n"
                          "Load history.csv with pandas float_precision='round_trip', construct\n"
                          "MarketConfig(**manifest['model']['config']), then call\n"
                          "simulate_market(prices, config, anchor=manifest['model']['reference_close']).\n"
                          "This bundle contains all terminal outcomes, but only the first 100 full paths.\n"
                          "The bands are pointwise simulation percentiles, not a simultaneous confidence region.\n")
    return buffer.getvalue()
