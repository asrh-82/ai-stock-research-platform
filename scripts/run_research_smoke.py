"""Exercise historical research on predeclared example datasets, not recommendations.

No raw prices are published. Provider failures remain DATA_UNAVAILABLE; there is
no synthetic replacement. Engineering CI must not confuse these with market tests.
"""
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from Utils.research_data import download_daily
from Utils.research_protocol import Protocol, Rule, develop, evaluate_holdout


def main():
    folder = Path("artifacts")
    folder.mkdir(exist_ok=True)
    evidence = {"status": "NOT_VALIDATED", "purpose": "historical engineering exercise, not an investment recommendation",
                "commit": os.environ.get("GITHUB_SHA", "local"), "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "datasets": []}
    # All examples and parameters are declared before seeing any of their outcomes.
    for symbol in ("SPY", "QQQ", "IWM"):
        try:
            snapshot = download_daily(symbol, "2010-01-01", "2026-01-01")
            development = develop(snapshot, Protocol())
            (folder / f"{symbol}_frozen_plan.json").write_text(json.dumps(development.plan, indent=2, allow_nan=False))
            holdout, passive, stress = evaluate_holdout(snapshot, development.plan)
            row = {"symbol": symbol, "data_status": "HISTORICAL_DATA_RETRIEVED", "strategy_status": "NOT_VALIDATED",
                   "source": snapshot.manifest(), "folds": len(development.folds),
                   "frozen_rule": Rule(**development.plan["selected_rule"]).label,
                   "holdout_first_session": development.plan["holdout_first_session"],
                   "holdout_last_session": development.plan["holdout_last_session"],
                   "development": development.simulation.metrics, "development_passive": development.passive.metrics,
                   "holdout": holdout.metrics, "holdout_passive": passive.metrics,
                   "cost_stress": stress.to_dict("records"), "plan_sha256": development.plan["plan_sha256"]}
        except Exception as exc:
            row = {"symbol": symbol, "data_status": "DATA_UNAVAILABLE_OR_RUN_FAILED",
                   "error_type": type(exc).__name__, "error": str(exc)}
        evidence["datasets"].append(row)
        print("HISTORICAL_EVIDENCE " + json.dumps(row, allow_nan=False), flush=True)
    evidence["historical_runs_completed"] = sum(x["data_status"] == "HISTORICAL_DATA_RETRIEVED" for x in evidence["datasets"])
    (folder / "historical_research.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
    print("HISTORICAL_RUNS_COMPLETED=" + str(evidence["historical_runs_completed"]), flush=True)
    return 0 if evidence["historical_runs_completed"] == 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
