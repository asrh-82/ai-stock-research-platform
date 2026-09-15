"""Execute the checked-in opportunity scanner and preserve success/failure evidence."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Utils.opportunities import OpportunityConfig, VERSION
from Utils.opportunity_scan import export_scan, run_scan, scan_table


def main() -> int:
    output = Path("opportunity_evidence")
    output.mkdir(exist_ok=True)
    config = OpportunityConfig()
    now = datetime.now(timezone.utc)
    record = {"version": VERSION, "captured": now.isoformat(), "config": asdict(config),
              "requested_universe": "PLTR,USO,SPCX", "benchmark": "SPY",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                  ("Utils/opportunities.py", "Utils/opportunity_scan.py", "Utils/market_data.py")}}
    try:
        scan = run_scan(record["requested_universe"], "SPY", config, years=5, now=now)
        record["results"] = {s: r.summary for s, r in scan.results.items()}
        record["errors"] = scan.errors
        (output/"complete_scan.zip").write_bytes(export_scan(scan))
        table = scan_table(scan)
        (output/"scan.csv").write_text(table.to_csv(index=False))
        print(table.to_string(index=False))
        for s,r in scan.results.items():
            print("AXION_OPPORTUNITY " + json.dumps(r.summary, allow_nan=False), flush=True)
        for s,e in scan.errors.items():
            print("AXION_OPPORTUNITY_BLOCKED " + json.dumps({"symbol":s,"error":e}), flush=True)
        record["run_status"] = "completed" if scan.results else "all_blocked"
    except Exception as exc:
        record["run_status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(record["error"], flush=True)
    (output/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    print("AXION_OPPORTUNITY_RUN " + json.dumps(record, allow_nan=False), flush=True)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("# Opportunity evidence\n\nRead-only price-based forecasts, not proven mispricing.\n\n```json\n"
                    + json.dumps(record, indent=2, allow_nan=False) + "\n```\n")
    return 0 if record["run_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
