"""Generate reports from a company name or ticker using the same workflow as the app."""

import argparse
from pathlib import Path

from .pipeline import research_company
from .price_study import MARKET_ERRORS, run_price_study
from .sources import SourceClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("company")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--replay", type=Path, help="Read the recorded filing sources instead of downloading")
    parser.add_argument("--with-study", action="store_true", help="Include the fixed market study")
    parser.add_argument(
        "--replay-prices", type=Path, help="Replay a market-study.zip together with its dated filing sources"
    )
    args = parser.parse_args()
    if args.replay_prices and not args.replay:
        parser.error("--replay-prices needs --replay for the same dated run")
    if args.with_study and args.replay and not args.replay_prices:
        parser.error("For offline reproduction, supply --replay-prices with --with-study")
    if args.output.exists():
        parser.error("Choose a new output directory for this run.")
    client = SourceClient(args.replay)
    report = research_company(args.company, client)
    args.output.mkdir(parents=True, exist_ok=False)
    client.save(args.output / "filing-sources.zip")
    (args.output / "evidence.json").write_text(report.audit())
    path = args.output / "report.md"
    path.write_text(report.markdown())
    if args.with_study or args.replay_prices:
        try:
            study = run_price_study(report, args.replay_prices)
        except MARKET_ERRORS as exc:
            parser.exit(2, f"Filing report saved to {path}. Price study unavailable: {exc}\n")
        (args.output / "market-study.zip").write_bytes(study.audit)
        path.write_text(report.markdown() + "\n" + study.markdown)
    print(f"{report.ticker} · {report.document} · {len(report.facts)} reported measures · {path}")


if __name__ == "__main__":
    main()
