"""Versioned private research records and paired 21-session paper cohorts.

No broker, execution API, unattended scheduling, or claim of audited timestamps.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd

from Utils.research_data import data_hash, validate_daily
from Utils.selection import BENCHMARK, NY, UniverseData, aware, canonical, digest, utc_now

LEDGER_SCHEMA = 1
MAX_BYTES = 2_000_000
MAX_EVENTS = 200
PAPER_VERSION = "paper-cohort-1.0"


def empty_ledger() -> dict:
    return {"schema": LEDGER_SCHEMA, "revision": 0, "events": []}


def verify_snapshot(snapshot: dict) -> None:
    if not isinstance(snapshot, dict) or "id" not in snapshot:
        raise ValueError("Invalid scan snapshot.")
    body = {k: v for k, v in snapshot.items() if k != "id"}
    if digest(body) != snapshot["id"]:
        raise ValueError("Scan content hash does not match.")
    if snapshot.get("version") != "selection-1.0":
        raise ValueError("Unsupported scan version.")
    if not isinstance(snapshot.get("ranked"), list) or len(snapshot["ranked"]) > 25:
        raise ValueError("Invalid ranking size.")
    rows = snapshot["ranked"]
    names = [r["symbol"] for r in rows]
    if len(names) != len(set(names)) or [r["rank"] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError("Invalid ranking order or duplicate symbols.")
    if rows != sorted(rows, key=lambda r: (-r["momentum"], r["symbol"])):
        raise ValueError("Rank order differs from the declared momentum formula.")
    numeric = ("momentum", "percentile", "reported_close", "volatility", "drawdown", "dollar_volume", "recent_return")
    for row in rows:
        if any(not isinstance(row[k], Real) or isinstance(row[k], bool) or not isfinite(row[k]) for k in numeric):
            raise ValueError("Nonfinite or malformed ranking metrics.")
        if not 0 <= row["percentile"] <= 100 or row["reported_close"] <= 0 or row["momentum"] <= -1:
            raise ValueError("Out-of-range ranking metrics.")
    exclusions = [r["symbol"] for r in snapshot["excluded"]]
    if len(names + exclusions) != len(set(names + exclusions)) or sorted(names + exclusions) != snapshot["universe"]:
        raise ValueError("Every universe member must be ranked or explicitly excluded.")
    if not isinstance(snapshot["synthetic"], bool):
        raise ValueError("Scan must explicitly identify synthetic data.")
    pd.Timestamp(snapshot["asof"])
    aware(datetime.fromisoformat(snapshot["created_at"]))


def _positive_money(value: object) -> bool:
    return (isinstance(value, Real) and not isinstance(value, bool)
            and isfinite(value) and 0 < value <= 1_000_000)


def freeze_selection(snapshot: dict, chosen: list[str], initial_cash: float = 10_000.,
                     now: datetime | None = None) -> dict:
    """Freeze model and discretionary choices together, before a future entry.

    Live entries are no earlier than the next NY calendar date AFTER recording,
    even when recording before today's open. Synthetic examples use an explicit
    hypothetical clock. Each cohort is independent; capital is not stitched.
    """
    verify_snapshot(snapshot)
    registered = aware(now or utc_now())
    if not _positive_money(initial_cash):
        raise ValueError("Virtual capital must be finite, positive, and at most 1,000,000.")
    eligible = [r["symbol"] for r in snapshot["ranked"]]
    if len(eligible) < 2:
        raise ValueError("At least two eligible names are needed for comparison.")
    slots = min(5, len(eligible))
    if not isinstance(chosen, list) or len(set(chosen)) != len(chosen):
        raise ValueError("Selections must be a list without duplicates.")
    if len(chosen) > slots or not set(chosen).issubset(eligible):
        raise ValueError(f"Choose at most {slots} names from this frozen ranking.")
    asof = pd.Timestamp(snapshot["asof"]).date()
    if snapshot["synthetic"]:
        decision_date = asof
    else:
        decision_date = registered.astimezone(NY).date()
        if not 0 <= (decision_date - asof).days <= 7:
            raise ValueError("Scan is future-dated or stale; refresh before freezing a paper cohort.")
        if aware(datetime.fromisoformat(snapshot["created_at"])) > registered:
            raise ValueError("Cannot record a decision before the scan was created.")
    record = {"version": PAPER_VERSION, "scan_id": snapshot["id"],
              "registered_at": registered.isoformat(), "decision_date": str(decision_date),
              "scan_asof": str(asof), "synthetic": snapshot["synthetic"],
              "earliest_entry_date": str(decision_date + timedelta(days=1)),
              "horizon_sessions": 21, "initial_cash": float(initial_cash),
              "commission_bps": 5.0, "slippage_bps": 10.0,
              "slots": slots, "model": eligible[:slots], "user": sorted(chosen),
              "eligible": eligible, "benchmark": BENCHMARK,
              "ranking": [{"symbol": r["symbol"], "momentum": r["momentum"]}
                          for r in snapshot["ranked"]],
              "rules": "Top five eligible names, equal allocation on entry; unfilled user slots remain cash.",
              "units": "fractional adjusted research units; not executable shares",
              "evidence": "synthetic demonstration" if snapshot["synthetic"] else "self-recorded prospective paper cohort",
              "timestamp_assurance": "not independently attested; browser records are user-editable"}
    record["id"] = digest(record)
    return record


def verify_decision(decision: dict) -> None:
    if not isinstance(decision, dict) or decision.get("version") != PAPER_VERSION:
        raise ValueError("Unsupported paper record.")
    if digest({k: v for k, v in decision.items() if k != "id"}) != decision.get("id"):
        raise ValueError("Paper decision hash does not match.")
    if not _positive_money(decision.get("initial_cash")):
        raise ValueError("Invalid virtual capital.")
    if decision.get("horizon_sessions") != 21 or decision.get("commission_bps") != 5.0 or decision.get("slippage_bps") != 10.0:
        raise ValueError("Unsupported paper protocol; use a distinct version for different rules.")
    eligible = decision.get("eligible", [])
    if not 2 <= len(eligible) <= 25 or len(eligible) != len(set(eligible)):
        raise ValueError("Invalid frozen universe.")
    if decision.get("slots") != min(5, len(eligible)) or decision.get("model") != eligible[:decision["slots"]]:
        raise ValueError("Model selection does not match the declared rule.")
    user = decision.get("user", [])
    if len(user) != len(set(user)) or len(user) > decision["slots"] or not set(user).issubset(eligible):
        raise ValueError("Invalid user selection.")
    if [r["symbol"] for r in decision["ranking"]] != eligible:
        raise ValueError("Rank order does not match the frozen universe.")
    aware(datetime.fromisoformat(decision["registered_at"]))
    if pd.Timestamp(decision["earliest_entry_date"]) != pd.Timestamp(decision["decision_date"]) + pd.Timedelta(days=1):
        raise ValueError("Entry must follow the decision date.")


def evaluate_cohort(decision: dict, batch: UniverseData, now: datetime | None = None) -> dict:
    """Evaluate all frozen names on the same reference sessions; no silent drops.

    Entry is a next-session-open fill proxy. A completed cohort's last close
    uses a predeclared 21st-session liquidation proxy. In-progress cohorts stay
    marked; hypothetical liquidation is separate. No full-fill realism claim.
    """
    verify_decision(decision)
    recorded = aware(now or utc_now())
    if bool(decision["synthetic"]) != bool(batch.synthetic):
        raise ValueError("Synthetic and market data cannot be mixed.")
    if BENCHMARK not in batch.histories:
        raise ValueError("Reference history unavailable.")
    reference = validate_daily(batch.histories[BENCHMARK].prices)
    if not batch.synthetic:
        if recorded < aware(datetime.fromisoformat(decision["registered_at"])):
            raise ValueError("Observation precedes registration.")
        reference = reference.loc[reference.index < pd.Timestamp(recorded.astimezone(NY).date())]
    reference = reference.loc[reference.index >= pd.Timestamp(decision["earliest_entry_date"])]
    window = reference.iloc[:21].index
    base = {"decision_id": decision["id"], "observed_at": recorded.isoformat(),
            "synthetic": batch.synthetic, "sessions": len(window), "horizon_sessions": 21,
            "status": "COMPLETE" if len(window) == 21 else "IN_PROGRESS" if len(window) else "AWAITING_NEXT_SESSION"}
    if len(window) == 0:
        return {**base, "books": {}, "curve": [], "rank_ic": None}
    if reference.index[0] > pd.Timestamp(decision["earliest_entry_date"]) + pd.Timedelta(days=7):
        raise ValueError("Reference coverage starts too late; the actual first eligible session is unknown.")
    all_symbols = sorted(set(decision["eligible"]) | {BENCHMARK})
    frames, hashes = {}, {}
    for symbol in all_symbols:
        if symbol in batch.errors or symbol not in batch.histories:
            raise ValueError(f"{symbol}: missing frozen-universe member; evaluation blocked, not dropped.")
        data = validate_daily(batch.histories[symbol].prices)
        if not window.isin(data.index).all():
            raise ValueError(f"{symbol}: missing required outcome sessions; no forward fill or survivor-only result.")
        frame = data.loc[window]
        if (frame.Volume <= 0).any():
            raise ValueError(f"{symbol}: zero-volume session; this simplified fill model cannot evaluate the cohort.")
        frames[symbol] = frame
        hashes[symbol] = data_hash(frame)
    fee, slip = decision["commission_bps"] / 10000, decision["slippage_bps"] / 10000
    initial = decision["initial_cash"]
    groups = {"Model": {s: 1 / decision["slots"] for s in decision["model"]},
              "User": {s: 1 / decision["slots"] for s in decision["user"]},
              "Equal universe": {s: 1 / len(decision["eligible"]) for s in decision["eligible"]},
              "Reference": {BENCHMARK: 1.0}}
    books, curves = {}, {}
    for name, weights in groups.items():
        cash = initial * (1 - sum(weights.values()))
        equity = pd.Series(cash, index=window, dtype=float)
        entry_fees = slippage = 0.0
        for symbol, weight in weights.items():
            first_open = float(frames[symbol].Open.iloc[0])
            quantity = initial * weight / (first_open * (1 + slip) * (1 + fee))
            equity += quantity * frames[symbol].Close
            entry_fees += quantity * first_open * (1 + slip) * fee
            slippage += quantity * first_open * slip
        marked = float(equity.iloc[-1])
        proceeds = (marked - cash) * (1 - slip) * (1 - fee)
        liquidation = cash + proceeds
        if base["status"] == "COMPLETE":
            equity.iloc[-1] = liquidation
            entry_fees += (marked - cash) * (1 - slip) * fee
            slippage += (marked - cash) * slip
        drawdown = equity / equity.cummax().clip(lower=initial) - 1
        books[name] = {"ending_equity": float(equity.iloc[-1]),
                       "net_return": float(equity.iloc[-1] / initial - 1),
                       "estimated_liquidation_equity": liquidation,
                       "maximum_drawdown": float(min(0, drawdown.min())),
                       "fees": float(entry_fees), "modeled_slippage": float(slippage),
                       "cash_allocation": max(0.0, 1 - sum(weights.values())),
                       "weights": weights}
        curves[name] = equity
    curve = pd.DataFrame(curves)
    curve.index.name = "Date"
    ranks = pd.Series({r["symbol"]: r["momentum"] for r in decision["ranking"]})
    returns = pd.Series({s: float(frames[s].Close.iloc[-1] / frames[s].Open.iloc[0] - 1)
                         for s in decision["eligible"]})
    ic = None
    if len(window) == 21 and len(ranks) >= 3 and ranks.nunique() > 1 and returns.nunique() > 1:
        ic = float(ranks.rank().corr(returns.rank()))
    return {**base, "first_session": str(window[0].date()), "asof": str(window[-1].date()),
            "books": books, "rank_ic": ic,
            "curve": [{"date": str(d.date()), **{k: float(v) for k, v in r.items()}}
                      for d, r in curve.iterrows()],
            "outcome_hashes": hashes,
            "limitations": "Independent cohort, not a continuous rebalanced account. Fractional adjusted-unit fill proxies; no tax, interest, spread calibration, market impact or settlement. Reference sessions are not an independently verified exchange calendar."}


def validate_ledger(document: dict) -> dict:
    """Detect accidental changes and invalid links, NOT malicious re-hashing."""
    try:
        if len(canonical(document).encode()) > MAX_BYTES:
            raise ValueError("Record exceeds the 2 MB browser-vault limit.")
        if not isinstance(document, dict) or document.get("schema") != LEDGER_SCHEMA:
            raise ValueError("Unsupported ledger schema.")
        events = document.get("events")
        if not isinstance(events, list) or len(events) > MAX_EVENTS or isinstance(document.get("revision"), bool) or document.get("revision") != len(events):
            raise ValueError("Invalid event count or revision.")
        last, scans, decisions, seen, outcomes = "ROOT", {}, {}, set(), {}
        for event in events:
            if event.get("previous") != last or digest({k: v for k, v in event.items() if k != "id"}) != event.get("id"):
                raise ValueError("Record hash chain is broken.")
            aware(datetime.fromisoformat(event["recorded_at"]))
            body = event["payload"]
            kind = event["kind"]
            if kind == "scan":
                verify_snapshot(body)
                if body["id"] in scans:
                    raise ValueError("Duplicate scan event.")
                scans[body["id"]] = body
            elif kind == "decision":
                verify_decision(body)
                if body["scan_id"] not in scans or body["scan_id"] in seen:
                    raise ValueError("Decision has no unique frozen scan.")
                s = scans[body["scan_id"]]
                if body["synthetic"] != s["synthetic"] or body["eligible"] != [r["symbol"] for r in s["ranked"]]:
                    raise ValueError("Decision differs from its scan.")
                decisions[body["id"]] = body
                seen.add(body["scan_id"])
            elif kind == "observation":
                d = decisions.get(body["decision_id"])
                if d is None or d["synthetic"] != body["synthetic"]:
                    raise ValueError("Observation has no matching decision.")
                if not isinstance(body["sessions"], int) or isinstance(body["sessions"], bool) or not 0 <= body["sessions"] <= 21:
                    raise ValueError("Invalid observation horizon.")
                if body["status"] not in ("AWAITING_NEXT_SESSION", "IN_PROGRESS", "COMPLETE"):
                    raise ValueError("Invalid observation status.")
                expected = "COMPLETE" if body["sessions"] == 21 else "IN_PROGRESS" if body["sessions"] else "AWAITING_NEXT_SESSION"
                if body["status"] != expected or len(body["curve"]) != body["sessions"]:
                    raise ValueError("Observation status or curve length differs from its horizon.")
                prior = outcomes.get(body["decision_id"], -1)
                if body["sessions"] <= prior:
                    raise ValueError("Outcome sessions must advance; saved observations cannot be overwritten.")
                outcomes[body["decision_id"]] = body["sessions"]
                aware(datetime.fromisoformat(body["observed_at"]))
                if body["sessions"]:
                    if set(body["books"]) != {"Model", "User", "Equal universe", "Reference"}:
                        raise ValueError("Missing comparison book.")
                    dates = [pd.Timestamp(row["date"]) for row in body["curve"]]
                    if dates != sorted(set(dates)) or dates[0] < pd.Timestamp(d["earliest_entry_date"]):
                        raise ValueError("Outcome dates must follow registration and be ordered.")
                    for book in body["books"].values():
                        if any(not isinstance(book[k], Real) or not isfinite(book[k]) for k in
                               ("ending_equity", "net_return", "maximum_drawdown")):
                            raise ValueError("Malformed performance metrics.")
            else:
                raise ValueError("Unknown record type.")
            last = event["id"]
        return copy.deepcopy(document)
    except (KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed research record.") from exc


def append_event(document: dict, kind: str, payload: dict, now: datetime | None = None) -> dict:
    result = validate_ledger(document)
    if len(result["events"]) >= MAX_EVENTS:
        raise ValueError("Vault limit reached. Export a backup before starting a new workspace.")
    event = {"kind": kind, "payload": copy.deepcopy(payload),
             "recorded_at": aware(now or utc_now()).isoformat(),
             "previous": result["events"][-1]["id"] if result["events"] else "ROOT"}
    event["id"] = digest(event)
    result["events"].append(event)
    result["revision"] += 1
    return validate_ledger(result)


def parse_backup(payload: bytes) -> dict:
    if not isinstance(payload, bytes) or len(payload) > MAX_BYTES:
        raise ValueError("Provide a JSON backup no larger than 2 MB.")
    try:
        value = json.loads(payload.decode("utf-8"))
        return validate_ledger(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid JSON backup.") from exc


def records(document: dict, kind: str) -> list[dict]:
    return [e["payload"] for e in document["events"] if e["kind"] == kind]
