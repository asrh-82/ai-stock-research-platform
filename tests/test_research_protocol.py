"""Regression tests for chronology, invariants, accounting, and frozen experiments."""
import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from Utils.research_data import demo_snapshot, parse_daily_csv, validate_daily
from Utils.research_protocol import (DEFAULT_RULES, Execution, Protocol, Rule, _metrics,
                                    _rank, _simulate, develop, evaluate_holdout, signal, stress_test)

RULES = (Rule("cash"), Rule("sma", 20, 5), Rule("momentum", 20), Rule("rsi", 14))


def protocol(**changes):
    return replace(Protocol(train=100, test=40, holdout=60, rules=RULES), **changes)


@pytest.fixture
def snapshot():
    return demo_snapshot(360)


@pytest.mark.parametrize("rule", DEFAULT_RULES)
def test_signals_are_prefix_invariant(snapshot, rule):
    whole = signal(snapshot.prices.Close, rule)
    for stop in (50, 100, 280):
        pd.testing.assert_series_equal(whole.iloc[:stop], signal(snapshot.prices.Close.iloc[:stop], rule))


@pytest.mark.parametrize("kind", ["sma", "momentum", "rsi"])
def test_signal_warmup_stays_cash(snapshot, kind):
    rule = Rule(kind, 20, 5)
    first_valid = 19 if kind == "sma" else 20
    assert signal(snapshot.prices.Close, rule).iloc[:first_valid].eq(0).all()


def test_rsi_extreme_and_flat_prices():
    dates = pd.date_range("2020-01-01", periods=50)
    rising = pd.Series(np.arange(1, 51, dtype=float), index=dates)
    falling = pd.Series(np.arange(50, 0, -1, dtype=float), index=dates)
    flat = pd.Series(100.0, index=dates)
    assert signal(rising, Rule("rsi", 14)).eq(0).all()
    assert signal(falling, Rule("rsi", 14)).iloc[14:].eq(1).all()
    assert signal(flat, Rule("rsi", 14)).eq(0).all()


def test_holdout_changes_cannot_change_development(snapshot):
    first = develop(snapshot, protocol())
    changed = copy.deepcopy(snapshot)
    changed.prices.iloc[-60:, :4] *= np.linspace(.05, 8.0, 60)[:, None]
    second = develop(changed, protocol())
    pd.testing.assert_frame_equal(first.training_scores, second.training_scores)
    pd.testing.assert_frame_equal(first.folds, second.folds)
    pd.testing.assert_frame_equal(first.simulation.curve, second.simulation.curve)
    assert first.plan["selected_rule"] == second.plan["selected_rule"]
    assert first.plan["plan_sha256"] != second.plan["plan_sha256"]


def test_later_validation_cannot_change_earlier_choices(snapshot):
    first = develop(snapshot, protocol())
    changed = copy.deepcopy(snapshot)
    changed.prices.iloc[200:, :4] *= .2
    second = develop(changed, protocol())
    pd.testing.assert_frame_equal(first.training_scores.query("fold <= 2"), second.training_scores.query("fold <= 2"))
    pd.testing.assert_frame_equal(first.simulation.curve.iloc[:99], second.simulation.curve.iloc[:99])


def test_boundaries_gaps_and_candidate_counts(snapshot):
    cfg = protocol(gap=3)
    result = develop(snapshot, cfg)
    assert result.folds.iloc[0].test_start == cfg.train + cfg.gap
    for row in result.folds.itertuples():
        assert row.test_start - row.train_stop_exclusive == cfg.gap
        assert row.test_stop_exclusive <= result.plan["development_stop_exclusive"]
    starts = result.folds.test_start.to_numpy()[1:]
    stops = result.folds.test_stop_exclusive.to_numpy()[:-1]
    np.testing.assert_array_equal(starts, stops)
    assert len(result.training_scores) == len(result.folds) * len(cfg.rules)
    assert result.simulation.curve.index.max() < snapshot.prices.index[result.plan["holdout_start"]]
    assert result.plan["holdout_start"] - result.plan["development_stop_exclusive"] == 3


def test_compounded_fold_returns_reconcile(snapshot):
    result = develop(snapshot, protocol())
    assert np.prod(1 + result.folds.oos_net_return) - 1 == pytest.approx(result.simulation.metrics["net_return"])
    assert np.prod(1 + result.folds.passive_net_return) - 1 == pytest.approx(result.passive.metrics["net_return"])


def test_freeze_round_trip_and_tamper_detection(snapshot):
    result = develop(snapshot, protocol())
    plan = json.loads(json.dumps(result.plan, allow_nan=False))
    h1, _, _ = evaluate_holdout(snapshot, plan)
    h2, _, _ = evaluate_holdout(snapshot, plan)
    pd.testing.assert_frame_equal(h1.curve, h2.curve)
    assert len(h1.curve) == 60
    plan["protocol"]["execution"]["slippage_bps"] = 0
    with pytest.raises(ValueError, match="modified"):
        evaluate_holdout(snapshot, plan)


def test_frozen_data_cannot_be_replaced(snapshot):
    result = develop(snapshot, protocol())
    changed = copy.deepcopy(snapshot)
    changed.prices.iloc[-1, :4] *= 1.1
    with pytest.raises(ValueError, match="Snapshot changed"):
        evaluate_holdout(changed, result.plan)


def test_holdout_never_calls_rank(snapshot, monkeypatch):
    result = develop(snapshot, protocol())
    def prohibited(*args, **kwargs):
        raise AssertionError("Holdout must not select parameters.")
    monkeypatch.setattr("Utils.research_protocol._rank", prohibited)
    evaluate_holdout(snapshot, result.plan)


def test_plan_is_deterministic_and_has_no_holdout_metrics(snapshot):
    first = develop(snapshot, protocol()).plan
    second = develop(snapshot, protocol()).plan
    assert first == second
    assert first["status"] == "FROZEN_NOT_EVALUATED"
    assert "holdout_net_return" not in first
    assert len(first["source"]["data_sha256"]) == 64


@pytest.mark.parametrize("allocation", [0.25, 1.0])
def test_accounting_cash_and_pnl_reconcile(snapshot, allocation):
    prices = validate_daily(snapshot.prices)
    targets = signal(prices.Close, Rule("momentum", 20)).shift(1).fillna(0)
    cfg = Execution(allocation=allocation)
    result = _simulate(prices, targets, cfg, 21, len(prices))
    m = result.metrics
    assert result.curve.Cash.min() >= -1e-6
    assert m["ending_equity"] - cfg.initial_cash == pytest.approx(m["realized_pnl"] + m["open_position_pnl"])
    assert result.orders["Cash flow"].sum() + cfg.initial_cash == pytest.approx(result.curve.Cash.iloc[-1], abs=1e-8)
    assert (result.orders["Signal session"] < result.orders.Session).all()
    assert result.curve["Adjusted units"].ge(0).all()


def test_next_open_does_not_capture_previous_overnight_gap():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"Open": [100., 200., 200.], "Close": [100., 220., 200.],
                           "Volume": 1000}, index=dates)
    targets = pd.Series([0., 1., 0.], index=dates)
    result = _simulate(prices, targets, Execution(commission_bps=0, slippage_bps=0), 1, 3)
    assert result.curve.Equity.iloc[0] == pytest.approx(11000)
    assert result.curve.Equity.iloc[-1] == pytest.approx(10000)
    assert result.orders.iloc[0].Fill == 200


def test_zero_volume_blocks_fills(snapshot):
    prices = validate_daily(snapshot.prices)
    prices.Volume = 0
    result = _simulate(prices, pd.Series(1.0, index=prices.index), Execution(), 1, 360)
    assert result.orders.empty
    assert result.metrics["ending_equity"] == 10000


def test_initial_loss_is_in_drawdown():
    m = _metrics(pd.Series([90., 91.]), 100)
    assert m["maximum_drawdown"] == pytest.approx(-.10)


def test_price_scale_invariance_for_adjusted_units(snapshot):
    prices = validate_daily(snapshot.prices)
    scaled = prices.copy()
    scaled.iloc[:, :4] *= .13
    targets = pd.Series(1.0, index=prices.index)
    a = _simulate(prices, targets, Execution(), 1, 360)
    b = _simulate(scaled, targets, Execution(), 1, 360)
    np.testing.assert_allclose(a.curve.Equity, b.curve.Equity)


def test_stress_fixes_signals_and_increases_cost_drag(snapshot):
    prices = validate_daily(snapshot.prices)
    targets = pd.Series((np.arange(len(prices)) // 10) % 2, index=prices.index)
    result = stress_test(prices, targets, Execution(), 1, 360)
    assert result.net_return.is_monotonic_decreasing
    assert result.slippage_multiplier.tolist() == [1, 2, 4]


def test_cash_wins_when_all_risk_candidates_have_negative_training_score(snapshot):
    prices = snapshot.prices.iloc[:100].copy()
    prices.iloc[:, :4] = 100
    winner, scores = _rank(prices, protocol())
    assert winner.kind == "cash"
    assert scores[0]["selection_score"] == 0


@pytest.mark.parametrize("changes", [dict(train=10), dict(test=0), dict(holdout=0),
                                         dict(gap=0), dict(gap=True), dict(train=100.5),
                                         dict(rules=(Rule("momentum", 20),)),
                                         dict(rules=(Rule("cash"), Rule("cash"))),
                                         dict(rules=(Rule("cash"), Rule("unknown"))),
                                         dict(execution=Execution(slippage_bps=300))])
def test_invalid_protocols_fail_closed(snapshot, changes):
    with pytest.raises(ValueError):
        develop(snapshot, protocol(**changes))


@pytest.mark.parametrize("changes", [dict(initial_cash=0), dict(allocation=1.01),
                                         dict(commission_bps=float("nan")), dict(slippage_bps=-1),
                                         dict(initial_cash=True)])
def test_invalid_execution(changes):
    with pytest.raises(ValueError):
        replace(Execution(), **changes).validate()


@pytest.mark.parametrize("problem", ["duplicate_date", "reversed", "missing", "infinite", "negative",
                                     "bad_ohlc", "negative_volume", "timezone", "intraday", "weekly", "signal"])
def test_bad_data_is_not_silently_repaired(snapshot, problem):
    data = snapshot.prices.copy()
    if problem == "duplicate_date":
        data.index = pd.DatetimeIndex([data.index[0]] * len(data))
    elif problem == "reversed":
        data = data.iloc[::-1]
    elif problem == "missing":
        data.iloc[0, 0] = np.nan
    elif problem == "infinite":
        data.iloc[0, 0] = np.inf
    elif problem == "negative":
        data.iloc[0, 0] = -1
    elif problem == "bad_ohlc":
        data.iloc[0, 1] = 1
    elif problem == "negative_volume":
        data.iloc[0, 4] = -1
    elif problem == "timezone":
        data.index = data.index.tz_localize("UTC")
    elif problem == "intraday":
        data.index = data.index + pd.Timedelta(hours=1)
    elif problem == "weekly":
        data.index = pd.date_range("2010-01-01", periods=len(data), freq="W")
    else:
        data["Signal"] = 0
    with pytest.raises(ValueError):
        validate_daily(data)


def test_adjusted_csv_confirmation_and_round_trip(snapshot):
    payload = snapshot.prices.to_csv().encode()
    with pytest.raises(ValueError, match="Confirm"):
        parse_daily_csv(payload)
    loaded = parse_daily_csv(payload, True)
    pd.testing.assert_frame_equal(loaded.prices, validate_daily(snapshot.prices), check_dtype=False, check_freq=False)
    assert loaded.metadata["provenance_verified"] is False


@pytest.mark.parametrize("payload", [b"", b"Date,Open,Open\n2020-01-01,1,1\n", b"\xff\xfe", b"x\n1\n"])
def test_bad_csv(payload):
    with pytest.raises(ValueError):
        parse_daily_csv(payload, True)


def test_manifest_does_not_claim_complete_or_verified_data(snapshot):
    manifest = snapshot.manifest()
    assert manifest["calendar_completeness"] == "not independently verified"
    assert manifest["provenance_verified"] is False
    assert manifest["source"] == "SYNTHETIC SOFTWARE FIXTURE"


def test_metric_overflow_fails_explicitly():
    with pytest.raises(ValueError, match="unstable"):
        _metrics(pd.Series([1e200, 1e201]), 100.0)
