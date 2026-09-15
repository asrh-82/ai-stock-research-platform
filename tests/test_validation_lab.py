"""Deterministic accounting and execution tests; not tests of trading profitability."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from Utils.validation_lab import (
    ReplayConfig, arithmetic_break_even, bootstrap_returns, expanding_splits,
    replay, synthetic_fixture, validate_bars,
)


def bars(signals=(1, 1, 0, 0, 0)):
    n = len(signals)
    return pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100.,
                         "Volume": 10000., "Signal": signals},
                        index=pd.date_range("2020-01-01", periods=n, tz="UTC", name="Timestamp"))


def config(**kwargs):
    return replace(ReplayConfig(initial_cash=1000., entry_fraction=1., commission_bps=0.,
                                slippage_bps=0., prior_volume_fraction=1., halt_drawdown=None),
                   **kwargs)


def set_prices(data, i, opening, high=None, low=None, close=None):
    data.loc[data.index[i], ["Open", "High", "Low", "Close"]] = (
        opening, opening if high is None else high, opening if low is None else low,
        opening if close is None else close)


def test_signal_executes_at_next_open_not_same_close():
    data = bars((1, 0, 0, 0))
    set_prices(data, 1, 120)
    result = replay(data, config())
    trade = result.trades.iloc[0]
    assert result.equity.Shares.iloc[0] == 0
    assert trade["Signal Time"] == data.index[0]
    assert trade["Entry Time"] == data.index[1]
    assert trade["Entry Price"] == 120
    assert trade["Exit Time"] == data.index[2]
    assert trade["Shares"] == 8


def test_explicit_execution_delay():
    data = bars((1, 0, 0, 0, 0))
    result = replay(data, config(execution_delay=2))
    assert result.trades.iloc[0]["Entry Time"] == data.index[2]
    assert result.trades.iloc[0]["Exit Time"] == data.index[3]


def test_gap_stop_fills_from_open_not_ideal_stop_price():
    data = bars((1, 1, 0, 0))
    set_prices(data, 2, 70, 75, 68, 72)
    result = replay(data, config(stop_loss=.10))
    assert result.trades.iloc[0]["Exit Price"] == 70
    assert result.trades.iloc[0]["Reason"] == "stop gap"
    assert result.metrics["ending_equity"] == 700


def test_dual_touch_uses_stop_first():
    data = bars((1, 0, 0, 0))
    set_prices(data, 1, 100, 120, 80, 100)
    result = replay(data, config(stop_loss=.10, profit_target=.05))
    assert result.metrics["ambiguous_bars"] == 1
    assert result.trades.iloc[0]["Exit Price"] == 90
    assert result.trades.iloc[0]["Reason"] == "stop-first ambiguous"


def test_known_target_open_gap_precedes_unknown_intrabar_stop():
    data = bars((1, 1, 0, 0))
    set_prices(data, 2, 120, 125, 50, 100)
    result = replay(data, config(stop_loss=.1, profit_target=.05))
    assert result.trades.iloc[0]["Exit Price"] == 120
    assert result.trades.iloc[0]["Reason"] == "target gap"


def test_round_trip_accounting_includes_both_fees_and_slippage():
    result = replay(bars((1, 0, 0)), config(commission_bps=100, slippage_bps=100))
    quantity = 9
    expected = 1000 - quantity * 101 * 1.01 + quantity * 99 * .99
    assert result.metrics["ending_equity"] == pytest.approx(expected)
    assert result.metrics["fees_paid"] == pytest.approx(18)
    assert result.metrics["modeled_slippage"] == pytest.approx(18)
    assert result.trades.iloc[0]["Net PnL"] == pytest.approx(expected - 1000)
    assert result.trades.iloc[0]["Fees"] == pytest.approx(18)


def test_initial_fee_counts_in_drawdown():
    result = replay(bars((1, 1, 1)), config(commission_bps=100))
    assert result.metrics["maximum_drawdown"] == pytest.approx(-.009)


def test_entry_capacity_uses_previous_bar_volume():
    data = bars((1, 0, 0))
    data.iloc[0, data.columns.get_loc("Volume")] = 100
    data.iloc[1, data.columns.get_loc("Volume")] = 1_000_000
    result = replay(data, config(prior_volume_fraction=.01))
    assert result.trades.iloc[0]["Shares"] == 1


def test_unaffordable_or_zero_capacity_entry_is_rejected():
    result = replay(bars((1, 0, 0)), config(initial_cash=50))
    assert result.metrics["rejected_entries"] == 1
    assert result.metrics["open_shares"] == 0


def test_zero_volume_blocks_entry():
    data = bars((1, 0, 0))
    data.iloc[1, data.columns.get_loc("Volume")] = 0
    assert replay(data, config()).trades.empty


def test_zero_volume_blocks_exit_until_next_tradable_bar():
    data = bars((1, 0, 0, 0))
    data.iloc[2, data.columns.get_loc("Volume")] = 0
    result = replay(data, config())
    assert result.equity.Shares.iloc[2] == 10
    assert result.trades.iloc[0]["Exit Time"] == data.index[3]


def test_drawdown_halt_queues_next_open_not_current_close():
    data = bars((1, 1, 1, 1, 1))
    set_prices(data, 1, 100, 100, 70, 80)
    set_prices(data, 2, 60)
    result = replay(data, config(halt_drawdown=.1))
    assert result.equity.Halted.iloc[1]
    assert result.trades.iloc[0]["Exit Time"] == data.index[2]
    assert result.trades.iloc[0]["Exit Price"] == 60
    assert result.metrics["ending_equity"] == 600
    assert result.metrics["completed_trades"] == 1
    assert result.metrics["open_shares"] == 0


def test_no_signal_means_no_trades_and_no_fabricated_win_rate():
    result = replay(bars((0, 0, 0)), config())
    assert result.metrics["ending_equity"] == 1000
    assert result.metrics["win_rate"] is None
    assert result.metrics["profit_factor"] is None


def test_open_position_is_marked_not_recorded_as_closed():
    data = bars((1, 1, 1))
    result = replay(data, config(commission_bps=10, slippage_bps=10))
    assert result.trades.empty
    assert result.metrics["open_shares"] == 9
    assert result.metrics["estimated_liquidation_equity"] < result.metrics["ending_equity"]
    assert result.metrics["open_position_pnl"] == pytest.approx(result.metrics["ending_equity"] - 1000)


def test_reproducibility_manifest_is_json_safe_and_data_sensitive():
    data = bars()
    a, b = replay(data, config()), replay(data, config())
    assert a.manifest == b.manifest
    json.dumps({"manifest": a.manifest, "metrics": a.metrics}, allow_nan=False)
    assert a.manifest["status"] == "NOT_VALIDATED"
    data.iloc[-1, data.columns.get_loc("Signal")] = 1
    assert replay(data, config()).manifest["data_sha256"] != a.manifest["data_sha256"]


def test_future_data_cannot_change_prefix_execution():
    data = synthetic_fixture(80)
    altered = data.copy()
    altered.loc[altered.index[40]:, ["Open", "High", "Low", "Close"]] *= 7
    altered.loc[altered.index[40]:, "Signal"] = 1
    pd.testing.assert_frame_equal(replay(data).equity.iloc[:40], replay(altered).equity.iloc[:40])
    pd.testing.assert_frame_equal(replay(data.iloc[:40]).equity, replay(data).equity.iloc[:40])


@pytest.mark.parametrize("seed", range(10))
def test_reconciles_pnl_and_never_borrows(seed):
    cfg = config(entry_fraction=.7, commission_bps=7, slippage_bps=11,
                 stop_loss=.08, profit_target=.12)
    result = replay(synthetic_fixture(120, seed), cfg)
    assert result.equity.Cash.min() >= -1e-7
    assert (result.equity.Shares >= 0).all()
    pnl = result.trades["Net PnL"].sum() + result.metrics["open_position_pnl"]
    assert pnl == pytest.approx(result.metrics["ending_equity"] - cfg.initial_cash)


@pytest.mark.parametrize("field,value", [
    ("initial_cash", 0), ("initial_cash", float("nan")), ("initial_cash", "bad"),
    ("entry_fraction", 1.1), ("commission_bps", -1), ("slippage_bps", float("inf")),
    ("execution_delay", 0), ("execution_delay", True), ("execution_delay", 1.5),
    ("stop_loss", 0), ("profit_target", 1), ("halt_drawdown", float("nan")),
    ("prior_volume_fraction", 0),
])
def test_invalid_config_fails_closed(field, value):
    with pytest.raises(ValueError):
        replay(bars(), config(**{field: value}))


@pytest.mark.parametrize("mutation", ["missing", "nan", "infinity", "negative_price", "bounds",
                                      "negative_volume", "bad_signal", "duplicate_time", "unsorted",
                                      "naive_time", "duplicate_columns", "multiple_assets"])
def test_invalid_data_fails_closed(mutation):
    data = bars()
    if mutation == "missing":
        data = data.drop(columns="High")
    elif mutation in ("nan", "infinity", "negative_price"):
        data.iloc[0, 0] = {"nan": np.nan, "infinity": np.inf, "negative_price": -1}[mutation]
    elif mutation == "bounds":
        data.iloc[0, 1] = 90
    elif mutation == "negative_volume":
        data.iloc[0, 4] = -1
    elif mutation == "bad_signal":
        data.iloc[0, 5] = 2
    elif mutation == "duplicate_time":
        data.index = pd.DatetimeIndex([data.index[0]] * len(data))
    elif mutation == "unsorted":
        data = data.iloc[::-1]
    elif mutation == "naive_time":
        data.index = data.index.tz_localize(None)
    elif mutation == "duplicate_columns":
        data = pd.concat([data, data[["High"]]], axis=1)
    elif mutation == "multiple_assets":
        data["Symbol"] = ["A", "B", "A", "B", "A"]
    with pytest.raises(ValueError):
        validate_bars(data)


def test_break_even_includes_cost_and_does_not_cap_impossible_values():
    assert arithmetic_break_even(.05, .10) == pytest.approx(2 / 3)
    assert arithmetic_break_even(.05, .10, .005) == pytest.approx(.7)
    assert arithmetic_break_even(.05, .10, .06) > 1


def test_expanding_splits_are_chronological_and_nonoverlapping():
    folds = expanding_splits(100, 40, 10, 5)
    assert len(folds) == 5
    assert folds[0] == {"train_start": 0, "train_stop": 40, "test_start": 45, "test_stop": 55}
    assert all(f["train_stop"] + 5 == f["test_start"] for f in folds)
    assert all(a["test_stop"] == b["test_start"] for a, b in zip(folds, folds[1:]))


@pytest.mark.parametrize("args", [(10, 8, 3, 1), (100, 20, 10, 0), (True, 1, 1, 1)])
def test_invalid_splits_rejected(args):
    with pytest.raises(ValueError):
        expanding_splits(*args)


def test_bootstrap_is_deterministic_and_preserves_constant_returns():
    values = np.full(30, .001)
    a = bootstrap_returns(values, paths=100)
    assert a == bootstrap_returns(values, paths=100)
    assert a["return_p05"] == pytest.approx(1.001 ** 30 - 1)
    assert a["return_p95"] == pytest.approx(a["return_p05"])
    assert a["drawdown_p05"] == pytest.approx(0)


def test_bootstrap_counts_loss_from_initial_capital():
    result = bootstrap_returns(np.full(20, -.01), paths=100)
    assert result["drawdown_p05"] == pytest.approx(.99 ** 20 - 1)
    assert result["loss_fraction_in_resamples"] == 1


@pytest.mark.parametrize("values", [[.01] * 19, [np.nan] * 20, [-1] * 20, [[.01] * 20]])
def test_invalid_bootstrap_data_rejected(values):
    with pytest.raises(ValueError):
        bootstrap_returns(values, paths=100)


def test_bootstrap_resource_limit():
    with pytest.raises(ValueError, match="memory limit"):
        bootstrap_returns(np.zeros(1000), paths=20_000)


def test_synthetic_fixture_reproducible_and_valid():
    pd.testing.assert_frame_equal(synthetic_fixture(), synthetic_fixture())
    assert len(validate_bars(synthetic_fixture())) == 120


def test_csv_round_trip_preserves_ohlcv_and_timezone():
    from Utils.validation_lab import parse_replay_csv
    data = synthetic_fixture()
    restored = parse_replay_csv(data.to_csv().encode())
    pd.testing.assert_frame_equal(validate_bars(restored), validate_bars(data), check_freq=False)


@pytest.mark.parametrize("payload", [
    b"", b"\xff", b"Timestamp,Open,Open\n2020-01-01Z,1,1\n", b"Open,High\n1,2\n",
])
def test_invalid_csv_rejected(payload):
    from Utils.validation_lab import parse_replay_csv
    with pytest.raises(ValueError):
        parse_replay_csv(payload)


def test_csv_requires_explicit_timezone():
    from Utils.validation_lab import parse_replay_csv
    payload = synthetic_fixture().to_csv().replace("+00:00", "").encode()
    with pytest.raises(ValueError, match="UTC offset"):
        parse_replay_csv(payload)


def test_csv_rejects_oversized_payload():
    from Utils.validation_lab import parse_replay_csv
    with pytest.raises(ValueError, match="10 MiB"):
        parse_replay_csv(b"x" * (10 * 1024 * 1024 + 1))
