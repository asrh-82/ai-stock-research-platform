from dataclasses import replace
from io import BytesIO
import json
from statistics import NormalDist
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from Utils.market_data import demo_snapshot
from Utils.market_export import evidence_bundle
from Utils.market_monte_carlo import (
    MODELS, MarketConfig, MarketResult, path_bands, price_digest, simulate_market,
    summarize_market, tail_loss, validate_config, validate_prices, walk_forward,
)


def history(n=756, seed=7):
    rng = np.random.Generator(np.random.PCG64(seed))
    return pd.Series(100 * np.exp(np.r_[0, np.cumsum(rng.normal(0.0004, 0.015, n))]),
                     index=pd.bdate_range("2022-01-03", periods=n + 1))


@pytest.mark.parametrize("changes", [
    {"simulations": 0}, {"simulations": 20_001}, {"simulations": True}, {"simulations": 100.0},
    {"horizon": 0}, {"horizon": 253}, {"seed": -1}, {"seed": 2**32},
    {"block_length": 0}, {"block_length": 64}, {"model": "dcf"}, {"drift": "unknown"},
    {"annual_log_drift": np.nan}, {"annual_log_drift": 2}, {"historical_weight": -0.1},
    {"ewma_decay": 1}, {"ewma_decay": np.inf},
])
def test_invalid_configs(changes):
    with pytest.raises(ValueError):
        validate_config(replace(MarketConfig(), **changes))


@pytest.mark.parametrize("model", MODELS)
def test_seeded_positive_and_readonly(model):
    config = MarketConfig(model=model, simulations=300, horizon=21)
    first = simulate_market(history(), config, anchor=123)
    second = simulate_market(history(), config, anchor=123)
    np.testing.assert_array_equal(first.paths, second.paths)
    assert first.paths.shape == (300, 22)
    assert np.isfinite(first.paths).all() and (first.paths > 0).all()
    np.testing.assert_array_equal(first.paths[:, 0], 123)
    assert not first.paths.flags.writeable
    other = simulate_market(history(), replace(config, seed=43), anchor=123)
    assert not np.array_equal(first.paths, other.paths)


@pytest.mark.parametrize("model", MODELS)
def test_constant_history_zero_log_drift(model):
    prices = history()
    prices[:] = 100
    result = simulate_market(prices, MarketConfig(model=model, horizon=63, simulations=100))
    np.testing.assert_allclose(result.paths, 100, atol=1e-10)


def test_manual_log_drift_has_exact_deterministic_compounding():
    prices = history()
    prices[:] = 100
    config = MarketConfig(model="gbm", drift="manual", annual_log_drift=0.10, horizon=252, simulations=100)
    result = simulate_market(prices, config)
    np.testing.assert_allclose(result.paths[:, -1], 100 * np.exp(0.10), rtol=1e-12)


def test_zero_policy_does_not_extrapolate_past_price_growth():
    prices = pd.Series(100 * np.exp(np.arange(200) * 0.01), index=pd.bdate_range("2024-01-02", periods=200))
    config = MarketConfig(model="gbm", horizon=21, simulations=100)
    result = simulate_market(prices, config)
    np.testing.assert_allclose(result.paths[:, -1], prices.iloc[-1], rtol=1e-12)
    shrunk = simulate_market(prices, replace(config, drift="shrunk_historical", historical_weight=0.25))
    np.testing.assert_allclose(shrunk.paths[:, -1], prices.iloc[-1] * np.exp(0.25 * 0.01 * 21), rtol=1e-12)


def test_gbm_matches_analytical_log_distribution():
    prices = history(1000)
    config = MarketConfig(model="gbm", horizon=63, simulations=20_000, drift="manual", annual_log_drift=0.06)
    result = simulate_market(prices, config, anchor=100)
    sigma = np.diff(np.log(prices)).std(ddof=1)
    expected_mu, expected_sigma = 0.06 * 63 / 252, sigma * np.sqrt(63)
    log_returns = np.log(result.paths[:, -1] / 100)
    assert abs(log_returns.mean() - expected_mu) < 4 * expected_sigma / np.sqrt(20_000)
    assert abs(log_returns.std() / expected_sigma - 1) < 0.025
    for q in [0.05, 0.50, 0.95]:
        target = 100 * np.exp(expected_mu + expected_sigma * NormalDist().inv_cdf(q))
        assert np.quantile(result.paths[:, -1], q) == pytest.approx(target, rel=0.015)
    assert result.paths[:, -1].mean() == pytest.approx(100 * np.exp(expected_mu + expected_sigma**2 / 2), rel=0.01)


def test_stationary_bootstrap_only_resamples_observed_centered_returns():
    prices = history()
    result = simulate_market(prices, MarketConfig(simulations=100, horizon=21, block_length=1))
    observed = np.diff(np.log(prices))
    observed -= observed.mean()
    simulated = np.diff(np.log(result.paths), axis=1).ravel()
    assert np.max(np.min(abs(simulated[:, None] - observed[None, :]), axis=1)) < 1e-12


def test_blocks_preserve_volatility_clustering_more_than_iid_resampling():
    rng = np.random.default_rng(9)
    vol = np.repeat([0.005, 0.05] * 5, 60)
    prices = pd.Series(100 * np.exp(np.r_[0, np.cumsum(vol * rng.standard_normal(len(vol)))]),
                       index=pd.bdate_range("2020-01-02", periods=len(vol) + 1))
    correlations = []
    for block in [1, 20]:
        result = simulate_market(prices, MarketConfig(simulations=500, horizon=126, block_length=block))
        absolute = abs(np.diff(np.log(result.paths), axis=1))
        correlations.append(np.corrcoef(absolute[:, :-1].ravel(), absolute[:, 1:].ravel())[0, 1])
    assert correlations[1] > correlations[0] + 0.15


def test_ewma_changes_volatility_with_recent_regime():
    prices = history()
    log_returns = np.diff(np.log(prices))
    higher = log_returns.copy()
    higher[-30:] *= 5
    high_prices = pd.Series(100 * np.exp(np.r_[0, np.cumsum(higher)]), index=prices.index)
    config = MarketConfig(model="ewma_bootstrap", horizon=21, simulations=5_000)
    low, high = simulate_market(prices, config), simulate_market(high_prices, config)
    assert high.diagnostics["annualized_ewma_volatility"] > 3 * low.diagnostics["annualized_ewma_volatility"]
    expected = high.diagnostics["annualized_ewma_volatility"] / np.sqrt(252)
    assert np.std(np.log(high.paths[:, 1] / high.paths[:, 0])) == pytest.approx(expected, rel=0.06)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_price_values(bad):
    prices = history()
    prices.iloc[50] = bad
    with pytest.raises(ValueError):
        validate_prices(prices)


def test_invalid_dates_and_lengths():
    prices = history()
    invalid = [prices.iloc[:126], prices.iloc[::-1], pd.Series(prices.values),
               pd.concat([prices.iloc[:50], prices.iloc[49:]]),
               prices.set_axis(prices.index + pd.Timedelta(hours=1))]
    for sample in invalid:
        with pytest.raises(ValueError):
            validate_prices(sample)
    with pytest.raises(ValueError, match="gap"):
        validate_prices(prices.drop(prices.index[100:110]))


@pytest.mark.parametrize("anchor", [0, -1, np.nan, np.inf, 1e12])
def test_invalid_reference_closes(anchor):
    with pytest.raises(ValueError):
        simulate_market(history(), anchor=anchor)


def test_extreme_tail_is_not_silently_clipped():
    prices = pd.Series(np.resize([1e-300, 1e300], 200), index=pd.bdate_range("2024-01-02", periods=200))
    with pytest.raises(ValueError, match="overflow/underflow"):
        simulate_market(prices, MarketConfig(simulations=100, horizon=21), anchor=100)


def test_summary_drawdowns_and_level_touches_include_time_zero():
    paths = np.array([[100, 200, 100], [100, 50, 80], [100, 100, 120], [100, 100, 100]], dtype=float)
    result = MarketResult(paths, MarketConfig(), {}, ())
    summary = summarize_market(result, upper=150, lower=60)
    assert summary["median_max_drawdown"] == pytest.approx(0.25)
    assert summary["p95_max_drawdown"] == pytest.approx(0.5)
    assert summary["probability_touch_upper"] == 0.25
    assert summary["probability_touch_lower"] == 0.25
    assert summary["probability_finish_above_upper"] == 0
    assert summary["probability_gain"] == 0.25
    assert summary["probability_loss"] == 0.25
    assert summarize_market(result, upper=90)["probability_touch_upper"] == 1
    with pytest.raises(ValueError):
        summarize_market(result, upper=80, lower=90)


def test_signed_var_and_fractional_mass_expected_shortfall():
    var, es = tail_loss(np.array([-0.5, -0.3, 0.1]), confidence=0.5)
    assert var == pytest.approx(0.3)
    assert es == pytest.approx((0.5 + 0.5 * 0.3) / 1.5)
    assert tail_loss(np.full(100, 0.1)) == pytest.approx((-0.1, -0.1))
    assert tail_loss(np.full(100, -0.2)) == pytest.approx((0.2, 0.2))


def test_pointwise_bands_are_ordered_and_start_at_anchor():
    result = simulate_market(history(), MarketConfig(simulations=100, horizon=21), anchor=80)
    bands = path_bands(result)
    assert bands.shape == (22, 5)
    assert (np.diff(bands.values, axis=1) >= 0).all()
    np.testing.assert_array_equal(bands.iloc[0].values, 80)


def test_walk_forward_only_uses_training_history_and_nonoverlapping_targets():
    prices = demo_snapshot().prices
    config = MarketConfig(model="gbm", horizon=63, simulations=100)
    first = walk_forward(prices, config)
    changed = prices.copy()
    changed.iloc[-1] *= 1.2
    second = walk_forward(changed, config)
    pd.testing.assert_series_equal(first["Training SHA256"], second["Training SHA256"])
    pd.testing.assert_series_equal(first["P05 return"], second["P05 return"])
    assert first.iloc[-1]["Realized return"] != second.iloc[-1]["Realized return"]
    assert (first["Train end"] == first.Origin).all()
    assert (first.Target > first.Origin).all()
    assert all(first.Target.iloc[i] <= first.Origin.iloc[i + 1] for i in range(len(first) - 1))


def test_walk_forward_small_sample_is_rejected():
    with pytest.raises(ValueError, match="three"):
        walk_forward(history(600), MarketConfig(horizon=63, simulations=100))


def test_evidence_roundtrip_and_mismatch_rejection():
    snapshot = demo_snapshot()
    result = simulate_market(snapshot.prices, MarketConfig(simulations=100, horizon=21), anchor=snapshot.reference_close)
    bundle = evidence_bundle(snapshot, result)
    with ZipFile(BytesIO(bundle)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        recovered = pd.read_csv(archive.open("history.csv"), index_col="Date", parse_dates=True,
                                float_precision="round_trip")["Adjusted Close"]
        assert price_digest(recovered) == manifest["model"]["history_sha256"]
        again = simulate_market(recovered, MarketConfig(**manifest["model"]["config"]), anchor=snapshot.reference_close)
        np.testing.assert_array_equal(result.paths, again.paths)
        assert len(pd.read_csv(archive.open("terminal_outcomes.csv"))) == 100
    with pytest.raises(ValueError, match="reference"):
        evidence_bundle(replace(snapshot, reference_close=1.0), result)


def test_engine_has_no_dcf_or_network_imports():
    import ast
    from pathlib import Path
    tree = ast.parse(Path("Utils/market_monte_carlo.py").read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert all(not (name or "").startswith(("Utils.dcf", "Utils.monte_carlo", "yfinance")) for name in imports)


def test_weekly_data_is_not_treated_as_daily():
    prices = pd.Series(np.linspace(100, 120, 150), index=pd.date_range("2022-01-07", periods=150, freq="W-FRI"))
    with pytest.raises(ValueError, match="daily"):
        validate_prices(prices)


def test_calibration_path_count_is_bounded():
    frame = walk_forward(demo_snapshot().prices, MarketConfig(model="gbm", simulations=5000, horizon=252))
    assert frame.Simulations.eq(2000).all()
    assert len(frame) == 3


def test_evidence_rejects_different_history_even_when_anchor_is_same():
    snapshot = demo_snapshot()
    result = simulate_market(snapshot.prices, MarketConfig(simulations=100, horizon=21), anchor=snapshot.reference_close)
    altered = snapshot.prices.copy()
    altered.iloc[50] *= 1.01
    with pytest.raises(ValueError, match="snapshot"):
        evidence_bundle(replace(snapshot, prices=altered), result)
