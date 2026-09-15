"""Seeded market-return simulation. No valuation, provider calls, or trading actions.

Paths are rebased adjusted-return indices, NOT intrinsic values or executable quotes.
The log-return drift is explicit; zero log drift is not zero arithmetic return.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd

ENGINE_VERSION = "market-mc-1.0.0"
TRADING_DAYS = 252
MIN_RETURNS = 126
MODELS = ("stationary_bootstrap", "ewma_bootstrap", "gbm")


@dataclass(frozen=True)
class MarketConfig:
    model: str = "stationary_bootstrap"
    horizon: int = 63
    simulations: int = 5_000
    seed: int = 42
    drift: str = "zero"
    historical_weight: float = 0.25
    annual_log_drift: float = 0.0
    block_length: int = 10
    ewma_decay: float = 0.94


@dataclass(frozen=True)
class MarketResult:
    paths: np.ndarray
    config: MarketConfig
    diagnostics: dict
    warnings: tuple[str, ...]


def validate_config(config: MarketConfig) -> None:
    if config.model not in MODELS:
        raise ValueError("Unknown market simulation model.")
    for label, value, low, high in (
        ("Horizon", config.horizon, 1, 252),
        ("Simulations", config.simulations, 100, 20_000),
        ("Seed", config.seed, 0, 2**32 - 1),
        ("Block length", config.block_length, 1, 63),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
            raise ValueError(f"{label} must be an integer in [{low}, {high}].")
    if config.drift not in {"zero", "shrunk_historical", "manual"}:
        raise ValueError("Unknown drift policy.")
    for label, value, low, high in (
        ("Historical weight", config.historical_weight, 0.0, 1.0),
        ("Annual log drift", config.annual_log_drift, -1.0, 1.0),
        ("EWMA decay", config.ewma_decay, 0.80, 0.995),
    ):
        if isinstance(value, bool) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label} must be finite and in [{low}, {high}].")


def validate_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Supply a Series indexed by dated daily observations.")
    if prices.index.hasnans or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be present, unique, and increasing; no automatic sorting or deduplication.")
    if not prices.index.equals(prices.index.normalize()):
        raise ValueError("Daily session labels, not intraday timestamps, are required.")
    if (prices.index.dayofweek >= 5).any():
        raise ValueError("Only weekday equity/ETF sessions are supported.")
    if len(prices) < MIN_RETURNS + 1:
        raise ValueError(f"At least {MIN_RETURNS + 1} closes ({MIN_RETURNS} returns) are required.")
    if len(prices) > 2_521:
        raise ValueError("At most 2,521 daily closes are supported per fit.")
    try:
        values = prices.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prices must be numeric.") from exc
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive; missing values are not filled or dropped.")
    gaps = prices.index.to_series().diff().dropna()
    if gaps.median() > pd.Timedelta(days=1):
        raise ValueError("History is not a sufficiently regular daily series; weekly observations are not daily returns.")
    if (gaps > pd.Timedelta(days=7)).any():
        raise ValueError("History contains a gap longer than seven calendar days; investigate the data.")
    return pd.Series(values, index=prices.index.copy(), name="Adjusted Close")


def price_digest(prices: pd.Series) -> str:
    """Hash exact observed dates and float64 prices for reproducible provenance."""
    records = [[date.isoformat(), float(value).hex()] for date, value in prices.items()]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _ewma_fit(centered: np.ndarray, decay: float) -> tuple[np.ndarray, float]:
    # Warm-up is entirely within training data. Exclude its first 20 innovations.
    warmup = 20
    variance = float(np.mean(centered[:warmup] ** 2))
    if variance <= 1e-20:
        # A flat warm-up must not erase a later observed jump. Still training-only.
        variance = float(np.mean(centered**2))
    residuals = []
    for value in centered[warmup:]:
        residuals.append(value / np.sqrt(variance) if variance > 1e-20 else 0.0)
        variance = decay * variance + (1 - decay) * value**2
    residuals = np.asarray(residuals)
    residuals -= residuals.mean()
    scale = float(np.sqrt(np.mean(residuals**2)))
    if scale > 1e-12:
        residuals /= scale
    elif np.any(np.abs(centered) > 1e-12):
        raise ValueError("Cannot standardize the EWMA residuals; use another model.")
    return residuals, variance


def simulate_market(prices: pd.Series, config: MarketConfig | None = None,
                    anchor: float | None = None) -> MarketResult:
    config = config or MarketConfig()
    validate_config(config)
    prices = validate_prices(prices)
    spot = float(prices.iloc[-1] if anchor is None else anchor)
    if not np.isfinite(spot) or not 0 < spot < 1e12:
        raise ValueError("The starting reference close must be positive, finite, and below 1e12.")
    returns = np.diff(np.log(prices.to_numpy()))
    historical_mean = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    centered = returns - historical_mean
    mean = {"zero": 0.0,
            "shrunk_historical": historical_mean * config.historical_weight,
            "manual": config.annual_log_drift / TRADING_DAYS}[config.drift]
    rng = np.random.Generator(np.random.PCG64(config.seed))
    paths = np.empty((config.simulations, config.horizon + 1), dtype=np.float64)
    paths[:, 0] = spot
    log_level = np.full(config.simulations, np.log(spot))
    indices = rng.integers(0, len(centered), config.simulations)
    residuals, variance = _ewma_fit(centered, config.ewma_decay)
    conditional_variance = np.full(config.simulations, variance)
    for day in range(1, config.horizon + 1):
        if config.model == "gbm":
            shock = sigma * rng.standard_normal(config.simulations)
        elif config.model == "stationary_bootstrap":
            if day > 1:
                restart = rng.random(config.simulations) < 1 / config.block_length
                indices = np.where(restart, rng.integers(0, len(centered), config.simulations),
                                   (indices + 1) % len(centered))
            shock = centered[indices]
        else:
            innovation = residuals[rng.integers(0, len(residuals), config.simulations)]
            shock = np.sqrt(conditional_variance) * innovation
            conditional_variance = (config.ewma_decay * conditional_variance
                                    + (1 - config.ewma_decay) * shock**2)
        log_level += mean + shock
        # Do not clip returns, truncate tails, discard paths, or conceal numerical failure.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            paths[:, day] = np.exp(log_level)
        if not np.isfinite(paths[:, day]).all() or (paths[:, day] <= 0).any():
            raise ValueError("Simulation overflow/underflow; no paths were clipped. Review data and assumptions.")
    warnings = [
        "Model-conditional scenarios, not calibrated real-world probabilities or intrinsic values.",
        "Adjusted-return paths rebased at the reference close include historical dividend effects; "
        "they are not forecasts of unadjusted exchange quotes.",
        "No future jumps, liquidity costs, taxes, delisting mechanism, or changing fundamentals are modeled.",
        "Fitted volatility and the chosen drift are treated as parameters, not estimated with full parameter uncertainty.",
    ]
    if sigma < 1e-8:
        warnings.append("Almost no observed price variation; this is not evidence of zero future risk.")
    if len(returns) < 504:
        warnings.append("Fewer than two trading years of returns; regime and tail estimates are fragile.")
    if config.horizon > len(returns) / 2:
        warnings.append("The forecast horizon is long relative to the observed history.")
    if np.max(np.abs(returns)) > np.log(1.5):
        warnings.append("A historical move exceeds 50% in multiplicative magnitude; inspect corporate actions.")
    if config.model == "gbm":
        warnings.append("GBM assumes independent Gaussian log returns and constant volatility.")
    if config.model == "stationary_bootstrap":
        warnings.append("Block resampling cannot generate unseen one-day shocks and assumes a stationary history.")
    if config.model == "ewma_bootstrap":
        warnings.append("EWMA decay is an explicit prior, not a fitted GARCH or regime-switching model.")
    diagnostics = {
        "engine_version": ENGINE_VERSION, "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "rng": "PCG64", "history_sha256": price_digest(prices),
        "history_start": prices.index[0].isoformat(), "as_of": prices.index[-1].isoformat(),
        "return_observations": len(returns), "reference_close": spot,
        "annualized_historical_log_mean": historical_mean * TRADING_DAYS,
        "annualized_assumed_log_mean": mean * TRADING_DAYS,
        "annualized_historical_volatility": sigma * np.sqrt(TRADING_DAYS),
        "annualized_ewma_volatility": np.sqrt(variance * TRADING_DAYS),
        "config": asdict(config),
    }
    paths.flags.writeable = False
    return MarketResult(paths, config, diagnostics, tuple(warnings))


def tail_loss(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Loss VaR and ES; signed, so a negative value denotes a modeled gain.

ES averages exactly the worst (1-confidence) probability mass, including the
fractional boundary observation. This remains defined for tied outcomes.
"""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Returns must be a nonempty finite one-dimensional array.")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one.")
    losses = np.sort(-values)[::-1]
    mass = (1 - confidence) * len(losses)
    whole = int(np.floor(mass))
    fraction = mass - whole
    es = (float(losses[:whole].sum()) + fraction * losses[min(whole, len(losses) - 1)]) / mass
    return float(np.quantile(losses, confidence)), float(es)


def summarize_market(result: MarketResult, upper: float | None = None,
                     lower: float | None = None) -> dict:
    paths = result.paths
    spot = float(paths[0, 0])
    for label, barrier in (("Upper", upper), ("Lower", lower)):
        if barrier is not None and (not np.isfinite(barrier) or barrier <= 0):
            raise ValueError(f"{label} level must be positive and finite.")
    if upper is not None and lower is not None and upper <= lower:
        raise ValueError("Upper level must exceed lower level.")
    terminal = paths[:, -1]
    returns = terminal / spot - 1
    quantiles = np.quantile(terminal, [0.05, 0.25, 0.50, 0.75, 0.95])
    drawdowns = np.max(1 - paths / np.maximum.accumulate(paths, axis=1), axis=1)
    var95, es95 = tail_loss(returns)
    return {
        "p05": float(quantiles[0]), "p25": float(quantiles[1]), "median": float(quantiles[2]),
        "p75": float(quantiles[3]), "p95": float(quantiles[4]), "mean": float(terminal.mean()),
        "probability_gain": float(np.mean(returns > 0)),
        "probability_loss": float(np.mean(returns < 0)),
        "mean_return": float(returns.mean()), "loss_var95": var95, "loss_es95": es95,
        "median_max_drawdown": float(np.median(drawdowns)),
        "p95_max_drawdown": float(np.quantile(drawdowns, 0.95)),
        "probability_touch_upper": None if upper is None else float(np.mean(np.max(paths, axis=1) >= upper)),
        "probability_touch_lower": None if lower is None else float(np.mean(np.min(paths, axis=1) <= lower)),
        "probability_finish_above_upper": None if upper is None else float(np.mean(terminal >= upper)),
        "probability_finish_below_lower": None if lower is None else float(np.mean(terminal <= lower)),
    }


def path_bands(result: MarketResult) -> pd.DataFrame:
    bands = np.quantile(result.paths, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0).T
    return pd.DataFrame(bands, columns=["P05", "P25", "Median", "P75", "P95"],
                        index=pd.Index(range(result.config.horizon + 1), name="Trading step"))


def terminal_samples(result: MarketResult) -> pd.DataFrame:
    paths = result.paths
    return pd.DataFrame({
        "Simulation": np.arange(1, len(paths) + 1), "Terminal price-equivalent": paths[:, -1],
        "Horizon return": paths[:, -1] / paths[:, 0] - 1,
        "Maximum drawdown": np.max(1 - paths / np.maximum.accumulate(paths, axis=1), axis=1),
    })


def walk_forward(prices: pd.Series, config: MarketConfig, train_window: int = 504,
                 max_folds: int = 12) -> pd.DataFrame:
    """Non-overlapping forward windows. Every fit sees only its past training slice.

Observed adjusted returns must match simulated adjusted returns. This is a
coverage diagnostic, not a trading backtest or an automatic model selector.
"""
    prices = validate_prices(prices)
    validate_config(config)
    if not isinstance(train_window, Integral) or not MIN_RETURNS <= train_window <= 1260:
        raise ValueError("Training window must contain 126 to 1,260 returns.")
    if not isinstance(max_folds, Integral) or not 1 <= max_folds <= 24:
        raise ValueError("Use 1 to 24 folds.")
    origins = list(range(train_window, len(prices) - config.horizon, config.horizon))[-max_folds:]
    if len(origins) < 3:
        raise ValueError("At least three complete, non-overlapping holdout windows are required.")
    rows = []
    for fold, origin in enumerate(origins):
        training = prices.iloc[origin - train_window:origin + 1]
        fold_config = replace(config, simulations=min(config.simulations, 2_000),
                              seed=(config.seed + origin) % (2**32))
        simulation = simulate_market(training, fold_config)
        values = simulation.paths[:, -1] / simulation.paths[:, 0]
        low, high = np.quantile(values, [0.05, 0.95])
        realized = float(prices.iloc[origin + config.horizon] / prices.iloc[origin])
        # Central 90% interval score (lower is better, same horizon and return basis).
        score = high - low + 20 * max(low - realized, 0) + 20 * max(realized - high, 0)
        rows.append({
            "Origin": prices.index[origin].isoformat(),
            "Target": prices.index[origin + config.horizon].isoformat(),
            "Train start": training.index[0].isoformat(), "Train end": training.index[-1].isoformat(),
            "P05 return": float(low - 1), "P95 return": float(high - 1),
            "Realized return": realized - 1, "Covered": bool(low <= realized <= high),
            "Interval score": float(score), "Seed": fold_config.seed,
            "Simulations": fold_config.simulations, "Training SHA256": price_digest(training),
        })
    return pd.DataFrame(rows)
