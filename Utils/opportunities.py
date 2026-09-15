"""Price-based opportunity research with purged, chronological forecast evaluation.

No valuation claims, provider calls, order placement, or fabricated directional drift.
Returns are adjusted total returns. Excess is asset minus benchmark, not beta alpha.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
from numbers import Integral
import platform
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

VERSION = "opportunity-1.0.0"
FEATURES = ("relative_21", "relative_63", "relative_126_skip21", "reversal_5",
            "relative_trend63", "volatility21", "volatility_change",
            "benchmark_momentum63", "relative_drawdown126")
TARGETS = ("total_return", "excess_return")


@dataclass(frozen=True)
class OpportunityConfig:
    horizon: int = 21
    min_train: int = 504
    max_train: int = 1008
    ridge_penalty: float = 0.1  # objective: mean squared error + penalty * ||coef||^2
    min_oos: int = 24
    min_signals: int = 12
    round_trip_bps: float = 20.0
    hurdle_bps: float = 100.0
    simulations: int = 5000
    seed: int = 42
    confidence: float = 0.95
    diagnostic_block: int = 3


@dataclass(frozen=True)
class OpportunityResult:
    summary: dict
    oos: pd.DataFrame
    contributions: pd.DataFrame
    scenarios: pd.DataFrame
    manifest: dict
    history: pd.DataFrame


def validate_config(c: OpportunityConfig) -> None:
    if c.horizon not in (21, 63):
        raise ValueError("Choose a 21- or 63-session forecast horizon.")
    for label, value, lo, hi in (
        ("horizon", c.horizon, 21, 63), ("min_train", c.min_train, 252, 1008),
        ("max_train", c.max_train, 504, 2000), ("min_oos", c.min_oos, 24, 60),
        ("min_signals", c.min_signals, 12, 60), ("simulations", c.simulations, 2000, 20000),
        ("seed", c.seed, 0, 2**32-1), ("diagnostic_block", c.diagnostic_block, 1, 12),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or not lo <= value <= hi:
            raise ValueError(f"{label} must be an integer in [{lo}, {hi}].")
    if c.max_train < c.min_train:
        raise ValueError("max_train must be at least min_train.")
    for label, value, lo, hi in (
        ("ridge_penalty", c.ridge_penalty, 0.01, 10.),
        ("round_trip_bps", c.round_trip_bps, 0., 1000.),
        ("hurdle_bps", c.hurdle_bps, 0., 1000.), ("confidence", c.confidence, .90, .99),
    ):
        if isinstance(value, bool) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{label} must be finite and in [{lo}, {hi}].")


def validate_history(history: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame) or list(history.columns) != ["asset", "benchmark"]:
        raise ValueError("Supply aligned asset and benchmark columns in that order.")
    idx = history.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None or idx.hasnans
            or idx.has_duplicates or not idx.is_monotonic_increasing
            or not idx.equals(idx.normalize()) or (idx.dayofweek >= 5).any()):
        raise ValueError("Dates must be unique increasing timezone-naive weekday session labels.")
    if not 127 <= len(history) <= 2521:
        raise ValueError("Require 127 to 2,521 aligned daily closes; longer training is needed to forecast.")
    gaps = idx.to_series().diff().dropna()
    if gaps.median() > pd.Timedelta(days=1) or (gaps > pd.Timedelta(days=7)).any():
        raise ValueError("History is irregular or contains a gap longer than seven calendar days.")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("All closes must be finite and positive. No filling, dropping, or repair.")
    return history.astype(float).copy()


def history_digest(history: pd.DataFrame) -> str:
    records = [[d.isoformat(), *(float(v).hex() for v in row)]
               for d, row in zip(history.index, history.to_numpy())]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def make_dataset(history: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """At row t, features use closes through t; target enters t+1 and exits t+1+h."""
    history = validate_history(history)
    if isinstance(horizon, bool) or horizon not in (21, 63):
        raise ValueError("Unsupported horizon.")
    a, b = history.asset, history.benchmark
    ar, br = np.log(a).diff(), np.log(b).diff()
    a21, b21 = a / a.shift(21)-1, b / b.shift(21)-1
    a63, b63 = a / a.shift(63)-1, b / b.shift(63)-1
    s21, s126 = ar.rolling(21).std(), ar.rolling(126).std()
    # Epsilon defines the feature for flat histories; it never changes source prices.
    x = pd.DataFrame({
        "relative_21": a21-b21,
        "relative_63": a63-b63,
        "relative_126_skip21": a.shift(21)/a.shift(126)-b.shift(21)/b.shift(126),
        "reversal_5": -(a/a.shift(5)-b/b.shift(5)),
        "relative_trend63": a/a.rolling(63).mean()-b/b.rolling(63).mean(),
        "volatility21": s21 * np.sqrt(252),
        "volatility_change": np.log((s21+1e-12)/(s126+1e-12)),
        "benchmark_momentum63": b63,
        "relative_drawdown126": a/a.rolling(126).max()-b/b.rolling(126).max(),
    }, index=history.index)
    total = a.shift(-(horizon+1))/a.shift(-1)-1
    bench = b.shift(-(horizon+1))/b.shift(-1)-1
    y = pd.DataFrame({"total_return": total, "excess_return": total-bench}, index=history.index)
    return x.loc[:, FEATURES], y


def training_indices(x: pd.DataFrame, y: pd.DataFrame, origin: int, c: OpportunityConfig) -> np.ndarray:
    idx = np.arange(len(x))
    finite = np.isfinite(x.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy()).all(axis=1)
    # Strictly before the origin. Purges labels that extend into/through prediction time.
    allowed = idx[finite & (idx+c.horizon+1 < origin)]
    return allowed[-c.max_train:]


def fit_at(x: pd.DataFrame, y: pd.DataFrame, origin: int, c: OpportunityConfig) -> dict:
    chosen = training_indices(x, y, origin, c)
    if len(chosen) < c.min_train:
        raise ValueError(f"Need {c.min_train} fully matured training labels; only {len(chosen)} available.")
    train = x.iloc[chosen].to_numpy()
    target = y.iloc[chosen].to_numpy()
    center, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (train-center)/scale
    intercept = target.mean(axis=0)
    coefficients = np.linalg.solve(z.T@z/len(z)+c.ridge_penalty*np.eye(len(FEATURES)),
                                   z.T@(target-intercept)/len(z))
    latest = (x.iloc[origin].to_numpy()-center)/scale
    prediction = intercept + latest@coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite model forecast; no substituted or clipped output.")
    return {"prediction": prediction, "baseline": intercept, "coefficients": coefficients,
            "center": center, "scale": scale, "z": latest, "indices": chosen}


def walk_predictions(history: pd.DataFrame, c: OpportunityConfig) -> tuple[pd.DataFrame, dict]:
    validate_config(c)
    x, y = make_dataset(history, c.horizon)
    valid = np.flatnonzero(np.isfinite(x.to_numpy()).all(axis=1))
    if not len(valid):
        raise ValueError("No complete past-only feature vectors.")
    latest = fit_at(x, y, len(history)-1, c)
    # Fixed first origin, no choice based on performance. Step ensures disjoint target windows.
    first = int(valid[0])+c.min_train+c.horizon+1
    origins = list(range(first, len(history)-c.horizon-1, c.horizon+1))[-60:]
    rows = []
    cost, hurdle = c.round_trip_bps/10000, c.hurdle_bps/10000
    for origin in origins:
        fitted = fit_at(x, y, origin, c)
        p, actual, baseline = fitted["prediction"], y.iloc[origin].to_numpy(), fitted["baseline"]
        selected = bool(p[0] > cost and p[1] > cost+hurdle)
        train_idx = fitted["indices"]
        rows.append({
            "origin": history.index[origin], "entry": history.index[origin+1],
            "exit": history.index[origin+c.horizon+1],
            "train_start": history.index[train_idx[0]],
            "train_last_feature": history.index[train_idx[-1]],
            "train_last_label_end": history.index[train_idx[-1]+c.horizon+1],
            "train_count": len(train_idx), "forecast_total": float(p[0]),
            "forecast_excess": float(p[1]), "realized_total": float(actual[0]),
            "realized_excess": float(actual[1]), "baseline_total": float(baseline[0]),
            "baseline_excess": float(baseline[1]), "selected": selected,
            "net_total_if_selected": float(actual[0]-cost) if selected else 0.,
            "net_excess_if_selected": float(actual[1]-cost) if selected else 0.,
        })
    return pd.DataFrame(rows), {"x": x, "y": y, "latest": latest}


def _bootstrap_indices(n: int, c: OpportunityConfig) -> np.ndarray:
    """Moving-block resampling of chronological OOS folds, not daily training labels."""
    rng = np.random.Generator(np.random.PCG64(c.seed))
    length = min(c.diagnostic_block, n)
    blocks = int(np.ceil(n/length))
    starts = rng.integers(0, n, size=(c.simulations, blocks))
    return ((starts[..., None]+np.arange(length)) % n).reshape(c.simulations, -1)[:, :n]


def _skill(actual: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> float | None:
    denom = float(np.mean((actual-baseline)**2))
    return 1-float(np.mean((actual-pred)**2))/denom if denom > 1e-14 else None


def evaluate_evidence(oos: pd.DataFrame, c: OpportunityConfig, family_size: int) -> dict:
    """Exploratory gates, not certification. No tuning against these evaluation folds."""
    validate_config(c)
    if isinstance(family_size, bool) or not isinstance(family_size, Integral) or not 1 <= family_size <= 12:
        raise ValueError("Prespecify a scan family of 1 to 12 requested assets.")
    n = len(oos)
    reasons = []
    result = {"oos_folds": n, "family_size": family_size, "evidence_pass": False}
    if n < c.min_oos:
        reasons.append(f"Only {n} non-overlapping target windows; require at least {c.min_oos}.")
    if n == 0:
        return {**result, "reasons": reasons or ["No OOS predictions."]}
    actual, pred = oos.realized_excess.to_numpy(), oos.forecast_excess.to_numpy()
    baseline = oos.baseline_excess.to_numpy()
    errors = actual-pred
    indices = _bootstrap_indices(n, c)
    # Three statistical gates per requested asset. Does not correct prior scans/configuration search.
    tail = (1-c.confidence)/(3*family_size)
    imp_zero = actual**2-errors**2
    imp_mean = (actual-baseline)**2-errors**2
    zero_lower = float(np.quantile(imp_zero[indices].mean(axis=1), tail))
    mean_lower = float(np.quantile(imp_mean[indices].mean(axis=1), tail))
    selected = oos.selected.to_numpy(dtype=bool)
    payoff = oos.net_excess_if_selected.to_numpy()
    counts = selected[indices].sum(axis=1)
    # A resample with no selections contributes -infinity, not a silently dropped favorable subset.
    conditional = np.divide(payoff[indices].sum(axis=1), counts,
                            out=np.full(c.simulations, -np.inf), where=counts > 0)
    payoff_lower = float(np.quantile(conditional, tail, method="inverted_cdf"))
    nsignals = int(selected.sum())
    recent = slice(n//2, n)
    recent_skill = _skill(actual[recent], pred[recent], np.zeros_like(actual[recent]))
    result.update({
        "skill_vs_zero": _skill(actual, pred, np.zeros_like(actual)),
        "skill_vs_historical_mean": _skill(actual, pred, baseline),
        "excess_rmse": float(np.sqrt(np.mean(errors**2))),
        "direction_hit_rate": float(np.mean(np.sign(actual)==np.sign(pred))),
        "zero_mse_improvement_lower": zero_lower, "mean_mse_improvement_lower": mean_lower,
        "selected_count": nsignals,
        "selected_net_excess_mean": float(payoff[selected].mean()) if nsignals else None,
        "selected_net_excess_lower": payoff_lower if np.isfinite(payoff_lower) else None,
        "recent_skill_vs_zero": recent_skill, "one_sided_tail": tail,
    })
    if zero_lower <= 0:
        reasons.append("No positive lower bound on forecasting improvement versus zero excess return.")
    if mean_lower <= 0:
        reasons.append("No positive lower bound on improvement versus rolling historical-mean excess return.")
    if nsignals < c.min_signals:
        reasons.append(f"Only {nsignals} prior rule-selected observations; require {c.min_signals}.")
    if not np.isfinite(payoff_lower) or payoff_lower <= 0:
        reasons.append("Prior rule-selected net outperformance lacks a positive lower diagnostic bound.")
    if recent_skill is None or recent_skill <= 0:
        reasons.append("Recent-half forecast error does not improve on zero excess return.")
    return {**result, "evidence_pass": not reasons, "reasons": reasons}


def forecast_opportunity(history: pd.DataFrame, symbol: str, benchmark: str = "SPY",
                         config: OpportunityConfig | None = None, family_size: int = 1,
                         provenance: dict | None = None) -> OpportunityResult:
    c = config or OpportunityConfig()
    history = validate_history(history)
    if symbol == benchmark:
        raise ValueError("Asset and benchmark must be different securities.")
    oos, work = walk_predictions(history, c)
    ev = evaluate_evidence(oos, c, family_size)
    fitted = work["latest"]
    prediction = fitted["prediction"]
    cost, hurdle = c.round_trip_bps/10000, c.hurdle_bps/10000
    reasons = list(ev["reasons"])
    if float(np.max(np.abs(fitted["z"]))) > 6:
        reasons.append("Current feature vector is more than six training standard deviations from its center.")
    if prediction[0] <= -1 or prediction[0]-prediction[1] <= -1:
        reasons.append("Linear forecast implies a physically invalid asset or benchmark return.")
    if prediction[1] <= cost+hurdle:
        reasons.append("Forecast excess return does not exceed costs plus the prespecified research hurdle.")
    if prediction[0] <= cost:
        reasons.append("Forecast total return does not exceed assumed round-trip costs.")
    scenarios = pd.DataFrame(columns=TARGETS)
    scenario_summary = {}
    if len(oos) >= c.min_oos:
        residuals = oos[["realized_total", "realized_excess"]].to_numpy()-oos[["forecast_total", "forecast_excess"]].to_numpy()
        rng = np.random.Generator(np.random.PCG64(c.seed))
        # Preserve paired total/excess errors and their empirical bias. No in-sample fit residuals.
        draws = prediction+residuals[rng.integers(0, len(residuals), c.simulations)]
        if not np.isfinite(draws).all() or (draws[:, 0] <= -1).any() or (draws[:, 0]-draws[:, 1] <= -1).any():
            reasons.append("OOS-error scenarios imply invalid returns; distribution withheld rather than clipped.")
        else:
            scenarios = pd.DataFrame(draws, columns=TARGETS)
            net = draws[:, 1]-cost
            scenario_summary = {
                "net_excess_p05": float(np.quantile(net, .05)),
                "net_excess_median": float(np.median(net)), "net_excess_p95": float(np.quantile(net, .95)),
                "outperform_after_cost_frequency": float(np.mean(net > 0)),
                "total_loss_frequency": float(np.mean(draws[:, 0]-cost < 0)),
                "residual_observations": len(residuals),
            }
    synthetic = bool((provenance or {}).get("synthetic", False))
    if synthetic:
        reasons.append("Synthetic demonstration cannot qualify as market evidence.")
    status = ("Research candidate" if not reasons else
              "Unvalidated estimate" if not ev["evidence_pass"] else "No qualifying gap")
    rmse = ev.get("excess_rmse", 0.)
    summary = {"symbol": symbol, "benchmark": benchmark, "as_of": str(history.index[-1].date()),
               "status": status, "horizon": c.horizon,
               "forecast_total": float(prediction[0]), "forecast_excess": float(prediction[1]),
               "net_forecast_excess": float(prediction[1]-cost),
               "gap_to_hurdle": float(prediction[1]-cost-hurdle),
               "gap_per_oos_error": float((prediction[1]-cost)/rmse) if rmse > 1e-12 else None,
               "cost_bps": c.round_trip_bps, "hurdle_bps": c.hurdle_bps,
               **scenario_summary, "evidence": ev, "reasons": reasons}
    contributions = pd.DataFrame({"feature": ["Intercept", *FEATURES],
                                  "total_contribution": [fitted["baseline"][0], *(fitted["z"]*fitted["coefficients"][:, 0])],
                                  "excess_contribution": [fitted["baseline"][1], *(fitted["z"]*fitted["coefficients"][:, 1])]})
    manifest = {"version": VERSION, "config": asdict(c), "family_size": family_size,
                "history_sha256": history_digest(history), "summary": summary,
                "dependencies": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "provenance": provenance or {}, "feature_names": FEATURES,
                "fitted_model": {k: fitted[k].tolist() for k in ("center", "scale", "coefficients", "baseline")},
                "interpretation": "Conditional price-based return forecast, not intrinsic value or proven mispricing. "
                  "Hypothetical entry next session close, exit horizon sessions later. Adjusted total-return basis. "
                  "Costs are assumptions, not measured fills; excess is not beta-adjusted alpha. "
                  "OOS block-bootstrap gates are exploratory and do not correct prior scans or survivor selection. "
                  "Residual scenarios are terminal outcomes, not market paths or calibrated probabilities. "
                  "No orders, live execution, position-sizing advice, or profitability guarantee."}
    return OpportunityResult(summary, oos, contributions, scenarios, manifest, history)


def export_opportunity(result: OpportunityResult) -> bytes:
    if history_digest(result.history) != result.manifest["history_sha256"]:
        raise ValueError("History changed after forecasting; evidence export rejected.")
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(result.manifest, indent=2, allow_nan=False))
        z.writestr("history.csv", result.history.to_csv(index_label="Date", float_format="%.17g"))
        z.writestr("oos_predictions.csv", result.oos.to_csv(index=False))
        z.writestr("feature_contributions.csv", result.contributions.to_csv(index=False))
        z.writestr("conditional_terminal_scenarios.csv", result.scenarios.to_csv(index=False))
        z.writestr("REPRODUCE.txt", "Read history.csv with parse_dates=['Date'], index_col='Date', float_precision='round_trip'.\n"
                    "Use recorded engine/dependency versions, config, symbols, family_size and provenance with forecast_opportunity.\n"
                    "Forecast target: next-session-close entry, exit h sessions later. Not a spot-price target.\n")
    return buffer.getvalue()
