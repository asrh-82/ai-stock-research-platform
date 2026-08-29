from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
import pandas as pd

from Utils.dcf import DCFInputs, calculate_dcf


@dataclass(frozen=True)
class MonteCarloConfig:
    simulation_count: int = 5_000
    seed: int = 42
    growth_std: float = 0.02
    margin_std: float = 0.015
    wacc_std: float = 0.0075
    terminal_growth_std: float = 0.005
    minimum_wacc_spread: float = 0.0025
    max_attempt_multiplier: int = 20


@dataclass(frozen=True)
class MonteCarloResult:
    samples: pd.DataFrame
    rejected_draws: int
    config: MonteCarloConfig


def validate_monte_carlo_config(config: MonteCarloConfig) -> None:
    if not 100 <= config.simulation_count <= 50_000:
        raise ValueError("Simulation count must be between 100 and 50,000.")
    if config.seed < 0:
        raise ValueError("Random seed cannot be negative.")

    standard_deviations = (
        config.growth_std,
        config.margin_std,
        config.wacc_std,
        config.terminal_growth_std,
    )
    if any(not isfinite(value) or value < 0 for value in standard_deviations):
        raise ValueError("Simulation standard deviations must be finite and non-negative.")
    if not 0 <= config.minimum_wacc_spread < 1:
        raise ValueError("Minimum WACC spread must be between 0% and 100%.")
    if config.max_attempt_multiplier < 1:
        raise ValueError("Maximum attempt multiplier must be at least one.")


def _economically_valid(inputs: DCFInputs, minimum_wacc_spread: float) -> bool:
    return (
        all(growth > -0.95 for growth in inputs.revenue_growth)
        and all(-0.50 <= margin <= 0.80 for margin in inputs.ebit_margin)
        and 0.01 <= inputs.wacc <= 0.35
        and -0.03 <= inputs.terminal_growth <= 0.08
        and inputs.wacc > inputs.terminal_growth + minimum_wacc_spread
    )


def run_monte_carlo(
    base_inputs: DCFInputs,
    config: MonteCarloConfig | None = None,
) -> MonteCarloResult:
    config = config or MonteCarloConfig()
    validate_monte_carlo_config(config)

    rng = np.random.default_rng(config.seed)
    rows = []
    attempts = 0
    maximum_attempts = config.simulation_count * config.max_attempt_multiplier

    while len(rows) < config.simulation_count and attempts < maximum_attempts:
        remaining = config.simulation_count - len(rows)
        batch_size = min(max(remaining * 2, 512), maximum_attempts - attempts)

        growth_draws = rng.normal(0, config.growth_std, batch_size)
        margin_draws = rng.normal(0, config.margin_std, batch_size)
        wacc_draws = rng.normal(base_inputs.wacc, config.wacc_std, batch_size)
        terminal_growth_draws = rng.normal(
            base_inputs.terminal_growth,
            config.terminal_growth_std,
            batch_size,
        )

        for growth_delta, margin_delta, wacc, terminal_growth in zip(
            growth_draws,
            margin_draws,
            wacc_draws,
            terminal_growth_draws,
        ):
            if len(rows) >= config.simulation_count:
                break

            attempts += 1
            simulated_inputs = replace(
                base_inputs,
                revenue_growth=tuple(
                    growth + float(growth_delta)
                    for growth in base_inputs.revenue_growth
                ),
                ebit_margin=tuple(
                    margin + float(margin_delta)
                    for margin in base_inputs.ebit_margin
                ),
                wacc=float(wacc),
                terminal_growth=float(terminal_growth),
            )

            if not _economically_valid(
                simulated_inputs,
                config.minimum_wacc_spread,
            ):
                continue

            try:
                valuation = calculate_dcf(simulated_inputs)
            except ValueError:
                continue

            if not isfinite(valuation.implied_value_per_share):
                continue

            rows.append(
                {
                    "Simulation": len(rows) + 1,
                    "Growth Adjustment": float(growth_delta),
                    "Margin Adjustment": float(margin_delta),
                    "WACC": float(wacc),
                    "Terminal Growth": float(terminal_growth),
                    "Enterprise Value": valuation.enterprise_value,
                    "Equity Value": valuation.equity_value,
                    "Implied Value Per Share": valuation.implied_value_per_share,
                }
            )

    if len(rows) < config.simulation_count:
        raise ValueError(
            "Too many simulation draws were economically invalid. "
            "Reduce the assumption ranges and try again."
        )

    return MonteCarloResult(
        samples=pd.DataFrame(rows),
        rejected_draws=attempts - len(rows),
        config=config,
    )


def summarize_monte_carlo(
    result: MonteCarloResult,
    current_price: float | None = None,
) -> dict[str, float | int | None]:
    values = result.samples["Implied Value Per Share"].astype(float)
    current_price = float(current_price) if current_price else None

    return {
        "simulations": len(values),
        "rejected_draws": result.rejected_draws,
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "probability_above_market": (
            float((values > current_price).mean())
            if current_price is not None and current_price > 0
            else None
        ),
    }
