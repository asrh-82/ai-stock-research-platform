from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class DCFInputs:
    base_revenue: float
    revenue_growth: tuple[float, ...]
    ebit_margin: tuple[float, ...]
    tax_rate: float
    depreciation_pct_revenue: float
    capex_pct_revenue: float
    nwc_pct_incremental_revenue: float
    wacc: float
    terminal_growth: float
    cash: float
    debt: float
    diluted_shares: float


@dataclass(frozen=True)
class DCFYear:
    year: int
    revenue_growth: float
    revenue: float
    ebit_margin: float
    ebit: float
    nopat: float
    depreciation: float
    capex: float
    change_in_nwc: float
    unlevered_fcf: float
    discount_factor: float
    present_value_fcf: float


@dataclass(frozen=True)
class DCFResult:
    forecast: tuple[DCFYear, ...]
    terminal_fcf: float
    terminal_value: float
    present_value_terminal: float
    enterprise_value: float
    equity_value: float
    implied_value_per_share: float


def _validate_finite(name: str, values: Iterable[float]) -> None:
    if not all(isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain only finite numbers.")


def validate_dcf_inputs(inputs: DCFInputs) -> None:
    if inputs.base_revenue <= 0:
        raise ValueError("Base revenue must be greater than zero.")
    if not inputs.revenue_growth:
        raise ValueError("At least one forecast year is required.")
    if len(inputs.revenue_growth) != len(inputs.ebit_margin):
        raise ValueError("Revenue growth and EBIT margin must have the same number of years.")

    _validate_finite("Revenue growth", inputs.revenue_growth)
    _validate_finite("EBIT margin", inputs.ebit_margin)
    _validate_finite(
        "DCF inputs",
        (
            inputs.base_revenue,
            inputs.tax_rate,
            inputs.depreciation_pct_revenue,
            inputs.capex_pct_revenue,
            inputs.nwc_pct_incremental_revenue,
            inputs.wacc,
            inputs.terminal_growth,
            inputs.cash,
            inputs.debt,
            inputs.diluted_shares,
        ),
    )

    if any(growth <= -1 for growth in inputs.revenue_growth):
        raise ValueError("Revenue growth must be greater than -100%.")
    if not 0 <= inputs.tax_rate <= 0.60:
        raise ValueError("Tax rate must be between 0% and 60%.")
    if not 0 <= inputs.depreciation_pct_revenue <= 1:
        raise ValueError("D&A as a percentage of revenue must be between 0% and 100%.")
    if not 0 <= inputs.capex_pct_revenue <= 1:
        raise ValueError("Capital expenditures as a percentage of revenue must be between 0% and 100%.")
    if not -1 <= inputs.nwc_pct_incremental_revenue <= 1:
        raise ValueError("Working-capital investment must be between -100% and 100% of incremental revenue.")
    if not 0 < inputs.wacc < 1:
        raise ValueError("WACC must be between 0% and 100%.")
    if not -0.10 < inputs.terminal_growth < 0.10:
        raise ValueError("Terminal growth must be between -10% and 10%.")
    if inputs.wacc <= inputs.terminal_growth:
        raise ValueError("WACC must be greater than terminal growth.")
    if inputs.cash < 0 or inputs.debt < 0:
        raise ValueError("Cash and debt cannot be negative.")
    if inputs.diluted_shares <= 0:
        raise ValueError("Diluted shares must be greater than zero.")


def calculate_dcf(inputs: DCFInputs) -> DCFResult:
    validate_dcf_inputs(inputs)

    forecast = []
    prior_revenue = float(inputs.base_revenue)

    for year, (growth, margin) in enumerate(
        zip(inputs.revenue_growth, inputs.ebit_margin),
        start=1,
    ):
        revenue = prior_revenue * (1 + growth)
        ebit = revenue * margin
        nopat = ebit * (1 - inputs.tax_rate)
        depreciation = revenue * inputs.depreciation_pct_revenue
        capex = revenue * inputs.capex_pct_revenue
        change_in_nwc = (revenue - prior_revenue) * inputs.nwc_pct_incremental_revenue
        unlevered_fcf = nopat + depreciation - capex - change_in_nwc
        discount_factor = (1 + inputs.wacc) ** year
        present_value_fcf = unlevered_fcf / discount_factor

        forecast.append(
            DCFYear(
                year=year,
                revenue_growth=growth,
                revenue=revenue,
                ebit_margin=margin,
                ebit=ebit,
                nopat=nopat,
                depreciation=depreciation,
                capex=capex,
                change_in_nwc=change_in_nwc,
                unlevered_fcf=unlevered_fcf,
                discount_factor=discount_factor,
                present_value_fcf=present_value_fcf,
            )
        )
        prior_revenue = revenue

    terminal_fcf = forecast[-1].unlevered_fcf * (1 + inputs.terminal_growth)
    terminal_value = terminal_fcf / (inputs.wacc - inputs.terminal_growth)
    present_value_terminal = terminal_value / ((1 + inputs.wacc) ** len(forecast))
    enterprise_value = sum(year.present_value_fcf for year in forecast) + present_value_terminal
    equity_value = enterprise_value + inputs.cash - inputs.debt
    implied_value_per_share = equity_value / inputs.diluted_shares

    return DCFResult(
        forecast=tuple(forecast),
        terminal_fcf=terminal_fcf,
        terminal_value=terminal_value,
        present_value_terminal=present_value_terminal,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        implied_value_per_share=implied_value_per_share,
    )


def apply_scenario_adjustments(
    inputs: DCFInputs,
    growth_delta: float = 0,
    margin_delta: float = 0,
    wacc_delta: float = 0,
) -> DCFInputs:
    return replace(
        inputs,
        revenue_growth=tuple(value + growth_delta for value in inputs.revenue_growth),
        ebit_margin=tuple(value + margin_delta for value in inputs.ebit_margin),
        wacc=inputs.wacc + wacc_delta,
    )


def calculate_sensitivity(
    inputs: DCFInputs,
    wacc_values: Iterable[float],
    terminal_growth_values: Iterable[float],
) -> dict[float, dict[float, float | None]]:
    table = {}

    for wacc in wacc_values:
        row = {}
        for terminal_growth in terminal_growth_values:
            if wacc <= terminal_growth:
                row[terminal_growth] = None
                continue

            scenario = replace(
                inputs,
                wacc=float(wacc),
                terminal_growth=float(terminal_growth),
            )
            row[terminal_growth] = calculate_dcf(scenario).implied_value_per_share
        table[wacc] = row

    return table
