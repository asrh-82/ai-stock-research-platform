from dataclasses import dataclass
from math import isfinite

import pandas as pd


@dataclass(frozen=True)
class DCFDefaults:
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
    sources: dict[str, str]
    warnings: tuple[str, ...]


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _line_item_series(statement: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    if statement is None or statement.empty:
        return pd.Series(dtype=float)

    for name in names:
        if name not in statement.index:
            continue

        row = statement.loc[name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        values = pd.to_numeric(row, errors="coerce").dropna()
        if not values.empty:
            return values

    return pd.Series(dtype=float)


def _latest(statement: pd.DataFrame, names: tuple[str, ...]):
    values = _line_item_series(statement, names)
    return _number(values.iloc[0]) if not values.empty else None


def _info_value(info: dict, names: tuple[str, ...]):
    for name in names:
        value = _number(info.get(name))
        if value is not None:
            return value
    return None


def _bounded_default(value, low, high, fallback, label, warnings):
    if value is None:
        warnings.append(f"{label} was unavailable; using a manual fallback of {fallback:.1%}.")
        return fallback
    if value < low or value > high:
        bounded = min(max(value, low), high)
        warnings.append(
            f"{label} ({value:.1%}) was outside the default range and was capped at {bounded:.1%}; review it manually."
        )
        return bounded
    return value


def build_dcf_defaults(
    info: dict,
    income_statement: pd.DataFrame,
    cashflow_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    forecast_years: int = 5,
) -> DCFDefaults:
    warnings = []
    sources = {}

    revenue_series = _line_item_series(
        income_statement,
        ("Total Revenue", "Operating Revenue"),
    )
    statement_revenue = _number(revenue_series.iloc[0]) if not revenue_series.empty else None
    info_revenue = _info_value(info, ("totalRevenue",))
    base_revenue = statement_revenue or info_revenue or 0.0
    sources["base_revenue"] = (
        "Latest annual income statement"
        if statement_revenue is not None
        else "Company summary data"
        if info_revenue is not None
        else "Missing — enter manually"
    )
    if base_revenue <= 0:
        warnings.append("Revenue was unavailable; enter the latest annual revenue manually.")

    latest_growth = None
    if len(revenue_series) >= 2:
        latest_revenue = _number(revenue_series.iloc[0])
        prior_revenue = _number(revenue_series.iloc[1])
        if latest_revenue is not None and prior_revenue not in {None, 0}:
            latest_growth = (latest_revenue / prior_revenue) - 1
            sources["revenue_growth"] = "Latest annual income-statement growth"
    if latest_growth is None:
        latest_growth = _info_value(info, ("revenueGrowth",))
        sources["revenue_growth"] = (
            "Company summary data"
            if latest_growth is not None
            else "Manual fallback"
        )

    latest_growth = _bounded_default(
        latest_growth,
        -0.20,
        0.40,
        0.05,
        "Revenue growth",
        warnings,
    )
    year_five_growth = 0.04
    if forecast_years == 1:
        revenue_growth = (latest_growth,)
    else:
        revenue_growth = tuple(
            latest_growth + (year_five_growth - latest_growth) * year / (forecast_years - 1)
            for year in range(forecast_years)
        )

    latest_ebit = _latest(income_statement, ("EBIT", "Operating Income"))
    statement_margin = latest_ebit / base_revenue if latest_ebit is not None and base_revenue else None
    info_margin = _info_value(info, ("operatingMargins",))
    latest_margin = statement_margin if statement_margin is not None else info_margin
    sources["ebit_margin"] = (
        "Latest annual EBIT / revenue"
        if statement_margin is not None
        else "Company summary operating margin"
        if info_margin is not None
        else "Manual fallback"
    )
    latest_margin = _bounded_default(
        latest_margin,
        -0.30,
        0.60,
        0.15,
        "EBIT margin",
        warnings,
    )
    ebit_margin = tuple(latest_margin for _ in range(forecast_years))

    pretax_income = _latest(income_statement, ("Pretax Income", "Income Before Tax"))
    tax_provision = _latest(income_statement, ("Tax Provision", "Income Tax Expense"))
    observed_tax_rate = None
    if pretax_income is not None and pretax_income > 0 and tax_provision is not None and tax_provision >= 0:
        observed_tax_rate = tax_provision / pretax_income
    tax_rate = _bounded_default(
        observed_tax_rate,
        0,
        0.50,
        0.21,
        "Effective tax rate",
        warnings,
    )
    sources["tax_rate"] = "Latest annual tax provision / pretax income" if observed_tax_rate is not None else "Manual fallback"

    depreciation = _latest(
        cashflow_statement,
        (
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
            "Depreciation",
        ),
    )
    depreciation_ratio = abs(depreciation) / base_revenue if depreciation is not None and base_revenue else None
    depreciation_ratio = _bounded_default(
        depreciation_ratio,
        0,
        0.25,
        0.03,
        "D&A as a percentage of revenue",
        warnings,
    )
    sources["depreciation"] = "Latest annual cash-flow statement" if depreciation is not None else "Manual fallback"

    capex = _latest(
        cashflow_statement,
        ("Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"),
    )
    capex_ratio = abs(capex) / base_revenue if capex is not None and base_revenue else None
    capex_ratio = _bounded_default(
        capex_ratio,
        0,
        0.30,
        0.04,
        "Capital expenditures as a percentage of revenue",
        warnings,
    )
    sources["capex"] = "Latest annual cash-flow statement" if capex is not None else "Manual fallback"

    cash_effect_of_nwc = _latest(
        cashflow_statement,
        ("Change In Working Capital", "Change In Other Working Capital"),
    )
    nwc_ratio = None
    if cash_effect_of_nwc is not None and len(revenue_series) >= 2:
        current_revenue = _number(revenue_series.iloc[0])
        prior_revenue = _number(revenue_series.iloc[1])
        revenue_change = current_revenue - prior_revenue if current_revenue is not None and prior_revenue is not None else None
        if revenue_change is not None and abs(revenue_change) > max(abs(current_revenue or 0) * 0.01, 1):
            nwc_ratio = -cash_effect_of_nwc / revenue_change
    observed_nwc_ratio = nwc_ratio
    nwc_ratio = _bounded_default(
        nwc_ratio,
        -0.50,
        0.50,
        0.02,
        "Working-capital investment",
        warnings,
    )
    sources["working_capital"] = (
        "Latest annual cash-flow statement and revenue change"
        if observed_nwc_ratio is not None
        else "Manual fallback"
    )

    statement_cash = _latest(
        balance_sheet,
        (
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash",
        ),
    )
    cash = _info_value(info, ("totalCash",))
    if cash is None:
        cash = statement_cash or 0.0
    sources["cash"] = "Company summary data" if _info_value(info, ("totalCash",)) is not None else "Latest annual balance sheet" if statement_cash is not None else "Missing — defaults to zero"

    statement_debt = _latest(balance_sheet, ("Total Debt",))
    debt = _info_value(info, ("totalDebt",))
    if debt is None:
        debt = statement_debt or 0.0
    sources["debt"] = "Company summary data" if _info_value(info, ("totalDebt",)) is not None else "Latest annual balance sheet" if statement_debt is not None else "Missing — defaults to zero"

    statement_shares = _latest(
        income_statement,
        ("Diluted Average Shares", "Basic Average Shares"),
    )
    summary_shares = _info_value(info, ("sharesOutstanding", "impliedSharesOutstanding"))
    diluted_shares = statement_shares or summary_shares or 0.0
    sources["diluted_shares"] = (
        "Latest annual diluted average shares"
        if statement_shares is not None
        else "Company summary shares outstanding"
        if summary_shares is not None
        else "Missing — enter manually"
    )
    if statement_shares is None and summary_shares is not None:
        warnings.append(
            "Fully diluted shares were unavailable; the default uses shares outstanding and should be reviewed manually."
        )
    if diluted_shares <= 0:
        warnings.append("Diluted shares were unavailable; enter a valid share count manually.")

    sources["wacc"] = "Manual default of 9.0%"
    sources["terminal_growth"] = "Manual default of 2.5%"

    return DCFDefaults(
        base_revenue=base_revenue,
        revenue_growth=revenue_growth,
        ebit_margin=ebit_margin,
        tax_rate=tax_rate,
        depreciation_pct_revenue=depreciation_ratio,
        capex_pct_revenue=capex_ratio,
        nwc_pct_incremental_revenue=nwc_ratio,
        wacc=0.09,
        terminal_growth=0.025,
        cash=max(cash, 0.0),
        debt=max(debt, 0.0),
        diluted_shares=diluted_shares,
        sources=sources,
        warnings=tuple(warnings),
    )
