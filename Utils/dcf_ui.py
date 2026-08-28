import pandas as pd
import streamlit as st

from Utils.data_utils import get_balance_sheet_cached, get_cashflow_cached
from Utils.dcf import (
    DCFInputs,
    apply_scenario_adjustments,
    calculate_dcf,
    calculate_sensitivity,
)
from Utils.dcf_data import build_dcf_defaults


def _is_financial_company(info: dict) -> bool:
    sector = str(info.get("sector", "")).lower()
    industry = str(info.get("industry", "")).lower()
    financial_terms = ("bank", "insurance", "financial services", "capital markets")
    return sector == "financial services" or any(term in industry for term in financial_terms)


def _amount_input(label, value, key, help_text=None):
    return st.number_input(
        label,
        min_value=0.0,
        value=max(float(value), 0.0),
        step=max(abs(float(value)) * 0.01, 1.0),
        format="%.2f",
        key=key,
        help=help_text,
    )


def _forecast_dataframe(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Year": year.year,
                "Revenue Growth %": year.revenue_growth * 100,
                "Revenue ($M)": year.revenue,
                "EBIT Margin %": year.ebit_margin * 100,
                "EBIT ($M)": year.ebit,
                "NOPAT ($M)": year.nopat,
                "D&A ($M)": year.depreciation,
                "CapEx ($M)": year.capex,
                "Change in NWC ($M)": year.change_in_nwc,
                "UFCF ($M)": year.unlevered_fcf,
                "PV of UFCF ($M)": year.present_value_fcf,
            }
            for year in result.forecast
        ]
    )


def _format_sensitivity_value(value):
    if value is None or pd.isna(value):
        return "—"
    return f"${value:,.2f}"


def render_dcf(context):
    ticker = context["ticker"]
    info = context["info"]
    current_price = context["current_price"]

    st.subheader("Discounted Cash Flow Valuation")
    st.caption(
        "Editable five-year unlevered free-cash-flow model. All figures are annual and in USD millions unless stated otherwise."
    )

    if _is_financial_company(info):
        st.warning(
            "This UFCF model is not designed for banks, insurers, or other financial companies. "
            "Use a dividend-discount, residual-income, or sector-specific model instead."
        )
        return

    cashflow_statement = get_cashflow_cached(ticker)
    balance_sheet = get_balance_sheet_cached(ticker)
    defaults = build_dcf_defaults(
        info=info,
        income_statement=context["financials"],
        cashflow_statement=cashflow_statement,
        balance_sheet=balance_sheet,
    )

    if defaults.warnings:
        with st.expander("Input warnings — review before using the valuation", expanded=True):
            for warning in defaults.warnings:
                st.warning(warning)

    st.markdown("#### Forecast assumptions")
    forecast_editor = pd.DataFrame(
        {
            "Year": [1, 2, 3, 4, 5],
            "Revenue Growth %": [value * 100 for value in defaults.revenue_growth],
            "EBIT Margin %": [value * 100 for value in defaults.ebit_margin],
        }
    )
    edited_forecast = st.data_editor(
        forecast_editor,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=["Year"],
        key=f"dcf_forecast_assumptions_{ticker}",
        column_config={
            "Year": st.column_config.NumberColumn(format="Year %d"),
            "Revenue Growth %": st.column_config.NumberColumn(format="%.2f%%"),
            "EBIT Margin %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    operating_col, reinvestment_col, capital_col = st.columns(3)
    with operating_col:
        st.markdown("##### Operating")
        base_revenue = _amount_input(
            "Base Revenue ($M)",
            defaults.base_revenue / 1_000_000,
            f"dcf_base_revenue_{ticker}",
        )
        tax_rate = st.number_input(
            "Effective Tax Rate (%)",
            min_value=0.0,
            max_value=60.0,
            value=float(defaults.tax_rate * 100),
            step=0.25,
            key=f"dcf_tax_rate_{ticker}",
        )

    with reinvestment_col:
        st.markdown("##### Reinvestment")
        depreciation_ratio = st.number_input(
            "D&A / Revenue (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.depreciation_pct_revenue * 100),
            step=0.25,
            key=f"dcf_depreciation_{ticker}",
        )
        capex_ratio = st.number_input(
            "CapEx / Revenue (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.capex_pct_revenue * 100),
            step=0.25,
            key=f"dcf_capex_{ticker}",
        )
        nwc_ratio = st.number_input(
            "NWC Investment / Incremental Revenue (%)",
            min_value=-100.0,
            max_value=100.0,
            value=float(defaults.nwc_pct_incremental_revenue * 100),
            step=0.25,
            key=f"dcf_nwc_{ticker}",
        )

    with capital_col:
        st.markdown("##### Capital and terminal value")
        wacc = st.number_input(
            "WACC (%)",
            min_value=0.01,
            max_value=99.99,
            value=float(defaults.wacc * 100),
            step=0.25,
            key=f"dcf_wacc_{ticker}",
        )
        terminal_growth = st.number_input(
            "Terminal Growth (%)",
            min_value=-9.99,
            max_value=9.99,
            value=float(defaults.terminal_growth * 100),
            step=0.25,
            key=f"dcf_terminal_growth_{ticker}",
        )

    bridge_col1, bridge_col2, bridge_col3 = st.columns(3)
    with bridge_col1:
        cash = _amount_input(
            "Cash and Short-Term Investments ($M)",
            defaults.cash / 1_000_000,
            f"dcf_cash_{ticker}",
        )
    with bridge_col2:
        debt = _amount_input(
            "Total Debt ($M)",
            defaults.debt / 1_000_000,
            f"dcf_debt_{ticker}",
        )
    with bridge_col3:
        diluted_shares = _amount_input(
            "Diluted Shares ($M)",
            defaults.diluted_shares / 1_000_000,
            f"dcf_diluted_shares_{ticker}",
            "Use fully diluted shares outstanding, expressed in millions.",
        )

    try:
        revenue_growth = tuple(
            float(value) / 100 for value in edited_forecast["Revenue Growth %"]
        )
        ebit_margin = tuple(
            float(value) / 100 for value in edited_forecast["EBIT Margin %"]
        )
        base_inputs = DCFInputs(
            base_revenue=float(base_revenue),
            revenue_growth=revenue_growth,
            ebit_margin=ebit_margin,
            tax_rate=float(tax_rate) / 100,
            depreciation_pct_revenue=float(depreciation_ratio) / 100,
            capex_pct_revenue=float(capex_ratio) / 100,
            nwc_pct_incremental_revenue=float(nwc_ratio) / 100,
            wacc=float(wacc) / 100,
            terminal_growth=float(terminal_growth) / 100,
            cash=float(cash),
            debt=float(debt),
            diluted_shares=float(diluted_shares),
        )
        base_result = calculate_dcf(base_inputs)
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return

    st.divider()
    st.markdown("#### Base-case output")
    market_difference = None
    if current_price and current_price > 0:
        market_difference = (base_result.implied_value_per_share / current_price - 1) * 100

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    metric1.metric("Enterprise Value", f"${base_result.enterprise_value:,.1f}M")
    metric2.metric("Equity Value", f"${base_result.equity_value:,.1f}M")
    metric3.metric(
        "Implied Value / Share",
        f"${base_result.implied_value_per_share:,.2f}",
        delta=f"{market_difference:+.1f}% vs market" if market_difference is not None else None,
    )
    metric4.metric("Current Price", f"${current_price:,.2f}" if current_price else "N/A")
    terminal_share = (
        base_result.present_value_terminal / base_result.enterprise_value * 100
        if base_result.enterprise_value
        else 0
    )
    metric5.metric("Terminal Value / EV", f"{terminal_share:.1f}%")

    forecast_df = _forecast_dataframe(base_result)
    st.dataframe(
        forecast_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Revenue Growth %": st.column_config.NumberColumn(format="%.2f%%"),
            "EBIT Margin %": st.column_config.NumberColumn(format="%.2f%%"),
            "Revenue ($M)": st.column_config.NumberColumn(format="$%.1f"),
            "EBIT ($M)": st.column_config.NumberColumn(format="$%.1f"),
            "NOPAT ($M)": st.column_config.NumberColumn(format="$%.1f"),
            "D&A ($M)": st.column_config.NumberColumn(format="$%.1f"),
            "CapEx ($M)": st.column_config.NumberColumn(format="$%.1f"),
            "Change in NWC ($M)": st.column_config.NumberColumn(format="$%.1f"),
            "UFCF ($M)": st.column_config.NumberColumn(format="$%.1f"),
            "PV of UFCF ($M)": st.column_config.NumberColumn(format="$%.1f"),
        },
    )

    st.markdown("#### Scenario analysis")
    scenario_editor = pd.DataFrame(
        {
            "Scenario": ["Bear", "Base", "Bull"],
            "Growth Adjustment (pp)": [-2.0, 0.0, 2.0],
            "Margin Adjustment (pp)": [-2.0, 0.0, 2.0],
            "WACC Adjustment (pp)": [1.0, 0.0, -1.0],
        }
    )
    edited_scenarios = st.data_editor(
        scenario_editor,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=["Scenario"],
        key=f"dcf_scenarios_{ticker}",
        column_config={
            "Growth Adjustment (pp)": st.column_config.NumberColumn(format="%+.2f"),
            "Margin Adjustment (pp)": st.column_config.NumberColumn(format="%+.2f"),
            "WACC Adjustment (pp)": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

    scenario_rows = []
    for _, row in edited_scenarios.iterrows():
        try:
            scenario_inputs = apply_scenario_adjustments(
                base_inputs,
                growth_delta=float(row["Growth Adjustment (pp)"]) / 100,
                margin_delta=float(row["Margin Adjustment (pp)"]) / 100,
                wacc_delta=float(row["WACC Adjustment (pp)"]) / 100,
            )
            result = calculate_dcf(scenario_inputs)
            difference = (
                (result.implied_value_per_share / current_price - 1) * 100
                if current_price and current_price > 0
                else None
            )
            scenario_rows.append(
                {
                    "Scenario": row["Scenario"],
                    "Implied Value / Share": result.implied_value_per_share,
                    "Difference vs Market %": difference,
                    "Enterprise Value ($M)": result.enterprise_value,
                    "Equity Value ($M)": result.equity_value,
                }
            )
        except ValueError as error:
            scenario_rows.append(
                {
                    "Scenario": row["Scenario"],
                    "Implied Value / Share": None,
                    "Difference vs Market %": None,
                    "Enterprise Value ($M)": None,
                    "Equity Value ($M)": None,
                    "Error": str(error),
                }
            )

    st.dataframe(
        pd.DataFrame(scenario_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Implied Value / Share": st.column_config.NumberColumn(format="$%.2f"),
            "Difference vs Market %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Enterprise Value ($M)": st.column_config.NumberColumn(format="$%.1f"),
            "Equity Value ($M)": st.column_config.NumberColumn(format="$%.1f"),
        },
    )

    st.markdown("#### WACC / terminal-growth sensitivity")
    wacc_values = tuple(
        max(base_inputs.wacc + adjustment, 0.0001)
        for adjustment in (-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02)
    )
    terminal_growth_values = tuple(
        min(max(base_inputs.terminal_growth + adjustment, -0.099), 0.099)
        for adjustment in (-0.01, -0.005, 0, 0.005, 0.01)
    )
    sensitivity = calculate_sensitivity(
        base_inputs,
        wacc_values=wacc_values,
        terminal_growth_values=terminal_growth_values,
    )
    sensitivity_df = pd.DataFrame(sensitivity).T
    sensitivity_df.index = [f"{value:.2%}" for value in sensitivity_df.index]
    sensitivity_df.columns = [f"{value:.2%}" for value in sensitivity_df.columns]
    sensitivity_df.index.name = "WACC"
    sensitivity_df.columns.name = "Terminal Growth"
    formatted_sensitivity = sensitivity_df.apply(
        lambda column: column.map(_format_sensitivity_value)
    )
    st.dataframe(formatted_sensitivity, width="stretch")

    with st.expander("Input sources and methodology"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Input": key.replace("_", " ").title(), "Source": source}
                    for key, source in defaults.sources.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            "**Formula:** UFCF = EBIT × (1 − tax rate) + D&A − CapEx − change in NWC. "
            "Terminal value uses the perpetuity-growth method. Cash is added and debt is subtracted to bridge from enterprise value to equity value."
        )

    st.download_button(
        "Download Base-Case Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_dcf_forecast.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption("Educational research model only. Model outputs are not investment recommendations.")
