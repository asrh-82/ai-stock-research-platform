from html import escape

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

SCENARIO_ADJUSTMENTS = {
    "Bear": {
        "growth_delta": -0.02,
        "margin_delta": -0.02,
        "wacc_delta": 0.01,
        "summary": "Growth −2 pp · EBIT margin −2 pp · WACC +1 pp",
    },
    "Neutral": {
        "growth_delta": 0.0,
        "margin_delta": 0.0,
        "wacc_delta": 0.0,
        "summary": "Automatic company data and model defaults",
    },
    "Bull": {
        "growth_delta": 0.02,
        "margin_delta": 0.02,
        "wacc_delta": -0.01,
        "summary": "Growth +2 pp · EBIT margin +2 pp · WACC −1 pp",
    },
}


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


def _format_currency(value, decimals=2):
    value = float(value)
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def _render_implied_price_hero(ticker, result, current_price, scenario):
    market_difference = None
    if current_price and current_price > 0:
        market_difference = (result.implied_value_per_share / current_price - 1) * 100

    if market_difference is None:
        comparison_text = "Current-market comparison unavailable"
        comparison_class = "neutral"
    else:
        comparison_text = f"{market_difference:+.1f}% model gap vs current market price"
        comparison_class = "positive" if market_difference >= 0 else "negative"

    st.markdown(
        """
        <style>
        .dcf-value-hero {
            padding: 1.6rem 1.8rem;
            margin: 0.5rem 0 1rem 0;
            border: 1px solid #2563eb;
            border-radius: 16px;
            background: linear-gradient(135deg, #0b1730 0%, #111827 60%, #0f2447 100%);
            box-shadow: 0 16px 48px rgba(37, 99, 235, 0.18);
        }

        .dcf-value-hero.bear {
            border-color: #ea580c;
            background: linear-gradient(135deg, #2b140d 0%, #111827 60%, #32180d 100%);
            box-shadow: 0 16px 48px rgba(234, 88, 12, 0.16);
        }

        .dcf-value-hero.bull {
            border-color: #16a34a;
            background: linear-gradient(135deg, #08291a 0%, #111827 60%, #0b321e 100%);
            box-shadow: 0 16px 48px rgba(22, 163, 74, 0.16);
        }

        .dcf-value-eyebrow {
            color: #93c5fd;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .dcf-value-label {
            color: #cbd5e1;
            font-size: 1rem;
            margin-top: 0.65rem;
        }

        .dcf-value-price {
            color: #f8fafc;
            font-size: clamp(3rem, 8vw, 5.25rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            line-height: 1;
            margin: 0.3rem 0 0.7rem 0;
        }

        .dcf-value-comparison {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 650;
        }

        .dcf-value-comparison.positive {
            color: #bbf7d0;
            background: rgba(22, 101, 52, 0.35);
            border: 1px solid #166534;
        }

        .dcf-value-comparison.negative {
            color: #fed7aa;
            background: rgba(154, 52, 18, 0.3);
            border: 1px solid #9a3412;
        }

        .dcf-value-comparison.neutral {
            color: #cbd5e1;
            background: rgba(51, 65, 85, 0.45);
            border: 1px solid #475569;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    scenario_class = str(scenario).lower()
    st.markdown(
        f"""
        <div class="dcf-value-hero {scenario_class}">
            <div class="dcf-value-eyebrow">{escape(str(scenario))} scenario DCF · {escape(str(ticker))}</div>
            <div class="dcf-value-label">Implied value per share</div>
            <div class="dcf-value-price">{_format_currency(result.implied_value_per_share)}</div>
            <div class="dcf-value-comparison {comparison_class}">{comparison_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    terminal_share = (
        result.present_value_terminal / result.enterprise_value * 100
        if result.enterprise_value
        else 0
    )
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric(
        "Current Market Price",
        _format_currency(current_price) if current_price else "N/A",
    )
    metric2.metric("Enterprise Value", f"{_format_currency(result.enterprise_value, 1)}M")
    metric3.metric("Equity Value", f"{_format_currency(result.equity_value, 1)}M")
    metric4.metric("Terminal Value / EV", f"{terminal_share:.1f}%")
    st.caption(
        "Automatically generated from the latest available company data and model defaults. "
        "This is a research-model output, not a price target or recommendation."
    )


def render_dcf(context):
    ticker = context["ticker"]
    info = context["info"]
    current_price = context["current_price"]

    st.subheader("Discounted Cash Flow Valuation")
    st.caption("Automatic five-year DCF. Pick an outlook; no assumptions need to be entered.")

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

    st.markdown("#### Valuation scenario")
    selected_scenario = st.select_slider(
        "Scenario outlook",
        options=tuple(SCENARIO_ADJUSTMENTS),
        value="Neutral",
        key=f"dcf_selected_scenario_{ticker}",
        label_visibility="collapsed",
        help="Moves the full DCF between consistent automatic bear, neutral, and bull assumptions.",
    )
    selected_adjustments = SCENARIO_ADJUSTMENTS[selected_scenario]
    st.caption(selected_adjustments["summary"])

    result_section = st.container()
    quality_section = st.container()
    critical_inputs_missing = defaults.base_revenue <= 0 or defaults.diluted_shares <= 0

    with st.expander(
        "Adjust assumptions (optional)",
        expanded=critical_inputs_missing,
    ):
        st.info(
            "The neutral model is already filled in. Change these only when you want to override its automatic assumptions."
        )

        st.markdown("##### Revenue growth and EBIT margin")
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
            st.markdown("##### Company data")
            base_revenue = _amount_input(
                "Base Revenue ($M)",
                defaults.base_revenue / 1_000_000,
                f"dcf_base_revenue_{ticker}",
            )
            cash = _amount_input(
                "Cash ($M)",
                defaults.cash / 1_000_000,
                f"dcf_cash_{ticker}",
            )
            debt = _amount_input(
                "Debt ($M)",
                defaults.debt / 1_000_000,
                f"dcf_debt_{ticker}",
            )
            diluted_shares = _amount_input(
                "Diluted Shares ($M)",
                defaults.diluted_shares / 1_000_000,
                f"dcf_diluted_shares_{ticker}",
                "Use fully diluted shares outstanding, expressed in millions.",
            )

        with reinvestment_col:
            st.markdown("##### Operating and reinvestment")
            tax_rate = st.number_input(
                "Effective Tax Rate (%)",
                min_value=0.0,
                max_value=60.0,
                value=float(defaults.tax_rate * 100),
                step=0.25,
                key=f"dcf_tax_rate_{ticker}",
            )
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
                "NWC / Incremental Revenue (%)",
                min_value=-100.0,
                max_value=100.0,
                value=float(defaults.nwc_pct_incremental_revenue * 100),
                step=0.25,
                key=f"dcf_nwc_{ticker}",
            )

        with capital_col:
            st.markdown("##### Discount and terminal value")
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

        reset_keys = (
            f"dcf_selected_scenario_{ticker}",
            f"dcf_forecast_assumptions_{ticker}",
            f"dcf_base_revenue_{ticker}",
            f"dcf_cash_{ticker}",
            f"dcf_debt_{ticker}",
            f"dcf_diluted_shares_{ticker}",
            f"dcf_tax_rate_{ticker}",
            f"dcf_depreciation_{ticker}",
            f"dcf_capex_{ticker}",
            f"dcf_nwc_{ticker}",
            f"dcf_wacc_{ticker}",
            f"dcf_terminal_growth_{ticker}",
        )
        if st.button(
            "Reset to automatic assumptions",
            key=f"dcf_reset_assumptions_{ticker}",
            width="stretch",
        ):
            for key in reset_keys:
                st.session_state.pop(key, None)
            st.rerun()

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
        selected_inputs = apply_scenario_adjustments(
            base_inputs,
            growth_delta=selected_adjustments["growth_delta"],
            margin_delta=selected_adjustments["margin_delta"],
            wacc_delta=selected_adjustments["wacc_delta"],
        )
        selected_result = calculate_dcf(selected_inputs)
    except (TypeError, ValueError) as error:
        with result_section:
            st.error(f"The automatic valuation could not be generated: {error}")
        with quality_section:
            if defaults.warnings:
                with st.expander(
                    f"Data quality notes ({len(defaults.warnings)})",
                    expanded=True,
                ):
                    for warning in defaults.warnings:
                        st.warning(warning)
        return

    with result_section:
        _render_implied_price_hero(
            ticker,
            selected_result,
            current_price,
            selected_scenario,
        )

    with quality_section:
        if defaults.warnings:
            with st.expander(
                f"Data quality notes ({len(defaults.warnings)})",
                expanded=critical_inputs_missing,
            ):
                for warning in defaults.warnings:
                    st.warning(warning)

    forecast_df = _forecast_dataframe(selected_result)
    st.markdown(f"#### {selected_scenario} scenario details")
    with st.expander("Forecast cash flows"):
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
        st.download_button(
            f"Download {selected_scenario} Forecast CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_{selected_scenario.lower()}_dcf_forecast.csv",
            mime="text/csv",
            width="stretch",
        )

    scenario_assumptions = pd.DataFrame(
        [
            {
                "Scenario": scenario,
                "Growth Adjustment (pp)": adjustments["growth_delta"] * 100,
                "Margin Adjustment (pp)": adjustments["margin_delta"] * 100,
                "WACC Adjustment (pp)": adjustments["wacc_delta"] * 100,
            }
            for scenario, adjustments in SCENARIO_ADJUSTMENTS.items()
        ]
    )
    scenario_rows = []
    for _, row in scenario_assumptions.iterrows():
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

    with st.expander("Automatic bear, neutral, and bull range"):
        st.caption(
            "These scenarios are generated automatically using ±2 percentage points for growth and margin and ∓1 percentage point for WACC."
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

    wacc_values = tuple(
        max(selected_inputs.wacc + adjustment, 0.0001)
        for adjustment in (-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02)
    )
    terminal_growth_values = tuple(
        min(max(selected_inputs.terminal_growth + adjustment, -0.099), 0.099)
        for adjustment in (-0.01, -0.005, 0, 0.005, 0.01)
    )
    sensitivity = calculate_sensitivity(
        selected_inputs,
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
    with st.expander("WACC / terminal-growth sensitivity"):
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

    st.caption("Educational research model only. Model outputs are not investment recommendations.")
