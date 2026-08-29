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


def _render_dcf_styles():
    st.markdown(
        """
        <style>
        .dcf-scenario-copy {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            margin: 1rem 0 0.35rem;
        }

        .dcf-scenario-copy span {
            color: #e5e7eb;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .dcf-scenario-copy p {
            color: #7f8b9d;
            font-size: 0.8rem;
            margin: 0;
        }

        div[data-testid="stSegmentedControl"] {
            margin-bottom: 0.2rem;
            width: 100%;
        }

        div[data-testid="stSegmentedControl"] > div,
        div[data-testid="stSegmentedControl"] [role="radiogroup"] {
            width: 100%;
        }

        div[data-testid="stSegmentedControl"] [role="radiogroup"] {
            display: flex;
            gap: 3px;
            padding: 3px;
            border: 1px solid #273449;
            border-radius: 8px;
            background: #0b1220;
        }

        div[data-testid="stSegmentedControl"] button {
            flex: 1 1 0;
            min-height: 2.35rem;
            border: 0 !important;
            border-radius: 5px !important;
            background: transparent !important;
            color: #8591a3 !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.075em;
            text-transform: uppercase;
            box-shadow: none !important;
        }

        div[data-testid="stSegmentedControl"] button:hover {
            color: #e5e7eb !important;
            background: #151f2e !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            color: #0b1220 !important;
            background: #e5e7eb !important;
        }

        .dcf-scenario-assumptions {
            color: #7f8b9d;
            font-size: 0.78rem;
            min-height: 1.2rem;
            margin: 0.15rem 0 0.85rem;
        }

        .dcf-value-panel {
            --scenario-accent: #6b9ff8;
            margin: 0.35rem 0 0.85rem;
            overflow: hidden;
            border: 1px solid #273449;
            border-left: 3px solid var(--scenario-accent);
            border-radius: 8px;
            background: #0e1726;
        }

        .dcf-value-panel.bear {
            --scenario-accent: #d97757;
        }

        .dcf-value-panel.bull {
            --scenario-accent: #43a675;
        }

        .dcf-value-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: end;
            gap: 2rem;
            padding: 1.25rem 1.35rem 1.1rem;
        }

        .dcf-value-eyebrow {
            color: #7f8b9d;
            font-size: 0.7rem;
            font-weight: 650;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .dcf-value-label {
            color: #b4bdca;
            font-size: 0.82rem;
            margin-top: 0.75rem;
        }

        .dcf-value-price {
            color: #f5f7fa;
            font-size: clamp(2.85rem, 7vw, 4.35rem);
            font-weight: 720;
            letter-spacing: -0.055em;
            line-height: 0.98;
            margin-top: 0.2rem;
        }

        .dcf-market-comparison {
            min-width: 10rem;
            padding-bottom: 0.25rem;
            text-align: right;
        }

        .dcf-market-comparison span {
            display: block;
            color: #7f8b9d;
            font-size: 0.67rem;
            font-weight: 650;
            letter-spacing: 0.075em;
            text-transform: uppercase;
        }

        .dcf-market-comparison strong {
            display: block;
            color: #c8d0dc;
            font-size: 1.35rem;
            font-weight: 680;
            line-height: 1.15;
            margin-top: 0.28rem;
        }

        .dcf-market-comparison strong.positive {
            color: #69bd8e;
        }

        .dcf-market-comparison strong.negative {
            color: #df876e;
        }

        .dcf-market-comparison small {
            color: #7f8b9d;
            font-size: 0.72rem;
        }

        .dcf-value-stats {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            border-top: 1px solid #273449;
            background: #0b1320;
        }

        .dcf-value-stat {
            min-width: 0;
            padding: 0.72rem 1rem 0.78rem;
            border-right: 1px solid #273449;
        }

        .dcf-value-stat:last-child {
            border-right: 0;
        }

        .dcf-value-stat span {
            display: block;
            color: #778397;
            font-size: 0.64rem;
            font-weight: 650;
            letter-spacing: 0.065em;
            text-transform: uppercase;
        }

        .dcf-value-stat strong {
            display: block;
            overflow-wrap: anywhere;
            color: #dbe1e9;
            font-size: 0.93rem;
            font-weight: 620;
            margin-top: 0.18rem;
        }

        @media (max-width: 700px) {
            .dcf-scenario-copy p {
                display: none;
            }

            .dcf-value-top {
                grid-template-columns: 1fr;
                gap: 0.85rem;
            }

            .dcf-market-comparison {
                padding-bottom: 0;
                text-align: left;
            }

            .dcf-value-stats {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .dcf-value-stat:nth-child(2) {
                border-right: 0;
            }

            .dcf-value-stat:nth-child(-n + 2) {
                border-bottom: 1px solid #273449;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_implied_price_hero(ticker, result, current_price, scenario):
    market_difference = None
    if current_price and current_price > 0:
        market_difference = (result.implied_value_per_share / current_price - 1) * 100

    if market_difference is None:
        comparison_value = "—"
        comparison_note = "Market comparison unavailable"
        comparison_class = "neutral"
    else:
        comparison_value = f"{market_difference:+.1f}%"
        comparison_note = "model premium" if market_difference >= 0 else "model discount"
        comparison_class = "positive" if market_difference >= 0 else "negative"

    scenario_class = str(scenario).lower()
    terminal_share = (
        result.present_value_terminal / result.enterprise_value * 100
        if result.enterprise_value
        else 0
    )
    st.markdown(
        f"""
        <div class="dcf-value-panel {scenario_class}">
            <div class="dcf-value-top">
                <div>
                    <div class="dcf-value-eyebrow">{escape(str(ticker))} / DCF / {escape(str(scenario))}</div>
                    <div class="dcf-value-label">Implied share value</div>
                    <div class="dcf-value-price">{_format_currency(result.implied_value_per_share)}</div>
                </div>
                <div class="dcf-market-comparison">
                    <span>Versus market</span>
                    <strong class="{comparison_class}">{comparison_value}</strong>
                    <small>{comparison_note}</small>
                </div>
            </div>
            <div class="dcf-value-stats">
                <div class="dcf-value-stat">
                    <span>Market price</span>
                    <strong>{_format_currency(current_price) if current_price else "N/A"}</strong>
                </div>
                <div class="dcf-value-stat">
                    <span>Enterprise value</span>
                    <strong>{_format_currency(result.enterprise_value, 1)}M</strong>
                </div>
                <div class="dcf-value-stat">
                    <span>Equity value</span>
                    <strong>{_format_currency(result.equity_value, 1)}M</strong>
                </div>
                <div class="dcf-value-stat">
                    <span>Terminal value / EV</span>
                    <strong>{terminal_share:.1f}%</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Automatic research-model output · not a recommendation")


def render_dcf(context):
    ticker = context["ticker"]
    info = context["info"]
    current_price = context["current_price"]

    _render_dcf_styles()
    st.subheader("Discounted cash flow")
    st.caption("Five-year unlevered cash-flow model with automatic inputs.")

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

    st.markdown(
        """
        <div class="dcf-scenario-copy">
            <span>Scenario</span>
            <p>Switch the full model without editing assumptions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_scenario = st.segmented_control(
        "Scenario outlook",
        options=tuple(SCENARIO_ADJUSTMENTS),
        default="Neutral",
        key=f"dcf_scenario_switch_{ticker}",
        label_visibility="collapsed",
        help="Moves the full DCF between consistent automatic bear, neutral, and bull assumptions.",
        format_func=str.upper,
        required=True,
        width="stretch",
    )
    selected_adjustments = SCENARIO_ADJUSTMENTS[selected_scenario]
    st.markdown(
        f'<div class="dcf-scenario-assumptions">{selected_adjustments["summary"]}</div>',
        unsafe_allow_html=True,
    )

    result_section = st.container()
    quality_section = st.container()
    critical_inputs_missing = defaults.base_revenue <= 0 or defaults.diluted_shares <= 0

    with st.expander(
        "Model assumptions · optional",
        expanded=critical_inputs_missing,
    ):
        st.caption(
            "The neutral model is already filled in. Change these only when you want to override its automatic assumptions."
        )

        st.markdown("**Growth and margin**")
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
            st.markdown("**Company data**")
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
            st.markdown("**Operating and reinvestment**")
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
            st.markdown("**Discount and terminal value**")
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
            f"dcf_scenario_switch_{ticker}",
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
    st.markdown(f"##### {selected_scenario} scenario details")
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

    with st.expander("Scenario comparison"):
        st.caption(
            "Growth and EBIT margin move ±2 percentage points; WACC moves ∓1 percentage point."
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
