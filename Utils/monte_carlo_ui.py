from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from Utils.data_utils import get_balance_sheet_cached, get_cashflow_cached
from Utils.dcf import DCFInputs, apply_scenario_adjustments
from Utils.dcf_data import build_dcf_defaults
from Utils.monte_carlo import (
    MonteCarloConfig,
    run_monte_carlo,
    summarize_monte_carlo,
)

SCENARIO_CENTERS = {
    "Bear": (-0.02, -0.02, 0.01),
    "Neutral": (0.0, 0.0, 0.0),
    "Bull": (0.02, 0.02, -0.01),
}


@st.cache_data(show_spinner=False)
def _run_cached_simulation(inputs: DCFInputs, config: MonteCarloConfig):
    return run_monte_carlo(inputs, config)


def _is_financial_company(info: dict) -> bool:
    sector = str(info.get("sector", "")).lower()
    industry = str(info.get("industry", "")).lower()
    terms = ("bank", "insurance", "financial services", "capital markets")
    return sector == "financial services" or any(term in industry for term in terms)


def _automatic_inputs(context) -> tuple[DCFInputs, object]:
    ticker = context["ticker"]
    defaults = build_dcf_defaults(
        info=context["info"],
        income_statement=context["financials"],
        cashflow_statement=get_cashflow_cached(ticker),
        balance_sheet=get_balance_sheet_cached(ticker),
    )
    inputs = DCFInputs(
        base_revenue=defaults.base_revenue / 1_000_000,
        revenue_growth=defaults.revenue_growth,
        ebit_margin=defaults.ebit_margin,
        tax_rate=defaults.tax_rate,
        depreciation_pct_revenue=defaults.depreciation_pct_revenue,
        capex_pct_revenue=defaults.capex_pct_revenue,
        nwc_pct_incremental_revenue=defaults.nwc_pct_incremental_revenue,
        wacc=defaults.wacc,
        terminal_growth=defaults.terminal_growth,
        cash=defaults.cash / 1_000_000,
        debt=defaults.debt / 1_000_000,
        diluted_shares=defaults.diluted_shares / 1_000_000,
    )
    return inputs, defaults


def _render_styles():
    st.markdown(
        """
        <style>
        .mc-result-panel {
            margin: 0.35rem 0 0.9rem;
            overflow: hidden;
            border: 1px solid #273449;
            border-left: 3px solid #6b9ff8;
            border-radius: 8px;
            background: #0e1726;
        }

        .mc-result-top {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) repeat(2, minmax(8rem, 0.65fr));
            gap: 1px;
            background: #273449;
        }

        .mc-result-primary,
        .mc-result-secondary {
            background: #0e1726;
            padding: 1rem 1.15rem;
        }

        .mc-result-primary span,
        .mc-result-secondary span {
            display: block;
            color: #7f8b9d;
            font-size: 0.65rem;
            font-weight: 650;
            letter-spacing: 0.075em;
            text-transform: uppercase;
        }

        .mc-result-primary strong {
            display: block;
            color: #f5f7fa;
            font-size: clamp(2.35rem, 5vw, 3.35rem);
            font-weight: 720;
            letter-spacing: -0.045em;
            line-height: 1;
            margin-top: 0.3rem;
        }

        .mc-result-secondary strong {
            display: block;
            color: #dbe1e9;
            font-size: 1.25rem;
            font-weight: 680;
            margin-top: 0.35rem;
        }

        .mc-result-secondary small {
            color: #7f8b9d;
            font-size: 0.7rem;
        }

        @media (max-width: 700px) {
            .mc-result-top {
                grid-template-columns: 1fr 1fr;
            }

            .mc-result-primary {
                grid-column: 1 / -1;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _distribution_figure(samples: pd.DataFrame, summary: dict, current_price):
    values = samples["Implied Value Per Share"]
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=55,
            name="Simulations",
            marker={"color": "#6b9ff8", "line": {"color": "#8fb5fa", "width": 0.3}},
            opacity=0.82,
            hovertemplate="Implied value: $%{x:,.2f}<br>Draws: %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=summary["median"],
        line_color="#f5f7fa",
        line_width=1.5,
        annotation_text="Median",
        annotation_position="top right",
    )
    if current_price and current_price > 0:
        figure.add_vline(
            x=current_price,
            line_color="#d97757",
            line_width=1.5,
            line_dash="dash",
            annotation_text="Market",
            annotation_position="top left",
        )
    figure.update_layout(
        height=390,
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1320",
        font={"color": "#aeb8c6", "size": 12},
        bargap=0.02,
        showlegend=False,
        xaxis={
            "title": "Implied value per share",
            "tickprefix": "$",
            "gridcolor": "#1f2b3d",
            "zeroline": False,
        },
        yaxis={"title": "Simulation count", "gridcolor": "#1f2b3d"},
    )
    return figure


def render_monte_carlo(context):
    ticker = context["ticker"]
    current_price = context["current_price"]

    _render_styles()
    st.subheader("Monte Carlo valuation")
    st.caption(
        "The DCF is rerun across distributions of growth, margin, WACC, and terminal-growth assumptions."
    )

    if _is_financial_company(context["info"]):
        st.warning(
            "This simulation uses the UFCF DCF and is not designed for banks, insurers, or other financial companies."
        )
        return

    try:
        automatic_inputs, defaults = _automatic_inputs(context)
    except (TypeError, ValueError) as error:
        st.error(f"Automatic valuation inputs could not be prepared: {error}")
        return

    critical_inputs_missing = (
        automatic_inputs.base_revenue <= 0 or automatic_inputs.diluted_shares <= 0
    )
    with st.expander(
        "Simulation settings · optional",
        expanded=critical_inputs_missing,
    ):
        st.caption("Defaults are ready to run. Change these only to test a different uncertainty range.")
        settings_left, settings_middle, settings_right = st.columns(3)
        with settings_left:
            scenario_center = st.selectbox(
                "Scenario center",
                tuple(SCENARIO_CENTERS),
                index=1,
                key=f"mc_scenario_{ticker}",
            )
            simulation_count = st.select_slider(
                "Simulation count",
                options=(1_000, 2_500, 5_000, 10_000, 20_000),
                value=5_000,
                key=f"mc_count_{ticker}",
            )
            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1,
                key=f"mc_seed_{ticker}",
            )
        with settings_middle:
            growth_std = st.number_input(
                "Growth range (1σ, pp)",
                min_value=0.0,
                max_value=25.0,
                value=2.0,
                step=0.25,
                key=f"mc_growth_std_{ticker}",
            )
            margin_std = st.number_input(
                "Margin range (1σ, pp)",
                min_value=0.0,
                max_value=25.0,
                value=1.5,
                step=0.25,
                key=f"mc_margin_std_{ticker}",
            )
        with settings_right:
            wacc_std = st.number_input(
                "WACC range (1σ, pp)",
                min_value=0.0,
                max_value=10.0,
                value=0.75,
                step=0.25,
                key=f"mc_wacc_std_{ticker}",
            )
            terminal_growth_std = st.number_input(
                "Terminal-growth range (1σ, pp)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.25,
                key=f"mc_terminal_std_{ticker}",
            )

    growth_delta, margin_delta, wacc_delta = SCENARIO_CENTERS[scenario_center]
    centered_inputs = apply_scenario_adjustments(
        automatic_inputs,
        growth_delta=growth_delta,
        margin_delta=margin_delta,
        wacc_delta=wacc_delta,
    )
    config = MonteCarloConfig(
        simulation_count=int(simulation_count),
        seed=int(seed),
        growth_std=float(growth_std) / 100,
        margin_std=float(margin_std) / 100,
        wacc_std=float(wacc_std) / 100,
        terminal_growth_std=float(terminal_growth_std) / 100,
    )

    try:
        with st.spinner(f"Running {simulation_count:,} DCF simulations…"):
            result = _run_cached_simulation(centered_inputs, config)
        summary = summarize_monte_carlo(result, current_price)
    except (TypeError, ValueError) as error:
        st.error(f"The simulation could not be completed: {error}")
        if defaults.warnings:
            with st.expander("Data quality notes", expanded=True):
                for warning in defaults.warnings:
                    st.warning(warning)
        return

    probability = summary["probability_above_market"]
    probability_label = f"{probability:.1%}" if probability is not None else "N/A"
    market_label = f"${current_price:,.2f}" if current_price else "N/A"
    st.markdown(
        f"""
        <div class="mc-result-panel">
            <div class="mc-result-top">
                <div class="mc-result-primary">
                    <span>{escape(ticker)} · median simulated value</span>
                    <strong>${summary["median"]:,.2f}</strong>
                </div>
                <div class="mc-result-secondary">
                    <span>Probability above market</span>
                    <strong>{probability_label}</strong>
                    <small>Current price {market_label}</small>
                </div>
                <div class="mc-result-secondary">
                    <span>90% valuation range</span>
                    <strong>${summary["p05"]:,.2f}–${summary["p95"]:,.2f}</strong>
                    <small>{summary["simulations"]:,} accepted draws</small>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        _distribution_figure(result.samples, summary, current_price),
        width="stretch",
        config={"displayModeBar": False},
    )

    percentile_table = pd.DataFrame(
        {
            "Percentile": ["5th", "25th", "Median", "Mean", "75th", "95th"],
            "Implied Value Per Share": [
                summary["p05"],
                summary["p25"],
                summary["median"],
                summary["mean"],
                summary["p75"],
                summary["p95"],
            ],
        }
    )
    with st.expander("Distribution statistics"):
        st.dataframe(
            percentile_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Implied Value Per Share": st.column_config.NumberColumn(format="$%.2f")
            },
        )
        st.caption(
            f"Rejected {result.rejected_draws:,} draws because they violated economic or model constraints."
        )

    export_frame = result.samples.copy()
    for column in (
        "Growth Adjustment",
        "Margin Adjustment",
        "WACC",
        "Terminal Growth",
    ):
        export_frame[column] = export_frame[column] * 100
    st.download_button(
        "Download simulation CSV",
        data=export_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_monte_carlo_dcf.csv",
        mime="text/csv",
        width="stretch",
    )

    if defaults.warnings:
        with st.expander(f"Data quality notes ({len(defaults.warnings)})"):
            for warning in defaults.warnings:
                st.warning(warning)

    with st.expander("Methodology and limitations"):
        st.markdown(
            "Each draw applies independent normal changes to the full revenue-growth path, "
            "EBIT-margin path, WACC, and terminal growth, then reruns the same deterministic DCF engine. "
            "Draws are rejected when WACC is too close to terminal growth or assumptions fall outside "
            "the model's economic bounds. Correlations between assumptions are not modeled in this version."
        )
    st.caption("Educational research model only. Simulation outputs are not investment recommendations.")
