from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from Utils.backtesting import (
    SUPPORTED_EVALUATION_MODES,
    SUPPORTED_POSITION_MODES,
    SUPPORTED_REBALANCING,
    SUPPORTED_STRATEGIES,
    BacktestConfig,
    build_stability_table,
    run_backtest,
)
from Utils.data_utils import get_price_history_cached

PERIOD_OPTIONS = {
    "1 year": "1y",
    "3 years": "3y",
    "5 years": "5y",
    "10 years": "10y",
    "Maximum": "max",
}


def _extract_close(history: pd.DataFrame, label: str) -> pd.Series:
    if history is None or history.empty:
        raise ValueError(f"{label} price history is unavailable.")

    if "Close" in history.columns:
        close = history["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close

    if isinstance(history.columns, pd.MultiIndex):
        for level in range(history.columns.nlevels):
            if "Close" in history.columns.get_level_values(level):
                close = history.xs("Close", axis=1, level=level)
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return close

    raise ValueError(f"{label} history does not contain a closing-price series.")


def _render_styles():
    st.markdown(
        """
        <style>
        .bt-summary {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0.45rem 0 0.95rem;
            overflow: hidden;
            border: 1px solid #273449;
            border-left: 3px solid #6b9ff8;
            border-radius: 8px;
            background: #273449;
            gap: 1px;
        }

        .bt-summary-item {
            min-width: 0;
            padding: 0.8rem 0.9rem;
            background: #0e1726;
        }

        .bt-summary-item span {
            display: block;
            color: #778397;
            font-size: 0.62rem;
            font-weight: 650;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .bt-summary-item strong {
            display: block;
            overflow-wrap: anywhere;
            color: #e5e9ef;
            font-size: 1.05rem;
            font-weight: 680;
            margin-top: 0.22rem;
        }

        .bt-summary-item small {
            display: block;
            color: #778397;
            font-size: 0.66rem;
            margin-top: 0.08rem;
        }

        @media (max-width: 800px) {
            .bt-summary {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _percent(value):
    return "N/A" if value is None else f"{value:+.1%}"


def _ratio(value):
    return "N/A" if value is None else f"{value:.2f}"


def _performance_figure(performance, ticker, benchmark):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=performance.index,
            y=performance["Strategy Equity"] * 100,
            name="Strategy",
            line={"color": "#6b9ff8", "width": 2.2},
            hovertemplate="%{x|%Y-%m-%d}<br>Strategy: $%{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=performance.index,
            y=performance["Benchmark Equity"] * 100,
            name=benchmark,
            line={"color": "#8a96a8", "width": 1.5, "dash": "dot"},
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{escape(benchmark)}: $%{{y:,.2f}}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": f"Growth of $100 · {escape(ticker)} strategy vs {escape(benchmark)}", "x": 0},
        height=430,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1320",
        font={"color": "#aeb8c6", "size": 12},
        legend={"orientation": "h", "y": 1.05, "x": 1, "xanchor": "right"},
        xaxis={"gridcolor": "#1f2b3d"},
        yaxis={"title": "Value", "tickprefix": "$", "gridcolor": "#1f2b3d"},
        hovermode="x unified",
    )
    return figure


def _drawdown_figure(performance):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=performance.index,
            y=performance["Strategy Drawdown"] * 100,
            fill="tozeroy",
            line={"color": "#d97757", "width": 1.2},
            fillcolor="rgba(217, 119, 87, 0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": "Strategy drawdown", "x": 0},
        height=260,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1320",
        font={"color": "#aeb8c6", "size": 12},
        showlegend=False,
        xaxis={"gridcolor": "#1f2b3d"},
        yaxis={"title": "Drawdown", "ticksuffix": "%", "gridcolor": "#1f2b3d"},
    )
    return figure


def render_backtesting(context):
    ticker = context["ticker"]
    _render_styles()
    st.subheader("Strategy backtest")
    st.caption(
        "Price-only strategies with next-period execution, explicit trading costs, and benchmark comparison."
    )

    control_left, control_middle, control_right = st.columns([1.4, 0.8, 0.8])
    with control_left:
        strategy = st.selectbox(
            "Strategy",
            SUPPORTED_STRATEGIES,
            key=f"bt_strategy_{ticker}",
        )
    with control_middle:
        period_label = st.selectbox(
            "History",
            tuple(PERIOD_OPTIONS),
            index=2,
            key=f"bt_period_{ticker}",
        )
    with control_right:
        benchmark = st.text_input(
            "Benchmark",
            value="SPY",
            key=f"bt_benchmark_{ticker}",
        ).strip().upper()

    with st.expander("Strategy and execution settings · optional"):
        strategy_column, execution_column, evaluation_column = st.columns(3)
        with strategy_column:
            fast_window = 50
            slow_window = 200
            momentum_lookback = 126
            momentum_threshold = 0.0
            rsi_window = 14
            rsi_entry = 30.0
            rsi_exit = 55.0

            if strategy == "SMA Crossover":
                fast_window = st.number_input(
                    "Fast average (days)",
                    min_value=2,
                    max_value=500,
                    value=50,
                    step=1,
                    key=f"bt_fast_{ticker}",
                )
                slow_window = st.number_input(
                    "Slow average (days)",
                    min_value=3,
                    max_value=1_000,
                    value=200,
                    step=1,
                    key=f"bt_slow_{ticker}",
                )
            elif strategy == "Price Momentum":
                momentum_lookback = st.number_input(
                    "Momentum lookback (days)",
                    min_value=2,
                    max_value=1_000,
                    value=126,
                    step=1,
                    key=f"bt_momentum_lookback_{ticker}",
                )
                momentum_threshold = st.number_input(
                    "Minimum return (%)",
                    min_value=-100.0,
                    max_value=500.0,
                    value=0.0,
                    step=0.5,
                    key=f"bt_momentum_threshold_{ticker}",
                )
            elif strategy == "RSI Mean Reversion":
                rsi_window = st.number_input(
                    "RSI window (days)",
                    min_value=2,
                    max_value=100,
                    value=14,
                    step=1,
                    key=f"bt_rsi_window_{ticker}",
                )
                rsi_entry = st.number_input(
                    "Enter below RSI",
                    min_value=0.0,
                    max_value=99.0,
                    value=30.0,
                    step=1.0,
                    key=f"bt_rsi_entry_{ticker}",
                )
                rsi_exit = st.number_input(
                    "Exit above RSI",
                    min_value=1.0,
                    max_value=100.0,
                    value=55.0,
                    step=1.0,
                    key=f"bt_rsi_exit_{ticker}",
                )

        with execution_column:
            rebalancing = st.selectbox(
                "Rebalancing",
                SUPPORTED_REBALANCING,
                key=f"bt_rebalancing_{ticker}",
            )
            position_mode = st.selectbox(
                "Position rule",
                SUPPORTED_POSITION_MODES,
                key=f"bt_position_mode_{ticker}",
            )
            transaction_cost_bps = st.number_input(
                "Transaction cost (bps)",
                min_value=0.0,
                max_value=1_000.0,
                value=5.0,
                step=1.0,
                key=f"bt_cost_{ticker}",
            )
            slippage_bps = st.number_input(
                "Slippage (bps)",
                min_value=0.0,
                max_value=1_000.0,
                value=2.0,
                step=1.0,
                key=f"bt_slippage_{ticker}",
            )

        with evaluation_column:
            evaluation_mode = st.selectbox(
                "Evaluation",
                SUPPORTED_EVALUATION_MODES,
                key=f"bt_evaluation_{ticker}",
            )
            holdout_fraction = 0.30
            if evaluation_mode == "Holdout":
                holdout_fraction = st.slider(
                    "Holdout share",
                    min_value=0.10,
                    max_value=0.80,
                    value=0.30,
                    step=0.05,
                    format="%.0f%%",
                    key=f"bt_holdout_{ticker}",
                )
            risk_free_rate = st.number_input(
                "Risk-free rate (%)",
                min_value=-50.0,
                max_value=100.0,
                value=0.0,
                step=0.25,
                key=f"bt_risk_free_{ticker}",
            )

    if not benchmark:
        st.warning("Enter a benchmark ticker.")
        return

    period = PERIOD_OPTIONS[period_label]
    try:
        asset_history = get_price_history_cached(ticker, period)
        benchmark_history = get_price_history_cached(benchmark, period)
        config = BacktestConfig(
            strategy=strategy,
            fast_window=int(fast_window),
            slow_window=int(slow_window),
            momentum_lookback=int(momentum_lookback),
            momentum_threshold=float(momentum_threshold) / 100,
            rsi_window=int(rsi_window),
            rsi_entry=float(rsi_entry),
            rsi_exit=float(rsi_exit),
            rebalancing=rebalancing,
            position_mode=position_mode,
            transaction_cost_bps=float(transaction_cost_bps),
            slippage_bps=float(slippage_bps),
            risk_free_rate=float(risk_free_rate) / 100,
            evaluation_mode=evaluation_mode,
            holdout_fraction=float(holdout_fraction),
        )
        result = run_backtest(
            _extract_close(asset_history, ticker),
            _extract_close(benchmark_history, benchmark),
            config,
        )
    except (TypeError, ValueError) as error:
        st.error(f"The backtest could not be completed: {error}")
        return

    strategy_metrics = result.strategy_metrics
    benchmark_metrics = result.benchmark_metrics
    st.markdown(
        f"""
        <div class="bt-summary">
            <div class="bt-summary-item">
                <span>Strategy return</span>
                <strong>{_percent(strategy_metrics.total_return)}</strong>
                <small>{escape(strategy)} · since {result.evaluation_start:%Y-%m-%d}</small>
            </div>
            <div class="bt-summary-item">
                <span>{escape(benchmark)} return</span>
                <strong>{_percent(benchmark_metrics.total_return)}</strong>
                <small>Buy-and-hold benchmark</small>
            </div>
            <div class="bt-summary-item">
                <span>CAGR</span>
                <strong>{_percent(strategy_metrics.cagr)}</strong>
                <small>Benchmark {_percent(benchmark_metrics.cagr)}</small>
            </div>
            <div class="bt-summary-item">
                <span>Maximum drawdown</span>
                <strong>{_percent(strategy_metrics.maximum_drawdown)}</strong>
                <small>Benchmark {_percent(benchmark_metrics.maximum_drawdown)}</small>
            </div>
            <div class="bt-summary-item">
                <span>Sharpe ratio</span>
                <strong>{_ratio(strategy_metrics.sharpe_ratio)}</strong>
                <small>Sortino {_ratio(strategy_metrics.sortino_ratio)}</small>
            </div>
            <div class="bt-summary-item">
                <span>Annualized volatility</span>
                <strong>{_percent(strategy_metrics.annualized_volatility)}</strong>
                <small>Benchmark {_percent(benchmark_metrics.annualized_volatility)}</small>
            </div>
            <div class="bt-summary-item">
                <span>Turnover / year</span>
                <strong>{result.annualized_turnover:.1f}×</strong>
                <small>{result.trade_count} trade segments</small>
            </div>
            <div class="bt-summary-item">
                <span>Market exposure</span>
                <strong>{result.market_exposure:.1%}</strong>
                <small>Win rate {_percent(result.win_rate)}</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        _performance_figure(result.performance, ticker, benchmark),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        _drawdown_figure(result.performance),
        width="stretch",
        config={"displayModeBar": False},
    )

    for warning in result.warnings:
        st.info(warning)

    stability = build_stability_table(result)
    with st.expander("Period stability"):
        st.caption(
            "Fixed-rule returns across sequential periods. This is a stability check, not parameter optimization."
        )
        if stability.empty:
            st.info("Not enough observations for a stability table.")
        else:
            st.dataframe(
                stability,
                hide_index=True,
                width="stretch",
                column_config={
                    "Start": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "End": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Strategy Return": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Benchmark Return": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Active Return": st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )

    with st.expander(f"Trade ledger ({len(result.trades)})"):
        if result.trades.empty:
            st.info("No trades were generated.")
        else:
            st.dataframe(
                result.trades,
                hide_index=True,
                width="stretch",
                column_config={
                    "Entry Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Exit Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Entry Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Exit Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Net Return": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )
            st.download_button(
                "Download trade ledger CSV",
                data=result.trades.to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_{strategy.lower().replace(' ', '_')}_trades.csv",
                mime="text/csv",
                width="stretch",
            )

    performance_export = result.performance.reset_index(names="Date")
    st.download_button(
        "Download performance CSV",
        data=performance_export.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_{strategy.lower().replace(' ', '_')}_backtest.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Methodology and bias controls"):
        st.markdown(
            "Signals are calculated from each closing price and shifted one trading period before returns are applied. "
            "Transaction costs and slippage are charged whenever the position changes. The benchmark uses the same aligned dates. "
            "This version uses historical prices only; it does not use restated fundamentals. A single current ticker can still carry "
            "survivorship and selection bias, and fixed-rule results do not replace walk-forward parameter testing."
        )
    st.caption("Educational historical analysis only. Backtest results are not investment recommendations.")
