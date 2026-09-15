# Phase 10: Daily walk-forward research

## Scope and separation

The **Walk Forward** sidebar page adds deterministic daily research baselines, a historical-data adapter, chronological candidate selection, and a separately evaluated final holdout. It depends on Phase 9's branch but does not modify its engine or any existing DCF, Monte Carlo, or Backtest calculations. No broker integration, live orders, subscriptions, leverage, or personal capital settings are added.

This is not a reconstruction of a discretionary minute-level strategy. No source strategy's undefined indicator settings, screening rules, or risk parameters are silently invented. Minute-level strategies require independently specified rules and appropriately timestamped historical minute data, including the historical eligible universe.

## Fixed candidate registry

The default registry is declared before evaluation and includes all seven candidates in every training comparison:

| Candidate | Close-of-session rule |
| --- | --- |
| Cash | Always uninvested, earning zero interest in this model. |
| SMA 20/100 | Long when the trailing 20-session mean exceeds the 100-session mean. |
| SMA 50/200 | Long when the trailing 50-session mean exceeds the 200-session mean. |
| Momentum 63 | Long when close / close 63 sessions earlier is above 1. |
| Momentum 126 | Same rule over 126 sessions. |
| Momentum 252 | Same rule over 252 sessions. |
| RSI 14 | Enter at RSI <= 30, exit at RSI >= 55, otherwise retain the signal. |

RSI uses gain/loss exponential means, alpha = 1 / window, adjust=False, with a full-window warmup. It is an explicitly defined EWM variant, not a claim of identical output to every chart package's Wilder initialization. Rising-only histories map to 100, falling-only histories to 0, and flat histories to 50. Signals remain cash before enough observations exist. Each eligible signal is lagged one full session before execution at the next open.

## Research protocol

Defaults are 504 initial training sessions, 126-session test windows, a one-session gap, and 252 reserved final sessions. All candidate training metrics start after the same 253-session common warmup. Candidates are ranked by net training daily Sharpe, assuming a zero risk-free rate; cash receives score 0, undefined non-cash Sharpe is ineligible, and ties resolve by rule label. These are software defaults, not empirically optimized recommendations.

Each walk-forward choice sees only its training prefix. Gap rows may update indicators once their closes are known, but are excluded from that selection's training metrics. Earlier validation periods can enter later expanding training sets after they become history. Test windows do not overlap; the final development fold may be partial. Cash and holdings carry across folds in one continuous out-of-sample simulation, without free resets or repeated starting-capital injections.

The final rule is selected on the full development prefix, excluding the pre-holdout gap. `develop()` does not compute holdout performance. It returns a plan containing the complete configuration, candidate set, training scores, chosen rule, boundaries, source manifest, dataset hash, and plan hash. `evaluate_holdout()` verifies this plan and snapshot and evaluates the fixed rule without any ranking or refitting. The holdout begins a separate virtual account; its returns are not stitched onto development returns.

A plan hash is a reproducibility check, not authenticated preregistration. The interface records repeated experiments and prior holdout exposure only within the current session. Resetting the browser/server or examining history elsewhere is not prevented. Once viewed and used to change the model, the same holdout is no longer untouched. Historical hindsight, asset selection, repeated experiments, and multiple testing still require independent controls. There is no significance threshold or automatic profitability certification.

## Data contract and adjusted-unit accounting

Historical downloads explicitly request daily auto-adjusted OHLC through yfinance, keep missing values visible, disable repair, and use an exclusive end date. Requests occur only on user action. Failures remain failures; the loader never falls back to synthetic data. Record retrieval time, provider version, requested and actual dates, row count, adjustment basis, source timezone, and SHA-256 of normalized OHLCV.

For uploads, use `Date,Open,High,Low,Close,Volume`, one asset, YYYY-MM-DD session dates, at most 8,000 observations / 10 MiB, and consistently adjusted OHLC. Do not supply a Signal column: the engine generates its own signals. Dates are session labels rather than real midnight executions. Duplicate, unsorted, intraday, nonfinite, nonpositive-price, and malformed-OHLC inputs are rejected, not repaired. Sparse and zero-volume histories are disclosed. Exchange-calendar completeness, vendor accuracy, delistings, and historical universe membership are not independently verified.

Retrospectively adjusted prices must not be treated as actual historical share prices for whole-share affordability. This engine uses **fractional adjusted research units**. It models next-open entry/exit with per-side commissions and adverse slippage; allocation is applied on entry, without daily rebalancing. Cash cannot be borrowed. Zero-volume sessions do not fill. No volume capacity, quote/depth, settlement, taxes, cash interest, intraday stops, or market-impact calibration is claimed. Adjustments represent corporate actions only through the supplied adjusted price history; no separate dividends are credited.

Equity includes open positions marked at the close. Hypothetical liquidation equity is separate and includes assumed exit costs. Closed-position PnL plus open-position PnL reconciles to equity minus starting capital. Maximum drawdown includes initial capital as the first peak. Annualization assumes 252 sessions per year. The passive comparison uses the same asset, allocation-on-entry, initial equity, dates, and cost assumptions, not a cross-asset market-alpha estimate.

Cost stress uses 1x, 2x, and 4x slippage with the frozen signals unchanged. It does not reselect a better-looking rule for each stress scenario. Charts, scores, order ledgers, the frozen plan, and experiment reports are exportable. Full snapshot export becomes available after opening the holdout; provider data redistribution rights are not granted by this app.

## Verification and evidence

Run `python -m pytest tests -q`. Tests cover signal prefix invariance, warmup/RSI edges, holdout-mutation isolation, earlier-fold isolation, chronology and gaps, accounting and compounding, next-open gap treatment, frozen-plan/data modification, no refitting in holdout, provider failures, and interface state invalidation. UI tests also check navigation from the actual app when the main market-data context is unavailable.

The Phase 10 evidence workflow runs the full suite, produces JUnit results, attempts predeclared historical exercises on SPY/QQQ/IWM for 2010-01-01 through exclusive 2026-01-01, and runs an actual browser against a CI-local Streamlit server using synthetic data. Example tickers are test datasets, not recommendations. Historical failures are explicitly reported and never replaced by demonstration prices. The workflow retains aggregate research reports and browser evidence for seven days; it does not publish raw price files. A browser pass on that server is not verification of the protected Vercel deployment. Consult actual run outcomes rather than treating a workflow definition as a passing result.

## Remaining research gates

No profitable edge is certified. Outstanding work includes independently validated point-in-time data, a precise minute-level strategy specification when applicable, calibrated execution, formal multiple-testing controls, durable preregistration, and an auditable forward paper record. Promotion to live trading is not automated.

## Primary technical references

- yfinance history contract and adjustment options: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
- yfinance download interval and exclusive-end semantics: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- Chronological train/test gaps: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Streamlit AppTest and multipage navigation: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
