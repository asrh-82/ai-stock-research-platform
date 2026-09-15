# Phase 9: Validation Lab

## Scope

This phase adds an independent Streamlit page and a deterministic long/cash OHLC replay engine. It does not change the existing Dashboard, DCF, Monte Carlo, or Backtest calculations. It has no broker connection, live orders, paid data subscription, or automatic capital deployment.

Open **Validation Lab** in the sidebar. Enable the explicitly labeled synthetic demonstration to inspect the interface, or upload your own CSV. Starting cash is virtual and editable. Default capital and cost settings are generic software examples, not allocation recommendations. Uploaded data and capital settings are not written to repository files by this code; exports are user-initiated downloads.

## Input contract

Required columns: `Timestamp,Open,High,Low,Close,Volume,Signal`.

- One asset per file; `Symbol` or `Ticker`, when present, must contain exactly one value.
- UTF-8 CSV, no duplicate headers, at most 10 MiB and 100,000 rows.
- ISO 8601 timestamps must include `Z` or a UTC offset. CSV timestamps normalize to UTC.
- Each timestamp labels the **bar open**. Signal is known only after that bar closes.
- Signal is desired long/cash exposure: `1` or `0`. It is not an order-event flag.
- Prices must be finite and positive, OHLC bounds consistent, and volume nonnegative.
- Timestamps must be sorted and unique. Missing values are rejected rather than silently repaired.
- Supply consistently adjusted OHLC and volume. Corporate actions and dividends are not separately modeled.

A daily input file cannot substantiate a strategy that requires minute-level observations. Importing a signal column does not prove the signals were generated without future information.

## Execution and accounting

A signal from bar t first becomes eligible at the open of bar t + delay, where delay is at least one bar. Entry sizing includes commission and adverse slippage, uses whole shares, and cannot borrow. Entry capacity is capped by previous-bar volume. Zero-volume bars cannot fill entries or exits; nonzero-volume exits assume full execution and have no depth model.

Stops and targets are simplified price-triggered market-fill proxies, **not** exact limit-order simulations. They activate immediately after entry. Known opening gaps take precedence over unknown intrabar paths. When both levels are touched in the same bar, the simulator takes the stop first. It does not re-enter in that same bar. A continuing long target can cause re-entry on a later bar.

The drawdown rule observes bar-close equity and queues an exit for the next tradable open. It is not a guaranteed maximum loss. Equity and drawdown include initial capital as the starting peak. Open positions remain marked rather than being silently closed; hypothetical liquidation equity is reported separately. Closed-trade PnL plus open-position PnL reconciles to equity minus initial capital.

## Research output

The page shows virtual equity, net return, bar-close maximum drawdown, completed trades, entry rejections, ambiguous bars, fee totals, and modeled slippage. Comparisons are same-sizing passive exposure in the same asset and non-interest-bearing cash, not a diversified total-market benchmark. Cost stress multiplies slippage by 1x, 2x, and 4x while holding commission fixed.

Exports include the complete equity ledger, closed-trade ledger, and a JSON report with engine version, input-data hash, configuration, timestamp semantics, and limitations. Chart sampling does not change replay or export calculations.

Circular block resampling operates on supplied daily net returns and has a fixed seed and memory cap. Its results describe resampling uncertainty conditional on the observed dataset, not future-return probabilities. A chronological expanding-fold helper provides train/test intervals with a gap; it does **not** fit models, tune parameters, purge labels automatically, or establish out-of-sample validity.

## Validation status

The page deliberately reports **NOT VALIDATED**. Software correctness is not trading evidence. Missing evidence includes point-in-time universe membership, delistings, verifiable signal provenance, calibrated execution costs, untouched holdout performance, and a forward paper record.

Local checks in the implementation environment: 71 engine/CSV tests passed; all new Python files compiled. The Streamlit UI test module was skipped locally because Streamlit was unavailable. The accompanying CI workflow installs the application dependencies and runs the full test suite, including UI smoke tests. Check the actual run status; a workflow definition is not a passing result.

Standalone UI tests do not constitute full browser/navigation verification. A Vercel READY status also does not establish successful interactive execution.

## Subsequent phases, not implemented here

1. Formalize a deterministic strategy specification and acquire licensed, timestamped data with appropriate history and universe coverage.
2. Add automated walk-forward fitting, untouched holdouts, comparisons against appropriate benchmarks, and multiple-testing controls.
3. Add an auditable forward paper ledger and execution-quality reconciliation. Live deployment remains a separate decision, not an automatic promotion after a green test.
