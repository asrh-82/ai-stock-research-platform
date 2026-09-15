# Phase 13: opportunity research, not neutral risk cones

## Product contract

The application opens on **Opportunities**, a bounded scanner of up to twelve prespecified US equity/ETF tickers. Market Monte Carlo remains under Research methods as a risk tool. DCF and DCF assumption uncertainty remain separate and unchanged.

This is a **price-based conditional return model**, not fundamental fair value, a market-implied required-return calculation, or proof of economic mispricing. Excess return means asset total return minus the selected benchmark's total return. It is not beta-adjusted alpha. SPY is the default comparator, not necessarily the right economic comparator for every asset (particularly a commodity fund).

No zero or historical drift is inserted into opportunity forecasts. A regularized regression learns coefficients and an intercept from matured historical outcomes. A positive number alone never qualifies an opportunity. There are no orders, account connections, position sizing, or automatic trades.

## Target and information timing

At observation t, features use adjusted closes through t. The hypothetical entry is the **next session's close t+1**. The exit is t+1+h, where h is 21 (default) or 63 sessions. The entry delay avoids claiming execution at a close used to construct the signal. The target is therefore a holding-period return from an unknown future entry close, not a dollar price target anchored at the current quote.

For every historical fit, a label is admissible only if its exit is **strictly before** the forecast origin. Training data with unfinished/overlapping future labels are purged. Scalers and coefficients are fitted again inside each fold. No random train/test shuffle or current fundamentals backfill is used.

Daily training targets can overlap; their count is not an independent sample count. Evaluation origins are spaced h+1 sessions apart, producing non-overlapping target windows. Non-overlap does not imply independence.

## Fixed first specification

Nine past-only features: relative 21/63-session momentum; 126-session relative momentum skipping the latest 21; five-session relative reversal; relative 63-session moving-average distance; 21-session volatility; 21/126 volatility change; benchmark 63-session momentum; relative 126-session drawdown.

Two arithmetic-return targets are fitted jointly: asset total return and asset-minus-benchmark excess return. Each target's intercept is its training mean. Train-standardized features use the objective mean squared error + 0.1 times the squared coefficient norm; the intercept is not penalized. No hyperparameter is selected on the displayed evaluation outcomes. At least 504 matured training labels are required; the most recent 1,008 are used when available. The initial 126 sessions provide feature warm-up.

Price-derived inputs are a starting specification, not evidence that momentum/reversal works for any given ticker. Trying multiple horizons, benchmarks, histories or universes after seeing results introduces researcher selection bias. These choices must eventually be locked for prospective paper validation.

## Opportunity and evidence gates

The predeclared hypothetical long-selection rule requires a positive total-return estimate after costs and an excess-return estimate greater than costs plus the extra hurdle. Defaults: 20 bps cost deduction from entry capital and 100 bps excess-return hurdle. Costs are assumed, not estimated actual fills; benchmark costs are assumed zero. No short signals are implemented.

Candidate eligibility additionally requires:

- At least 24 non-overlapping OOS targets and 12 prior observations selected by that rule.
- Positive lower diagnostic bounds for mean squared-error improvement against BOTH zero-excess and rolling historical-mean excess forecasts, evaluated on the same dates.
- A positive lower diagnostic bound for mean net excess return on prior rule-selected observations.
- Positive forecast skill versus zero excess in the most recent half of the OOS ledger.
- No current standardized feature beyond six training standard deviations; no invalid implied asset/benchmark returns.

Bounds use 5,000 moving-block resamples of chronological OOS folds with block length three. The one-sided tail threshold is (1-confidence)/(3 * requested-family-size), using default confidence 0.95. Failed tickers remain in the prespecified family size. The adjustment addresses three diagnostic gates across the CURRENT requested list only. It is not a remedy for prior scans, survivor selection, data revisions, model construction choices or all multiple testing. These small-sample bootstrap diagnostics are exploratory, not certified confidence statements or a profitability backtest of the complete gated scanner.

A selection-free bootstrap replicate gets an adverse infinite lower payoff, not silent deletion. Signal counts, raw error improvements and gate failures are visible. Only assets clearing every gate appear in the candidate table. Ranking within that table uses net predicted excess return divided by historical OOS excess forecast RMSE. Other estimates retain **Unvalidated estimate** or **No qualifying gap** labels; data/model failures stay visible.

## Forecast uncertainty and Monte Carlo

Terminal scenarios are the current learned pair of return forecasts plus a resampled pair of historical **OOS prediction errors**. Error pairs preserve total/excess relationships and their empirical bias; they are not centered to make a forecast look better. In-sample residuals are not used. The resulting interval can be shifted from the regression point estimate.

The actual number of held-out errors appears next to the 5,000 simulation count. Simulations do not multiply evidence. These are empirical error-conditioned terminal outcomes, not market price paths, a fundamental valuation distribution, full parameter uncertainty, or calibrated real-world probabilities. If resampling implies an asset or benchmark return at/below -100%, the distribution is withheld, not clipped, and candidacy is blocked. No intervals are shown before 24 OOS targets.

The older price-path Monte Carlo remains available for separate volatility/drawdown research. This implementation deliberately does not convert the total-return point estimate into an arbitrary daily drift and call the resulting cone an independently validated forecast.

## Data and reproducibility

The existing `load_market_snapshot` handles identity, supported asset/currency, date, freshness and missing-value checks. The full current New York day is excluded. Each scan captures a single cutoff and loads the benchmark once. Within the common listed history, asset and benchmark dates must match exactly with the same final session. Internal intersections/fills/repairs are prohibited. Rows preceding the later first observation are excluded and their counts recorded.

History remains provider-adjusted/revised, not independently point-in-time certified. A selected current universe is susceptible to survivorship bias; no market-wide validation claim is made. Short histories, including recent listings, cannot be backfilled by substituting another ticker.

An explicit submission clears any prior result before the request, so failures cannot leave old answers displayed as new. Exports include every requested asset and failure, exact aligned history with round-trip float recovery, config/dependency versions, fitted model, feature contributions, complete OOS ledger and conditional terminal scenarios. A history digest mismatch aborts export.

## Verification and evidence

`tests/test_opportunities.py` tests timing, purging, scaler isolation, synthetic learnability, unchanged forecasts across cost assumptions, reproducibility, uncertainty provenance, gate success/failure and archive replay. `tests/test_opportunity_scan.py` tests identity/alignment, consistent cutoffs, benchmark reuse, failed-request retention and family size. `tests/test_opportunity_ui.py` contains offline Streamlit tests; its fixtures are not market evidence.

`scripts/run_opportunity_evidence.py` executes the actual modules against live data for a prespecified universe. Its archive records the code commit, inputs, parameters, outputs and failures. Automated tests establish tested implementation behavior, not market predictive advantage. The live runner is read-only and does not alter main.

## Method references

- Training-only rolling-origin evaluation: https://otexts.com/fpp3/tscv.html
- Temporal splits and gaps: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Ridge objective (our implementation scales the penalty against mean squared error): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

## Deliberately not claimed

No fundamental/news/earnings conditioning, market-implied valuation gap, fitted transaction-cost model, market-wide alpha proof, learned regime changes, portfolio optimization, fully calibrated error distributions, or validated profits. Those require additional data and evidence rather than changing a label.
