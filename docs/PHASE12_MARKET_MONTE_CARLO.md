# Phase 12: market Monte Carlo, separate from DCF

## Scope and navigation

`Workspace > Market Monte Carlo` is a fundamentals-independent research page at
`/market-monte-carlo`. Company tools also has a Market Monte Carlo tab. Neither
entry point feeds revenue, margins, WACC, terminal growth, or DCF outputs into
market paths.

The original `Utils/monte_carlo.py` and `Utils/monte_carlo_ui.py` remain unchanged
for backward compatibility. Their DCF simulation is available only inside the
DCF valuation section, under an explicit enable control. It retains its own
baseline assumptions and does **not** inherit manual DCF edits above it. The
interface states this explicitly. Corporate DCF is gated to identified equities;
funds are not treated as operating businesses.

## What the market output means

The engine samples future log-return sequences and compounds them into positive
paths. It is a conditional scenario model, **not intrinsic value, a calibrated
probability oracle, a trading recommendation, or risk-neutral option pricing**.

Training uses Yahoo's split/dividend-adjusted closes. Displayed dollar paths are
those adjusted-return paths rebased at the most recent reported close. Therefore
they are **price equivalents**, not literal future unadjusted exchange quotes.
This distinction is visible above results and in every export. Daily sampling
does not estimate intraday barrier crossings or executable order fills.

The current NY calendar day is excluded in its entirety, even after market close.
The as-of session, anchor, requested dates, provider version, return basis, exact
history hash, RNG, parameters, and dependency versions are recorded. No live quote
is spliced onto an older training history.

## Models

All models share an explicitly chosen mean daily log return `m`. Available policies:

- **Zero**: `m = 0`, the default risk baseline. This is not a zero expected arithmetic return.
- **Shrunk historical**: `m = weight * mean(observed log returns)`. The default weight
  is 0.25, an exposed prior, not a fitted or validated forecast.
- **Manual**: `m = annual_log_drift / 252`. The input is an annual log mean, not a target CAGR.

### Historical blocks (default)

Stationary bootstrap with geometric block lengths: begin at a uniform historical
index; at each subsequent step restart uniformly with probability `1 / block_length`,
or advance to the next return with circular wraparound. Center historical returns
at their observed mean, then add `m`. This preserves some dependence within blocks
and empirical tails, but cannot generate an unseen one-day shock. Expected block
length defaults to 10 sessions and is not automatically optimized.

### Volatility-adaptive bootstrap

Filter centered returns using `v_next = lambda * v + (1-lambda) * shock^2`.
Initialize with 20 training observations (use whole-training variance if that
warm-up is flat), exclude those warm-up innovations, then center and scale the
remaining standardized innovations to unit second moment. Sample those innovations
and evolve a separate EWMA variance for each simulated path. Decay defaults to 0.94.
This is a filtered historical/EWMA scenario model, not a fitted GARCH or a learned
regime-switching model.

### Gaussian / GBM baseline

For each day, `log(S_next / S) = m + sample_std(log returns) * Z`, where `Z` is
standard normal. This is a transparent constant-volatility baseline. Because drift
is specified as a **log** mean, subtracting `sigma^2/2` again would be incorrect.
The implied continuous price drift is `252*m + annual_variance/2`.

Model comparison holds data, horizon, path count, seed, and drift policy fixed.
It does not select a winner or average models into a supposedly validated forecast.

## Risk statistics and bounds

Default: 5,000 paths, 63 trading steps, PCG64 seed 42. UI horizons are 21, 63, 126,
and 252 sessions; the engine accepts 1–252. Maximum 20,000 paths bounds allocations.
All paths include time zero. Underflow/overflow aborts the run; returns and paths
are never clipped, discarded, or replaced to obtain a more attractive distribution.

Results include terminal median/mean and P5/P25/P75/P95, pointwise bands, simulated
gain/loss frequency, terminal loss VaR and expected shortfall, and per-path maximum
drawdown. ES integrates exactly the worst 5% empirical probability mass, including a
fractional boundary observation. Loss metrics are signed: negative means a modeled
gain. Level-touch and terminal-beyond-level frequencies are separate. Touches are
monitored only at simulated daily closes, including time zero.

The 90% pointwise band is **not** a simultaneous 90% confidence region for the whole
path. Frequencies have simulation noise as well as much larger possible model
error. Zero simulated events do not establish zero real-world risk.

## Data gates

Only identity-matched, USD equities/ETFs with US Eastern sessions are supported.
A recent listing without 126 returns is rejected, not backfilled or substituted.
Both reported and adjusted Close must be present. No NaN filling, sorting,
deduplication, stale-price substitution, ticker guessing, or corporate-action repair.
Dates must be unique, ordered daily weekday labels. Predominantly non-daily data,
gaps longer than seven calendar days, and a last bar over seven days old are rejected.

Exchange-calendar completeness, corporate actions, and point-in-time provenance
are **not independently certified**. Short gaps can escape detection. These limits
are exposed rather than represented as clean/audited data. A historical move larger
than 50% in multiplicative magnitude gets an explicit review warning, not winsorization.

## Walk-forward coverage diagnostic

Each origin fits only its preceding 504 returns. Evaluate the next chosen horizon
on observations excluded from that fit. Forward evaluation windows do not overlap;
training windows can overlap. Require at least three complete folds, take at most
12 recent folds, and cap each fold at 2,000 paths. Store training and target dates,
training hash, fold seed, realized return, coverage, and the central-90% interval score.

The score is interval width plus `20 * distance outside the interval`; lower is
better only for comparable horizons and data. No automatic parameter selection or
model selection uses these holdouts. Small coverage counts are not proof of
calibration, and trying many settings after inspecting them creates selection bias.
Provider-adjusted history may be revised. This is not a profitability backtest.

## UI and evidence

No provider request or simulation occurs on initial render. Form submission runs
explicitly and clears any old result first, so a failed request cannot display an
old run as its answer. Synthetic demo data is opt-in and conspicuously labeled;
it is never used as a fallback for a failed live request.

Exports contain exact training history, JSON metadata/configuration, every terminal
outcome, pointwise bands, first 100 full paths, and any completed coverage diagnostic.
For exact CSV recovery use pandas `float_precision="round_trip"`. Reproduction also
requires the recorded engine and dependency versions; a seed alone is insufficient.

## Tests

New tests cover configuration/data rejection, seeded determinism, positive paths,
analytical GBM moments/quantiles, log-drift compounding, block dependence, EWMA regime
response, no silent tail clipping, signed VaR/ES and fractional tail mass, drawdowns,
daily barrier semantics, non-overlapping forward windows, future-data leakage,
bundle replay and mismatched-snapshot rejection. Streamlit AppTest exercises initial
render, synthetic runs, comparison, coverage, export, and failed-request state clearing.
Existing DCF, legacy Monte Carlo, backtesting, and workspace tests remain in the suite.

## Limitations and intentionally deferred work

No full parameter-uncertainty integration, fundamental/news conditioning, future
corporate-action schedule, calibrated jumps/default events, multivariate portfolio
correlations, trading costs, or automated execution. No fitted Student-t or GARCH
model is claimed. These need explicit model design and out-of-sample evidence, not
an unsupported accuracy score.

## Method references

- NumPy Generator/PCG64: https://numpy.org/doc/stable/reference/random/generator.html
- Stationary bootstrap (Politis–Romano), official arch documentation:
  https://arch.readthedocs.io/en/stable/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html
- Streamlit forms and explicit execution:
  https://docs.streamlit.io/develop/concepts/architecture/forms
