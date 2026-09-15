# Phase 11: Stock Selection Workspace

## Product workflow

The default app opens **Stock selection**, not an immediate quote request. Three views share one frozen scan: **Screen**, **Research**, and **Paper & results**. Legacy company tools and the execution/walk-forward labs remain in sidebar groups. No valuation or prior strategy engine is replaced. The Docker image now includes `pages/` and `components/`; earlier images did not include the multipage files.

## Screening contract

The starting ticker list is an illustrative, user-editable current universe, not an investment recommendation or historical constituent database. Choose 2–25 distinct ticker symbols. A manual scan fetches bounded daily histories with up to four concurrent requests. Prices exclude the entire current New York calendar date; requests are cached for 15 minutes. Failures are listed, never replaced by demo prices.

SPY supplies reference session labels. The latest reference date must be 1–7 calendar days old. Each ranked name needs 253 consecutive observations matching the reference dates, positive finite prices, a latest reported close at least $5, a 20-session average of reported close times volume at least $2 million, and no zero-volume session in the latest 20. Thresholds are disclosed and editable. Invalid, missing, stale and misaligned histories are explicitly excluded. Exchange-calendar completeness and instrument classification are not independently verified; this is not a licensed point-in-time universe.

The score is only `AdjustedClose[t-21] / AdjustedClose[t-252] - 1`. This is a daily-session approximation inspired by the prior-months-2-through-12 convention, not a replication of the Fama–French portfolios. Higher values rank first. Ticker order breaks exact ties; tied values have the same percentile. Percentiles are relative to eligible names, not probabilities or calibrated expected returns. Rank changes appear only when the universe, rules, data kind and version match the preceding scan.

OHLC adjustment is explicit: the provider's reported OHLC is multiplied by `Adj Close / Close`. Reported Close and Volume are used for the liquidity screen. Scanner records retain data hashes, formula, rules, full universe, exclusions, ranks, source and timestamps. They do not retain or publish full vendor price histories.

## Connected research

Selecting a candidate carries its ticker and frozen rank into Research. Price context, recent returns and historical correlations come from the same in-memory batch; after a browser reload, the saved rank remains available but the chart requires another scan. Current company context is fetched only on request. Company summary data is identified as current, not historically available fundamental signal data. Unverified earnings dates are not invented. The DCF and Monte Carlo views reuse existing modules for the selected company. Financial businesses remain blocked from the existing UFCF model. Valuation inputs do not change the momentum ranking. Monte Carlo varies assumptions; it is not a probability of profitable trading.

## Private persistence

A small dependency-free Streamlit component stores the event record in browser localStorage under a stable application key, separately on each origin. Web Locks plus expected revisions guard against cross-tab lost updates. A save is acknowledged only after successful storage. Storage denial, quota exhaustion and conflicts are visible. Nothing is written to a shared repository file, a shared server database or a broker.

The record survives reloads in the same browser/origin but is **not cloud-synced**. Incognito mode, clearing site data, another device or another preview origin can make records unavailable. JSON backup/export is built in. Import validates the schema and hash chain and works only into an empty record; existing records are not overwritten. Limits are 200 events and 2 MB. Export backups for retention.

Hash-linked events detect accidental corruption, not deliberate editing and re-hashing. Server-created timestamps stored in a user-editable browser are not independently attested preregistration. JavaScript-compatible numeric normalization prevents the JSON roundtrip from changing hashes solely because `1.0` becomes `1`.

## Paired paper cohorts

A saved scan can create one immutable paired decision. The model selects the first five eligible names, or all eligible names when fewer than five. It divides initial virtual capital equally across those slots. User selections can use the same or fewer slots; unused user slots stay in non-interest-bearing cash. A same-universe equal-weight comparator and an SPY reference are recorded separately. User/model differences include exposure changes, not just stock-selection skill.

For market data, the earliest entry date is the calendar date **after registration in New York**, even if the decision is recorded before today's open. This deliberately conservative rule avoids assigning prices from before registration. The first subsequent reference session supplies an open-fill proxy. Decisions do not generate a broker order. Every frozen member must have the required outcome sessions; missing names, dates or zero-volume sessions block the entire evaluation rather than drop losing assets.

Each cohort covers 21 reference sessions. The simulator uses fractional adjusted research units, 5 basis points commission and 10 basis points adverse slippage per side. Entry budgets include costs. In-progress positions remain marked; hypothetical liquidation equity is separate. On the predefined 21st session, a closing-price liquidation proxy includes exit costs. Outcomes retain session-specific data hashes. Saved observations can advance, not overwrite earlier observations; a completed outcome remains frozen.

**These are independent cohorts, not an automatically rebalanced continuous account.** There is no capital carry between cohorts, automated monthly rebalance, broker integration, unattended recording, tax or settlement model. Do not add overlapping cohort returns. Whole-share affordability is not established by fractional adjusted-unit simulation. Small-account virtual capital is an editable simulation scale, not a deployment recommendation.

Results include model/user/comparator returns, fees, modeled slippage, equity and initial-capital-inclusive drawdown. Completed cohorts report Spearman rank correlation between the frozen momentum score and subsequent open-to-close return when at least three nonconstant observations exist. One small, correlated cross-section is not significance evidence. No formal multiple-testing control or historical-universe validation is claimed.

## Demonstration versus evidence

The synthetic demo creates a prefix with no future rows and exposes separate generated future rows only when the paper evaluation is requested. Its fictional ticker names, hypothetical decision date and outcomes stay explicitly labeled. Demo and market records cannot be mixed. A browser workflow pass demonstrates software behavior, not market performance.

## Verification

Run `python -m pytest tests -q`. The workspace workflow runs the full suite, an optional bounded current-data provider probe, a real-browser sequence against the app entrypoint, and a JSON backup readback. Browser checks cover scan, linked research, freezing both books, outcome evaluation, reload restoration, export, and separate-browser isolation. The provider probe's status is retained independently; provider errors do not become synthetic successes.

Inspect the actual CI artifact and current commit checks. A workflow definition is not a passing result. CI-local browser verification does not prove access to the protected Vercel preview.

## Primary references

- Kenneth French prior 2–12 momentum convention: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_10_port_form_pr_12_2.html
- yfinance history adjustment and exclusive end date: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
- Streamlit custom components: https://docs.streamlit.io/develop/api-reference/custom-components
