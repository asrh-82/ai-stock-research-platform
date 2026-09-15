# AI Stock Research Platform

A quant-assisted equity research workspace built with Python and Streamlit. Start with a transparent cross-sectional ranking, inspect the selected company, and record paired model-versus-user paper decisions.

## Run the workspace

```bash
git clone https://github.com/asrh-82/ai-stock-research-platform.git
cd ai-stock-research-platform
python -m pip install -r requirements.txt
streamlit run app.py
```

The default **Stock selection** workspace has three views:

| View | Workflow |
| --- | --- |
| Screen | Choose 2–25 tickers. Review momentum ranks, recent returns, volatility, drawdowns, eligibility failures, and timestamps. |
| Research | Open a ranked name with the same price context. Load current company fundamentals, DCF scenarios, or assumption uncertainty explicitly. |
| Paper & results | Freeze the model's top-five picks and your selections together. Compare their next 21-session paper outcomes with an equal-universe portfolio and SPY. |

To explore without a provider request, select **Synthetic demo**, scan, open a candidate, then freeze and update a paper example. Demo companies and outcomes are fictional and explicitly labeled.

## Quantitative baseline

The ranking uses one declared formula:

```text
AdjustedClose[t-21] / AdjustedClose[t-252] - 1
```

This is a 252/21-session approximation of prior-year momentum excluding the most recent month, not a replication of published factor portfolios. Eligibility uses aligned historical coverage, reported price, and recent dollar volume. Thresholds, exclusions, ties, and data hashes are visible. Percentiles are ranks within the chosen universe, not probabilities of profit.

Fundamentals and valuation are research context. They do not silently change the momentum score. Current company data is not inserted into historical decisions.

## Private records

Scans and decisions are saved in this browser on this deployment origin, not in a cloud account. Export the JSON backup under **Record storage and backups**. Restore is available only into an empty browser record. Clearing site data or opening another device or preview does not automatically carry records over.

Writes require browser acknowledgment and use revision checks and a browser lock. The app preserves prior events instead of editing them. Hashes detect accidental corruption, not intentional re-hashing; timestamps are not independently attested. Limits are 200 events and 2 MB per record.

## Paper evaluation scope

Paper outcomes are **independent 21-session cohorts**, not an automatically rebalanced continuous account. The model chooses up to five equally allocated names. User selections share the same allocation slots; unused slots remain cash. Market-data records begin no earlier than a session after the decision date. Missing frozen names block evaluation instead of disappearing from the results.

Accounting uses fractional adjusted research units, 5 bps commission and 10 bps adverse slippage per side. These are explicit simulation assumptions, not calibrated real-world fills. Completed cohorts include a predetermined closing-price liquidation proxy. In-progress positions remain marked with hypothetical liquidation reported separately. Overlapping cohort returns must not be added together.

No broker connection, real orders, leverage, paid subscription, or automatic capital deployment is included. A passing software test or a favorable paper outcome does not demonstrate an investment edge.

## Existing research tools

**Company tools** retains the original dashboard, company analysis, peer comparison, watchlist, DCF, Monte Carlo, and price-strategy backtests. The existing mathematical engines retain their calculations.

**Research methods** groups execution validation and walk-forward research. The latter selects predefined daily rules from earlier data and evaluates later windows and a separately frozen final holdout. It is distinct from the cross-sectional stock scanner.

DCF implies a value under stated cash-flow assumptions. Monte Carlo varies those assumptions; the fraction of simulated values above market price is not a probability of a profitable trade. The DCF's initial scope excludes banks and insurers.

## Methodology and implementation

- [Stock-selection workspace](docs/PHASE11_WORKSPACE.md): ranking, eligibility, browser persistence, paper-cohort accounting, and limitations.
- [Daily walk-forward protocol](docs/PHASE10_RESEARCH.md): training, frozen holdouts, adjusted-unit accounting, and data requirements.
- [Execution lab contract](docs/PHASE9_VALIDATION.md): whole-share OHLC replay and execution assumptions.
- [Numerical notes](docs/PHASE10_NUMERICAL_NOTES.md): provider-adjustment tolerance without rewriting price data.

```text
app.py                         Navigation and workspace entrypoint
pages/                         Workspace, legacy company tools, advanced labs
Utils/selection.py             Transparent ranking and exclusions
Utils/selection_data.py        Bounded on-demand historical data loading
Utils/selection_ui.py          Connected screen/research/paper workflow
Utils/paper_record.py           Frozen decisions, outcomes, record validation
Utils/paper_vault.py            Browser-storage acknowledgment bridge
components/paper_vault/         Dependency-free private storage component
Utils/dcf*.py                  Existing valuation calculations and interface
Utils/monte_carlo*.py           Existing assumption simulation
Utils/research_*.py             Daily data and walk-forward methodology
Utils/validation_lab.py         Whole-share execution replay
scripts/                       Provider and real-browser verification
```

## Verification

```bash
python -m pip install pytest
python -m pytest tests -q
```

A Node.js runtime is required for the browser-JSON and component-protocol regression tests. CI installs the full application and browser dependencies. The workspace workflow checks the full suite, a separately reported current-data probe, real Chromium interactions, reload persistence, and backup export/import. CI-local browser verification does not verify a protected Vercel URL. Inspect the actual workflow status and evidence artifacts, not only this description.

The Docker image includes both `pages/` and `components/`. Historical universe membership, independently checked vendor data/calendar, calibrated execution, durable attestation, continuous portfolio rebalancing, and a demonstrated trading edge remain outside this release.

## Legacy company-tool screenshots

![Company dashboard](screenshots/dashboard.png)
![Company analysis](screenshots/analysis1.png)

## Disclaimer

This project is for education, software experimentation, and research. Outputs are not investment recommendations. No trained ML stock-selection model is included in the current baseline.
