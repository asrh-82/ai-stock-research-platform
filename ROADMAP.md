# Product Roadmap

The platform is becoming a dedicated equity research workspace. Portfolio tracking is intentionally out of scope so development can focus on research depth, modeling quality, and reproducible outputs.

## Product Principles

- Separate data retrieval, financial calculations, and Streamlit rendering.
- Show data sources, dates, assumptions, and units wherever they affect a result.
- Keep every model editable and reproducible.
- Never present a model output as an investment recommendation.
- Prefer correct, testable calculations over feature count.

## Milestone 1: Financial Data Foundation

Build a normalized financial-data layer before building valuation UI.

Acceptance criteria:

- Retrieve annual income statement, balance sheet, and cash-flow data.
- Normalize line items needed for free cash flow, net debt, and diluted shares.
- Keep market data separate from financial-statement data.
- Record source, filing period, retrieval time, units, and missing fields.
- Return explicit validation errors instead of silently substituting zeros.
- Cover normalization and calculation helpers with unit tests.

## Milestone 2: Editable DCF Generator

Create a five-year unlevered free-cash-flow model for U.S.-listed, non-financial companies.

Acceptance criteria:

- Display historical revenue, operating income, taxes, D&A, capital expenditures, working capital, and free cash flow.
- Let users edit revenue growth, margins, tax rate, reinvestment assumptions, WACC, and terminal growth.
- Support base, bull, and bear cases without duplicating calculation logic.
- Calculate enterprise value, equity value, and implied value per diluted share.
- Show a WACC versus terminal-growth sensitivity table.
- Validate assumptions, including `WACC > terminal growth`.
- Export assumptions and forecast schedules to Excel.

## Milestone 3: Monte Carlo Valuation

Run the DCF engine across distributions of key assumptions rather than creating a separate valuation model.

Acceptance criteria:

- Let users define distributions for growth, margins, WACC, and terminal growth.
- Support a configurable random seed and simulation count.
- Reject economically invalid draws.
- Show the valuation distribution, key percentiles, and probability relative to the current market price.
- Allow CSV export of simulation results and inputs.

## Milestone 4: Backtesting Engine

Start with strategies that can be tested from trustworthy historical price data, then add fundamental signals only when point-in-time fundamentals are available.

Acceptance criteria:

- Ensure a signal can use only information available at its decision date.
- Support benchmark selection, rebalancing frequency, position rules, transaction costs, and slippage.
- Report cumulative return, CAGR, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, turnover, and trade statistics.
- Support in-sample, holdout, and walk-forward evaluation.
- Display warnings for look-ahead bias, survivorship bias, and insufficient data.
- Export an auditable trade ledger and performance series.

## Milestone 5: Research Workflow Expansion

- Comparable-company valuation and peer screening
- Earnings history, estimate revisions, and surprise analysis
- Company-specific KPI dashboards
- Thesis, catalyst, risk, and assumption notes
- Downloadable research reports
- Model versioning and saved research workspaces

## Delivery Order

1. Financial data foundation
2. Deterministic DCF engine and tests
3. DCF Streamlit interface and sensitivity analysis
4. Monte Carlo layer on top of the DCF engine
5. Backtesting engine
6. Reporting, comps, and workflow expansion
