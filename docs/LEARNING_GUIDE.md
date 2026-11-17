# Build and understand your stock research platform

Prepared for Arnav, November 17, 2026.

The goal is to independently build, explain and audit this project. You already have enough calculus background to start. The first version depends most on Python, accounting, probability, statistics and careful handling of time. Fluency comes from implementing small pieces and checking them by hand.

## 1. Start with the system you actually built

A research platform has separate jobs. Company identity determines what security exists. Data ingestion gathers observations and records their origin. Accounting normalizes filings. The report presents facts, calculations and uncertainty. A strategy turns available observations into a target position. A simulator turns that target into orders and cash movements. Evaluation asks whether the result is repeatable. The interface lets someone inspect all of it.

In this project, Kahala and MTY expose the first problem. A business name is not automatically a ticker. Kahala is an operating subsidiary. MTY.TO is the quoted parent. If that identity is wrong, correct code can still produce a completely misleading report. Your system now resolves Kahala to its explicitly labeled public parent and leaves unavailable standalone figures unknown.

| Layer | Main file | What you must be able to explain |
|---|---|---|
| Identity and report | Utils/research.py | Why MTY is the tradable proxy and which numbers belong to it |
| Verified evidence | Data/research/kahala.json | Entity, unit, period, source and verification date for every figure |
| Daily market data | Utils/market_data.py | Raw versus adjusted prices, timestamps, metadata and file hashes |
| Signals and simulator | Utils/backtesting.py | What was known before each order and how cash changed |
| Evaluation | Utils/walk_forward.py | Why training data cannot include later test observations |
| Reproduction | scripts/run_kahala_study.py | How another person gets the same result from the same snapshot |
| Report screen | Utils/research_ui.py | What is verified and what is merely a dated snapshot |
| Backtest screen | Utils/backtesting_ui.py | Which assumptions the user controls and what their units mean |
| Automated checks | tests/test_backtesting.py and other tests | How an incorrect engine is forced to fail |

Read the files in that order. Change one behavior per branch. Write down your expected result before running it.

## 2. Python foundations you will use constantly

Learn values, variables, arithmetic, strings, lists, dictionaries, loops, conditionals, functions, imports and exceptions. Then learn type annotations, dataclasses, pathlib, JSON and context managers. Functions should receive explicit inputs and return outputs. Avoid putting network calls or interface state inside accounting calculations.

A dictionary is useful for a source record. A dataclass makes a strategy configuration explicit. A list collects orders. An exception should stop a calculation when a required input is unavailable. Returning zero for missing revenue would quietly turn a data error into a financial claim.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CostAssumptions:
    commission_bps: float = 5.0
    slippage_bps: float = 10.0

def basis_points_to_fraction(value: float) -> float:
    return value / 10_000

assert basis_points_to_fraction(10) == 0.001
```

Exercise: write a function accepting price and number of shares, returning market capitalization. Reject missing, infinite or negative inputs. Test 36.38 times 22.841361 million. You should get approximately CAD830.9687 million. Explain why a stale share count makes that an illustration rather than a verified current market cap.

Use a debugger or print one row at a time. Before asking why a large program fails, reduce the input until you have the smallest failing case. Read tracebacks from the last line upward to find the actual failure and then its caller.

## 3. Git and reproducible environments

Learn clone, status, diff, add, commit, branch, fetch, pull and push. A branch isolates a change. A commit is a recoverable snapshot. A pull request lets you review the combined behavior and validation before release. Avoid committing secrets, API tokens or private account files.

The project's pyproject.toml describes dependencies. uv.lock pins their resolved versions. A virtual environment isolates them from other projects. An experiment also needs its raw data, retrieval time, provider, exact parameters, random seed, date boundaries, costs, code revision and output. A seed alone cannot reproduce a study if the provider later changes its historical prices.

```bash
uv sync --frozen
uv run --with pytest python -m pytest -q
uv run streamlit run app.py
```

To reproduce the frozen study using a saved snapshot, run from the repository root. Replace the example paths with the extracted data directory and your chosen output directory.

```bash
uv run python scripts/run_kahala_study.py \
  --reuse /path/to/saved_snapshot \
  --output /path/to/reproduced_results
```

The snapshot directory needs MTY.TO.csv, XIU.TO.csv and matching metadata JSON files. Without --reuse, the script downloads data again. That tests a new provider snapshot, so check its manifest and differences. Keep the original final test untouched while you learn.

Exercise: change a chart title on a branch, inspect the diff, commit it, and restore the original on another commit. You pass when you can recover the first state without copying a whole folder.

## 4. pandas, NumPy and dates

A Series is one indexed column. A DataFrame is a table. Its index often represents trading sessions. Learn selecting rows, joining tables, rolling windows, shift, pct_change, missing values, groupby and resampling. NumPy gives fast array operations and reproducible random generation.

```python
returns = close.pct_change(fill_method=None)
fast = close.rolling(20, min_periods=20).mean()
slow = close.rolling(100, min_periods=100).mean()
signal_at_close = (fast > slow).astype(float)
order_target_next_open = signal_at_close.shift(1)
```

This code creates a decision schedule. It does not calculate a fully correct portfolio return by itself. You still need opening fills, existing overnight exposure, fees, cash and quantities.

An inner join retains only shared dates. It can silently delete a market session. The current engine requires equal calendars so you must investigate differences. Never forward-fill an unavailable execution price just to keep the program running. Distinguish a market holiday, a missing vendor row and a trading halt.

Timezones matter. A Toronto session is a local calendar date. Converting every timestamp to UTC and then normalizing can shift another market's session date. The loader preserves local dates. A historical earnings release also needs a publication timestamp, not just the fiscal period-end date.

Exercise: deliberately delete one benchmark date. Confirm the engine rejects the input. Then explain why silently dropping that asset date could change a breakout signal, a fill and the measured return.

## 5. Returns and risk mathematics

Simple return is new value divided by old value minus one. Multi-period return is the product of one plus each simple return, minus one. Adding simple returns is generally wrong. Log returns add over time, but portfolio accounting still needs actual currency values and positions.

If an account gains 20% then loses 20%, it ends at 0.96 times its starting value. The average simple return is zero; the compound return is negative 4%. CAGR converts an endpoint ratio into a constant annual growth rate. It says nothing about the path or the probability of repeating it.

Volatility measures dispersion. Drawdown measures loss from the running peak. Sharpe divides average excess return by return variability. Sortino uses downside deviation. Turnover measures traded notional relative to portfolio value. Exposure measures time or capital at risk, depending on the definition. Always state the definition.

The square-root-of-252 convention is common for daily volatility and Sharpe. Serial dependence and irregular data weaken its interpretation. A high Sharpe from three trades has substantial uncertainty. A win rate can look excellent while the sample is tiny or the losing tail has not appeared yet.

FX example: if CAD assets return 10% and one Canadian dollar loses 5% of its USD value, the USD return is 1.10 x 0.95 - 1 = 4.5%, before FX costs. CAD and USD series cannot be compared as though currency did not exist.

Exercises: calculate a 50% loss followed by the gain needed to break even; draw a five-point equity curve and compute its drawdown; explain why the first transaction cost must appear in drawdown. Answers: a 100% gain is needed to recover from a 50% loss, and initial capital is part of the high-water mark.

## 6. Financial statements and valuation

Learn the income statement, balance sheet and cash-flow statement together. Revenue is not cash receipts. Net income includes accruals and noncash expenses. Operating cash flow incorporates working-capital movements. Capital spending and acquisitions are different uses of cash. Share dilution changes the denominator of per-share value.

For a franchisor, distinguish restaurant system sales from the franchisor's recognized revenue. A franchisee sells food to customers; the parent might collect only a royalty or other fee. Adding system sales to corporate revenue would double count different economic measures. Advertising funds can also inflate reported revenue without a corresponding profit contribution.

Read management's non-GAAP definitions and reconcile them to financial statements. EBITDA omits cash needs such as capital spending, interest and taxes. A free-cash-flow measure may include asset sale proceeds. IFRS lease accounting affects EBITDA, debt and cash flow. You must treat leases consistently in both the numerator and denominator of a valuation multiple.

Unlevered free cash flow is commonly modeled as EBIT after tax plus depreciation and amortization, minus capital expenditure and working-capital investment. Discount it at WACC to estimate enterprise value, then bridge to equity using debt, cash and other claims. Equity cash flows instead require a cost of equity. Do not subtract debt twice when the modeled cash flows already belong to equity holders.

A DCF needs explicit operating forecasts, reinvestment, taxes, discount rate, terminal assumptions and share count. A single precise target can conceal broad uncertainty. Use scenarios and explain which evidence could change each input. Check that WACC exceeds terminal growth. Show the fraction of valuation coming from the terminal value.

For MTY, the report gives transparent trailing arithmetic. It does not call management FCF yield a promised return, or a reported P/E a normalized P/E. No standalone Kahala valuation is supported without standalone cash flows and claims.

Exercise: reconstruct one reported quarter from its filing, including units. If the statement says thousands of CAD, 277,726 means CAD277.726 million. Enter the source page and period beside the number. Then calculate the year-over-year change using comparable figures from the same reporting basis.

## 7. Data engineering and source-backed reports

Every fact should carry an entity, metric, value, currency, scale, fiscal period, source, publication time and retrieval time. Those last two times are different. A report published in October about August was unavailable to a trader in August. Later restatements were also unavailable in the original period.

Maintain separate layers for raw observations, validated data, derived calculations and presentation. Save original data before cleaning. Record every correction as a transformation with a reason. Never overwrite a questionable corporate action merely because a different number makes performance look plausible.

The study rejected XIC's original Yahoo snapshot after a material 2021 discrepancy against issuer annual returns. XIU was chosen before strategy results were viewed and passed a coarse annual comparison. Market-price and NAV returns may differ, so the comparison is not a daily-data certification. The audit retains both the issue and the replacement reason.

A report generator should label reported facts, calculated quantities, inferences and unknowns. An LLM can help draft explanations or locate candidate sources, but it must not silently supply missing financial values. The current report is a reviewed evidence pack and deterministic renderer. Other companies need their own verified packs or a separately validated extraction pipeline.

Exercise: change the evidence quote currency to USD. The report must refuse the mismatch. Change a source date to next year and request today's report. It must fail. Delete standalone Kahala EBITDA. The report should say unknown, not borrow MTY's figure.

## 8. Strategy rules and their economic meaning

A strategy is a precise map from available information to an order. Define the signal, warm-up, entry, exit, position size, rebalance schedule, execution timestamp and all costs before measuring results.

SMA crossover follows a trend. Momentum uses a past percentage move. RSI describes the balance of recent gains and losses; it does not directly measure intrinsic value. A Donchian channel compares price with a past range. A mean-reversion rule assumes reversals after weakness, while a breakout rule assumes continuation. Each assumption can fail during prolonged regime changes.

Wilder RSI initializes average gains and losses using a full window, then smooths them. If all observed changes are positive and average loss is zero, RSI is 100. If all are negative, it is zero. If both are zero after warm-up, it is 50. Undefined warm-up observations must remain unavailable, not become fabricated neutral data.

For a breakout, exclude the current high from the prior channel. Otherwise your condition may become impossible or depend on the wrong time. For monthly rules, use the first observed session of a new month and the prior close. That avoids recognizing a future last trading day using later data.

Exercise: write the SMA signal yourself and verify it on a hand-made rising series. Then change every price after a cutoff. Earlier signals must remain identical.

## 9. Build the cash ledger by hand

Take CAD10,000 and a stock whose adjusted opening price is CAD100. For an intentionally large test, use 1% slippage and 1% commission per side. The buy fill is CAD101. Quantity is 10,000 / (101 x 1.01), approximately 98.0296 units. On an unchanged reference price, the sell fill is CAD99. After the sell commission, ending cash is approximately CAD9,607.88. The model must report a loss despite a flat market.

This is an excellent test because you can calculate it without trusting the simulator. At ordinary study assumptions of 10 bps slippage and 5 bps commission, use the same formulas with 0.001 and 0.0005.

Now add a gap. Yesterday's close is CAD100 and today's open is CAD110. A decision based on yesterday's completed close cannot buy yesterday at CAD100. It buys today's open plus slippage. If you already held the asset yesterday, your existing units earn or lose that overnight change. These are different cases.

The original engine shifted a signal but credited the whole close-to-close return at the prior closing price. A shift alone did not make that execution achievable. The replacement tracks quantity and cash, using next-open fills. Buy and hold pays comparable entry and exit costs.

Dividends and splits require a declared convention. Here, the adjusted OHLC ratio embeds the provider's reinvestment model. You must not add cash dividends again. A production broker ledger would instead track actual shares, splits, dividend receivables, payment dates, cash and tax. That is a different implementation, requiring different data.

Exercise: create a constant-price dataset, add costs and confirm a loss; add a dividend offset by the price drop and confirm no spurious loss or double gain; create a 10% overnight jump before a new entry and confirm the buyer does not receive it.

## 10. Statistics and avoiding false discoveries

Start with distributions, conditional probability, expectation, variance, covariance, sampling error, estimation, confidence intervals and hypothesis testing. Then learn regression, residuals, autocorrelation, heteroskedasticity and robust inference. Multiple testing matters because the best result among many attempts is likely to be flattered by chance.

Training chooses a rule or estimates parameters. Validation evaluates choices on later observations. A final holdout is withheld until the process is frozen. Once you inspect it and change the system, it becomes development data. You then need new unseen data for an honest final test.

The study uses 504-session training windows followed by 126-session validation windows. Its last 252 sessions are reserved. Six candidates are compared under the same costs. A training window needs at least two completed trades and positive Sharpe for a candidate to beat cash. Two trades is an operational eligibility threshold, not statistically sufficient evidence.

For machine learning, scale and impute using training data only. Fit preprocessing inside each fold. If a label is a future 20-day return, nearby train/test labels can overlap. Remove observations whose outcomes extend into the test period, and consider a gap or embargo where justified. A random train/test split usually breaks the chronology of financial prediction.

A block bootstrap resamples adjacent chunks to retain some serial structure. The study's 20-session blocks give a wide interval for active return. That interval includes zero. It does not account for every idea a researcher considered, and it is not a probability of making money next year.

Exercise: generate many random strategies on random prices, select the best training performer and evaluate it later. Repeat. Record how often a beautiful training chart fails. Use a fixed seed for reproducibility, but do not repeatedly choose the seed that produces your favorite demonstration.

## 11. Interpret this actual MTY study

The final rule, selected using earlier data, was RSI 14 entering at 30 and exiting at 55. From November 14, 2025 through November 16, 2026 it returned approximately 18.62% after modeled costs. Holding MTY returned 11.11%. Holding XIU returned 23.24% on the same dates and cost convention.

There were three completed trades. All were positive, but three wins cannot establish a reliable win rate. The first trade accounted for approximately 59.3% of dollar profit and held for two sessions. The largest portfolio day was November 17, 2025, during the period in which the strategic-review announcement became public. Do not infer that RSI caused the news-driven return.

The fixed RSI rule lost approximately 33.68% over the earlier common development window and had a drawdown near 71.92%. The walk-forward sequence of selected rules had about 4.67% annualized return and a 42.50% maximum drawdown during validation. The withheld year is therefore insufficient to declare a stable edge.

The paired block-bootstrap interval for annualized arithmetic excess over XIU was approximately -28.5% to +27.7%. A wide interval crossing zero supports uncertainty, not a victory claim. Higher slippage also lowers returns. Every one of those observations belongs in the report alongside the good year.

Your next meaningful validation is a prospective paper-trading record with frozen rules and timestamped orders. Revising indicators until this same year looks better cannot produce new evidence.

## 12. Testing, review and operating the app

Unit tests check one piece against known arithmetic. Integration tests check that connected components behave together. Property tests check invariants across many inputs. App tests exercise controls and error states. Different categories find different failures.

Useful invariants include cash plus marked holdings equals equity, closed-trade PnL reconciles to the account, changing future prices does not change past decisions, no order fills before its signal exists, and missing inputs never become unexplained zeros. Compare the event loop against a separately written reference ledger on simple scenarios.

Production operations require timeouts, rate-limit handling, clear data freshness, predictable caching, structured logging, access controls and secret management. Cache keys should include ticker, dates, provider and price-adjustment convention. A failed data request should stop the backtest and explain why. It should never switch to sample prices without an explicit demo mode.

Before deploying, run the test suite and an app smoke test, inspect the diff, pin dependencies and record the code revision. A Streamlit app needs a host that supports its server process and websocket connection. Verify that the deployment platform supports your runtime; a successful Git push does not prove the app is serving correctly.

Exercise: block a data request and verify the app shows a useful error while the dated evidence report still opens. Then restore it and compare the generated file hash with the saved manifest.

## 13. A twelve-week sequence with concrete checkpoints

Budget roughly 6-8 focused hours per week. Adjust the pace based on whether you can pass each checkpoint without copying code. A productive session is 15 minutes of learning, 35 minutes of implementation and 10 minutes of explaining what you built.

| Week | Learn | Build | Pass condition |
|---|---|---|---|
| 1 | Python functions, collections, exceptions | Return and basis-point calculator | Explain and test all inputs and outputs |
| 2 | Git, environments, JSON, files | Saved configuration and data manifest | Reproduce a run from a clean checkout |
| 3 | pandas, NumPy, dates | Price loader and validator | Reject missing, duplicate and impossible bars |
| 4 | Three statements and franchisor accounting | Source-linked MTY fact table | Reconcile five figures to filing pages and units |
| 5 | Returns, risk, compounding | Independent metrics module | Match hand-calculated equity and drawdown examples |
| 6 | Signals, indicators, warm-up | SMA, RSI and breakout rules | Future-price mutation cannot change past signals |
| 7 | Execution and portfolio accounting | Next-open cash ledger | Reconcile every trade and account cent within tolerance |
| 8 | Validation, statistics, selection bias | Chronological training and test splits | Holdout mutations leave selection unchanged |
| 9 | Robustness and data audits | Cost, regime and benchmark checks | Explain every material discrepancy and limitation |
| 10 | Reporting and Streamlit | Search, report and backtest screens | Every displayed metric has its unit and source |
| 11 | Integration tests and deployment | Repeatable build and smoke test | Fresh environment passes checks and serves the app |
| 12 | Research discipline | Freeze a new hypothesis and begin paper tracking | Timestamp rules before observing new outcomes |

## 14. The next mathematical and engineering levels

For cross-sectional quant work, learn linear algebra for factor exposures, covariance matrices, regression and portfolio optimization. Learn constrained optimization, transaction-cost objectives and risk budgeting. Your planned linear algebra course is directly relevant. Probability and statistical inference deserve equal priority.

For larger research systems, learn SQL, table design, primary keys, migrations, point-in-time joins, data versioning, job queues and monitoring. Historical company identifiers change through mergers, ticker changes and delistings. A current list of surviving companies is not a historical universe.

For machine learning, begin with simple baselines and interpretable models. Evaluate predictions through an economic decision rule after costs. A low prediction error does not guarantee profitable trades. Calibrate probabilities, track drift and keep all tuning within development folds. Model complexity should earn its place through new evidence.

For real execution research, obtain historical quotes and trades, simulate market and limit orders, model partial fills and queue priority, inspect market impact and compare simulation with timestamped paper fills. Model borrow fees, recalls and margin before enabling shorts. Do not extend a long-only accounting formula by simply negating or inverting returns.

For portfolio claims, include a survivorship-aware universe, corporate-action histories, delisting returns, portfolio constraints and overlapping exposures. Track benchmark-relative risk, factor exposure and liquidity at portfolio scale. A successful single-ticker example cannot establish broad generalization.

## 15. Your first session

Open tests/test_backtesting.py and read the flat-price cost example. Predict the expected cash by hand. Run that one test. Change its slippage from 1% to 0.5% and recalculate the answer before running again. Then open _simulate in Utils/backtesting.py and trace the buy and sell line by line.

Write a short explanation answering three questions. What information existed before the order? What price was actually available when the order executed? Where did every change in account value come from? If you can answer those, you understand the core of a backtesting engine.

## References and free learning material

These are primary documentation and research sources checked for this guide. Use them to learn a specific concept, then apply it to a small project task.

- Python tutorial: https://docs.python.org/3/tutorial/
- pandas getting started: https://pandas.pydata.org/docs/getting_started/
- pytest getting started: https://docs.pytest.org/en/stable/getting-started.html
- Pro Git: https://git-scm.com/book/en/v2
- Streamlit tutorials: https://docs.streamlit.io/develop/tutorials
- yfinance price-history contract: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
- scikit-learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Bailey, Borwein, Lopez de Prado and Zhu, The Probability of Backtest Overfitting: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- MTY financial reports: https://mtygroup.com/en/financial-reports/
- BlackRock XIU issuer page: https://www.blackrock.com/ca/investors/en/products/239832/ishares-sp-tsx-60-index-etf
