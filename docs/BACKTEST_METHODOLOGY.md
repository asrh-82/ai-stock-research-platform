# What this engine measures

Version dated November 17, 2026. Scope is daily, single-asset, unlevered long/cash research. It is not an execution simulator for live trading. It cannot measure a separately traded Kahala stock because Kahala is an MTY subsidiary.

## Data contract

Supply daily Open, High, Low, Close, Adj Close and Volume. The Yahoo loader explicitly requests `auto_adjust=False`, `back_adjust=False`, `actions=True`, `repair=False`, `keepna=True`. The engine validates finite positive prices, nonnegative volume, OHLC consistency and unique ordered exchange-local session dates. Asset and benchmark currencies must be explicitly supplied and equal. Their session calendars must match exactly. Missing values, mismatched calendars and close-only series raise errors instead of silently changing the sample.

All OHLC prices are multiplied by Adj Close / Close. This makes them adjusted research units, including the provider's distribution adjustment convention. The engine must not add a second dividend payment or apply a second split adjustment. Fractional units are economic units, not original historical shares. The engine does not model dividend payable dates or tax. It uses only scale-invariant price signals. Absolute price thresholds would need point-in-time adjustment factors.

Daily data must be independently checked for corporate-action anomalies, stale dates and major discrepancies. Equal calendars do not prove both feeds contain every required exchange session. The current data screen does not certify every bar against an exchange feed.

## Sequence on each session

1. Accrue the configured cash rate over elapsed calendar days on cash carried from the last recorded close.
2. Read the previous completed session's signal. Weekly/monthly rules update on the first observed session of a new period, using the previous close.
3. Hold existing adjusted units through the overnight move. A new buyer does not receive a gap that happened before entry.
4. Fill an entry at the adjusted open multiplied by 1 + slippage. Fill an exit at open multiplied by 1 - slippage. Commission is a fraction of filled notional.
5. Reject an order exceeding the configured fraction of the prior 20-session average dollar volume. Never size using today's total volume. A zero-volume session cannot fill. This is an all-or-none capacity approximation, not a partial-fill order book.
6. Mark cash plus units at the adjusted close. On a predetermined final date, attempt liquidation at the close with adverse slippage and commission. A rejected terminal exit remains explicitly open and marked to market.

Buy quantity is cash / (adjusted open x (1 + slippage) x (1 + commission)). Fixed units are held until exit; the simulator does not silently rebalance the position daily. Entry and exit price, quantity, commission, proceeds and PnL are recorded. Sum of trade PnL plus cash interest reconciles to ending equity less starting capital, including unrealized PnL for explicitly open trades.

## Benchmarks and evaluation boundaries

The benchmark uses the same evaluation dates, starting capital, cost assumptions and capacity rules. Comparing CAD MTY with a USD ETF is rejected without explicit FX conversion. The study uses XIU.TO, a large-cap Canadian ETF, and separately reports MTY buy and hold. XIU is not a restaurant-sector or small-cap matched benchmark.

Every evaluation window starts with fresh cash. Signal warm-up does not import free existing holdings. The first entry is charged even if the signal was already active before the evaluation window. Each validation fold is a reset-capital trial; chaining its returns is a research summary, not a single continuously sized cash account. Scheduled fold liquidation adds trading costs. The final holdout is one continuous simulated account.

## Metrics

- Total return is ending equity / starting capital - 1.
- CAGR uses elapsed calendar days including the endpoint, divided by 365.25. Very short-window CAGR is unstable.
- Volatility is sample standard deviation of daily equity returns multiplied by sqrt(252).
- Sharpe uses the mean excess daily return and its sample standard deviation, annualized with sqrt(252). A configurable constant annual risk-free rate is converted geometrically. It is not a historical daily Treasury series.
- Sortino divides by the square root of the mean squared negative excess returns over ALL days, with nonnegative excess returns set to zero.
- Drawdown includes starting cash in the high-water mark, so the initial entry cost or first loss is counted.
- Win rate uses completed trades only. Open marks are not wins.
- Costs include modeled commissions plus adverse price movement from slippage. Cash interest is separately recorded.
- Exposure is the fraction of sessions with a position after the opening trade, not a risk-weighted or intraday measure.

## Frozen study protocol

Six candidate rules: SMA 20/100, SMA 50/200, momentum 63, momentum 126, Wilder RSI 14 entering at or below 30 and exiting at or above 55, and Donchian entry/exit 55/20. Donchian compares the close with a channel excluding the current bar. Cash is the fallback, not a tuned seventh strategy.

Use 504 trading sessions to train, 126 to validate, and reserve the final 252 sessions. Rank candidates by net training Sharpe only if they have at least two closed trades. Hold cash if none has a positive eligible Sharpe. The final rule uses only the last 504 development sessions. All candidates and 84 selection trials are logged. No holdout result enters selection. A mutation test changes all final-period prices and requires unchanged training scores and validation results.

The historical test was designed in this session. It was withheld from algorithmic selection, not prospectively preregistered years ago. Its dates are known history. Repeated human revisions after seeing its results would contaminate it. The report does not claim the backtest proves an edge.

## Pattern exploration

Development-only conditional forward returns use the next open as entry and the close of session 1, 5 or 20 as the endpoint. They exclude unobserved future endpoints. Samples overlap, cluster by regime and omit implementation costs. They are descriptive diagnostics, not independent trades, causal estimates or significance tests.

The paired circular-block bootstrap uses 20-session blocks, 2,000 draws and seed 42. Its interval describes uncertainty in annualized arithmetic active mean. It is not CAGR, a probability of profit, or a multiple-testing-adjusted significance result.

## Known limits and next work

Missing point-in-time fundamentals, a delisted universe, bid/ask histories, auction volume, payment-date dividends, taxes, actual FX conversion, borrow availability and margin rules remain outside this scope. Shorting is disabled rather than represented with incorrect inverse-price returns. A second independent daily price vendor, official exchange calendar validation, realistic partial fills, history-dependent rates and prospective paper trading should precede a live-use claim.

References: [yfinance history contract](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html), [BlackRock XIU](https://www.blackrock.com/ca/investors/en/products/239832/ishares-sp-tsx-60-index-etf), [Bailey et al. on backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
