"""Generate the report from verified facts and a completed reproducible study."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Utils.research import render_report


def study_markdown(folder):
    folder=Path(folder)
    s=json.loads((folder/'summary.json').read_text())
    m=s['holdout_metrics'];b=s['benchmark_metrics'];h=s['mty_buy_hold_metrics']
    dev=pd.read_csv(folder/'development_rules.csv')
    patterns=pd.read_csv(folder/'development_patterns.csv')
    trades=pd.read_csv(folder/'holdout_trades.csv')
    stress=pd.read_csv(folder/'holdout_cost_stress.csv')
    data=json.loads((folder/'data_manifest.json').read_text())
    checks=json.loads((folder/'benchmark_checks.json').read_text())
    selected=s['selected']
    lines=['# Backtesting MTY and testing its patterns',
           '## Result and evidence strength',
           f"The rule selected using earlier data was {selected}. The final withheld window runs {s['holdout_start']} through {s['holdout_end']}. This is a historical research result, not proof of a repeatable edge.",
           '| Same-period simulation | Net total return | CAGR | Max drawdown | Sharpe |',
           '|---|---:|---:|---:|---:|',
           f"| Selected rule | {m['total_return']:.2%} | {m['cagr']:.2%} | {m['maximum_drawdown']:.2%} | {m['sharpe_ratio']:.2f} |",
           f"| MTY buy and hold | {h['total_return']:.2%} | {h['cagr']:.2%} | {h['maximum_drawdown']:.2%} | {h['sharpe_ratio']:.2f} |",
           f"| XIU Canadian large-cap ETF | {b['total_return']:.2%} | {b['cagr']:.2%} | {b['maximum_drawdown']:.2%} | {b['sharpe_ratio']:.2f} |",
           f"The strategy beat holding MTY by {(m['total_return']-h['total_return'])*100:.2f} percentage points but trailed XIU by {(b['total_return']-m['total_return'])*100:.2f} points. There were only {s['holdout_closed_trades']} completed trades. The first trade generated {trades.PnL.iloc[0]/trades.PnL.sum():.1%} of dollar profit over two sessions. That concentration makes generalization especially weak.",
           '![Final-period portfolio values](holdout_equity.png)',
           '## How the experiment was frozen',
           'Daily history begins July 27, 2016, after the Kahala acquisition closed. Six rules were specified before results: SMA 20/100 and 50/200, momentum 63 and 126 sessions, RSI 14 with 30/55 thresholds, and Donchian 55/20. Each training window contains 504 sessions. It selects the highest net Sharpe rule with at least two completed trades for the next 126-session validation window. Cash is used if no eligible rule has positive Sharpe. The final 252 sessions are excluded from selection.',
           f"There are {s['walk_forward_folds']} validation folds and {s['trials_logged']} logged candidate evaluations including final selection. Final selection uses the last 504 development sessions only. The code physically truncates training data before calculating scores. The historical period was withheld from algorithmic selection, not prospectively registered before these events occurred.",
           'Starting capital is CAD10,000. Costs are 5 bps commission plus 10 bps adverse slippage per side. Positions are long or cash, with no leverage. Cash interest and the Sharpe reference rate are both set to zero. Strategy and benchmarks use identical evaluation dates and cost rules. Each fold resets capital and liquidates at its endpoint; chained validation returns do not represent a continuously resized live account.',
           '## Earlier evidence is less favorable',
           f"The sequence of rules selected during validation returned {s['validation_metrics']['total_return']:.2%} in total, with CAGR {s['validation_metrics']['cagr']:.2%} and maximum drawdown {s['validation_metrics']['maximum_drawdown']:.2%}. Fixed-rule development results below share the same start after the 200-session warm-up and end immediately before the final holdout.",
           '| Fixed rule, development only | Net return | Max drawdown | Trade segments |', '|---|---:|---:|---:|']
    for _,r in dev.iterrows():lines.append(f"| {r.Rule} | {r.total_return:.2%} | {r.maximum_drawdown:.2%} | {int(r.trades)} |")
    lines += ['The fixed RSI rule lost money over the earlier common sample. A profitable final year, especially one with only three trades, does not overturn that evidence. The other strategies are shown rather than hidden because they performed poorly.',
              '## Pattern diagnostics',
              'These are development-only, overlapping observations. Entry is the next open and exit is the close after the stated number of holding sessions. They omit trading costs and are not independent trades, p-values or actionable signals.',
              '| Condition | Sessions | Count | Mean return | Median return |', '|---|---:|---:|---:|---:|']
    for _,r in patterns.iterrows():lines.append(f"| {r['Pattern']} | {int(r['Holding Sessions'])} | {int(r.Observations)} | {r['Mean Forward Return']:.2%} | {r['Median Forward Return']:.2%} |")
    lines += ['Inference: the diagnostics offer little support for a simple, consistent continuation pattern in this sample. The slightly positive 20-session mean following RSI below 30 is small relative to uncertainty and does not establish a net profitable trading rule.',
              '## Trade audit',
              'Fill prices below are adjusted research-unit prices, including distribution adjustments and modeled slippage. They are not the historical tape prices a broker would have quoted.',
              '| Entry | Exit | Adjusted entry | Adjusted exit | Net trade return | PnL CAD |', '|---|---|---:|---:|---:|---:|']
    for _,r in trades.iterrows():lines.append(f"| {r['Entry Date'][:10]} | {r['Exit Date'][:10]} | {r['Entry Price']:.4f} | {r['Exit Price']:.4f} | {r['Net Return']:.2%} | {r.PnL:,.2f} |")
    lines += [f"Total modeled commission and slippage were CAD{s['costs']:.2f}. Ending equity minus starting capital minus trade PnL and cash interest reconciles to CAD{s['ledger_reconciliation_error']:.8f}. The complete order and daily cash ledgers are included in the data package.",
              '## Costs and uncertainty',
              '| Slippage per side | Commission per side | Selected-rule final return |', '|---:|---:|---:|']
    for _,r in stress.iterrows():lines.append(f"| {r.slippage_bps_per_side:.0f} bps | {r.commission_bps_per_side:.0f} bps | {r.total_return:.2%} |")
    ci=s['bootstrap']
    lines += [f"A paired circular-block bootstrap with {ci['block_sessions']}-session blocks, {ci['draws']:,} draws and seed {ci['seed']} estimates an annualized arithmetic active mean of {ci['annualized_arithmetic_active_mean']:.1%} versus XIU, with a descriptive 95% interval from {ci['lower_95']:.1%} to {ci['upper_95']:.1%}. It includes zero and is not corrected for all researcher choices. It is neither CAGR nor a probability of future profit.",
              '## Data checks and limitations',
              f"MTY and XIU each contain {data['MTY.TO']['rows']:,} sessions, ending {data['MTY.TO']['last_session']}. Both are CAD securities on Toronto's market. The raw snapshots and SHA-256 hashes are included. Yahoo OHLC and adjusted close are explicitly requested. All OHLC values are multiplied by adjusted close / close; dividends are not credited a second time.",
              'XIC was initially considered, then rejected before strategy results because its downloaded 2021 adjusted market-price return was 23.39% versus an issuer NAV return of 25.06%, alongside a vendor action factor of 0.987. The cause was not conclusively resolved, so no correction was guessed. XIU passed the following coarse annual screen. Market-price and NAV conventions naturally differ; passing does not certify every daily bar.',
              '| Year | Yahoo XIU market-price return | BlackRock NAV return | Difference, percentage points |', '|---|---:|---:|---:|']
    for r in checks['checks']:lines.append(f"| {r['year']} | {r['yahoo_market_price_return']:.2%} | {r['issuer_NAV_return']:.2%} | {r['difference_pp']:+.3f} |")
    lines += ['Issuer sources: [XIU performance](https://www.blackrock.com/ca/investors/en/products/239832/ishares-sp-tsx-60-index-etf) and [XIC performance](https://www.blackrock.com/ca/investors/en/products/239837/ishares-sptsx-capped-composite-index-etf). Data interface: [yfinance history documentation](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html).',
              'Fills use next-session adjusted opens, adverse slippage and commission. Orders exceeding 1% of prior 20-session dollar volume are rejected. This limit is a capacity screen, not an auction-depth or partial-fill model. Identical calendars are required, but a separate exchange-calendar audit and second independent daily vendor remain future work.',
              'No actual bid/ask spread history, payment-date dividend cash accounting, tax, short borrow, margin or intraday path is modeled. No fundamentals or news enter the tested signals. The study cannot isolate Kahala returns, prove causality, remove the selection bias of choosing MTY, or guarantee future performance.',
              '## What changed in the platform',
              'The revision adds explicit Kahala-to-parent identity resolution, a verified evidence schema, source-linked reports, currency checks, next-open fills, cash-and-unit accounting, comparable benchmark costs, holdout entry charges, corrected RSI edge cases, drawdown from initial capital, and all-period downside deviation. It adds chronological selection, a final holdout, order audits, pattern diagnostics, fixed seeds and data hashes. Unsupported short accounting is disabled.',
              'Regression tests include hand-calculated trades, flat-price costs, overnight gaps, dividend adjustment, account reconciliation, invalid data, zero volume, currency mismatch, future-data invariance and interface controls. Passing tests verifies the implemented model; it does not certify vendor prices or live fills.',
              '## Next useful step',
              'Preserve this result, including its weaknesses. Collect better independently reconciled data and start a prospective paper record with unchanged rules. Any new hypothesis should be developed on development data and tested on a new untouched period. The accompanying learning guide walks through how to build and audit each layer.']
    return '\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=render_report()+'\n'+study_markdown(a.study)
    (a.output/'Kahala_Brands_Report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=pd.read_csv(a.study/'holdout_performance.csv',index_col=0,parse_dates=True)
    hold=pd.read_csv(a.study/'holdout_mty_buy_hold_performance.csv',index_col=0,parse_dates=True)
    fig,ax=plt.subplots(figsize=(9,4.1))
    colors=['#12617b','#809093','#a56332']
    for series,label,color in [(data['Strategy Equity'],'Selected RSI rule',colors[0]),(hold['Strategy Equity'],'MTY buy and hold',colors[1]),(data['Benchmark Equity'],'XIU benchmark',colors[2])]:
        ax.plot(series.index,series*10000,label=label,color=color,lw=1.6)
    ax.set_ylabel('Portfolio value (CAD)');ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',frameon=False,fontsize=9);fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(a.output/'holdout_equity.png',dpi=200);plt.close(fig)
    print(a.output/'Kahala_Brands_Report.md')

if __name__=='__main__':main()
