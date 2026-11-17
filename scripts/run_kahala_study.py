"""Reproduce the frozen Kahala/MTY research protocol, or refresh its data explicitly."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Utils.market_data import fetch_daily_history
from Utils.backtesting import BacktestConfig, run_backtest, prepare_prices
from Utils.walk_forward import research_candidates, run_walk_forward, block_bootstrap_active_mean, pattern_diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse',type=Path,help='Explicit local snapshot directory containing MTY.TO.csv and XIU.TO.csv')
    parser.add_argument('--end',default='2026-11-17',help='Exclusive end; default freezes the original study')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=BacktestConfig(transaction_cost_bps=5,slippage_bps=10,initial_capital=10000,max_participation=.01)
    candidates=research_candidates(base)
    protocol={'entity':'Kahala Brands','traded_proxy':'MTY Food Group Inc.', 'asset':'MTY.TO','benchmark':'XIU.TO','currency':'CAD',
              'start':'2016-07-27','end_exclusive':args.end,'train_sessions':504,'test_sessions':126,'holdout_sessions':252,
              'candidates':{n:asdict(c) for n,c in candidates.items()},'selection':'Net training Sharpe; at least two closed trades; cash if none positive',
              'created_before_results_utc':datetime.now(timezone.utc).isoformat(),
              'benchmark_choice':'XIC rejected before strategy results due to 2021 return discrepancy and nonstandard 0.987 action. XIU large-cap Canadian ETF used after issuer return checks.',
              'independence':'Fixed historical protocol, not a prospective preregistration. Do not retune after viewing final holdout.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    data={}; manifests={}
    for ticker in ['MTY.TO','XIU.TO']:
        if args.reuse:
            frame=pd.read_csv(args.reuse/(ticker+'.csv'),index_col=0)
            frame.index=pd.DatetimeIndex([str(d)[:10] for d in frame.index])
            frame=frame.loc[protocol['start']:]
            frame=frame.loc[frame.index<pd.Timestamp(args.end)]
            meta_path=args.reuse/(ticker+'.metadata.json')
            raw=json.loads(meta_path.read_text())
            if raw.get('symbol')!=ticker or raw.get('currency')!='CAD':raise ValueError('Snapshot identity/currency mismatch')
            manifest={'provider':'Yahoo Finance via yfinance','ticker':ticker,'currency':'CAD','exchange':raw.get('exchangeName'),
                      'first_session':str(frame.index[0].date()),'last_session':str(frame.index[-1].date()),'rows':len(frame),
                      'snapshot_loaded_at_utc':datetime.now(timezone.utc).isoformat(),
                      'note':'Reused locally downloaded snapshot; retrieval provenance recorded in study audit.'}
        else:
            frame,manifest=fetch_daily_history(ticker,protocol['start'],args.end)
        prepare_prices(frame,ticker)
        if (pd.Timestamp(args.end)-frame.index[-1]).days>5:raise ValueError('Stale snapshot')
        frame.to_csv(args.output/(ticker+'.csv'),index_label='Date')
        manifest['sha256_csv']=hashlib.sha256((args.output/(ticker+'.csv')).read_bytes()).hexdigest()
        manifest['nonzero_split_events']=[{'date':str(d.date()),'factor':float(v)} for d,v in frame.get('Stock Splits',pd.Series(dtype=float)).items() if v!=0]
        manifests[ticker]=manifest;data[ticker]=frame
    issuer_returns={2021:.2790,2022:-.0636,2023:.1182,2024:.2077,2025:.2889}
    checks=[]
    close=data['XIU.TO']['Adj Close']
    for year,expected in issuer_returns.items():
        actual=float(close.loc[str(year)].iloc[-1]/close.loc[str(year-1)].iloc[-1]-1)
        difference=actual-expected
        checks.append({'year':year,'yahoo_market_price_return':actual,'issuer_NAV_return':expected,'difference_pp':difference*100})
        if abs(difference)>.0025:raise ValueError(f'Benchmark failed issuer annual-return reconciliation for {year}')
    (args.output/'benchmark_checks.json').write_text(json.dumps({'source':'https://www.blackrock.com/ca/investors/en/products/239832/ishares-sp-tsx-60-index-etf','tolerance_percentage_points':.25,'note':'Market-close versus NAV returns may differ; this is a coarse data screen, not daily certification.','checks':checks},indent=2))
    (args.output/'data_manifest.json').write_text(json.dumps(manifests,indent=2))
    a,b=data['MTY.TO'],data['XIU.TO']
    print('Data validated. Starting chronological selection.',flush=True)
    study=run_walk_forward(a,b,currency='CAD',candidates=candidates)
    for key in ['folds','trials','validation']:
        study[key].to_csv(args.output/(key+'.csv'),index=(key=='validation'))
    r=study['holdout'];bh=study['holdout_buy_hold']
    for label,result in [('holdout',r),('holdout_mty_buy_hold',bh)]:
        result.performance.to_csv(args.output/(label+'_performance.csv'))
        result.trades.to_csv(args.output/(label+'_trades.csv'),index=False)
        result.orders.to_csv(args.output/(label+'_orders.csv'),index=False)
    holdout_start=study['holdout_start']
    development=a.loc[a.index<holdout_start]
    pattern_diagnostics(development).to_csv(args.output/'development_patterns.csv',index=False)
    # Evaluate the prespecified rules on development only, all on the SAME start date.
    comparison=[];common_start=a.index[200]
    for name,cfg in {**candidates,'MTY buy and hold':replace(base,strategy='Buy and Hold')}.items():
        result=run_backtest(development,b.loc[development.index],cfg,evaluation_start=common_start,asset_currency='CAD',benchmark_currency='CAD')
        comparison.append({'Rule':name,**asdict(result.strategy_metrics),'trades':result.trade_count})
    pd.DataFrame(comparison).to_csv(args.output/'development_rules.csv',index=False)
    # Cost stress uses the already selected rule. It does not select a new winner.
    stress=[]
    for slip in [0,10,25,50]:
        cfg=replace(BacktestConfig(**study['selected_config']),slippage_bps=slip)
        s=run_backtest(a,b,cfg,evaluation_start=holdout_start,asset_currency='CAD',benchmark_currency='CAD')
        stress.append({'slippage_bps_per_side':slip,'commission_bps_per_side':cfg.transaction_cost_bps,**asdict(s.strategy_metrics)})
    pd.DataFrame(stress).to_csv(args.output/'holdout_cost_stress.csv',index=False)
    interval=block_bootstrap_active_mean(r.performance['Strategy Return'],r.performance['Benchmark Return'])
    summary={'selected':study['selected'],'selected_config':study['selected_config'],'holdout_start':str(holdout_start.date()),
             'holdout_end':str(r.performance.index[-1].date()),'validation_metrics':asdict(study['validation_metrics']),
             'holdout_metrics':asdict(r.strategy_metrics),'benchmark_metrics':asdict(r.benchmark_metrics),
             'mty_buy_hold_metrics':asdict(bh.strategy_metrics),'holdout_trades':r.trade_count,
             'holdout_closed_trades':int((r.trades.Status=='Closed').sum()),'holdout_win_rate':r.win_rate,
             'cash_interest':float(r.performance['Cash Interest'].sum()),'costs':float(r.performance.Costs.sum()),
             'ledger_reconciliation_error':float(r.performance.Equity.iloc[-1]-base.initial_capital-r.trades.PnL.sum()-r.performance['Cash Interest'].sum()),
             'candidate_count':study['candidate_count'],'walk_forward_folds':len(study['folds']),
             'trials_logged':len(study['trials']),'bootstrap':interval,'warnings':r.warnings,'limitations':study['limitations']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
