from dataclasses import replace
import numpy as np
import pandas as pd
from Utils.backtesting import BacktestConfig
from Utils.walk_forward import run_walk_forward, block_bootstrap_active_mean, pattern_diagnostics
from test_backtesting import make_prices


def candidates():
    base=BacktestConfig(fast_window=5,slow_window=20,momentum_lookback=10)
    return {'SMA':base,'Momentum':replace(base,strategy='Price Momentum')}


def test_holdout_cannot_change_selection_or_validation():
    a=make_prices(360)
    changed=a.copy()
    changed.iloc[-50:,changed.columns.get_indexer(['Open','High','Low','Close','Adj Close'])]*=2
    first=run_walk_forward(a,a,currency='CAD',candidates=candidates(),train_bars=100,test_bars=40,holdout_bars=50)
    second=run_walk_forward(changed,changed,currency='CAD',candidates=candidates(),train_bars=100,test_bars=40,holdout_bars=50)
    assert first['selected']==second['selected']
    pd.testing.assert_frame_equal(first['trials'],second['trials'])
    pd.testing.assert_frame_equal(first['validation'],second['validation'])
    assert first['validation'].index.max()<first['holdout_start']
    assert not first['validation'].index.has_duplicates
    assert (first['folds']['Train End']<first['folds']['Test Start']).all()


def test_bootstrap_is_seeded_and_paired():
    idx=pd.bdate_range('2020-01-01',periods=100)
    s=pd.Series(np.arange(100)/10000,index=idx)
    a=block_bootstrap_active_mean(s,s,draws=100)
    assert a['lower_95']==a['upper_95']==0
    assert a==block_bootstrap_active_mean(s,s,draws=100)


def test_pattern_diagnostics_does_not_fill_unobserved_future():
    result=pattern_diagnostics(make_prices(230))
    assert result.Observations.max()<=229
    assert set(result['Holding Sessions'])=={1,5,20}
