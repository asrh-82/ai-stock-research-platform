from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from Utils.opportunities import (FEATURES, OpportunityConfig, evaluate_evidence, export_opportunity,
    fit_at, forecast_opportunity, history_digest, make_dataset, training_indices,
    validate_config, validate_history, walk_predictions)


@pytest.fixture
def history():
    rng = np.random.default_rng(9)
    r = rng.normal([.0003, .0001], [.018, .01], (1500, 2))
    return pd.DataFrame(100*np.exp(r.cumsum(0)), columns=['asset', 'benchmark'],
                        index=pd.bdate_range('2020-01-02', periods=len(r)))


@pytest.mark.parametrize('field,value', [
    ('horizon', 1), ('horizon', True), ('min_train', 100), ('min_train', 252.5),
    ('max_train', 100), ('min_oos', 3), ('min_signals', 1), ('simulations', 100),
    ('seed', -1), ('seed', True), ('diagnostic_block', 0), ('ridge_penalty', 0),
    ('round_trip_bps', -1), ('round_trip_bps', np.nan), ('hurdle_bps', np.inf),
    ('confidence', .5), ('confidence', True),
])
def test_bad_config_rejected(field, value):
    with pytest.raises(ValueError):
        validate_config(replace(OpportunityConfig(), **{field: value}))


def test_train_order_rejected():
    with pytest.raises(ValueError):
        validate_config(OpportunityConfig(min_train=800, max_train=504))


@pytest.mark.parametrize('kind', ['zero', 'negative', 'nan', 'inf', 'duplicate', 'reverse', 'weekend', 'intraday', 'tz', 'columns', 'short', 'weekly', 'gap'])
def test_bad_history_rejected(history, kind):
    h = history.copy()
    if kind in ('zero','negative','nan','inf'):
        h.iloc[50,0] = dict(zero=0.,negative=-1.,nan=np.nan,inf=np.inf)[kind]
    elif kind == 'duplicate':
        h.index = pd.DatetimeIndex([h.index[0], h.index[0], *h.index[2:]])
    elif kind == 'reverse': h = h.iloc[::-1]
    elif kind == 'weekend': h.index = pd.date_range('2020-01-01', periods=len(h))
    elif kind == 'intraday': h.index = h.index + pd.Timedelta(hours=1)
    elif kind == 'tz': h.index = h.index.tz_localize('UTC')
    elif kind == 'columns': h = h.rename(columns={'asset':'price'})
    elif kind == 'short': h = h.iloc[:126]
    elif kind == 'weekly': h = h.iloc[::5]
    elif kind == 'gap': h = h.drop(h.index[30:50])
    with pytest.raises(ValueError): validate_history(h)


def test_target_has_real_execution_delay(history):
    _, y = make_dataset(history, 21)
    i = 700
    a = history.asset.iloc[i+22]/history.asset.iloc[i+1]-1
    b = history.benchmark.iloc[i+22]/history.benchmark.iloc[i+1]-1
    assert y.total_return.iloc[i] == pytest.approx(a)
    assert y.excess_return.iloc[i] == pytest.approx(a-b)
    assert y.tail(22).isna().all().all()


def test_features_do_not_see_future(history):
    altered = history.copy()
    altered.iloc[801:,0] *= np.exp(np.arange(len(altered)-801)*.01)
    a,_ = make_dataset(history,21)
    b,_ = make_dataset(altered,21)
    pd.testing.assert_frame_equal(a.iloc[:801],b.iloc[:801])


def test_training_scaler_and_predictions_do_not_see_future(history):
    c = OpportunityConfig()
    x,y = make_dataset(history, c.horizon)
    altered = history.copy()
    altered.iloc[901:] *= [17., .03]
    xx,yy = make_dataset(altered, c.horizon)
    a,b = fit_at(x,y,900,c),fit_at(xx,yy,900,c)
    for key in ('center','scale','coefficients','prediction'):
        np.testing.assert_array_equal(a[key],b[key])
    assert np.all(a['indices']+22 < 900)


def test_overlapping_unfinished_labels_are_purged(history):
    c = OpportunityConfig()
    x,y = make_dataset(history,21)
    chosen = training_indices(x,y,900,c)
    yy = y.copy()
    yy.iloc[chosen[-1]+1:] = 99999.
    np.testing.assert_array_equal(fit_at(x,y,900,c)['prediction'],fit_at(x,yy,900,c)['prediction'])


def test_non_overlapping_windows_and_training_boundaries(history):
    ledger,_ = walk_predictions(history, OpportunityConfig())
    assert (ledger.train_last_label_end < ledger.origin).all()
    assert (ledger.entry > ledger.origin).all()
    assert (ledger.exit.iloc[:-1].to_numpy() < ledger.entry.iloc[1:].to_numpy()).all()
    assert ledger.train_count.min() >= 504
    assert ledger.train_count.max() <= 1008


def test_ridge_learns_a_known_synthetic_relationship():
    rng=np.random.default_rng(42)
    x=pd.DataFrame(rng.normal(size=(1200,len(FEATURES))), columns=FEATURES)
    y=pd.DataFrame({'total_return': .02+.04*x.iloc[:,0], 'excess_return': -.01+.06*x.iloc[:,1]})
    fit=fit_at(x,y,1199,OpportunityConfig())
    target=y.iloc[-1].to_numpy()
    assert np.max(np.abs(fit['prediction']-target)) < .01
    assert np.linalg.norm(fit['prediction']-target) < np.linalg.norm(fit['baseline']-target)


def test_prediction_is_not_forced_to_zero(history):
    r=forecast_opportunity(history,'TEST')
    assert abs(r.summary['forecast_total']) > 1e-5
    assert abs(r.summary['forecast_excess']) > 1e-5
    np.testing.assert_allclose(r.contributions[['total_contribution','excess_contribution']].sum(),
                              [r.summary['forecast_total'],r.summary['forecast_excess']],atol=1e-14)


def test_costs_do_not_modify_the_learned_forecast(history):
    a=forecast_opportunity(history,'TEST')
    b=forecast_opportunity(history,'TEST',config=OpportunityConfig(round_trip_bps=100))
    assert a.summary['forecast_total'] == b.summary['forecast_total']
    assert a.summary['forecast_excess'] == b.summary['forecast_excess']
    assert a.summary['net_forecast_excess']-b.summary['net_forecast_excess'] == pytest.approx(.008)


def test_not_enough_training_never_fabricates(history):
    with pytest.raises(ValueError, match='fully matured'):
        forecast_opportunity(history.iloc[:500],'TEST')


def test_not_enough_oos_is_not_a_candidate(history):
    r=forecast_opportunity(history.iloc[:750],'TEST')
    assert r.summary['status']=='Unvalidated estimate'
    assert r.scenarios.empty
    assert r.summary['evidence']['oos_folds'] < 24


def test_identical_reference_rejected(history):
    with pytest.raises(ValueError): forecast_opportunity(history,'SPY','SPY')


def test_flat_history_cannot_pass_skill_gate(history):
    history[:]=100.
    r=forecast_opportunity(history,'TEST')
    assert r.summary['status']=='Unvalidated estimate'
    assert r.summary['evidence']['skill_vs_zero'] is None
    assert r.summary['forecast_total']==pytest.approx(0.)


def test_repeatability_and_inputs_unchanged(history):
    before=history.copy()
    a=forecast_opportunity(history,'TEST');b=forecast_opportunity(history,'TEST')
    assert a.summary==b.summary
    pd.testing.assert_frame_equal(a.scenarios,b.scenarios)
    pd.testing.assert_frame_equal(before,history)


def test_seed_changes_scenarios_not_point_forecast(history):
    a=forecast_opportunity(history,'TEST')
    b=forecast_opportunity(history,'TEST',config=OpportunityConfig(seed=19))
    assert a.summary['forecast_excess']==b.summary['forecast_excess']
    assert not a.scenarios.equals(b.scenarios)


def test_resampling_uses_paired_out_of_sample_errors_without_recentering(history):
    r=forecast_opportunity(history,'TEST')
    errors=r.oos[['realized_total','realized_excess']].to_numpy()-r.oos[['forecast_total','forecast_excess']].to_numpy()
    pred=np.array([r.summary['forecast_total'],r.summary['forecast_excess']])
    for v in r.scenarios.iloc[:100].to_numpy()-pred:
        assert np.any(np.all(np.isclose(errors,v,rtol=1e-12,atol=1e-12),axis=1))
    assert r.summary['residual_observations']==len(r.oos)
    assert len(r.scenarios)==5000


def _strong_evidence():
    actual=np.tile([.1,-.08,.07,-.06],15)
    pred=actual*.95
    selected=pred>.012
    return pd.DataFrame({'realized_excess':actual,'forecast_excess':pred,
                         'baseline_excess':np.zeros(60),'selected':selected,
                         'net_excess_if_selected':np.where(selected,actual-.002,0.)})


def test_gates_can_pass_real_signal_fixture_not_disabled_by_design():
    e=evaluate_evidence(_strong_evidence(),OpportunityConfig(),3)
    assert e['evidence_pass']
    assert e['skill_vs_zero']>.9


def test_bad_forecast_cannot_pass():
    oos=_strong_evidence();oos['forecast_excess']=-oos.realized_excess
    e=evaluate_evidence(oos,OpportunityConfig(),1)
    assert not e['evidence_pass']
    assert e['skill_vs_zero']<0


def test_family_correction_cannot_improve_evidence_bounds(history):
    oos,_=walk_predictions(history,OpportunityConfig())
    a=evaluate_evidence(oos,OpportunityConfig(),1)
    b=evaluate_evidence(oos,OpportunityConfig(),12)
    assert b['one_sided_tail'] < a['one_sided_tail']
    assert b['zero_mse_improvement_lower'] <= a['zero_mse_improvement_lower']
    assert b['mean_mse_improvement_lower'] <= a['mean_mse_improvement_lower']


@pytest.mark.parametrize('count',[0,13,True])
def test_bad_scan_family_rejected(count):
    with pytest.raises(ValueError): evaluate_evidence(_strong_evidence(),OpportunityConfig(),count)


def test_no_selections_cannot_pass():
    oos=_strong_evidence();oos['selected']=False;oos['net_excess_if_selected']=0.
    e=evaluate_evidence(oos,OpportunityConfig(),1)
    assert e['selected_net_excess_lower'] is None
    assert not e['evidence_pass']


def test_current_hurdle_gate_and_synthetic_gate(history):
    r=forecast_opportunity(history,'TEST',config=OpportunityConfig(hurdle_bps=1000),provenance={'synthetic':True})
    assert r.summary['status']!='Research candidate'
    assert any('Synthetic' in s for s in r.summary['reasons'])


def test_export_exact_replay_and_ledger(history):
    r=forecast_opportunity(history,'TEST')
    with ZipFile(BytesIO(export_opportunity(r))) as z:
        m=json.loads(z.read('manifest.json'))
        h=pd.read_csv(z.open('history.csv'),index_col='Date',parse_dates=['Date'],float_precision='round_trip')
        assert history_digest(h)==m['history_sha256']
        replay=forecast_opportunity(h,'TEST',config=OpportunityConfig(**m['config']))
        assert replay.summary==r.summary
        assert 'oos_predictions.csv' in z.namelist()
        assert 'conditional_terminal_scenarios.csv' in z.namelist()


def test_tampered_export_rejected(history):
    r=forecast_opportunity(history,'TEST')
    r.history.iloc[-1,0]*=2
    with pytest.raises(ValueError,match='changed'): export_opportunity(r)
