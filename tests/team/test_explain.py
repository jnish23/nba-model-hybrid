import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from team import explain as ex
from team import feature_groups as fg
from team import market_models as mm
from team import model_registry as mr

ROOT = Path(__file__).resolve().parent.parent.parent

SPREAD_FEATURES = ['odds_away_spread', 'diff_roster_bpm', 'off_offensiveRating_trend_home', 'def_OPP_FG3_PCT_20g_away', 'days_rest_home']
TOTAL_FEATURES = ['odds_total', 'odds_home_ml', 'off_fieldGoalsAttempted_5g_home', 'off_fieldGoalsAttempted_5g_away',
                  'off_turnovers_20g_away', 'inj_lost_new_home', 'season_game_number_home']


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    spread = np.round(rng.normal(-2, 6, n) * 2) / 2
    f = pd.DataFrame({
        'gameId': [f'00226{i:05d}' for i in range(n)],
        'odds_home_spread': spread, 'odds_away_spread': -spread, 'odds_total': np.round(rng.normal(225, 8, n)),
        'odds_home_ml': np.round(rng.normal(-120, 150, n)), 'diff_roster_bpm': rng.normal(0, 2, n),
        'off_offensiveRating_trend_home': rng.normal(0, 3, n), 'def_OPP_FG3_PCT_20g_away': rng.normal(0.36, 0.02, n),
        'days_rest_home': rng.integers(1, 5, n).astype(float), 'off_fieldGoalsAttempted_5g_home': rng.normal(88, 4, n),
        'off_fieldGoalsAttempted_5g_away': rng.normal(88, 4, n), 'off_turnovers_20g_away': rng.normal(13, 2, n),
        'inj_lost_new_home': rng.exponential(1, n), 'season_game_number_home': rng.integers(1, 80, n).astype(float),
        'season_game_number_away': rng.integers(1, 80, n).astype(float), 'days_rest_away': rng.integers(1, 5, n).astype(float),
        'inj_missing_home': 0.0, 'inj_missing_away': 0.0})
    f['target_point_diff'] = -f['odds_home_spread'] + 1.5 * f['diff_roster_bpm'] + rng.normal(0, 10, n)
    f['target_total_pts'] = f['odds_total'] + 0.5 * (f['off_fieldGoalsAttempted_5g_home'] - 88) + rng.normal(0, 15, n)
    return f


def _bundle(f):
    spread = lgb.LGBMRegressor(n_estimators=40, num_leaves=5, verbosity=-1, random_state=0).fit(f[SPREAD_FEATURES], mm.ResidualSpreadModel.make_target(f))
    total = lgb.LGBMRegressor(n_estimators=40, num_leaves=5, verbosity=-1, random_state=0).fit(f[TOTAL_FEATURES], f['target_total_pts'])
    return mr.ModelBundle(suffix='_test', ml=None, calibrator=None, spread=mm.ResidualSpreadModel(spread), total=total,
                          features={'ml': [], 'spread': SPREAD_FEATURES, 'total': TOTAL_FEATURES})


def _explain(f, bundle, move_total=-4.0, move_spread=1.0, **kw):
    current = pd.DataFrame({'spread_line': f['odds_home_spread'] + move_spread, 'total_line': f['odds_total'] + move_total})
    return ex.explain_markets(bundle, f, current, prediction_date='2026-10-21 09:00:00', **kw), current


# ------------------------------------------------------------------------------------------------ SHAP sums to the prediction

def test_shap_contributions_add_up_to_each_models_raw_prediction():
    f = _frame()
    b = _bundle(f)
    c, bias = ex.shap_contributions(b.total, f[TOTAL_FEATURES])
    assert np.allclose(c.sum(axis=1) + bias, b.total.predict(f[TOTAL_FEATURES]), atol=1e-8)
    c, bias = ex.shap_contributions(b.spread.model, f[SPREAD_FEATURES])
    assert np.allclose(c.sum(axis=1) + bias, b.spread.predict_residual(f[SPREAD_FEATURES]), atol=1e-8)
    assert np.allclose(c.sum(axis=1) + bias - f['odds_home_spread'], b.spread.predict(f[SPREAD_FEATURES], line=f['odds_home_spread'].to_numpy()), atol=1e-8)


# ------------------------------------------------------------------------------------------------ the bars add up to the edge

def test_breakdown_adds_up_to_model_minus_current_line_for_every_game():
    f = _frame(80)
    b = _bundle(f)
    e, current = _explain(f, b)
    for i in range(len(f)):
        gid = f['gameId'][i]
        for market, line, model in ((ex.TOTAL, current['total_line'][i], b.total.predict(f.loc[[i], TOTAL_FEATURES])[0]),
                                    (ex.SPREAD, -current['spread_line'][i],
                                     b.spread.predict(f.loc[[i], SPREAD_FEATURES], line=f['odds_home_spread'].to_numpy()[[i]])[0])):
            pieces = ex.game_pieces(e, gid, market)
            assert pieces['value'].iloc[0] == pytest.approx(line)
            assert ex.edge_of(pieces) == pytest.approx(model - line, abs=1e-6)
            # each step's value is the running total, and the steps (line + all contributions) reach the model's number
            assert pieces['contribution'].fillna(0).sum() + pieces['value'].iloc[0] == pytest.approx(model, abs=1e-6)
            steps = ex.chart_steps(pieces, top=3)
            assert steps[0][1] + sum(v for _, v, m in steps if m == 'relative') == pytest.approx(steps[-1][1], abs=1e-6)
            assert steps[-1][1] == pytest.approx(model, abs=1e-6)


def test_total_breakdown_separates_line_movement_from_trust_in_the_line():
    f = _frame(20)
    b = _bundle(f)
    e, current = _explain(f, b, move_total=-6.5)
    p = ex.game_pieces(e, f['gameId'][0], ex.TOTAL).set_index('feature')
    assert p.loc['line', 'value'] == pytest.approx(f['odds_total'][0] - 6.5)
    assert p.loc['movement', 'contribution'] == pytest.approx(6.5)               # open - current
    c, bias = ex.shap_contributions(b.total, f[TOTAL_FEATURES])
    assert p.loc['trust', 'contribution'] == pytest.approx(bias[0] + c[0, 0] - f['odds_total'][0])
    sentence = ex.summary_sentence(ex.game_pieces(e, f['gameId'][0], ex.TOTAL), ex.TOTAL)
    assert any(t in sentence for t in ('6.5 of that is the line moving since the open', 'works against it by 6.5', 'accounts for 6.5 points'))


def test_features_are_stored_per_game_with_labels_groups_and_percentiles():
    f = _frame(30)
    b = _bundle(f)
    ref = ex.build_reference(f, TOTAL_FEATURES + SPREAD_FEATURES)
    e, _ = _explain(f, b, reference=ref)
    rows = ex.game_features(e, f['gameId'][3], ex.TOTAL)
    assert set(rows['feature']) == set(TOTAL_FEATURES)
    assert (rows['group'] != '').all() and rows['label'].notna().all()
    assert rows['pctile'].between(0, 1).all()
    assert rows['contribution'].abs().is_monotonic_decreasing


def test_games_without_a_current_line_get_no_rows_for_that_market():
    f = _frame(10)
    b = _bundle(f)
    current = pd.DataFrame({'spread_line': f['odds_home_spread'], 'total_line': f['odds_total']})
    current.loc[2, 'total_line'] = np.nan
    e = ex.explain_markets(b, f, current, prediction_date='x')
    assert ex.game_pieces(e, f['gameId'][2], ex.TOTAL).empty
    assert not ex.game_pieces(e, f['gameId'][2], ex.SPREAD).empty


# ------------------------------------------------------------------------------------------------ flags

def test_flags_cover_line_move_early_season_missing_injuries_offseason_rest_and_history():
    f = _frame(10)
    f.loc[0, ['season_game_number_home', 'season_game_number_away', 'days_rest_home']] = [3, 2, 150]
    f.loc[0, 'inj_missing_away'] = 1.0
    f.loc[1, ['season_game_number_home', 'season_game_number_away']] = [40, 40]
    b = _bundle(f)
    hist = pd.DataFrame({'market': ['total'] * 100, 'abs_edge': np.full(100, 5.0), 'won': [True] * 53 + [False] * 47, 'season': ['2021-22'] * 100})
    e, _ = _explain(f, b, move_total=-4.0, history=hist)
    codes0 = set(ex.game_flags(e, f['gameId'][0], ex.TOTAL)['feature'])
    assert {'line_moved', 'early_season', 'injury_missing', 'offseason_rest'} <= codes0
    codes1 = set(ex.game_flags(e, f['gameId'][1], ex.TOTAL)['feature'])
    assert 'early_season' not in codes1 and 'injury_missing' not in codes1 and 'offseason_rest' not in codes1
    assert 'line_moved' in codes1
    assert 'line_moved' not in set(ex.game_flags(e, f['gameId'][1], ex.SPREAD)['feature'])      # spread moved 1.0, under the threshold
    e2, _ = _explain(f, b, move_total=-1.0)
    assert 'line_moved' not in set(ex.game_flags(e2, f['gameId'][1], ex.TOTAL)['feature'])


def test_history_context_uses_the_matching_edge_band_and_needs_enough_bets():
    hist = pd.DataFrame({'market': ['total'] * 60, 'abs_edge': np.full(60, 5.0), 'won': [True] * 33 + [False] * 27, 'season': ['2022-23'] * 60})
    rate, n, band, first, last = ex.history_for_edge(hist, 'total', 4.5)
    assert (round(rate, 3), n, band) == (0.55, 60, '4-6 pts')
    assert ex.history_for_edge(hist, 'total', 9.0) is None                       # no bets in the 8+ band
    assert ex.history_for_edge(hist, 'spread', 4.5) is None
    assert ex.history_for_edge(hist.head(10), 'total', 4.5) is None              # under 30 bets


def test_oos_history_loader_scores_the_models_side_and_drops_pushes(tmp_path):
    pd.DataFrame({'spread_pred': [5.0] * 4, 'odds_home_spread': [-2.0] * 4, 'home_score': [110, 100, 103, 102], 'away_score': [100] * 4,
                  'total_pred': [230.0, 230.0, 210.0, 230.0], 'odds_total': [220.0] * 4}).to_csv(tmp_path / 'oos_predictions_2024-25.csv', index=False)
    pd.DataFrame({'spread_pred': [1.0]}).to_csv(tmp_path / 'oos_predictions_2024-25_before_market_aware.csv', index=False)   # must not be read
    h = ex.load_oos_history(tmp_path)
    s = h[h['market'] == 'spread']
    assert list(s['abs_edge']) == [3.0, 3.0, 3.0] and list(s['won']) == [True, False, True]     # the 4th game is a push (home -2 wins by 2)
    t = h[h['market'] == 'total']
    assert list(t['won']) == [False, False, True, False]


# ------------------------------------------------------------------------------------------------ feature dictionary

def test_every_feature_of_the_deployed_model_set_has_a_mapping():
    cfg = mr.MODEL_DIR / f'feature_config{mr.DEPLOYED_SUFFIX}.json'
    if not cfg.exists():
        pytest.skip('deployed feature_config not available (model artifacts are not in the repo)')
    names = sorted({f for cols in json.loads(cfg.read_text()).values() for f in cols})
    unmapped = []
    for n in names:
        try:
            info = fg.describe_feature(n)
            assert info.group in fg.GROUP_ORDER and info.label
        except fg.UnmappedFeature:
            unmapped.append(n)
    assert not unmapped, f'add these to team/feature_groups.py: {unmapped}'


def test_feature_names_parse_into_group_label_and_side():
    i = fg.describe_feature('off_offensiveRating_trend_away')
    assert (i.group, i.label, i.side) == ('Shooting efficiency', 'Away offensive rating (recent trend)', 'away')
    i = fg.describe_feature('diff_def_OPP_FG3_PCT_20g')
    assert i.group == 'Defense allowed' and i.side == 'diff' and i.label.startswith('Home minus away') and '20 games' in i.label
    assert fg.describe_feature('odds_total').group == 'Market line'
    assert fg.describe_feature('days_rest_home').group == 'Rest / travel / schedule'
    assert fg.describe_feature('inj_missing_away').group == 'Injuries & roster'
    assert fg.describe_feature('win_pct_last10_home').group == 'Recent form'


def test_a_new_unknown_feature_fails_strictly_and_falls_back_to_other_in_production():
    with pytest.raises(fg.UnmappedFeature):
        fg.describe_feature('off_brandNewStat_5g_home')
    assert fg.describe_feature('off_brandNewStat_5g_home', strict=False).group == 'Other'
    assert fg.describe_feature('mystery', strict=False).group == 'Other'


# ------------------------------------------------------------------------------------------------ storage and export

def test_explanations_round_trip_through_the_database_and_refresh_per_game(tmp_path):
    f = _frame(12)
    b = _bundle(f)
    e, _ = _explain(f, b)
    db = tmp_path / 'x.db'
    assert ex.load_explanations(db).empty                                         # no table yet
    ex.save_explanations(db, e)
    back = ex.load_explanations(db)
    assert len(back) == len(e) and set(back.columns) == set(ex.COLUMNS)
    assert back['game_id'].str.len().eq(10).all()
    ex.save_explanations(db, e[e['game_id'] == f['gameId'][0]])                    # re-running one game does not duplicate it
    assert len(ex.load_explanations(db)) == len(e)


def _export_module():
    spec = importlib.util.spec_from_file_location('export_dashboard_data', ROOT / 'scripts' / 'export_dashboard_data.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_export_writes_both_files_and_only_the_recent_games_explanations(tmp_path):
    exp = _export_module()
    f = _frame(6)
    e, _ = _explain(f, _bundle(f))
    ids = e['game_id'].unique()
    db = tmp_path / 'nba.db'
    ex.save_explanations(db, e)
    preds = pd.DataFrame({'game_id': [ids[0], ids[1]], 'game_date': ['2026-10-21', '2026-09-01']})
    loaded = exp.load_explanations_table(preds, db_path=db, days=7)
    assert set(loaded['game_id']) == {ids[0]}                                      # the September game is older than 7 days
    exp.write_files(tmp_path / 'out', preds, loaded)
    assert (tmp_path / 'out' / 'predictions_log.csv').exists() and (tmp_path / 'out' / 'explanations.csv').exists()
    exp.write_files(tmp_path / 'out2', preds, pd.DataFrame(columns=ex.COLUMNS))     # nothing stored yet: predictions only, no empty file
    assert (tmp_path / 'out2' / 'predictions_log.csv').exists() and not (tmp_path / 'out2' / 'explanations.csv').exists()


def test_publish_puts_both_files_in_the_data_branch_commit(tmp_path, monkeypatch):
    exp = _export_module()
    f = _frame(4)
    e, _ = _explain(f, _bundle(f))
    calls = []

    def fake_git(*args, cwd=None, check=True):
        calls.append(args)
        if args[:2] == ('diff', '--cached'):
            return subprocess.CompletedProcess(args, 1)                            # something is staged
        return subprocess.CompletedProcess(args, 0, stdout='', stderr='')

    monkeypatch.setattr(exp, 'PUBLISH_DIR', tmp_path)
    monkeypatch.setattr(exp, 'ensure_worktree', lambda: None)
    monkeypatch.setattr(exp, 'git', fake_git)
    preds = pd.DataFrame({'game_id': ['0022600001'], 'game_date': ['2026-10-21']})
    assert exp.publish(preds, e) is True
    assert (tmp_path / 'predictions_log.csv').exists() and (tmp_path / 'explanations.csv').exists()
    assert any(c[0] == 'push' for c in calls)


# ------------------------------------------------------------------------------------------------ dashboard

def _load_dashboard():
    pytest.importorskip('streamlit')
    pytest.importorskip('plotly')
    spec = importlib.util.spec_from_file_location('dashboard_app', ROOT / 'dashboard' / 'app.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dashboard_handles_a_missing_explanations_file(tmp_path, monkeypatch):
    app = _load_dashboard()
    monkeypatch.setattr(app, 'DB_PATH', tmp_path / 'missing.db')
    monkeypatch.setattr(app, 'EXPLANATIONS_URL', (tmp_path / 'not_published.csv').as_uri())
    monkeypatch.setattr(app, 'EXPLANATIONS_SNAPSHOT_PATH', tmp_path / 'no_snapshot.csv')
    got = app._read_explanations()
    assert got.empty and list(got.columns) == ex.COLUMNS


def test_dashboard_falls_back_to_the_bundled_snapshot(tmp_path, monkeypatch):
    app = _load_dashboard()
    f = _frame(3)
    e, _ = _explain(f, _bundle(f))
    e.to_csv(tmp_path / 'explanations.csv', index=False)
    monkeypatch.setattr(app, 'DB_PATH', tmp_path / 'missing.db')
    monkeypatch.setattr(app, 'EXPLANATIONS_URL', (tmp_path / 'not_published.csv').as_uri())
    monkeypatch.setattr(app, 'EXPLANATIONS_SNAPSHOT_PATH', tmp_path / 'explanations.csv')
    assert len(app._read_explanations()) == len(e)


def test_panel_says_no_explanation_available_without_rows_and_draws_the_chart_with_them(tmp_path):
    pytest.importorskip('streamlit')
    pytest.importorskip('plotly')
    from streamlit.testing.v1 import AppTest
    f = _ml_frame(3)
    e, _ = _explain_ml(f, _ml_bundle(f))
    e.to_csv(tmp_path / 'explanations.csv', index=False)
    script = tmp_path / 'panel_script.py'
    script.write_text(
        "import importlib.util, os\n"
        "import pandas as pd\n"
        f"spec = importlib.util.spec_from_file_location('dashboard_app', r'{ROOT / 'dashboard' / 'app.py'}')\n"
        "app = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(app)\n"
        f"expl = pd.read_csv(r'{tmp_path / 'explanations.csv'}', dtype={{'game_id': str}})\n"
        "expl = expl[expl['game_id'] != os.environ['NO_EXPL_ID']] if os.environ.get('HIDE') else expl\n"
        "app.render_explanation(expl, {'game_id': os.environ['GAME_ID'], 'matchup': 'A @ B', 'home_team': 'B'})\n")
    gid = ex.normalize_game_ids(e)['game_id'].iloc[0]
    os.environ['GAME_ID'] = gid
    try:
        at = AppTest.from_file(str(script), default_timeout=60).run()
        assert not at.exception
        assert not [i for i in at.info if 'No explanation available' in i.value]
        os.environ['HIDE'], os.environ['NO_EXPL_ID'] = '1', gid
        at = AppTest.from_file(str(script), default_timeout=60).run()
        assert not at.exception
        assert len([i for i in at.info if 'No explanation available' in i.value]) == 3        # a Total, a Spread and a Moneyline tab
    finally:
        for k in ('GAME_ID', 'HIDE', 'NO_EXPL_ID'):
            os.environ.pop(k, None)


def test_headline_names_the_side_and_the_lean_strength():
    f = _frame(20)
    b = _bundle(f)
    e, current = _explain(f, b, move_total=-6.5, move_spread=1.0)
    gid = f['gameId'][0]
    for m in (ex.TOTAL, ex.SPREAD):
        p = ex.game_pieces(e, gid, m)
        edge = ex.edge_of(p)
        text = ex.summary_sentence(p, m, 'Home FC', 'Away FC')
        assert text.startswith(f"**{'Lean' if abs(edge) >= ex.LEAN_STRONG_PTS else 'Slight lean'} ")
        if m == ex.TOTAL:
            assert ('OVER' if edge > 0 else 'UNDER') in text.split('.')[0]
        else:
            line = float(p.loc[p['feature'] == 'line', 'value'].iloc[0])             # home margin the market implies
            want = f"Home FC {-line:+.1f}" if edge > 0 else f"Away FC {line:+.1f}"    # the side's own spread
            assert want in text.split('Model:')[0]


# ------------------------------------------------------------------------------------------------ phase 2: moneyline

ML_FEATURES = ['odds_away_spread', 'diff_roster_bpm', 'off_offensiveRating_trend_home', 'days_rest_home']


def _ml_bundle(f):
    b = _bundle(f)
    clf = lgb.LGBMClassifier(n_estimators=40, num_leaves=5, verbosity=-1, random_state=0).fit(f[ML_FEATURES], (f['target_point_diff'] > 0).astype(int))
    raw = clf.predict_proba(f[ML_FEATURES])[:, 1]
    margin = b.spread.predict(f[SPREAD_FEATURES], line=f['odds_home_spread'].to_numpy())
    cal = mm.MarginStackCalibrator().fit(raw, margin, 13.0, (f['target_point_diff'] > 0).astype(int))
    return mr.ModelBundle(suffix='_test', ml=clf, calibrator=cal, spread=b.spread, total=b.total,
                          features={'ml': ML_FEATURES, 'spread': SPREAD_FEATURES, 'total': TOTAL_FEATURES})


def _ml_frame(n=200):
    f = _frame(n)
    f['odds_away_ml'] = np.round(-f['odds_home_ml'] + 10)
    return f


def _explain_ml(f, b, shift=30.0, **kw):
    cur = pd.DataFrame({'spread_line': f['odds_home_spread'], 'total_line': f['odds_total'], 'home_ml': f['odds_home_ml'] + shift})
    return ex.explain_markets(b, f, cur, prediction_date='x', **kw), cur


def test_moneyline_breakdown_reproduces_the_calibrated_probability_exactly():
    f = _ml_frame()
    b = _ml_bundle(f)
    e, cur = _explain_ml(f, b)
    out = mm.predict_markets(b, f)
    for i in range(0, len(f), 7):
        p = ex.game_pieces(e, f['gameId'][i], ex.MONEYLINE)
        assert p['value'].iloc[-1] == pytest.approx(out['ml_prob'][i], abs=1e-9)                       # ends at the model's probability
        assert p['value'].iloc[0] == pytest.approx(float(ex.implied_prob(cur['home_ml'][i])))          # starts at what the current price implies
        steps = ex.chart_steps(p, top=3, scale=100.0)
        assert steps[0][1] + sum(v for _, v, m in steps if m == 'relative') == pytest.approx(steps[-1][1], abs=1e-6)


def test_moneyline_line_movement_step_goes_from_the_current_to_the_opening_price():
    f = _ml_frame(30)
    b = _ml_bundle(f)
    e, cur = _explain_ml(f, b, shift=40.0)
    p = ex.game_pieces(e, f['gameId'][0], ex.MONEYLINE).set_index('feature')
    assert p.loc['movement', 'value'] == pytest.approx(float(ex.implied_prob(f['odds_home_ml'][0])))
    assert p.loc['movement', 'contribution'] == pytest.approx(float(ex.implied_prob(f['odds_home_ml'][0]) - ex.implied_prob(cur['home_ml'][0])))


def test_moneyline_is_skipped_without_a_margin_calibrator_or_a_current_price():
    f = _ml_frame(10)
    b = _ml_bundle(f)
    no_ml = ex.explain_markets(b, f, pd.DataFrame({'spread_line': f['odds_home_spread'], 'total_line': f['odds_total']}), prediction_date='x')
    assert ex.game_pieces(no_ml, f['gameId'][0], ex.MONEYLINE).empty and not ex.game_pieces(no_ml, f['gameId'][0], ex.TOTAL).empty
    b.calibrator = type('Isotonic', (), {'uses_margin': False})()
    assert ex.game_pieces(_explain_ml(f, b)[0], f['gameId'][0], ex.MONEYLINE).empty


def test_implied_probability_and_moneyline_headline():
    assert ex.implied_prob(-150) == pytest.approx(0.6) and ex.implied_prob(150) == pytest.approx(0.4) and np.isnan(ex.implied_prob(0))
    f = _ml_frame(20)
    e, _ = _explain_ml(f, _ml_bundle(f))
    p = ex.game_pieces(e, f['gameId'][0], ex.MONEYLINE)
    text = ex.summary_sentence(p, ex.MONEYLINE, 'Home FC', 'Away FC')
    assert text.startswith('**') and 'moneyline.**' in text and ('Home FC' in text.split('.')[0] or 'Away FC' in text.split('.')[0])


def test_moneyline_history_context_is_built_from_the_oos_files(tmp_path):
    n = 200
    rng = np.random.default_rng(3)
    p = np.clip(rng.normal(0.6, 0.15, n), 0.05, 0.95)
    pd.DataFrame({'ml_prob': p, 'odds_home_ml': -150.0, 'odds_away_ml': 130.0, 'spread_pred': 1.0, 'odds_home_spread': -1.0, 'total_pred': 220.0,
                  'odds_total': 220.5, 'home_score': rng.integers(90, 130, n), 'away_score': rng.integers(90, 130, n)}).to_csv(tmp_path / 'oos_predictions_2024-25.csv', index=False)
    h = ex.load_oos_history(tmp_path)
    m = h[h['market'] == ex.MONEYLINE]
    assert len(m) > 50 and m['implied'].notna().all() and (m['abs_edge'] > 0).all()
    got = ex.history_for_ml_edge(h, float(m['abs_edge'].median()))
    assert got is None or (0 <= got[0] <= 1 and got[1] >= 30 and 0 < got[5] < 1)


# ------------------------------------------------------------------------------------------------ phase 2: results, drivers

def test_outcome_sentence_says_how_the_game_ended_and_whether_the_lean_was_right():
    f = _frame(10)
    e, _ = _explain(f, _bundle(f), move_total=0.0, move_spread=0.0)
    gid = f['gameId'][0]
    pt = ex.game_pieces(e, gid, ex.TOTAL)
    line = float(pt.loc[pt['feature'] == 'line', 'value'].iloc[0])
    lean_over = ex.edge_of(pt) > 0
    over = ex.outcome_sentence(pt, ex.TOTAL, 'H', 'A', 130, 100)               # 230 points: far over any line here
    assert 'OVER' in over and ('was right' if lean_over else 'was wrong') in over
    assert 'push' in ex.outcome_sentence(pt, ex.TOTAL, 'H', 'A', line, 0)
    assert ex.outcome_sentence(pt, ex.TOTAL, 'H', 'A', np.nan, np.nan) == ''
    ps = ex.game_pieces(e, gid, ex.SPREAD)
    assert 'covered' in ex.outcome_sentence(ps, ex.SPREAD, 'H', 'A', 130, 90)


def _drivers_fixture(n=600, seed=5):
    """A group whose pushes are right (signal), one whose pushes are random (noise), and one too small to count."""
    rng = np.random.default_rng(seed)
    rows, results = [], []
    for i in range(n):
        gid = f'{i:010d}'
        signal, noise = rng.normal(0, 3), rng.normal(0, 3)
        line = 220.0
        actual = line + 1.5 * signal + rng.normal(0, 10)
        for feature, label, value, c in (('line', 'Current line', line, np.nan), ('group:Signal', 'Signal', 0, signal),
                                         ('group:Noise', 'Noise', 0, noise), ('group:Tiny', 'Tiny', 0, 0.01)):
            rows.append(dict(game_id=gid, prediction_date='x', market='total', kind='piece', feature=feature, label=label, group='' if c != c else label,
                             value=value, contribution=c, pctile=np.nan, severity=''))
        results.append(dict(game_id=gid, home_score=actual / 2, away_score=actual / 2))
    return pd.DataFrame(rows, columns=ex.COLUMNS), pd.DataFrame(results)


def test_driver_stats_separates_a_group_that_works_from_noise_and_tiny_pushes():
    expl, results = _drivers_fixture()
    s = ex.driver_stats(expl, results, ex.TOTAL).set_index('label')
    assert s.loc['Signal', 'verdict'] == 'signal' and s.loc['Signal', 'hit_rate'] > 0.55 and s.loc['Signal', 'ci_low'] > 0
    assert s.loc['Noise', 'verdict'] == 'indistinguishable from noise'
    assert s.loc['Tiny', 'verdict'] == 'too few games' and s.loc['Tiny', 'games'] == 0
    assert s['share'].sum() == pytest.approx(1.0)
    assert list(s.index)[0] in ('Signal', 'Noise')                       # sorted by how much they push
    assert ex.driver_stats(expl, results.iloc[0:0], ex.TOTAL).empty       # no settled games: nothing to say


def test_driver_stats_for_the_moneyline_uses_win_minus_market_probability():
    rng = np.random.default_rng(8)
    rows, results = [], []
    for i in range(400):
        gid = f'{i:010d}'
        c = rng.normal(0, 0.05)
        p0 = 0.55
        home_win = rng.random() < p0 + 2 * c
        for feature, label, value, contrib in (('line', 'Current price', p0, np.nan), ('group:Signal', 'Signal', 0, c)):
            rows.append(dict(game_id=gid, prediction_date='x', market='moneyline', kind='piece', feature=feature, label=label, group='', value=value,
                             contribution=contrib, pctile=np.nan, severity=''))
        results.append(dict(game_id=gid, home_score=100 + 5 * home_win, away_score=100 + 5 * (not home_win)))
    s = ex.driver_stats(pd.DataFrame(rows, columns=ex.COLUMNS), pd.DataFrame(results), ex.MONEYLINE)
    assert s['hit_rate'].isna().all() and s['verdict'].iloc[0] == 'signal'


# ------------------------------------------------------------------------------------------------ phase 2: backfill and history files

def test_history_shards_keep_pieces_whole_and_only_the_biggest_features():
    exp = _export_module()
    f = _frame(8)
    e, _ = _explain(f, _bundle(f))
    e = ex.normalize_game_ids(e)
    preds = pd.DataFrame({'game_id': e['game_id'].unique(), 'game_date': ['2026-10-21', '2026-10-22', '2026-11-02', '2026-11-03', '2026-11-04',
                                                                         '2026-11-05', '2026-11-06', '2026-11-07']})
    shards = exp.history_shards(preds, e, top=3)
    assert set(shards) == {'2026-10.csv.gz', '2026-11.csv.gz'}
    allrows = pd.concat(shards.values())
    allrows = allrows[allrows['market'] != 'all']                                      # game-level flags have no feature rows
    for (gid, market), part in allrows.groupby(['game_id', 'market']):
        assert (part['kind'] == 'feature').sum() == 3
        full = e[(e['game_id'] == gid) & (e['market'] == market)]
        assert (part['kind'] == 'piece').sum() == (full['kind'] == 'piece').sum()                 # the pieces stay whole, so the chart still adds up
        feats = full[full['kind'] == 'feature']
        biggest = feats.reindex(feats['contribution'].abs().sort_values(ascending=False).index).head(3)
        assert set(part.loc[part['kind'] == 'feature', 'feature']) == set(biggest['feature'])


def test_export_writes_deterministic_monthly_history_files_and_a_manifest(tmp_path):
    exp = _export_module()
    f = _frame(4)
    e, _ = _explain(f, _bundle(f))
    e = ex.normalize_game_ids(e)
    preds = pd.DataFrame({'game_id': e['game_id'].unique(), 'game_date': ['2026-10-21'] * 4})
    exp.write_files(tmp_path / 'a', preds, e, e)
    exp.write_files(tmp_path / 'b', preds, e, e)
    manifest = json.loads((tmp_path / 'a' / 'explanations_manifest.json').read_text())
    assert [s['file'] for s in manifest['shards']] == ['2026-10.csv.gz']
    a = (tmp_path / 'a' / 'explanations_history' / '2026-10.csv.gz').read_bytes()
    assert a == (tmp_path / 'b' / 'explanations_history' / '2026-10.csv.gz').read_bytes()           # identical bytes: an unchanged month is no git change
    back = pd.read_csv(tmp_path / 'a' / 'explanations_history' / '2026-10.csv.gz', dtype={'game_id': str})
    assert len(back) == manifest['shards'][0]['rows']
    exp.write_files(tmp_path / 'c', preds, e)                                                       # no history given: no history files
    assert not (tmp_path / 'c' / 'explanations_manifest.json').exists()


def test_backfill_picks_only_unexplained_out_of_sample_games_and_counts_log_differences(tmp_path):
    import sqlite3
    spec = importlib.util.spec_from_file_location('backfill_explanations', ROOT / 'scripts' / 'backfill_explanations.py')
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    db = tmp_path / 'x.db'
    with sqlite3.connect(db) as conn:
        pd.DataFrame({'game_id': ['0022500001', '0022500002', '0022600001', '0022500003'],
                      'model_version': ['oos_backtest_2025-26'] * 2 + ['v1.0.0_hybrid', 'oos_backtest_2024-25'],
                      'spread_pred': 1.0, 'total_pred': 220.0, 'ml_prob': 0.5}).to_sql('predictions_log_v2', conn, index=False)
        todo = bf.games_to_explain(conn, {'0022500002'})
        assert list(todo['game_id']) == ['0022500001', '0022500003']                    # not the explained one, not the live model set
        assert list(bf.games_to_explain(conn, {'0022500002'}, season='2025-26')['game_id']) == ['0022500001']
        assert len(bf.games_to_explain(conn, {'0022500002'}, refresh=True)) == 3
        log = pd.read_sql("SELECT * FROM predictions_log_v2", conn)
    pieces = pd.DataFrame([dict(game_id='0022500001', market=m, kind='piece', feature='prediction', value=v)
                           for m, v in (('spread', 1.1), ('total', 223.0), ('moneyline', 0.5))])
    assert bf.differs_from_log(pieces, log) == (1, 1)                                   # the total is 3 points off the log


def test_dashboard_reads_published_history_files_when_there_is_no_database(tmp_path, monkeypatch):
    app = _load_dashboard()
    exp = _export_module()
    f = _frame(4)
    e, _ = _explain(f, _bundle(f))
    e = ex.normalize_game_ids(e)
    preds = pd.DataFrame({'game_id': e['game_id'].unique(), 'game_date': ['2026-10-21'] * 4})
    exp.write_files(tmp_path, preds, e, e)
    monkeypatch.setattr(app, 'DB_PATH', tmp_path / 'missing.db')
    monkeypatch.setattr(app, 'HISTORY_MANIFEST_URL', (tmp_path / 'explanations_manifest.json').as_uri())
    monkeypatch.setattr(app, 'HISTORY_DIR_URL', (tmp_path / 'explanations_history').as_uri() + '/')
    got = app._read_explanation_history()
    assert set(got['game_id']) == set(e['game_id']) and len(got) > 0
    monkeypatch.setattr(app, 'HISTORY_MANIFEST_URL', (tmp_path / 'nothing.json').as_uri())
    assert app._read_explanation_history().empty


def test_settled_panel_shows_the_result_and_the_logged_number_note(tmp_path):
    pytest.importorskip('streamlit')
    pytest.importorskip('plotly')
    from streamlit.testing.v1 import AppTest
    f = _ml_frame(3)
    e, _ = _explain_ml(f, _ml_bundle(f))
    e.to_csv(tmp_path / 'explanations.csv', index=False)
    script = tmp_path / 'settled_script.py'
    script.write_text(
        "import importlib.util, os\n"
        "import pandas as pd\n"
        f"spec = importlib.util.spec_from_file_location('dashboard_app', r'{ROOT / 'dashboard' / 'app.py'}')\n"
        "app = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(app)\n"
        f"expl = pd.read_csv(r'{tmp_path / 'explanations.csv'}', dtype={{'game_id': str}})\n"
        "game = {'game_id': os.environ['GAME_ID'], 'matchup': 'A @ B', 'home_team': 'B', 'away_team': 'A', 'home_score': 120, 'away_score': 100,\n"
        "        'total_pred': 150.0, 'spread_pred': 99.0, 'ml_prob': 0.01}\n"
        "app.render_explanation(expl, game, settled=True)\n")
    os.environ['GAME_ID'] = ex.normalize_game_ids(e)['game_id'].iloc[0]
    try:
        at = AppTest.from_file(str(script), default_timeout=60).run()
        assert not at.exception
        text = ' '.join(m.value for m in at.markdown)
        assert text.count('Final:') == 3                                                      # one result line per tab
        assert len([c for c in at.caption if 'the prediction log has' in c.value]) == 3         # the logged numbers are far from the explained ones
    finally:
        os.environ.pop('GAME_ID', None)


def test_first_game_of_the_season_is_flagged_as_a_roster_history_gap_not_a_missing_report():
    f = _frame(6)
    f.loc[0, ['season_game_number_home', 'inj_missing_home']] = [1, 1.0]            # first game: no roster history this season
    f.loc[0, ['season_game_number_away', 'inj_missing_away']] = [30, 1.0]           # report not submitted
    f.loc[1, ['season_game_number_home', 'season_game_number_away']] = [40, 40]
    f.loc[1, 'inj_missing_home'] = 1.0
    e, _ = _explain(f, _bundle(f))
    flags = ex.game_flags(e, f['gameId'][0], ex.TOTAL).set_index('feature')['label']
    assert 'First game of the season for the home team' in flags['injury_first_game'] and 'ignores injuries' in flags['injury_first_game']
    assert 'the away team' in flags['injury_missing'] and 'not submitted' in flags['injury_missing']
    only = set(ex.game_flags(e, f['gameId'][1], ex.TOTAL)['feature'])
    assert 'injury_missing' in only and 'injury_first_game' not in only
