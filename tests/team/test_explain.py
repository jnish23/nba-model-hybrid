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
    assert '6.5 of that is the line moving since the open' in sentence or 'works against it by 6.5' in sentence


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
    f = _frame(3)
    e, _ = _explain(f, _bundle(f))
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
        assert len([i for i in at.info if 'No explanation available' in i.value]) == 2        # a Total tab and a Spread tab
    finally:
        for k in ('GAME_ID', 'HIDE', 'NO_EXPL_ID'):
            os.environ.pop(k, None)
