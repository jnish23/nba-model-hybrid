"""WP4 of the props dashboard audit: date-clustered intervals, the design-effect band, model-version split (export + app), diagnostic tables, the freshness banner decision, labels, and that the
app still reads an OLD export without the new fields. No network (the live panel's ESPN fetch is stubbed) and no production data (the real-export check uses a copy and skips when none is found)."""
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import props_board_logic as L
from dashboard_props import logic as shim

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'dashboard_props' / 'app.py'
NOW = pd.Timestamp('2026-11-12T20:00:00Z')


def make(n_dates=20, per_date=30, seed=0, shock=0.0, win_p=0.5, dec=1.91, versions=None):
    """Live-style settled bets: one date-level shift in the win probability per date (`shock`) makes the bets of a date correlated."""
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for k in range(n_dates):
        p = float(np.clip(win_p + rng.normal(0, shock), 0.03, 0.97)) if shock else win_p
        for j in range(per_date):
            win = rng.random() < p
            side = 'over' if j % 3 == 0 else 'under'
            rows.append({'bet_id': f'b{i}', 'season': '2026-27', 'game_id': f'002260{k:02d}{j % 5:02d}', 'game_date': f'2026-10-{22 + k:02d}' if k < 10 else f'2026-11-{k - 9:02d}',
                         'tip_ref_utc': f'2026-10-{22 + k:02d}T23:00:00Z' if k < 10 else f'2026-11-{k - 9:02d}T23:00:00Z', 'minutes_before': [240, 60, 5][(j // 3) % 3], 'player_id': i, 'player_name': f'P{i}',
                         'stat': ['rebounds', 'assists', 'threes', 'ra'][j % 4], 'side': side, 'line': 5.5, 'book': 'draftkings', 'matchup': 'A @ B', 'dec_best': dec + 0.1, 'dec_median': dec,
                         'dec_flat': L.FLAT_110, 'ev_best': 0.03 + 0.001 * (j % 8) * 1.2, 'ev_median': 0.0, 'ev_flat': -0.01, 'p_over_model': 0.4, 'p_over_star': 0.45, 'p_over_market': 0.5,
                         'n_at_line': 5, 'n_quotes': 6, 'expected_minutes': 30.0, 'minutes_actual': 31.0, 'home_team': 'B', 'away_team': 'A', 'line_240': 5.5, 'line_60': 5.5, 'line_5': 5.5, 'mkt_240': 0.5, 'mkt_60': 0.5, 'mkt_5': 0.5, 'push': 0.0,
                         'status_stands': 'win' if win else 'loss', 'status_flagged': 'win' if win else 'loss', 'status_all': 'win' if win else 'loss', 'y': 6.0, 'played': 1,
                         'clv_prob': 0.004 if j % 2 else np.nan, 'line_move_favour': np.nan, 'close_line': 5.5, 'model_version': (versions or ['v1'])[k % len(versions or ['v1'])]})
            i += 1
    return pd.DataFrame(rows)


# --- date-clustered bootstrap -------------------------------------------------------------------------------------------------------------------------------------------------

def test_shim_exports_the_new_functions():
    for name in ('date_cluster_roi', 'freshness_status', 'diagnostic_table', 'ev_claim_table', 'slot_summary', 'line_move_summary', 'version_summary', 'version_meta_table', 'short_hash'):
        assert getattr(shim, name) is getattr(L, name)


def test_date_cluster_roi_is_deterministic_and_counts_only_stakes():
    d = make(shock=0.2, seed=1)
    d.loc[:20, 'status_stands'] = ['push', 'void', 'win'] * 7
    a, b = L.date_cluster_roi(d, 'best', n_boot=400, seed=3), L.date_cluster_roi(d, 'best', n_boot=400, seed=3)
    assert a == b and L.date_cluster_roi(d, 'best', n_boot=400, seed=4)['lo'] != a['lo']
    w = L.with_profit(d, 'best', 'stands')
    assert a['stakes'] == int(w['stake'].sum()) < len(d)                         # pushes and voids are not stakes
    assert a['roi'] == pytest.approx(w['profit'].sum() / w['stake'].sum()) and a['clusters'] == d['game_date'].nunique()
    assert a['lo'] < a['roi'] < a['hi']


def test_clustered_interval_is_wider_than_independent_on_clustered_data_and_similar_on_independent_data():
    clustered, indep = make(shock=0.25, seed=2), make(shock=0.0, seed=2)
    wc = L.date_cluster_roi(clustered, 'median', n_boot=600)
    ic = L.independent_roi_interval(clustered, 'median', n_boot=600)
    assert (wc['hi'] - wc['lo']) > 1.8 * (ic['hi'] - ic['lo'])
    wi = L.date_cluster_roi(indep, 'median', n_boot=600)
    ii = L.independent_roi_interval(indep, 'median', n_boot=600)
    assert 0.7 < (wi['hi'] - wi['lo']) / (ii['hi'] - ii['lo']) < 1.35
    assert wc['roi'] == pytest.approx(ic['roi'])                                  # same point estimate, only the interval differs


def test_roi_summary_uses_dates_and_keeps_its_old_keys():
    d = make(shock=0.2, seed=3)
    s = L.roi_summary(d, 'best', 'stands', n_boot=200)
    assert {'bets', 'win_rate', 'roi', 'lo', 'hi', 'units', 'player_games'} <= set(s) and s['clusters'] == 20
    by_game = L.roi_summary(d, 'best', 'stands', n_boot=200, cluster='game_id')
    assert by_game['clusters'] == d['game_id'].nunique() and by_game['roi'] == pytest.approx(s['roi'])
    assert L.roi_summary(d.assign(status_stands='void'), 'best')['bets'] == 0
    assert L.roi_summary(d.drop(columns='game_date'), 'best', n_boot=50)['clusters'] == d['game_id'].nunique()      # no dates: falls back to games


def test_design_effect_and_cumulative_band():
    clustered, indep = make(shock=0.25, seed=4), make(shock=0.0, seed=4)
    assert L.design_effect(clustered, 'median') > 2.0 and L.design_effect(indep, 'median') < 1.6
    wide = L.cumulative(clustered, 'median')
    narrow = L.cumulative(clustered, 'median', deff=None)
    assert wide.attrs['design_effect'] == pytest.approx(L.design_effect(clustered, 'median')) and narrow.attrs['design_effect'] == 1.0
    assert ((wide['hi'] - wide['lo']) >= (narrow['hi'] - narrow['lo']) - 1e-9).all() and (wide['hi'] - wide['lo']).iloc[-1] > 1.4 * (narrow['hi'] - narrow['lo']).iloc[-1]
    assert wide['n'].iloc[-1] == len(clustered) and (wide['lo'] <= wide['cum_units']).all()
    fixed = L.cumulative(clustered, 'median', deff=4.0)
    assert (fixed['hi'] - fixed['lo']).iloc[-1] == pytest.approx(2 * (narrow['hi'] - narrow['lo']).iloc[-1])


def test_noise_note_old_and_new_forms():
    assert '1.0 points' in L.noise_note(10000) and 'not' in L.noise_note(10000) and 'No settled bets' in L.noise_note(0)
    n = L.noise_note(1304, (-0.022, 0.085), 18)
    assert '18 dates' in n and '-2.2%' in n and '+8.5%' in n and 'includes zero' in n and 'too narrow' in n
    assert 'excludes zero' in L.noise_note(1000, (0.01, 0.05), 10)
    assert 'standard error' in L.noise_note(500, (np.nan, np.nan), 5)               # unusable interval: falls back to the labelled floor


# --- diagnostics ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_diagnostic_tables_by_side_slot_and_stat():
    d = make(shock=0.1, seed=5)
    d['p_over_market'] = np.where(d['side'] == 'over', 0.55, 0.45)                # market P of the bet side: 0.55 for overs, 0.55 for unders too -> side P 0.55 / 0.55
    d['p_over_star'] = 0.6
    t = L.diagnostic_table(d, 'side')
    assert list(t['side']) == ['over', 'under'] and t['stakes'].sum() == len(d)
    for side, row in t.set_index('side').iterrows():
        g = d[d['side'] == side]
        assert row['stakes'] == len(g) and row['win_rate'] == pytest.approx((g['status_stands'] == 'win').mean())
        assert row['p_market'] == pytest.approx(0.55 if side == 'over' else 0.55) and row['p_shrunk'] == pytest.approx(0.6 if side == 'over' else 0.4)
        w = L.with_profit(g, 'median', 'stands')
        assert row['roi_median'] == pytest.approx(w['profit'].sum() / w['stake'].sum()) and row['roi_median_lo'] < row['roi_median'] < row['roi_median_hi']
        assert row['clv_n'] == int(g['clv_prob'].notna().sum()) and row['clv_pts'] == pytest.approx(0.4) and not row['too_small']
    slot = L.diagnostic_table(d, 'slot')
    assert list(slot['slot']) == ['T-4h', 'T-1h', 'T-5m']                          # decision-time order, not alphabetical
    cell = L.diagnostic_table(d, ['side', 'slot'])
    assert len(cell) == 6 and cell['stakes'].sum() == len(d) and list(cell['side'].iloc[:3]) == ['over'] * 3
    assert L.diagnostic_table(d, 'stat')['stat'].tolist() == ['assists', 'ra', 'rebounds', 'threes']


def test_small_cells_are_flagged_not_dropped():
    d = make(n_dates=4, per_date=15, seed=6)                                        # 60 bets: side cells have 20 / 40, side x slot cells 6-14
    t = L.diagnostic_table(d, ['side', 'slot'])
    assert len(t) == 6 and t['too_small'].all()
    side = L.diagnostic_table(d, 'side')
    assert side['too_small'].all() and side['stakes'].sum() == 60                   # 20 and 40 stakes: both under 50, still present
    assert not L.diagnostic_table(make(n_dates=10, per_date=30), 'side')['too_small'].any()
    assert L.diagnostic_table(d, 'side', min_stakes=10)['too_small'].sum() == 0


def test_diagnostics_skip_pending_and_survive_missing_columns():
    d = make(n_dates=6, per_date=20)
    d.loc[:9, 'status_stands'] = np.nan
    assert L.diagnostic_table(d, 'side')['stakes'].sum() == len(d) - 10
    slim = d.drop(columns=['p_over_market', 'p_over_star', 'clv_prob', 'ev_best', 'close_line', 'line_move_favour'])
    t = L.diagnostic_table(slim, 'side')
    assert t['p_market'].isna().all() and t['clv_pts'].isna().all() and (t['clv_n'] == 0).all() and t['stakes'].sum() > 0
    assert L.ev_claim_table(slim).empty and L.line_move_summary(slim).empty
    assert L.diagnostic_table(d.iloc[0:0], 'side').empty and L.diagnostic_table(d.assign(status_stands=np.nan), ['side', 'slot']).empty
    assert L.diagnostic_table(d.drop(columns='status_stands'), 'side').empty


def test_ev_claim_table_buckets():
    d = make(n_dates=10, per_date=16, seed=7)
    d['ev_best'] = np.tile([0.031, 0.045, 0.055, 0.09], 40)
    t = L.ev_claim_table(d)
    assert list(t['ev_bucket']) == ['3-4%', '4-5%', '5-7%', '7%+'] and list(t['stakes']) == [40, 40, 40, 40]
    assert t['claimed_ev'].tolist() == pytest.approx([0.031, 0.045, 0.055, 0.09]) and bool(t['too_small'].all())
    big = d.loc[d.index.repeat(2)].reset_index(drop=True)
    assert not L.ev_claim_table(big)['too_small'].any()
    assert (t['roi_best'] > t['roi_median']).all()                                  # the best price is 0.1 higher than the median in make()
    below = L.ev_claim_table(d.assign(ev_best=0.02))
    assert list(below['ev_bucket'])[0] == '<3%' and below['stakes'].iloc[0] == len(d) and below['stakes'].iloc[1:].sum() == 0
    edge = L.ev_claim_table(d.assign(ev_best=0.04))                                  # an edge sits in the upper bucket
    assert edge.set_index('ev_bucket').loc['4-5%', 'stakes'] == len(d)


def test_slot_summary_last_seven_days_with_a_fixed_clock():
    log = pd.DataFrame({'game_id': list('abcdefg'), 'slot_minutes': [240, 240, 60, 60, 5, 5, 5], 'status': ['done', 'missed', 'done', 'no_quotes', 'missed', 'missed', 'done'],
                        'written_utc': ['2026-11-12T10:00:00Z', '2026-11-06T20:00:00Z', '2026-11-12T18:00:00Z', '2026-11-11T18:00:00Z', '2026-11-05T19:59:00Z', '2026-11-10T00:00:00Z', '2026-11-12T19:00:00Z']})
    s = L.slot_summary(log, NOW, days=7).set_index('slot')
    assert s.loc['T-4h', ['done', 'missed', 'no_quotes']].tolist() == [1, 1, 0]
    assert s.loc['T-1h', ['done', 'missed', 'no_quotes']].tolist() == [1, 0, 1]
    assert s.loc['T-5m', ['done', 'missed', 'no_quotes']].tolist() == [1, 1, 0]            # the row from Nov 5 19:59 is just outside 7 days
    assert s.loc['all', 'total'] == 6 and s.loc['all', 'done'] == 3 and s.loc['all', 'done_share'] == pytest.approx(0.5)
    for empty in (None, pd.DataFrame(), pd.DataFrame(columns=['slot_minutes'])):
        z = L.slot_summary(empty, NOW)
        assert list(z['slot']) == ['T-4h', 'T-1h', 'T-5m', 'all'] and (z['total'] == 0).all() and z['done_share'].isna().all()


def test_line_move_summary():
    d = pd.DataFrame({'minutes_before': [240, 240, 240, 60, 60, 5], 'close_line': [5.5, 5.5, 6.5, 4.5, np.nan, 5.5], 'line_move_favour': [np.nan, 1.0, -2.0, np.nan, np.nan, np.nan]})
    t = L.line_move_summary(d).set_index('group')
    assert t.loc['all', 'bets_with_close'] == 4 and t.loc['all', 'moved'] == 2 and t.loc['all', 'share_moved'] == 0.5      # T-5 bet and the bet without a close are excluded
    assert t.loc['all', 'share_in_favour'] == 0.5 and t.loc['all', 'mean_line_move_favour'] == pytest.approx(-0.5)
    assert t.loc['T-4h', 'moved'] == 2 and t.loc['T-1h', 'bets_with_close'] == 1 and np.isnan(t.loc['T-1h', 'mean_line_move_favour'])


# --- model versions -------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_version_tables_from_meta_and_from_an_old_export():
    d = make(n_dates=6, per_date=10, versions=['v1', 'v2'])
    meta = {'model_versions': [{'version': 'v1', 'first_game_date': '2026-10-22', 'last_game_date': '2026-10-26', 'bets': 30, 'provenance': L.RESTORED_PROVENANCE},
                               {'version': 'v2', 'first_game_date': '2026-10-23', 'last_game_date': '2026-10-27', 'bets': 30, 'provenance': L.LIVE_PROVENANCE}]}
    t = L.version_meta_table(meta, d)
    assert list(t['version']) == ['v1', 'v2'] and L.restored_versions(t) == ['v1']
    old = L.version_meta_table({}, d)                                               # an export without model_versions in meta
    assert list(old['version']) == ['v1', 'v2'] and set(old['provenance']) == {L.UNKNOWN_PROVENANCE} and L.restored_versions(old) == [] and list(old['bets']) == [30, 30]
    assert list(L.version_meta_table({}, d.drop(columns='model_version'))['version']) == ['unknown']
    assert L.version_meta_table({}, d.iloc[0:0]).empty and L.restored_versions(L.version_meta_table({'model_versions': 'junk'}, d.iloc[0:0])) == []
    vs = L.version_summary(d)
    assert list(vs['model_version']) == ['all', 'v1', 'v2'] and vs['bets'].tolist() == [60, 30, 30] and vs['stakes'].tolist() == [60, 30, 30] and vs.loc[1:, 'too_small'].all()
    assert list(L.version_summary(d.drop(columns='model_version'))['model_version']) == ['all', 'unknown']
    assert list(L.version_series(d.assign(model_version=[None] * 60))) == ['unknown'] * 60


def export_helpers():
    spec = importlib.util.spec_from_file_location('live_export_helpers', Path(__file__).with_name('test_live_export.py'))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


def test_export_writes_model_versions_meta_and_the_per_version_summary(tmp_path):
    from props.live import export, store
    h = export_helpers()
    conn = store.connect(tmp_path / 'live.db')
    old = [h._bet(1, model_version='v_old', game_date='2026-10-22'), h._bet(2, model_version='v_old', game_date='2026-10-23')]           # no snapshot_ts_utc: restored rows
    new = [h._bet(3, model_version='v_new', game_date='2026-11-04', snapshot_ts_utc='2026-11-04T23:00:00Z'), h._bet(4, model_version='v_new', game_date='2026-11-05', snapshot_ts_utc=None)]
    store.write_slot(conn, 'g1', 60, 'done', '2026-11-05T00:42:00Z', 'r1', bets=pd.concat(old + new))
    res = [{'bet_id': f'S|g1|{p}|assists', 'settled_utc': 'x', 'y': 6.0, 'played': 1, 'minutes_actual': 30.0, 'status_stands': s, 'status_flagged': s, 'status_all': s,
            'profit_stands_best': 1.0, 'clv_prob': 0.01, 'run_id': 'r2'} for p, s in ((1, 'win'), (2, 'loss'), (3, 'win'))]
    with conn:
        store._insert(conn, 'prop_results', pd.DataFrame(res))
    export.export_live(tmp_path / 'live.db', tmp_path / 'out', h.CONFIG, n_boot=30)
    meta = json.loads((tmp_path / 'out' / 'meta.json').read_text())
    assert meta['model_version'] == 'v'                                              # the pinned config's version is still reported
    mv = {r['version']: r for r in meta['model_versions']}
    assert list(mv) == ['v_old', 'v_new'] and mv['v_old']['bets'] == 2 and mv['v_old']['first_game_date'] == '2026-10-22' and mv['v_old']['last_game_date'] == '2026-10-23'
    assert mv['v_old']['provenance'] == 'restored from dashboard export 2026-11 (not re-derivable)'
    assert mv['v_new']['provenance'] == 'recorded live'                              # one of its two bets has a snapshot time
    sm = pd.read_csv(tmp_path / 'out' / 'season_summary.csv')
    assert list(sm['model_version']) == ['all', 'v_old', 'v_new'] and sm.loc[0, 'bets'] == 3 and sm['bets'].tolist() == [3, 2, 1]
    assert set(export.SUMMARY_COLS[:-1]) <= set(sm.columns) and 'roi_median_stands' in sm.columns
    assert json.dumps(meta)                                                          # serialisable
    assert export.model_versions_meta(pd.DataFrame()) == []
    nocol = export.model_versions_meta(pd.DataFrame({'game_date': ['2026-10-22'], 'model_version': [None]}))
    assert nocol[0]['version'] == 'unknown' and nocol[0]['provenance'] == L.RESTORED_PROVENANCE


# --- freshness banner -----------------------------------------------------------------------------------------------------------------------------------------------------------

def fresh_bets(tips, status):
    return pd.DataFrame({'tip_ref_utc': tips, 'status_stands': status, 'y': np.nan})


def test_freshness_decision_with_fixed_clocks():
    built_old, built_new = '2026-11-12T19:10:00Z', '2026-11-12T19:20:00Z'              # 50 and 40 minutes before NOW
    soon = fresh_bets(['2026-11-13T02:00:00Z'], [np.nan])                              # tips in 6 h, unsettled
    r = L.freshness_status(built_old, soon, NOW)
    assert r['age_min'] == pytest.approx(50.0) and r['stale'] and r['n_open_near'] == 1 and r['warn']
    assert not L.freshness_status(built_new, soon, NOW)['warn'] and not L.freshness_status(built_new, soon, NOW)['stale']       # 40 min old: only the caption
    assert not L.freshness_status('2026-11-12T19:15:00Z', soon, NOW)['stale']          # exactly 45 min is not older than 45
    assert L.freshness_status('2026-11-12T19:14:59Z', soon, NOW)['stale']
    far = fresh_bets(['2026-11-13T04:00:01Z'], [np.nan])                                # 8 h 0 min 1 s ahead: outside the window
    assert not L.freshness_status(built_old, far, NOW)['warn'] and L.freshness_status(built_old, far, NOW)['stale']
    assert L.freshness_status(built_old, fresh_bets(['2026-11-13T04:00:00Z'], [np.nan]), NOW)['warn']          # exactly 8 h ahead counts
    assert L.freshness_status(built_old, fresh_bets(['2026-11-12T16:00:00Z'], [np.nan]), NOW)['warn']          # tipped exactly 4 h ago counts
    assert not L.freshness_status(built_old, fresh_bets(['2026-11-12T15:59:59Z'], [np.nan]), NOW)['warn']      # tipped just over 4 h ago does not
    assert not L.freshness_status(built_old, fresh_bets(['2026-11-12T22:00:00Z'], ['win']), NOW)['warn']        # settled bets do not count
    mixed = fresh_bets(['2026-11-12T22:00:00Z', '2026-11-12T23:00:00Z', '2026-11-14T00:00:00Z'], ['win', np.nan, np.nan])
    assert L.freshness_status(built_old, mixed, NOW)['n_open_near'] == 1
    assert L.freshness_status(built_old, mixed, NOW, stale_after_min=60)['warn'] is False
    assert L.freshness_status(built_old, mixed.drop(columns='status_stands'), NOW)['n_open_near'] == 2          # no status column: open means no y


def test_freshness_decision_degrades_gracefully():
    assert L.freshness_status(None, None, NOW) == {'age_min': None, 'stale': False, 'n_open_near': 0, 'warn': False}
    assert L.freshness_status('not a time', fresh_bets(['2026-11-12T22:00:00Z'], [np.nan]), NOW)['warn'] is False
    r = L.freshness_status('2026-11-12T10:00:00Z', None, NOW)
    assert r['stale'] and r['n_open_near'] == 0 and not r['warn']
    assert not L.freshness_status('2026-11-12T10:00:00Z', pd.DataFrame({'a': [1]}), NOW)['warn']
    assert L.age_text(38) == '38 min' and L.age_text(125) == '2 h 05 min'


def test_short_hash_guards_none():
    assert L.short_hash(None) == 'n/a' and L.short_hash(float('nan')) == 'n/a' and L.short_hash('') == 'n/a'
    assert L.short_hash('761705dcca2b', 8) == '761705dc' and L.short_hash('abc', 8) == 'abc'


# --- the app: new export, old export, real export copy ----------------------------------------------------------------------------------------------------------------------------------

def write_export(path, bets, meta_extra=None, new=True, built_at=None):
    """A live-mode export folder. new=False writes the pre-WP4 shape: no model_versions in meta, no model_version column, no raw model probability, no summary model_version column."""
    path.mkdir(parents=True, exist_ok=True)
    b = bets.copy()
    if not new:
        b = b.drop(columns=['model_version', 'p_over_model', 'close_line', 'line_move_favour', 'clv_prob'])
    b.to_csv(path / 'bets.csv', index=False)
    row = {'season': '2026-27', 'kind': 'live paper record (settled bets only)', 'bets': 40, 'roi_best_stands': 0.03, 'roi_median_stands': -0.01, 'roi_flat_stands': -0.01, 'clv_real_pts': 0.3}
    if new:
        row['model_version'] = 'all'
    pd.DataFrame([row]).to_csv(path / 'season_summary.csv', index=False)
    pd.DataFrame(columns=['season', 'mean_p_model', 'hit_rate']).to_csv(path / 'reliability.csv', index=False)
    pd.DataFrame(columns=['season', 'month', 'stat', 'ratio']).to_csv(path / 'bias_monthly.csv', index=False)
    meta = {'mode': 'live', 'built_at_utc': built_at or pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), 'rule_sha256': 'a' * 64, 'features_db_sha256': 'b' * 64, 'code_commit': 'c' * 40,
            'model_version': 'v2', 'disclosure': 'test disclosure', 'record_started_utc': '2026-10-22T00:00:00Z', 'pending': 0,
            'rule': {'stats': ['rebounds', 'assists', 'threes', 'ra'], 'primary': {'e': 0.03, 'min_books': 3}, 'k_by_stat': {'rebounds': 0.26, 'assists': 0.29, 'threes': 0.31, 'ra': 0.23}}}
    if new:
        meta['model_versions'] = [{'version': 'v1', 'first_game_date': '2026-10-22', 'last_game_date': '2026-10-31', 'bets': int((b['model_version'] == 'v1').sum()), 'provenance': L.RESTORED_PROVENANCE},
                                  {'version': 'v2', 'first_game_date': '2026-11-01', 'last_game_date': '2026-11-12', 'bets': int((b['model_version'] == 'v2').sum()), 'provenance': L.LIVE_PROVENANCE}]
    meta.update(meta_extra or {})
    (path / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    (path / 'pmf.json').write_text(json.dumps({x: [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02] for x in b['bet_id']}), encoding='utf-8')
    (path / 'books.json').write_text(json.dumps({x: {'draftkings': 2.05, 'fanduel': 1.91, 'betmgm': 1.87} for x in b['bet_id']}), encoding='utf-8')
    pd.DataFrame(columns=['game_id', 'slot_minutes', 'status', 'detail', 'run_id', 'written_utc']).to_csv(path / 'slot_log.csv', index=False)


def live_bets():
    """60 settled bets (two versions) plus 6 pending bets that tip about 2 hours from now."""
    d = make(n_dates=20, per_date=3, versions=['v1', 'v1', 'v2'], seed=9)
    d['model_version'] = np.where(d['game_date'] < '2026-11-01', 'v1', 'v2')
    now = pd.Timestamp.now(tz='UTC')
    p = make(n_dates=1, per_date=6, seed=10, versions=['v2']).assign(bet_id=[f'pend{i}' for i in range(6)], player_id=range(900, 906), game_id='0022609999',
                                                                      game_date=str(now.tz_convert('America/New_York').date()), tip_ref_utc=(now + pd.Timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                                                                      status_stands=np.nan, status_flagged=np.nan, status_all=np.nan, y=np.nan, played=0, clv_prob=np.nan)
    return pd.concat([d, p], ignore_index=True)


def run_app(monkeypatch, folder):
    at = pytest.importorskip('streamlit.testing.v1')
    pytest.importorskip('plotly')
    import core.live_feed as F
    monkeypatch.setattr(F, '_get', lambda path, params=None: None)                  # never reach ESPN from a test
    monkeypatch.setenv('PROPS_DASH_LIVE', str(folder))
    monkeypatch.setenv('PROPS_DASH_DATA', str(folder) + '_no_backtest')
    app = at.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    return app


def check_all_pages(app):
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.tabs) == 4                                                       # Board, Tracker, Model health, Rules & about all rendered in the run
    text = ' '.join(m.value for m in app.markdown) + ' '.join(c.value for c in app.caption)
    labels = [m.label for m in app.metric]
    assert 'ROI, median price (primary)' in labels and 'ROI, best quoted price (not an assumed fill)' in labels
    assert 'Shrunk P(side)' in labels and not any(l == 'Model P(side)' for l in labels)
    assert 'median-price gate' in text and 'about 10 minutes before tip' in text and 'backfilled on 2026-11-12' in text and 'no longer a clean held-out season' in text
    assert 'date-clustered' in text and 'design effect' in text
    return text, labels


def test_app_on_a_new_format_export(tmp_path, monkeypatch):
    write_export(tmp_path / 'new', live_bets(), built_at=(pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ'))
    app = run_app(monkeypatch, tmp_path / 'new')
    text, labels = check_all_pages(app)
    assert any('Raw model P(side)' in l for l in labels)                           # the raw model probability is in the bet detail, labelled raw
    assert any('Data may be out of date' in w.value and '3 h' in w.value for w in app.warning)          # 3 h old export and open bets tip in 2 h
    ms = [s for s in app.selectbox if s.label == 'Model version'][0]
    assert ms.options == ['All', 'v1', 'v2'] and ms.value == 'All'
    assert any('restored from dashboard export 2026-11' in i.value for i in app.info)                   # restored rows are included by default
    ms.set_value('v2').run()
    assert not app.exception, [e.value for e in app.exception]
    assert not any('restored from dashboard export' in i.value for i in app.info)                       # and the note goes away when they are left out
    ms = [s for s in app.selectbox if s.label == 'Model version'][0]
    ms.set_value('v1').run()
    assert not app.exception and any('restored from dashboard export' in i.value for i in app.info)
    app.sidebar.radio[1].set_value('best').run()
    assert not app.exception, [e.value for e in app.exception]
    prov = ' '.join(str(df.value) for df in app.dataframe)
    assert 'recorded live' in prov and 'restored from dashboard export' in prov


def test_app_fresh_export_shows_only_the_caption(tmp_path, monkeypatch):
    write_export(tmp_path / 'fresh', live_bets())
    app = run_app(monkeypatch, tmp_path / 'fresh')
    check_all_pages(app)
    assert not any('Data may be out of date' in w.value for w in app.warning)
    assert any(c.value.startswith('Data built ') and 'ago' in c.value for c in app.caption)


def test_app_on_an_old_format_export(tmp_path, monkeypatch):
    write_export(tmp_path / 'old', live_bets(), new=False)
    app = run_app(monkeypatch, tmp_path / 'old')
    text, labels = check_all_pages(app)
    assert any(m.label == 'Raw model P(side)' and m.value == 'not in this export' for m in app.metric)
    assert not [s for s in app.selectbox if s.label == 'Model version']              # no model_version column: no filter to offer
    assert not any('restored from dashboard export' in i.value for i in app.info)
    app.sidebar.radio[1].set_value('best').run()
    assert not app.exception, [e.value for e in app.exception]


def test_app_on_an_old_export_that_has_the_column_but_no_meta_versions(tmp_path, monkeypatch):
    write_export(tmp_path / 'mid', live_bets(), meta_extra={'code_commit': None})      # null commit: the About page must not raise
    meta = json.loads((tmp_path / 'mid' / 'meta.json').read_text())
    meta.pop('model_versions')
    (tmp_path / 'mid' / 'meta.json').write_text(json.dumps(meta))
    pd.read_csv(tmp_path / 'mid' / 'season_summary.csv').drop(columns='model_version').to_csv(tmp_path / 'mid' / 'season_summary.csv', index=False)
    app = run_app(monkeypatch, tmp_path / 'mid')
    check_all_pages(app)
    ms = [s for s in app.selectbox if s.label == 'Model version'][0]
    assert ms.options == ['All', 'v1', 'v2']
    assert not any('restored from dashboard export' in i.value for i in app.info)    # provenance unknown: nothing claimed
    assert any(L.UNKNOWN_PROVENANCE in str(df.value) for df in app.dataframe)


def test_app_on_a_copy_of_the_real_export(tmp_path, monkeypatch):
    src = Path(os.environ.get('PROPS_REAL_EXPORT', ROOT / 'data' / 'props_dashboard_live'))
    if not (src / 'bets.csv').exists():
        pytest.skip(f'no real export at {src} (set PROPS_REAL_EXPORT to a props_dashboard_live folder)')
    dest = tmp_path / 'real'
    shutil.copytree(src, dest)                                                       # a copy: the original is never touched
    app = run_app(monkeypatch, dest)
    check_all_pages(app)
    ms = [s for s in app.selectbox if s.label == 'Model version']
    if ms:
        for opt in ms[0].options:
            ms[0].set_value(opt).run()
            assert not app.exception, (opt, [e.value for e in app.exception])
            ms = [s for s in app.selectbox if s.label == 'Model version']
