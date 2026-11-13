"""
Steps S10, S14 and S15 of reports/props/audit_2026-11/WP6_WP7_experiment_plan.md: scripts/a11_eval_models.py (6a, 7b) and scripts/a11_compare_dbs.py.
Everything runs on small synthetic prediction versions and SQLite files written to tmp_path in the real layouts; nothing under data/ is read and no model is trained.
"""
import hashlib
import json
import pickle
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import a11_compare_dbs as cdb  # noqa: E402
import a11_eval_models as em  # noqa: E402
from props.live import monitor as mon  # noqa: E402
from props.models import evaluate as ev, outcome as oc, registry  # noqa: E402

STATS = em.BET_STATS
SEASONS = ('2021-22', '2022-23', '2023-24', '2024-25')
QUOTED = ('2023-24', '2024-25')
BOOKS = ('draftkings', 'fanduel', 'betmgm', 'williamhill_us')
NB = 40                                                       # bootstrap resamples in the end-to-end runs
NBB = 60


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------------------------ synthetic world

def _truth(season, n_dates=8, games=3, players=5):
    rng = np.random.default_rng(int(season[:4]))
    yy = season[2:4]
    start = pd.Timestamp(f'{season[:4]}-11-01')
    idx, labels, env = [], [], []
    mus = {s: [] for s in STATS}
    for di in range(n_dates):
        date = (start + pd.Timedelta(days=di)).strftime('%Y-%m-%d')
        for gi in range(games):
            gid = f'002{yy}{di * games + gi:05d}'
            spread = {t: float(rng.normal(0, 8)) for t in (240, 60, 5)}
            for pid in range(players):
                pl = 1000 * int(yy) + 10 * gi + pid
                mu = {'rebounds': rng.uniform(2, 9), 'assists': rng.uniform(1, 7), 'threes': rng.uniform(0.5, 3)}
                y = {s: int(rng.poisson(m)) for s, m in mu.items()}
                mu['ra'] = mu['rebounds'] + mu['assists']
                labels.append({'game_id': gid, 'player_id': pl, 'lbl_played': 1, 'lbl_minutes': float(rng.uniform(10, 36)), 'lbl_team_margin': int(round(rng.normal(0, 12))),
                               'y_points': 10.0, 'y_rebounds': float(y['rebounds']), 'y_assists': float(y['assists']), 'y_threes': float(y['threes']),
                               'y_pra': 10.0 + y['rebounds'] + y['assists']})
                for t in (240, 60, 5):
                    idx.append({'game_id': gid, 'player_id': pl, 'minutes_before': t, 'season': season, 'game_date': date, 'minutes_mean': float(rng.uniform(12, 36))})
                    env.append({'game_id': gid, 'player_id': pl, 'minutes_before': t, 'env_team_spread': spread[t]})
                    for s in STATS:
                        mus[s].append(mu[s])
    return {'idx': pd.DataFrame(idx), 'mu': {s: np.asarray(v) for s, v in mus.items()}, 'labels': pd.DataFrame(labels), 'env': pd.DataFrame(env)}


def write_version(reg, name, truths, noise, seed, suffix='', seasons=SEASONS, db='data/db_a.db'):
    for season in seasons:
        tr = truths[season]
        rng = np.random.default_rng([seed, int(season[:4])])
        d = Path(reg) / (name + suffix) / season
        d.mkdir(parents=True, exist_ok=True)
        tr['idx'].to_csv(d / 'index.csv', index=False)
        for s in STATS:
            m = tr['mu'][s] * np.exp(rng.normal(0, noise, len(tr['idx']))) if noise else tr['mu'][s]
            m = np.clip(m, 0.2, 0.4 * oc.CAPS[s])                                    # a bad model, not a broken pmf
            k = np.arange(oc.CAPS[s] + 1)
            p = poisson.pmf(k[None, :], m[:, None])
            np.save(d / f'pred_{s}.npy', (p / p.sum(axis=1, keepdims=True)).astype(np.float32))
    (Path(reg) / (name + suffix)).mkdir(parents=True, exist_ok=True)
    (Path(reg) / (name + suffix) / 'manifest.json').write_text(json.dumps({'features_db': db, 'features_db_sha256': 'f' * 64}), encoding='utf-8')


def write_prices(prices, truths):
    Path(prices).mkdir(parents=True, exist_ok=True)
    for season in QUOTED:
        tr = truths[season]
        rng = np.random.default_rng(7 + int(season[:4]))
        cons_rows, bet_rows = [], []
        for i, r in tr['idx'].iterrows():
            t = int(r['minutes_before'])
            for s in STATS:
                mu = tr['mu'][s][i]
                line = np.floor(mu) + 0.5
                fair = float(np.clip(poisson.sf(line, mu * np.exp(rng.normal(0, 0.25))) + 0.02, 0.03, 0.97))
                snap = f'{r["game_date"]}T0{t % 10}:00:00Z'
                cons_rows.append({'game_id': r['game_id'], 'player_id': r['player_id'], 'minutes_before': t, 'stat': s, 'snapshot_ts_utc': snap, 'event_id_vendor': 'e' + r['game_id'],
                                  'player_name_raw': f'p{r["player_id"]}', 'consensus_line': line, 'consensus_fair_over': fair, 'n_at_line': 4, 'book_spread': 0.0})
                for book in BOOKS:
                    fo = float(np.clip(fair + rng.normal(0, 0.015), 0.02, 0.98))
                    bet_rows.append({'game_id': r['game_id'], 'player_id': r['player_id'], 'minutes_before': t, 'stat': s, 'book': book, 'line': line, 'over_price': np.nan, 'under_price': np.nan,
                                     'over_dec': 1 / (fo * 1.045), 'under_dec': 1 / ((1 - fo) * 1.045), 'fair_over': fo, 'consensus_fair_over': fair, 'n_at_line': 4,
                                     'snapshot_ts_utc': snap, 'quote_age_min': 3.0})
        with open(Path(prices) / f'prices_{season}.pkl', 'wb') as f:
            pickle.dump((pd.DataFrame(cons_rows), pd.DataFrame(bet_rows)), f)


def build_world(root, spec, p5=True, with_prices=True):
    """spec: {version: (noise, seed)}. Returns dict(reg, prices, db, frozen, truths)."""
    root = Path(root)
    truths = {s: _truth(s) for s in SEASONS}
    reg = root / 'models'
    for name, (noise, seed) in spec.items():
        write_version(reg, name, truths, noise, seed)
        if p5:
            write_version(reg, name, truths, noise, seed, suffix='_p5', seasons=QUOTED)
    if with_prices:
        write_prices(root / 'prices', truths)
    db = root / 'labels.db'
    with sqlite3.connect(db) as conn:
        pd.concat([truths[s]['labels'] for s in SEASONS] + [pd.DataFrame([{'game_id': '0022500001', 'player_id': 1, 'lbl_played': 1, 'lbl_minutes': 20.0, 'lbl_team_margin': 3,
                                                                           'y_points': 1.0, 'y_rebounds': 1.0, 'y_assists': 1.0, 'y_threes': 1.0, 'y_pra': 3.0}])]).to_sql('player_game_labels', conn, index=False)
        pd.concat([truths[s]['env'] for s in SEASONS]).to_sql('player_game_at_T', conn, index=False)
    frozen = root / 'frozen.json'
    frozen.write_text(json.dumps({'k_by_stat': {s: 0.2 for s in STATS}}), encoding='utf-8')
    return {'root': root, 'reg': reg, 'prices': root / 'prices', 'db': db, 'frozen': frozen, 'truths': truths}


def args_for(w, experiment, ref, cands, tag, extra=()):
    return ['--experiment', experiment, '--reference', ref, '--candidates', *cands, '--folds', *SEASONS[:3], '--seen', '2024-25', '--registry-root', str(w['reg']),
            '--prices-dir', str(w['prices']), '--features-db', str(w['db']), '--frozen-rule', str(w['frozen']), '--n-boot', str(NB), '--n-boot-bonf', str(NBB),
            '--out', str(w['root'] / f'res_{tag}'), '--report', str(w['root'] / f'rpt_{tag}' / 'report.md'), *extra]


R, A1, A2, A0, L = 'a11_r0_on_pg1', 'a11_pg1', 'a11_pg2', 'a11_noav', 'a11_r0'
BASE = {R: (0.30, 1), A1: (0.30, 1), L: (0.30, 5)}


def spec6a(a2=(0.0, 2), a0=None, a1=(0.30, 1)):
    s = {**BASE, A1: a1, A2: a2}
    if a0:
        s[A0] = a0
    return s


@pytest.fixture(scope='module')
def world_a2(tmp_path_factory):
    """A1 identical to R, A2 exact: A2 wins item 1 and item 2."""
    return build_world(tmp_path_factory.mktemp('w_a2'), spec6a())


@pytest.fixture(scope='module')
def run_a2(world_a2):
    w = world_a2
    before = {p: _sha(p) for p in w['root'].rglob('*') if p.is_file()}
    res = em.main(args_for(w, '6a', R, [A1, A2], 'full', ['--leaky', L]))
    return w, res, before


# ------------------------------------------------------------------------------------------------------------------------------------------------ paired CRPS

def test_paired_crps_equals_compare_versions(run_a2):
    w, res, _ = run_a2
    crps = res['crps']
    for cand, pair in ((A1, 'A1_vs_R'), (A2, 'A2_vs_R')):
        ra, rb, pa, pb = [], [], {s: [] for s in STATS}, {s: [] for s in STATS}
        for season in SEASONS[:3]:
            ia, ma = registry.load_predictions(R, season, root=w['reg'])
            ib, mb_ = registry.load_predictions(cand, season, root=w['reg'])
            lab = w['truths'][season]['labels']
            ra.append(ia.merge(lab, on=['game_id', 'player_id'], how='left'))
            rb.append(ib.merge(lab, on=['game_id', 'player_id'], how='left'))
            for s in STATS:
                pa[s].append(ma[s])
                pb[s].append(mb_[s])
        ref = ev.compare_versions(pd.concat(ra, ignore_index=True), {s: np.concatenate(v) for s, v in pa.items()}, pd.concat(rb, ignore_index=True),
                                  {s: np.concatenate(v) for s, v in pb.items()}, stats=list(STATS), n_boot=NB)
        for _, r in ref.iterrows():
            mine = em.pick(crps, pair, 'dev_pooled', r['stat'])
            assert mine['diff'] == pytest.approx(r['B_minus_A'], abs=1e-12)
            assert mine['lo'] == pytest.approx(r['lo'], abs=1e-12) and mine['hi'] == pytest.approx(r['hi'], abs=1e-12)
            assert mine['ref_mean'] == pytest.approx(r['crps_A'], abs=1e-12) and mine['cand_mean'] == pytest.approx(r['crps_B'], abs=1e-12)
            assert mine['rel_hi'] == pytest.approx(r['hi'] / r['crps_A'], rel=1e-9)
            assert np.isfinite(mine['hi_bonf']) and mine['hi_bonf'] >= mine['hi'] - 1e-12       # the Bonferroni interval is the wider one


def test_identical_versions_have_zero_difference_and_exact_is_better(run_a2):
    _, res, _ = run_a2
    c = res['crps']
    z = c[(c['pair'] == 'A1_vs_R') & (c['scope'] == 'dev_pooled') & c['metric'].isin(STATS)]
    assert (z['diff'] == 0).all() and (z['hi'] == 0).all()
    p = em.pick(c, 'A2_vs_A1', 'dev_pooled', 'pooled4_relative')
    assert p['hi'] < 0 and p['diff'] < 0
    assert set(c['scope']) >= {'dev_pooled', 'fold_2021-22', 'fold_2022-23', 'fold_2023-24', 'dev_pooled_T240', 'dev_pooled_T60', 'dev_pooled_T5', 'seen_2024-25'}
    assert set(c['metric']) >= {'pooled4_relative', 'pooled4_raw_sum', 'minutes_mae'}
    assert set(c['pair']) == {'A1_vs_R', 'A2_vs_R', 'A2_vs_A1', 'L_vs_R'}


def test_pooled_relative_is_the_equal_weight_mean(run_a2):
    _, res, _ = run_a2
    m, rows = res['metrics'], res['rows']
    dev = rows['season'].isin(SEASONS[:3]).to_numpy()
    rel = sum((m[A2][s] - m[R][s])[dev] / m[R][s][dev].mean() for s in STATS) / 4
    assert em.pick(res['crps'], 'A2_vs_R', 'dev_pooled', 'pooled4_relative')['diff'] == pytest.approx(rel.mean(), rel=1e-9)


# ------------------------------------------------------------------------------------------------------------------------------------------------ slope

def test_slope_pair_reproduces_monitor_and_is_paired():
    rng = np.random.default_rng(3)
    n = 800
    base = pd.DataFrame({'stat': rng.choice(['assists', 'rebounds'], n), 'game_date': rng.choice([f'2023-11-{d:02d}' for d in range(1, 21)], n), 'p_market': rng.uniform(0.3, 0.7, n)})
    base['hit_over'] = (rng.uniform(size=n) < base['p_market'] + 0.3 * rng.normal(0, 0.05, n)).astype(float)
    a = base.assign(p_model=np.clip(base['p_market'] + rng.normal(0, 0.05, n), 0.02, 0.98))
    b = base.assign(p_model=np.clip(base['p_market'] + 0.5 * (a['p_model'] - base['p_market']) + rng.normal(0, 0.02, n), 0.02, 0.98))
    pr = em.slope_pair(a, b, n_boot=300, seed=0)
    ma, mb_ = mon.information_slope(a, n_boot=300, seed=0), mon.information_slope(b, n_boot=300, seed=0)
    pooled = pr[pr['stat'] == 'pooled'].iloc[0]
    assert pooled['slope_a'] == pytest.approx(ma[ma['stat'] == 'pooled']['slope_icpt'].iloc[0], abs=1e-12)
    assert pooled['slope_b'] == pytest.approx(mb_[mb_['stat'] == 'pooled']['slope_icpt'].iloc[0], abs=1e-12)
    assert pooled['diff_b_minus_a'] == pytest.approx(pooled['slope_b'] - pooled['slope_a'], abs=1e-12)
    assert pooled['lo'] <= pooled['diff_b_minus_a'] <= pooled['hi']


# ------------------------------------------------------------------------------------------------------------------------------------------------ keep rule, clause by clause

def _ps(ratio=0.0005, stat=None, other=None):
    d = {s: {'hi': 0.0005 * 2.0, 'ref_mean': 2.0} for s in STATS}                 # ratio 0.0005 everywhere
    if stat:
        d[stat] = {'hi': ratio * 2.0, 'ref_mean': 2.0}
    return d


def _holds(cl, text=None):
    return {c['clause']: c['holds'] for c in cl if text is None or text in c['clause']}


@pytest.mark.parametrize('stat', STATS)
def test_nonharm_crps_clause_just_inside_and_outside(stat):
    inside = em.nonharm_clauses('A1', _ps(0.00099, stat), 0.5, 0.5)
    outside = em.nonharm_clauses('A1', _ps(0.00101, stat), 0.5, 0.5)
    assert all(c['holds'] for c in inside)
    failing = [c for c in outside if not c['holds']]
    assert len(failing) == 1 and stat in failing[0]['clause'] and failing[0]['value'] == pytest.approx(0.00101)
    assert em.nonharm_clauses('A1', _ps(-0.01, stat), 0.5, 0.5)[0]['holds']        # a gain is not harm


def test_nonharm_slope_clause_just_inside_and_outside():
    assert all(c['holds'] for c in em.nonharm_clauses('A1', _ps(), 0.4801, 0.50))        # -0.0199: at least R's minus 0.02
    out = em.nonharm_clauses('A1', _ps(), 0.4799, 0.50)                                  # -0.0201
    assert [c['holds'] for c in out] == [True] * 4 + [False]
    assert not em.nonharm_clauses('A1', _ps(), np.nan, 0.50)[-1]['holds']
    assert not em.nonharm_clauses('A1', {**_ps(), 'ra': {'hi': np.nan, 'ref_mean': 2.0}}, 0.5, 0.5)[3]['holds']


def test_a2_over_a1_clauses_each_threshold():
    pts = {s: 0.0004 for s in STATS}
    ok = em.a2_over_a1_clauses(True, -1e-6, pts)
    assert all(c['holds'] for c in ok)
    assert not em.a2_over_a1_clauses(False, -1e-6, pts)[0]['holds']
    assert not em.a2_over_a1_clauses(True, 0.0, pts)[1]['holds']                     # upper bound must be below 0, not at 0
    assert not em.a2_over_a1_clauses(True, 1e-6, pts)[1]['holds']
    for s in STATS:
        bad = em.a2_over_a1_clauses(True, -1e-6, {**pts, s: 0.00051})
        assert [c['holds'] for c in bad].count(False) == 1 and s in [c for c in bad if not c['holds']][0]['clause']
        assert all(c['holds'] for c in em.a2_over_a1_clauses(True, -1e-6, {**pts, s: 0.00049}))
        assert all(c['holds'] for c in em.a2_over_a1_clauses(True, -1e-6, {**pts, s: -0.01}))


def test_7b_keep_rule_each_clause():
    ps, ok = _ps(), dict(pooled_hi=-1e-6, fold_point=-1e-6)
    assert all(c['holds'] for c in em.keep_7b(ok['pooled_hi'], ok['fold_point'], ps, 0.5, 0.5))
    assert not em.keep_7b(0.0, -1e-6, ps, 0.5, 0.5)[0]['holds']                      # pooled upper bound must be below 0
    assert em.keep_7b(1e-6, -1e-6, ps, 0.5, 0.5)[0]['holds'] is False
    assert not em.keep_7b(-1e-6, 0.0, ps, 0.5, 0.5)[1]['holds']                      # the 2023-24 point must be below 0
    assert not em.keep_7b(-1e-6, 1e-6, ps, 0.5, 0.5)[1]['holds']
    assert [c['holds'] for c in em.keep_7b(-1e-6, -1e-6, _ps(0.00101, 'threes'), 0.5, 0.5)].count(False) == 1
    assert all(c['holds'] for c in em.keep_7b(-1e-6, -1e-6, _ps(0.00099, 'threes'), 0.5, 0.5))
    assert not em.keep_7b(-1e-6, -1e-6, ps, 0.4799, 0.5)[-1]['holds']
    assert em.keep_7b(-1e-6, -1e-6, ps, 0.4801, 0.5)[-1]['holds']


@pytest.mark.parametrize('passed, expect', [
    ({'A1': {'item1': True}, 'A2': {'item1': True, 'item2': True}}, 'A2'),                                  # several arms pass: the Out-kept arm first
    ({'A1': {'item1': True}, 'A2': {'item1': True, 'item2': False}}, 'A1'),                                 # A2 passes item 1 only: the v2 arm
    ({'A1': {'item1': True}, 'A2': {'item1': False, 'item2': True}}, 'A1'),
    ({'A1': {'item1': False}, 'A2': {'item1': True, 'item2': False}}, 'pending_A0'),                         # A0 not trained yet: the plan says train it first
    ({'A1': {'item1': False}, 'A2': {'item1': False, 'item2': False}}, 'pending_A0'),
    ({'A1': {'item1': False}, 'A2': {'item1': False, 'item2': False}, 'A0': {'item1': True}}, 'A0'),
    ({'A1': {'item1': False}, 'A2': {'item1': False, 'item2': False}, 'A0': {'item1': False}}, 'no_change'),
    ({'A1': {'item1': True}, 'A2': {'item1': True, 'item2': True}, 'A0': {'item1': True}}, 'A2'),            # A0 is never preferred to a passing arm
    ({'A1': {'item1': True}, 'A0': {'item1': True}}, 'A1'),
])
def test_preference_order(passed, expect):
    assert em.choose_6a(passed) == expect


# ------------------------------------------------------------------------------------------------------------------------------------------------ end to end, 6a

def test_6a_end_to_end_chooses_a2_and_writes_choice(run_a2):
    w, res, _ = run_a2
    cj = json.loads((w['root'] / 'res_full' / 'choice.json').read_text(encoding='utf-8'))
    assert cj['choice'] == 'A2' and res['choice'] == 'A2'
    assert cj['version'] == A2 and cj['database'] == 'data/db_a.db' and cj['database_sha256'] == 'f' * 64 and cj['roster_rule'] == 'pregame_all'
    assert cj['reference_for_7b'] == A2 and cj['needs_live_builder_change'] is True and cj['key_mismatch_override'] is False
    assert cj['slope_source'] == {'2023-24': '_p5', '2024-25': '_p5'}
    cl = pd.read_csv(w['root'] / 'res_full' / 'keep_rule.csv')
    assert (cl['result'] == 'PASS').sum() >= 1 and set(cl['item']) == {'B.9.1', 'B.9.2'}
    a2 = cl[cl['clause'].str.contains('A2')]
    assert (a2['result'] == 'PASS').all()
    out = w['root'] / 'res_full'
    for f in ('crps_paired.csv', 'slope.csv', 'slope_paired.csv', 'roi.csv', 'roi_paired.csv', 'blowout_roi.csv', 'scores_book_line.csv', 'calibration_by_side.csv', 'segments.csv',
              'key_check.csv', 'results.json', 'choice.json'):
        assert (out / f).exists(), f
    roi = pd.read_csv(out / 'roi.csv')
    assert set(roi['version']) == {R, A1, A2, L} and set(roi['season']) == set(QUOTED)
    bl = pd.read_csv(out / 'blowout_roi.csv')
    assert any('margin' in s for s in bl['segment'])
    slope = res['slope']
    assert set(slope[slope['stat'] == 'pooled']['version']) == {R, A1, A2, L}
    rep = (w['root'] / 'rpt_full' / 'report.md').read_text(encoding='utf-8')
    assert 'Keep rule, clause by clause' in rep and 'Bonferroni' in rep and 'Not computed here' in rep and 'lbl_team_margin' in rep


def test_outputs_only_under_out_and_report_and_inputs_untouched(run_a2):
    w, _, before = run_a2
    after = {p for p in w['root'].rglob('*') if p.is_file()}
    new = after - set(before)
    assert new
    for p in new:
        assert w['root'] / 'res_full' in p.parents or w['root'] / 'rpt_full' in p.parents, p
    assert {p: _sha(p) for p in before} == before


def test_a1_chosen_when_a2_only_ties(tmp_path):
    w = build_world(tmp_path, spec6a(a2=(0.30, 1)), p5=False)                    # A2 identical to A1: item 2 needs an upper bound below 0
    res = em.main(args_for(w, '6a', R, [A1, A2], 'a1', ['--skip-roi']))
    assert res['choice'] == 'A1'
    cl = res['clauses']
    assert (cl[cl['clause'].str.contains('A2 - A1: 95%')]['result'] == 'FAIL').all()
    cj = json.loads((w['root'] / 'res_a1' / 'choice.json').read_text(encoding='utf-8'))
    assert cj['version'] == A1 and cj['roster_rule'] == 'v2nq' and cj['slope_source']['2023-24'] == '(all-played versions)'


def test_pending_a0_then_a0_then_no_change(tmp_path):
    bad = (1.5, 3)
    w = build_world(tmp_path / 'x', spec6a(a2=bad, a1=bad), p5=False)
    res = em.main(args_for(w, '6a', R, [A1, A2], 'p', ['--skip-roi', '--leaky', L]))
    assert res['choice'] == 'pending_A0'
    cj = json.loads((w['root'] / 'res_p' / 'choice.json').read_text(encoding='utf-8'))
    assert cj['version'] is None and 'train A0' in cj['note']
    w2 = build_world(tmp_path / 'y', spec6a(a2=bad, a1=bad, a0=(0.0, 4)), p5=False)
    res = em.main(args_for(w2, '6a', R, [A1, A2, A0], 'a0', ['--skip-roi', '--leaky', L]))
    assert res['choice'] == 'A0'
    cj = json.loads((w2['root'] / 'res_a0' / 'choice.json').read_text(encoding='utf-8'))
    assert cj['version'] == A0 and cj['roster_rule'].startswith('none')
    w3 = build_world(tmp_path / 'z', spec6a(a2=bad, a1=bad, a0=bad), p5=False)
    res = em.main(args_for(w3, '6a', R, [A1, A2, A0], 'nc', ['--skip-roi', '--leaky', L]))
    assert res['choice'] == 'no_change'
    cj = json.loads((w3['root'] / 'res_nc' / 'choice.json').read_text(encoding='utf-8'))
    assert cj['version'] == L and cj['database'] == 'data/db_a.db' and cj['reference_for_7b'] == R


# ------------------------------------------------------------------------------------------------------------------------------------------------ key mismatch, holdout

def _drop_rows(reg, version, season, n):
    d = Path(reg) / version / season
    idx = pd.read_csv(d / 'index.csv', dtype={'game_id': str})
    keep = np.ones(len(idx), bool)
    keep[5:5 + n] = False
    idx[keep].to_csv(d / 'index.csv', index=False)
    for s in STATS:
        np.save(d / f'pred_{s}.npy', np.load(d / f'pred_{s}.npy')[keep])


def test_key_mismatch_is_reported_and_fails_loudly(tmp_path):
    w = build_world(tmp_path, spec6a(), p5=False)
    _drop_rows(w['reg'], A1, '2022-23', 3)
    with pytest.raises(SystemExit) as e:
        em.main(args_for(w, '6a', R, [A1, A2], 'km', ['--skip-roi']))
    msg = str(e.value)
    assert 'key mismatch' in msg and A1 in msg and '2022-23' in msg
    kc = pd.read_csv(w['root'] / 'res_km' / 'key_check.csv')
    bad = kc[(kc['only_in_reference'] > 0)]
    assert len(bad) == 1 and bad.iloc[0]['version'] == A1 and bad.iloc[0]['only_in_reference'] == 3 and bad.iloc[0]['season'] == '2022-23'
    assert not (w['root'] / 'res_km' / 'choice.json').exists()                       # nothing was decided


def test_allow_key_mismatch_continues_on_common_keys_and_says_so(tmp_path, capsys):
    w = build_world(tmp_path, spec6a(), p5=False)
    _drop_rows(w['reg'], A1, '2022-23', 3)
    res = em.main(args_for(w, '6a', R, [A1, A2], 'ok', ['--skip-roi', '--allow-key-mismatch']))
    out = capsys.readouterr().out
    assert 'KEY MISMATCH' in out
    cj = json.loads((w['root'] / 'res_ok' / 'choice.json').read_text(encoding='utf-8'))
    assert cj['key_mismatch_override'] is True and cj['rows_dropped_by_key_mismatch'][R] == 3
    n_full = sum(len(w['truths'][s]['idx']) for s in SEASONS)
    assert len(res['rows']) == n_full - 3


def test_2025_26_and_late_dates_are_refused(tmp_path):
    w = build_world(tmp_path, spec6a(), p5=False)
    for extra, key in ((['--folds', '2021-22', '2023-24', '2025-26'], 'folds'), (['--seen', '2025-26'], 'seen')):
        a = args_for(w, '6a', R, [A1, A2], 'h') + extra                            # argparse keeps the last --folds / --seen
        with pytest.raises(SystemExit) as e:
            em.main(a)
        assert '2025-26' in str(e.value)
    assert not (w['root'] / 'res_h' / 'choice.json').exists()
    d = Path(w['reg']) / A1 / '2024-25'
    idx = pd.read_csv(d / 'index.csv', dtype={'game_id': str})
    idx.loc[0, 'game_date'] = '2025-07-15'
    idx.to_csv(d / 'index.csv', index=False)
    with pytest.raises(SystemExit) as e:
        em.main(args_for(w, '6a', R, [A1, A2], 'late', ['--skip-roi']))
    assert '2025-06-30' in str(e.value)
    with pytest.raises(SystemExit):
        em.refuse_seasons(['2026-27'])
    em.refuse_seasons(['2024-25'])


def test_labels_of_later_seasons_are_never_selected(tmp_path):
    w = build_world(tmp_path, spec6a(), p5=False)
    lab = em.read_labels(w['db'], list(SEASONS))
    assert not lab['game_id'].str.startswith('00225').any() and len(lab) == sum(len(w['truths'][s]['labels']) for s in SEASONS)


def test_output_paths_may_not_overlap_inputs(tmp_path):
    w = build_world(tmp_path, spec6a(), p5=False)
    a = args_for(w, '6a', R, [A1, A2], 'o')
    a[a.index('--out') + 1] = str(w['reg'] / 'sub')
    with pytest.raises(SystemExit):
        em.main(a)


# ------------------------------------------------------------------------------------------------------------------------------------------------ 7b

def test_7b_adopts_a_clearly_better_environment_model_and_reports_spread_subsets(tmp_path):
    w = build_world(tmp_path, {R: (0.30, 1), 'a11_env': (0.0, 2)}, p5=False)
    res = em.main(args_for(w, '7b', R, ['a11_env'], 'env', ['--skip-roi']))
    assert res['choice'] == 'with_env'
    cj = json.loads((w['root'] / 'res_env' / 'choice.json').read_text(encoding='utf-8'))
    assert cj['version'] == 'a11_env' and cj['database'] == 'data/db_a.db' and cj['needs_live_builder_change'] is True and cj['experiment'] == '7b'
    cl = res['clauses']
    assert len(cl) == 7 and (cl['result'] == 'PASS').all() and set(cl['item']) == {'7b.6'}
    c = res['crps']
    assert {'fold_2021-22_T60', 'dev_pooled_spread>=10', 'dev_pooled_spread<10', 'seen_2024-25_spread>=10'} <= set(c['scope'])
    assert set(c['pair']) == {'with_env_vs_reference'}
    assert len(res['tables']['spread_calibration']) > 0
    with pytest.raises(SystemExit):
        em.main(args_for(w, '7b', R, ['a11_env', 'a11_env'], 'env2', ['--skip-roi']))


def test_7b_keeps_the_reference_when_the_environment_model_is_worse(tmp_path):
    w = build_world(tmp_path, {R: (0.30, 1), 'a11_env': (1.5, 2)}, p5=True)
    res = em.main(args_for(w, '7b', R, ['a11_env'], 'envbad'))
    assert res['choice'] == 'no_change'
    cj = json.loads((w['root'] / 'res_envbad' / 'choice.json').read_text(encoding='utf-8'))
    assert cj['version'] == R and cj['needs_live_builder_change'] is False
    assert (res['clauses']['result'] == 'FAIL').any()
    assert len(res['tables']['spread_calibration']) > 0 and (w['root'] / 'res_envbad' / 'spread_calibration.csv').exists()


# ------------------------------------------------------------------------------------------------------------------------------------------------ a11_compare_dbs

BASE_COLS = ['game_id TEXT', 'player_id INTEGER', 'team_id INTEGER', 'opp_team_id INTEGER', 'game_date TEXT', 'season TEXT', 'game_type TEXT', 'is_home INTEGER',
             'm_min_h15 REAL', 'r_points_h15 REAL', 'tm_off_rtg REAL', 'pos_group TEXT']
LAB_COLS = ['game_id TEXT', 'player_id INTEGER', 'lbl_played INTEGER', 'lbl_minutes REAL', 'y_points REAL', 'y_rebounds REAL']
AT_COLS = ['game_id TEXT', 'player_id INTEGER', 'minutes_before INTEGER', 'inj_p_play REAL', 'av_miss_minutes REAL', 'av_n_out INTEGER', 'opp_av_n_out INTEGER', 'env_team_spread REAL']


def _db_frames():
    rng = np.random.default_rng(11)
    base, lab, at = [], [], []
    for season, yy in (('2022-23', '22'), ('2023-24', '23'), ('2025-26', '25')):
        for g in range(12):
            gid = f'002{yy}{g:05d}'
            for p in range(5):
                pid = 100 + p
                base.append({'game_id': gid, 'player_id': pid, 'team_id': 1, 'opp_team_id': 2, 'game_date': f'20{yy}-11-{g + 1:02d}', 'season': season, 'game_type': 'RS', 'is_home': g % 2,
                             'm_min_h15': float(rng.uniform(10, 36)), 'r_points_h15': float(rng.uniform(0.1, 0.9)), 'tm_off_rtg': float(rng.uniform(100, 120)), 'pos_group': 'G'})
                lab.append({'game_id': gid, 'player_id': pid, 'lbl_played': 1, 'lbl_minutes': float(rng.uniform(10, 36)), 'y_points': float(rng.integers(0, 30)), 'y_rebounds': float(rng.integers(0, 12))})
                for t in (240, 60, 5):
                    at.append({'game_id': gid, 'player_id': pid, 'minutes_before': t, 'inj_p_play': float(rng.uniform(0.5, 1)), 'av_miss_minutes': float(rng.uniform(0, 30)),
                               'av_n_out': int(rng.integers(0, 3)), 'opp_av_n_out': int(rng.integers(0, 3)), 'env_team_spread': float(rng.normal(0, 6))})
    return {'player_game_base': pd.DataFrame(base), 'player_game_labels': pd.DataFrame(lab), 'player_game_at_T': pd.DataFrame(at)}


def _write_db(path, frames):
    path = Path(path)
    with sqlite3.connect(path) as conn:
        for name, cols in (('player_game_base', BASE_COLS), ('player_game_labels', LAB_COLS), ('player_game_at_T', AT_COLS)):
            conn.execute(f'CREATE TABLE "{name}" ({", ".join(cols)})')
            f = frames[name]
            conn.executemany(f'INSERT INTO "{name}" VALUES ({",".join("?" * len(f.columns))})', [tuple(None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, 'item') else v)
                                                                                                    for v in r) for r in f.itertuples(index=False)])
        conn.execute('CREATE INDEX ix_at ON player_game_at_T (game_id, player_id, minutes_before)')
    return path


def _cmp(tmp_path, fa, fb, extra=(), tag='c'):
    a, b = _write_db(tmp_path / f'a_{tag}.db', fa), _write_db(tmp_path / f'b_{tag}.db', fb)
    out = tmp_path / f'{tag}_out' / 'cmp.json'
    res = cdb.main(['--a', str(a), '--b', str(b), '--out', str(out), '--no-hash', *extra])
    return res, out


def test_compare_dbs_identical(tmp_path):
    fa = _db_frames()
    res, out = _cmp(tmp_path, fa, {k: v.copy() for k, v in fa.items()}, ['--expect-identical'])
    assert res['identical'] and res['exit_code'] == 0 and res['violations'] == []
    assert out.exists() and out.with_suffix('.md').exists() and (out.parent / 'cmp_tables' / 'player_game_base.csv').exists()
    assert (out.parent / 'cmp_tables' / 'row_counts.csv').exists() and (out.parent / 'cmp_tables' / 'group_summary.csv').exists()
    rc = pd.read_csv(out.parent / 'cmp_tables' / 'row_counts.csv')
    assert set(rc['season']) == set(cdb.default_seasons()) & set(rc['season']) and '2025-26' not in set(rc['season'])
    assert rc[(rc['table'] == 'player_game_at_T') & (rc['season'] == '2023-24')]['rows_a'].iloc[0] == 12 * 5 * 3
    assert json.loads(out.read_text(encoding='utf-8'))['base_cell_share_differing'] == 0


def test_compare_dbs_one_column_changed_in_three_rows(tmp_path):
    fa = _db_frames()
    fb = {k: v.copy() for k, v in fa.items()}
    rows = [3, 40, 41]
    fb['player_game_base'].loc[rows, 'r_points_h15'] += 0.5
    res, out = _cmp(tmp_path, fa, fb, ['--expect-identical'])
    t = pd.read_csv(out.parent / 'cmp_tables' / 'player_game_base.csv').set_index('column')
    r = t.loc['r_points_h15']
    assert r['n_diff_exact'] == 3 and r['n_diff_tol'] == 3 and r['n_compared'] == 120 and r['share_diff_tol'] == pytest.approx(3 / 120)
    assert r['max_abs_diff'] == pytest.approx(0.5) and r['group'] == 'rates'
    assert str(fa['player_game_base'].loc[3, 'game_id']) in r['first_differing_keys']
    assert (t.drop(index='r_points_h15')['n_diff_tol'] == 0).all()
    assert res['exit_code'] == 1 and any('r_points_h15' in v for v in res['violations'])
    assert res['groups_differing']['player_game_base'] == ['rates']
    # allowed by name, by prefix, by group; a different column does not help
    for i, allow in enumerate((['r_points_h15'], ['r_points'], ['group:rates'], ['tm_off_rtg,r_points_h15'])):
        r2, _ = _cmp(tmp_path, fa, fb, ['--expect-identical', '--allow-columns', *allow], tag=f'c{i}')
        assert r2['exit_code'] == 0 and r2['violations'] == [] and r2['identical'] is False
    r3, _ = _cmp(tmp_path, fa, fb, ['--expect-identical', '--allow-columns', 'tm_off_rtg'], tag='cx')
    assert r3['exit_code'] == 1
    r4, _ = _cmp(tmp_path, fa, fb, [], tag='cn')                                    # without --expect-identical the report is the product
    assert r4['exit_code'] == 0 and r4['violations']


def test_compare_dbs_missing_key_is_never_allowed_away(tmp_path):
    fa = _db_frames()
    fb = {k: v.copy() for k, v in fa.items()}
    fb['player_game_at_T'] = fb['player_game_at_T'].drop(index=[0, 1])
    res, out = _cmp(tmp_path, fa, fb, ['--expect-identical', '--allow-columns', 'av_', 'opp_av_', 'inj_', 'env_', 'group:teammates'])
    assert res['exit_code'] == 1 and any('keys in A only' in v for v in res['violations'])
    ko = pd.read_csv(out.parent / 'cmp_tables' / 'keys_only.csv')
    assert len(ko) == 2 and set(ko['only_in']) == {'A'} and set(ko['table']) == {'player_game_at_T'}
    rc = pd.read_csv(out.parent / 'cmp_tables' / 'row_counts.csv')
    assert rc['only_a'].sum() == 2 and rc['only_b'].sum() == 0


def test_compare_dbs_nan_handling_and_tolerance(tmp_path):
    fa = _db_frames()
    fa['player_game_base'].loc[[0, 1, 2], 'm_min_h15'] = np.nan
    fb = {k: v.copy() for k, v in fa.items()}
    fb['player_game_base'].loc[2, 'm_min_h15'] = 12.0                                # NaN against a number: a difference
    fb['player_game_base'].loc[5, 'tm_off_rtg'] = fa['player_game_base'].loc[5, 'tm_off_rtg'] * (1 + 1e-12)   # beyond exact, inside 1e-9 relative
    fb['player_game_base'].loc[6, 'tm_off_rtg'] = fa['player_game_base'].loc[6, 'tm_off_rtg'] * (1 + 1e-6)    # beyond tolerance
    res, out = _cmp(tmp_path, fa, fb)
    t = pd.read_csv(out.parent / 'cmp_tables' / 'player_game_base.csv').set_index('column')
    assert t.loc['m_min_h15', 'n_diff_exact'] == 1 and t.loc['m_min_h15', 'n_diff_tol'] == 1           # rows 0 and 1: NaN equals NaN
    assert t.loc['tm_off_rtg', 'n_diff_exact'] == 2 and t.loc['tm_off_rtg', 'n_diff_tol'] == 1
    assert t.loc['pos_group', 'n_diff_exact'] == 0
    strict, _ = _cmp(tmp_path, fa, fb, ['--expect-identical', '--exact', '--allow-columns', 'm_min_h15'], tag='ce')
    assert strict['exit_code'] == 1 and any('tm_off_rtg' in v and 'exactly' in v for v in strict['violations'])
    loose, _ = _cmp(tmp_path, fa, {**fb, 'player_game_base': fb['player_game_base'].assign(tm_off_rtg=fa['player_game_base']['tm_off_rtg'] * (1 + 1e-12))},
                    ['--expect-identical', '--allow-columns', 'm_min_h15'], tag='cl')
    assert loose['exit_code'] == 0


def test_compare_dbs_groups_and_extra_column_and_holdout(tmp_path):
    fa = _db_frames()
    fb = {k: v.copy() for k, v in fa.items()}
    fb['player_game_at_T'].loc[[10, 11], 'av_miss_minutes'] += 1.0
    fb['player_game_at_T'].loc[[10, 11], 'env_team_spread'] += 1.0
    fb['player_game_at_T'].loc[[12], 'opp_av_n_out'] += 1
    # rows of 2025-26 may differ without being read
    last = fb['player_game_at_T']['game_id'].str.startswith('00225')
    fb['player_game_at_T'].loc[last, 'inj_p_play'] = 0.0
    res, out = _cmp(tmp_path, fa, fb, ['--expect-identical', '--allow-columns', 'group:teammates'])
    gs = pd.read_csv(out.parent / 'cmp_tables' / 'group_summary.csv')
    g = gs[(gs['table'] == 'player_game_at_T')].set_index('group')
    assert g.loc['teammates', 'cells_differing'] == 3 and g.loc['market_env', 'cells_differing'] == 2 and g.loc['injury', 'cells_differing'] == 0
    assert g.loc['teammates', 'rows_any_column_differing'] == 3 and g.loc['teammates', 'share_rows_any_differing'] == pytest.approx(3 / 360)
    assert res['exit_code'] == 1 and all('env_team_spread' in v for v in res['violations'])         # the teammate group is allowed, the environment is not
    pos = pd.read_csv(out.parent / 'cmp_tables' / 'positive_share.csv')
    assert {'av_n_out', 'opp_av_n_out'} == set(pos['column']) and set(pos['side']) == {'A', 'B'}
    with pytest.raises(SystemExit) as e:
        cdb.main(['--a', str(tmp_path / 'a_c.db'), '--b', str(tmp_path / 'b_c.db'), '--out', str(tmp_path / 'o.json'), '--seasons', '2023-24', '2025-26'])
    assert '2025-26' in str(e.value)
    # an extra column in B is a schema difference
    b2 = tmp_path / 'b2.db'
    shutil.copy(tmp_path / 'b_c.db', b2)
    with sqlite3.connect(b2) as conn:
        conn.execute('ALTER TABLE player_game_at_T ADD COLUMN av_extra REAL')
    r2 = cdb.main(['--a', str(tmp_path / 'a_c.db'), '--b', str(b2), '--out', str(tmp_path / 'o2' / 'x.json'), '--expect-identical', '--allow-columns', 'group:teammates', 'env_', '--no-hash'])
    assert any('av_extra' in v for v in r2['violations'])                                # 'av_extra' is not in the dictionary, so group:teammates does not cover it
    assert r2['exit_code'] == 1


def test_compare_dbs_does_not_write_inputs_and_exit_status_of_the_script(tmp_path):
    fa = _db_frames()
    fb = {k: v.copy() for k, v in fa.items()}
    a, b = _write_db(tmp_path / 'a.db', fa), _write_db(tmp_path / 'b.db', fb)
    shas = (_sha(a), _sha(b))
    base = [sys.executable, str(ROOT / 'scripts' / 'a11_compare_dbs.py'), '--a', str(a), '--b', str(b)]
    cmd = base + ['--out', str(tmp_path / 'o' / 'r.json'), '--expect-identical', '--no-hash']
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    with sqlite3.connect(b) as conn:
        conn.execute("UPDATE player_game_base SET tm_off_rtg = tm_off_rtg + 1 WHERE game_id = (SELECT min(game_id) FROM player_game_base)")
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 1 and 'tm_off_rtg' in cp.stdout
    assert _sha(a) == shas[0]
    bad = subprocess.run(base + ['--out', str(a), '--no-hash'], capture_output=True, text=True)
    assert bad.returncode != 0 and _sha(a) == shas[0]
