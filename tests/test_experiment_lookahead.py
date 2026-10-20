"""Lookahead guards for the experiment features: corrupt every row dated on/after a game's date and check that the
feature for that game (and every earlier game) does not change."""
import numpy as np
import pandas as pd
import pytest

from experiments import features_e05 as e5


def synthetic_log(n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    teams = [1, 2, 3, 4, 5, 6]
    rows, gid = [], 0
    for day in range(n_days):
        date = pd.Timestamp('2025-11-01') + pd.Timedelta(days=day)
        order = rng.permutation(teams)
        for a, b in zip(order[::2], order[1::2]):
            gid += 1
            base = {c: rng.normal(30, 6, 2) for c in list(e5.STATS) + list(e5.STATS.values())}
            for k, t in enumerate((a, b)):
                row = {'teamId': int(t), 'gameId': f'{gid:010d}', 'gameDate': date, 'minutes': 240.0,
                       'pace': float(rng.normal(99, 3))}
                for c in list(e5.STATS) + list(e5.STATS.values()):
                    row[c] = float(base[c][k])
                rows.append(row)
    return pd.DataFrame(rows)


def corrupt_from(log, date, seed=1):
    rng = np.random.default_rng(seed)
    bad = log.copy()
    m = bad['gameDate'] >= date
    for c in bad.columns:
        if c in ('teamId', 'gameId', 'gameDate'):
            continue
        bad.loc[m, c] = bad.loc[m, c] * rng.uniform(0.2, 5.0, m.sum()) + rng.normal(0, 50, m.sum())
    return bad


def assert_no_lookahead(build, log, dates):
    full = build(log).set_index(['gameId', 'teamId']).sort_index()
    gdate = log.drop_duplicates('gameId').set_index('gameId')['gameDate']
    for d in dates:
        bad = build(corrupt_from(log, d)).set_index(['gameId', 'teamId']).sort_index()
        ids = [g for g in full.index.get_level_values(0).unique() if gdate[g] <= d]
        a, b = full.loc[ids], bad.loc[ids]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def test_e05_features_do_not_look_ahead():
    log = synthetic_log()
    dates = [pd.Timestamp('2025-11-15'), pd.Timestamp('2025-12-10'), pd.Timestamp('2025-12-28')]
    assert_no_lookahead(e5.build_features, log, dates)


def test_e05_features_do_react_to_the_past():
    """The guard is not vacuous: corrupting the PAST must change later features."""
    log = synthetic_log()
    full = e5.build_features(log).set_index(['gameId', 'teamId']).sort_index()
    bad_log = log.copy()
    early = bad_log['gameDate'] < pd.Timestamp('2025-11-20')
    bad_log.loc[early, 'points'] = bad_log.loc[early, 'points'] * 3
    bad = e5.build_features(bad_log).set_index(['gameId', 'teamId']).sort_index()
    late_ids = log[log['gameDate'] >= pd.Timestamp('2025-12-01')]['gameId'].unique()
    assert not np.allclose(full.loc[late_ids].fillna(0).to_numpy(), bad.loc[late_ids].fillna(0).to_numpy())


def test_e05_adjustment_uses_only_the_opponents_prior_games():
    log = synthetic_log(30)
    f = e5.build_features(log)
    first_day = log[log['gameDate'] == log['gameDate'].min()]['gameId'].unique()
    first = f[f['gameId'].isin(first_day)]
    assert first.filter(like='e5_').isna().all().all()       # nobody has history on day 1


# ------------------------------------------------------------------ E7 / E8 injury value variants

def _inj_inputs(seed=0):
    from experiments import inj_variants as iv
    rng = np.random.default_rng(seed)
    cand, rows = [], []
    for g in range(1, 7):
        for team in (1, 2):
            for p in range(12):
                pid = team * 100 + p
                em = float(max(4, 34 - 2.5 * p + rng.normal(0, 1)))
                bpm = float(rng.normal(0, 2))
                cand.append({'gameId': f'{g:010d}', 'teamId': team, 'personId': pid, 'e_minutes': em,
                             'value': em / 48 * (bpm + iv.REPLACEMENT_BPM * -1), 'gap': int(rng.integers(1, 15)),
                             'rotation': em >= 10, 'played': bool(rng.random() < 0.8), 'seasonYear': '2025-26'})
                if rng.random() < 0.2:
                    rows.append({'gameId': f'{g:010d}', 'teamId': team, 'personId': pid,
                                 'status': rng.choice(['Out', 'Questionable', 'Probable', 'Doubtful'])})
    cand, rows = pd.DataFrame(cand), pd.DataFrame(rows)
    cov = cand[['gameId', 'teamId']].drop_duplicates().assign(report_ts=pd.Timestamp('2025-12-01 12:00'), not_submitted=False)
    rates = pd.DataFrame({'mean': {'Out': 0.0, 'Questionable': 0.5, 'Probable': 0.9, 'Doubtful': 0.05}})
    return rows, cov, cand, rates


@pytest.mark.parametrize('kind', ['e7', 'e8'])
def test_injury_variants_ignore_the_realised_outcome(kind, monkeypatch):
    from experiments import inj_variants as iv
    monkeypatch.setattr(iv, 'position_groups', lambda db: {pid: 'GFC'[pid % 3] for pid in range(100, 300)})
    fn = iv.make_e7('x') if kind == 'e7' else iv.make_e8('x')
    rows, cov, cand, rates = _inj_inputs()
    a = fn(rows, cov, cand, rates).sort_values(['gameId', 'teamId']).reset_index(drop=True)
    shuffled = cand.copy()
    shuffled['played'] = np.random.default_rng(9).random(len(cand)) < 0.5      # change who actually played the game
    b = fn(rows, cov, shuffled, rates).sort_values(['gameId', 'teamId']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert a.filter(like='rep_lost').to_numpy().sum() > 0


def test_e7_replacement_level_is_the_teams_own_bench(monkeypatch):
    from experiments import inj_variants as iv
    monkeypatch.setattr(iv, 'position_groups', lambda db: {})
    rows, cov, cand, rates = _inj_inputs()
    base = cand.assign(value=lambda d: d['e_minutes'] / 48 * (4.0 + 2.0))        # every player BPM +4
    base['gap'] = 1
    r = rows[rows['gameId'] == rows['gameId'].iloc[0]]
    # all bench players are +4 too -> replacement is +4 -> nobody has value above replacement
    out = iv.make_e7('x')(r, cov, base, rates)
    assert np.allclose(out['rep_lost'], 0.0)
    out_const = pd.DataFrame(iv.irf.reported_features(r, cov, base, rates))
    assert out_const['rep_lost'].sum() > 0                                       # the constant -2 would have charged them


def test_e8_components_partition_the_total():
    from experiments import inj_variants as iv
    rows, cov, cand, rates = _inj_inputs()
    f = iv.make_e8('x')(rows, cov, cand, rates)
    parts = f['inj_fresh_lost'] + f['inj_chronic_lost'] + f['inj_mid_lost']
    np.testing.assert_allclose(parts, f['rep_lost'])
    assert (f['inj_chronic_lost_dec'].abs() <= f['inj_chronic_lost'].abs() + 1e-12).all()


# ------------------------------------------------------------------ E9 downgrade flag and injury sources

def test_downgrade_flag_uses_only_statuses_and_ignores_the_outcome():
    from models import injury_downgrades as dg
    rows, cov, cand, rates = _inj_inputs()
    early = rows[rows['status'].isin(['Probable', 'Questionable'])].copy()
    late = early.assign(status='Out')                                    # everyone listed worsens to Out
    a = dg.compute_downgrades(early, late, cand, rates)
    shuffled = cand.assign(played=np.random.default_rng(3).random(len(cand)) < 0.5)
    b = dg.compute_downgrades(early, late, shuffled, rates)
    pd.testing.assert_frame_equal(a, b)
    assert a['inj_downgrade_n'].sum() > 0
    same = dg.compute_downgrades(early, early, cand, rates)               # no change between the reports -> no downgrade
    assert same['inj_downgrade_lost'].abs().sum() == 0 and same['inj_upgrade_lost'].abs().sum() == 0


def test_downgrade_direction():
    from models import injury_downgrades as dg
    cand = pd.DataFrame({'gameId': ['g'], 'teamId': [1], 'personId': [7], 'value': [4.0], 'rotation': [True]})
    rates = pd.DataFrame({'mean': {'Out': 0.0, 'Questionable': 0.5, 'Probable': 0.9}})
    row = lambda s: pd.DataFrame({'gameId': ['g'], 'teamId': [1], 'personId': [7], 'status': [s]})
    down = dg.compute_downgrades(row('Probable'), row('Out'), cand, rates)
    assert down['inj_downgrade_lost'].iloc[0] == pytest.approx(4.0 * (1.0 - 0.1))
    up = dg.compute_downgrades(row('Out'), row('Questionable'), cand, rates)
    assert up['inj_downgrade_lost'].iloc[0] == 0 and up['inj_upgrade_lost'].iloc[0] == pytest.approx(4.0 * 0.5)
    unlisted_early = dg.compute_downgrades(row('Out').iloc[0:0], row('Out'), cand, rates)      # newly listed Out
    assert unlisted_early['inj_downgrade_lost'].iloc[0] == pytest.approx(4.0)


def test_source_categories_and_comparison():
    from models import injury_source_compare as sc
    assert [sc.pdf_category(s) for s in ('Out', 'Doubtful', 'Questionable', 'Probable', 'Available', None)] == \
        ['OUT', 'UNCERTAIN', 'UNCERTAIN', 'PLAYING', 'PLAYING', 'PLAYING']
    assert [sc.espn_category(s) for s in ('Out', 'Day-To-Day', 'Questionable', None)] == ['OUT', 'UNCERTAIN', 'UNCERTAIN', 'PLAYING']
    pdf = pd.DataFrame({'player': ['Williams III, Robert', 'Doe, John'], 'team': ['Boston Celtics', 'Boston Celtics'], 'status': ['Out', 'Questionable']})
    c = sc.compare_frames(pdf, {'Robert Williams III': 'Out', 'Jane Roe': 'Day-To-Day'}).set_index('player_key')
    assert c.loc['robertwilliamsiii', 'agree'] == 1
    assert c.loc['johndoe', 'agree'] == 0 and c.loc['janeroe', 'agree'] == 0         # listed by one source only


def test_espn_is_not_on_any_feature_path():
    """ESPN may only be touched by the comparator module; no model-input module or live row builder may reference it."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for rel in ['models/live_features.py', 'models/live_rotation_v2.py', 'models/injury_report_features.py', 'models/game_features_v2.py',
                'models/availability_features.py', 'models/injury_downgrades.py']:
        src = (root / rel).read_text(encoding='utf-8').lower()
        assert 'espn.com' not in src and 'beautifulsoup' not in src, rel
