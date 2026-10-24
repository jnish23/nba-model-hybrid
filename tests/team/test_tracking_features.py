"""Guards for experiments/features_tracking.py: no lookahead (everything from game t onward is overwritten with garbage and
the features at t must not move), a hand-computed example, the no-data path (NaN, never 0) and the training-set merge."""
import numpy as np
import pandas as pd
import pytest

from experiments import features_tracking as ft

HUSTLE = ft.HUSTLE_COLS


def synthetic_inputs(n_days=70, seed=0, teams=(1, 2, 3, 4, 5, 6)):
    """Random schedule (2-3 games a night); 10 players a team with fixed role minutes; distance = speed x hours played."""
    rng = np.random.default_rng(seed)
    tgf, tt, pt, th = [], [], [], []
    gid = 0
    for day in range(n_days):
        date = pd.Timestamp('2025-11-01') + pd.Timedelta(days=day)
        order = rng.permutation(list(teams))
        for a, b in list(zip(order[::2], order[1::2]))[: int(rng.integers(2, 4))]:
            gid += 1
            g = f'{gid:010d}'
            pace = float(rng.normal(99, 3))
            ot = bool(rng.random() < 0.1)
            for t, o, home in ((int(a), int(b), True), (int(b), int(a), False)):
                tm = 265.0 if ot else 240.0
                tgf.append({'seasonYear': '2025-26', 'teamId': t, 'teamTricode': f'T{t}', 'gameId': g, 'gameDate': date,
                            'MATCHUP': f'T{t} vs. T{o}' if home else f'T{t} @ T{o}', 'minutes': tm,
                            'plusMinus': float(rng.integers(-15, 16)), 'pace': pace})
                dist = 0.0
                for p in range(10):
                    mins = float(np.clip(rng.normal(34 if p < 5 else 12, 3), 1, 45))
                    speed = float(rng.normal(4.3 + 0.05 * p, 0.2))
                    d_ = speed * mins / 60.0 * float(rng.uniform(0.97, 1.03))
                    dist += d_
                    pt.append({'gameId': g, 'teamId': t, 'personId': t * 100 + p, 'minutes': f'{int(mins):02d}:{int((mins % 1) * 60):02d}', 'speed': speed, 'distance': d_})
                tt.append({'gameId': g, 'teamId': t, 'distance': dist, 'touches': int(rng.integers(300, 500))})
                th.append({'gameId': g, 'teamId': t, **{c: int(rng.integers(5, 30)) for c in HUSTLE}})
    arenas = pd.DataFrame({'teamId': list(teams), 'altitude': [5280.0] + [100.0] * (len(teams) - 1)})
    return {'tgf': pd.DataFrame(tgf), 'tt': pd.DataFrame(tt), 'pt': pd.DataFrame(pt), 'th': pd.DataFrame(th), 'arenas': arenas}


def corrupt_from(d, date, seed=1):
    """Overwrite every statistic of every row dated on/after `date`; the schedule (ids, dates, MATCHUP) stays, as the
    published schedule is allowed to be known."""
    rng = np.random.default_rng(seed)
    d = {k: v.copy() for k, v in d.items()}
    gdate = d['tgf'].drop_duplicates('gameId').set_index('gameId')['gameDate']
    stats = {'tgf': ['minutes', 'plusMinus', 'pace'], 'tt': ['distance', 'touches'], 'pt': ['speed', 'distance'], 'th': HUSTLE}
    for k, cols in stats.items():
        m = (d[k]['gameId'].map(gdate) >= date).to_numpy()
        for c in cols:
            d[k][c] = d[k][c].astype(float)
            d[k].loc[m, c] = d[k].loc[m, c] * rng.uniform(0.2, 5.0, m.sum()) + rng.normal(0, 40, m.sum())
    m = (d['pt']['gameId'].map(gdate) >= date).to_numpy()
    d['pt'].loc[m, 'minutes'] = [f'{int(x):02d}:30' for x in rng.integers(0, 48, m.sum())]
    return d


def feature_cols(f):
    return [c for c in f.columns if c.startswith('trk_')]


def test_features_do_not_look_ahead():
    d = synthetic_inputs()
    full = ft.build_team_features(d).set_index(['gameId', 'teamId']).sort_index()
    gdate = d['tgf'].drop_duplicates('gameId').set_index('gameId')['gameDate']
    for date in [pd.Timestamp('2025-12-05'), pd.Timestamp('2025-12-20'), pd.Timestamp('2026-01-05')]:
        bad = ft.build_team_features(corrupt_from(d, date)).set_index(['gameId', 'teamId']).sort_index()
        ids = [g for g in full.index.get_level_values(0).unique() if gdate[g] <= date]
        pd.testing.assert_frame_equal(full.loc[ids], bad.loc[ids], check_exact=False, rtol=1e-10, atol=1e-10)


def test_guard_is_not_vacuous():
    """Corrupting the PAST must change later features."""
    d = synthetic_inputs()
    full = ft.build_team_features(d).set_index(['gameId', 'teamId']).sort_index()
    gdate = d['tgf'].drop_duplicates('gameId').set_index('gameId')['gameDate']
    bad_d = corrupt_from(d, pd.Timestamp('2025-11-01'))
    keep = gdate[gdate >= pd.Timestamp('2025-12-20')].index
    for k in ('tgf', 'tt', 'pt', 'th'):                       # restore the later games, so only the past differs
        m = d[k]['gameId'].isin(keep)
        bad_d[k].loc[m] = d[k].loc[m]
    bad = ft.build_team_features(bad_d).set_index(['gameId', 'teamId']).sort_index()
    ids = list(keep)
    changed = [c for c in feature_cols(full) if not np.allclose(full.loc[ids, c].fillna(-99).to_numpy(float), bad.loc[ids, c].fillna(-99).to_numpy(float))]
    for c in ('trk_xload_last', 'trk_xload_4n', 'trk_rot_xload_last', 'trk_rot_speed_z_last', 'trk_prev_hustle_z', 'trk_dist_per_min_last'):
        assert c in changed, c


def test_one_side_tired_uses_only_earlier_days():
    rng = np.random.default_rng(3)
    n = 800
    dates = pd.Timestamp('2025-11-01') + pd.to_timedelta(np.sort(rng.integers(0, 120, n)), unit='D')
    h, a = rng.normal(0, 1, n), rng.normal(0, 1, n)
    base = ft.one_side_tired(dates, h, a)
    cut = pd.Timestamp('2026-01-05')
    h2, a2 = h.copy(), a.copy()
    late = dates >= cut
    h2[late], a2[late] = rng.normal(5, 9, late.sum()), rng.normal(-5, 9, late.sum())
    bad = ft.one_side_tired(dates, h2, a2)
    early = dates <= cut - pd.Timedelta(days=1)
    np.testing.assert_array_equal(base[early], bad[early])
    # a game on the cut date sees only earlier days' scores for its percentiles; its own score is the same in both runs
    assert np.isnan(base[:5]).all()                                          # no history yet
    assert set(np.unique(base[~np.isnan(base)])) <= {-1.0, 0.0, 1.0}


def two_team_inputs():
    """Teams 1 (altitude arena) and 2 play five games; team 1 is the subject of the hand computation."""
    rows = [  # date offset, home team, team-1 distance per 100 poss inputs
        (0, 1, 2), (1, 2, 1), (3, 1, 2), (4, 2, 1), (6, 2, 1)]                 # (day, home team id, away team id)
    base = pd.Timestamp('2025-11-01')
    tgf, tt = [], []
    mins = {1: [240, 240, 265, 240, 240], 2: [240, 240, 265, 240, 240]}
    pm = {1: [3, -4, 10, 5, 2], 2: [-3, 4, -10, -5, -2]}
    paces = [100.0, 98.0, 102.0, 99.0, 101.0]
    dist1 = [17.0, 17.5, 18.5, 16.5, 17.2]
    for i, (day, home, away) in enumerate(rows):
        g = f'{i + 1:010d}'
        for t, o in ((home, away), (away, home)):
            tgf.append({'seasonYear': '2025-26', 'teamId': t, 'teamTricode': f'T{t}', 'gameId': g, 'gameDate': base + pd.Timedelta(days=day),
                        'MATCHUP': f'T{t} vs. T{o}' if t == home else f'T{t} @ T{o}', 'minutes': float(mins[t][i]),
                        'plusMinus': float(pm[t][i]), 'pace': paces[i]})
            tt.append({'gameId': g, 'teamId': t, 'distance': dist1[i] if t == 1 else 17.0 + 0.1 * i, 'touches': 400})
    pt = []
    for i in range(5):
        g = f'{i + 1:010d}'
        for t in (1, 2):
            for p in range(2):
                pt.append({'gameId': g, 'teamId': t, 'personId': t * 10 + p, 'minutes': '41:00' if (p == 0 and i == 2) else '30:00', 'speed': 4.3, 'distance': 2.1})
    th = pd.DataFrame([{'gameId': f'{i + 1:010d}', 'teamId': t, **{c: 10 + i + (t == 1) for c in HUSTLE}} for i in range(5) for t in (1, 2)])
    arenas = pd.DataFrame({'teamId': [1, 2], 'altitude': [5280.0, 100.0]})
    return {'tgf': pd.DataFrame(tgf), 'tt': pd.DataFrame(tt), 'pt': pd.DataFrame(pt), 'th': th, 'arenas': arenas}, dist1, mins[1], paces


def test_hand_computed_example(monkeypatch):
    monkeypatch.setattr(ft, 'MIN_NORM', 2)
    monkeypatch.setattr(ft, 'MIN_SD', 2)
    monkeypatch.setattr(ft, 'SD_FLOOR_FRAC', 0.0)
    d, dist1, mins1, paces = two_team_inputs()
    f = ft.build_team_features(d)
    f = f[f['teamId'] == 1].set_index('gameId')
    g = [f'{i:010d}' for i in range(1, 6)]
    # distance per possession, possessions = pace x minutes / 240
    x = np.array([dist1[i] / (paces[i] * mins1[i] / 240.0) for i in range(5)])
    alpha = 2 / 21

    def ewma(v):
        w = (1 - alpha) ** np.arange(len(v) - 1, -1, -1)
        return float((w * np.asarray(v)).sum() / w.sum())

    z3 = (x[2] - ewma(x[:2])) / np.std(x[:2], ddof=1)
    z4 = (x[3] - ewma(x[:3])) / np.std(x[:3], ddof=1)
    z5 = (x[4] - ewma(x[:4])) / np.std(x[:4], ddof=1)
    # raw lagged values
    assert np.isnan(f.loc[g[0], 'trk_dist_per_poss_last'])                                         # NaN before any game
    assert f.loc[g[3], 'trk_dist_per_poss_last'] == pytest.approx(x[2])
    assert f.loc[g[3], 'trk_dist_per_min_last'] == pytest.approx(dist1[2] / mins1[2])
    # xload: games 1 and 2 have too little history (z needs 2 prior games), game 3 is the first with a z
    assert np.isnan(f.loc[g[2], 'trk_xload_last'])                       # last z-able game does not exist yet
    assert f.loc[g[3], 'trk_xload_last'] == pytest.approx(z3)
    assert f.loc[g[4], 'trk_xload_last'] == pytest.approx(z4)
    # xload over the 4 nights before: game 4 is on day 4 -> days 0..3 -> games 1,2,3 -> only game 3 has a z
    assert f.loc[g[3], 'trk_xload_4n'] == pytest.approx(z3)
    # game 5 is on day 6 -> days 2..5 -> games 3 (day 3) and 4 (day 4)
    assert f.loc[g[4], 'trk_xload_4n'] == pytest.approx(z3 + z4)
    assert z5 == z5                                                      # (z of game 5 is only ever used by game 6)
    # physical previous game
    assert f.loc[g[3], 'trk_prev_ot'] == 1.0 and f.loc[g[4], 'trk_prev_ot'] == 0.0
    assert f.loc[g[1], 'trk_prev_close'] == 1.0 and f.loc[g[3], 'trk_prev_close'] == 0.0      # margins 3, -4, 10, 5, 2: |3|<=5; |10|>5
    assert f.loc[g[4], 'trk_prev_close'] == 1.0                                                  # previous margin 5
    assert f.loc[g[3], 'trk_prev_n40'] == 1.0 and f.loc[g[4], 'trk_prev_n40'] == 0.0
    # previous game's pace relative to the team's own habit going into it (EWMA of the games BEFORE that game; needs 3 of them)
    assert np.isnan(f.loc[g[3], 'trk_prev_opp_pace'])                                              # only 2 games before game 3
    habit = pd.Series(paces[:3]).ewm(span=10).mean().iloc[-1]
    assert f.loc[g[4], 'trk_prev_opp_pace'] == pytest.approx(paces[3] - habit)
    # schedule density: games on days 0, 1, 3, 4, 6
    assert [f.loc[x, 'trk_g3in4'] for x in g] == [0.0, 0.0, 1.0, 1.0, 1.0]       # windows [t-3, t]: 1, 2, 3, 3 (1,3,4), 3 (3,4,6) games
    assert [f.loc[x, 'trk_g4in6'] for x in g] == [0.0, 0.0, 0.0, 1.0, 1.0]       # windows [t-5, t]: 1, 2, 3, 4 (0,1,3,4), 4 (1,3,4,6) games
    assert f.loc[g[4], 'trk_games_last7'] == 4.0                                        # days 0..5 hold games on 0, 1, 3, 4
    # venue pattern for team 1: home, away, home, away, away
    assert f.loc[g[0], 'trk_trip_last_game'] == 0.0 and f.loc[g[1], 'trk_trip_last_game'] == 1.0   # away on day 1, next game home on day 3
    assert f.loc[g[3], 'trk_trip_last_game'] == 0.0                                                  # away, next game also away
    assert f.loc[g[4], 'trk_trip_last_game'] == 0.0                                                  # no later game
    # altitude: team 1's home arena is at altitude; after an away game its venue is not
    assert f.loc[g[1], 'trk_prev_altitude'] == 1.0 and f.loc[g[2], 'trk_prev_altitude'] == 0.0


def test_no_data_gives_nan_not_zero(monkeypatch):
    monkeypatch.setattr(ft, 'MIN_NORM', 2)
    monkeypatch.setattr(ft, 'MIN_SD', 2)
    d = synthetic_inputs(n_days=40, seed=4)
    # team 3 never has tracking in its first 6 games: rows present but all zero (the 2025-26 pattern), and no hustle rows
    first = d['tgf'][d['tgf']['teamId'] == 3].sort_values('gameDate')['gameId'].head(6)
    m = d['tt']['gameId'].isin(first) & (d['tt']['teamId'] == 3)
    d['tt'].loc[m, ['distance', 'touches']] = 0
    mp = d['pt']['gameId'].isin(first) & (d['pt']['teamId'] == 3)
    d['pt'].loc[mp, ['speed', 'distance']] = 0
    d['th'] = d['th'][~(d['th']['gameId'].isin(first) & (d['th']['teamId'] == 3))]
    f = ft.build_team_features(d)
    t3 = f[f['teamId'] == 3].copy()
    order = d['tgf'][d['tgf']['teamId'] == 3].sort_values('gameDate')['gameId'].tolist()
    t3 = t3.set_index('gameId').loc[order]
    head = t3.iloc[:7]                                                   # game 7 (index 6) looks back at games 1-6, all without data
    for c in ('trk_dist_per_min_last', 'trk_dist_per_poss_last', 'trk_xload_last', 'trk_xload_4n', 'trk_rot_xload_last',
              'trk_rot_speed_z_last', 'trk_star_speed_z_last', 'trk_rot_speed_trend5', 'trk_prev_hustle_z'):
        assert head[c].isna().all(), c
    assert t3['trk_prev_ot'].iloc[1:].notna().all()                       # the box score is always there
    later = t3.iloc[10:]
    assert later['trk_xload_last'].notna().any()                         # data resumes -> features resume


def test_staleness_is_capped(monkeypatch):
    monkeypatch.setattr(ft, 'MIN_NORM', 2)
    monkeypatch.setattr(ft, 'MIN_SD', 2)
    d = synthetic_inputs(n_days=40, seed=5)
    order = d['tgf'][d['tgf']['teamId'] == 2].sort_values('gameDate')['gameId'].tolist()
    blank = order[10:15]                                                  # five consecutive games without tracking
    m = d['tt']['gameId'].isin(blank) & (d['tt']['teamId'] == 2)
    d['tt'].loc[m, ['distance', 'touches']] = 0
    tg = d['tgf'][d['tgf']['teamId'] == 2].set_index('gameId')
    tt = d['tt'][d['tt']['teamId'] == 2].set_index('gameId')
    raw = lambda i: tt.loc[order[i], 'distance'] / (tg.loc[order[i], 'pace'] * tg.loc[order[i], 'minutes'] / 240.0)
    f = ft.build_team_features(d)
    t2 = f[f['teamId'] == 2].set_index('gameId').loc[order, 'trk_dist_per_poss_last']
    for i in range(10, 14):                                               # games 10-12 are blank; game 9 is still at most 3 games old
        assert t2.iloc[i] == pytest.approx(raw(9))
    assert np.isnan(t2.iloc[14]) and np.isnan(t2.iloc[15])               # four blank games back: too stale
    assert t2.iloc[16] == pytest.approx(raw(15))


def test_add_to_games_keeps_existing_columns_and_rows():
    d = synthetic_inputs(n_days=50, seed=6)
    feats = ft.build_team_features(d)
    games = d['tgf'][d['tgf']['MATCHUP'].str.contains('vs.', regex=False)][['gameId', 'teamId', 'gameDate']].rename(columns={'teamId': 'teamId_home'})
    opp = d['tgf'][d['tgf']['MATCHUP'].str.contains('@', regex=False)][['gameId', 'teamId']].rename(columns={'teamId': 'teamId_away'})
    games = games.merge(opp, on='gameId')
    games['gameId'] = games['gameId'].astype(int)                         # the training set stores ids without leading zeros
    for s in ('home', 'away'):
        games[f'is_b2b_{s}'] = 0.0
        games[f'dist_miles_{s}'] = 500.0
    games['existing'] = np.arange(len(games))
    out = ft.add_to_games(games, feats)
    assert len(out) == len(games)
    pd.testing.assert_frame_equal(out[games.columns], games)
    for fam in ft.FAMILIES:
        cols = ft.family_columns(fam)
        assert all(c in out.columns for c in cols), fam
    assert (out['diff_trk_prev_ot'].dropna() == out['trk_prev_ot_home'].dropna() - out['trk_prev_ot_away'].dropna()).all()


def test_prev_opp_pace_is_a_shock_not_the_teams_own_pace_level(monkeypatch):
    monkeypatch.setattr(ft, 'MIN_NORM', 2)
    monkeypatch.setattr(ft, 'MIN_SD', 2)
    d, *_ = two_team_inputs()
    d['tgf'] = d['tgf'].assign(pace=101.0)                      # a constant-pace team: nothing in the previous game was unusual
    f = ft.build_team_features(d)
    f = f[f['teamId'] == 1].set_index('gameId')
    assert f['trk_prev_opp_pace'].dropna().abs().max() < 1e-9 and f['trk_prev_opp_pace'].notna().any()
