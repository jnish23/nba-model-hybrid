"""
WP6a (reports/props/audit_2026-11/WP6_WP7_experiment_plan.md, section B): pre-game teammate sets for av_* / opp_av_*.

Leakage tests of B.7 on a synthetic league that has the awkward cases (trades, a call-up, season openers, Out players listed or not, a report without a
team, a same-day placement on another team, a team whose report is not submitted):
  L1 roster-listing invariance (and its teeth under 'box'), L2 future-report invariance, L3 outcome invariance, L4 placeholder invariance,
  L5 equivalences (vectorised previous-game rosters = the live function; availability_over_members with the box set = teammate_availability, float for float;
  the training member set = the live rule's member set, date by date).
Default-unchanged tests compare the edited builder, roster rules and scripts with frozen copies of the code before the change.
"""
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from props.features import build, history, injury_at_t as inj, pregame_roster as pg, rows as rows_mod
from props.live import roster

ROOT = Path(__file__).resolve().parents[2]
GRID = (240, 60, 5)
AV = None                                   # filled lazily: the av_* / opp_av_* columns
STAT_COLS = list(rows_mod.BASIC_COLS.values()) + list(rows_mod.TRACK_COLS.values()) + list(rows_mod.USAGE_COLS.values())


# ---------------------------------------------------------------- synthetic league ----------------------------------------------------------------
def _season_dates(season, n=10):
    start = pd.Timestamp('2023-11-01') if season == '2023-24' else pd.Timestamp('2024-11-01')
    return [(start + pd.Timedelta(days=2 * i)).strftime('%Y-%m-%d') for i in range(n)]


def _team_of(pid, season, di):
    """Player 105 is traded from team 1 to team 3 at 2024-25 date 6; player 150 is a call-up who joins team 1 at 2024-25 date 7."""
    if pid == 105 and season == '2024-25' and di >= 6:
        return 3
    if pid == 150:
        return 1 if (season == '2024-25' and di >= 7) else None
    return pid // 100


def league(seed=7):
    """Returns (hist in rows.load_history layout, linked reports, tips)."""
    rng = np.random.default_rng(seed)
    players = [t * 100 + k for t in (1, 2, 3, 4) for k in range(1, 10)] + [150]
    hrecs, rep, tips = [], [], {}
    n = 0
    for season in ('2023-24', '2024-25'):
        yy = season[2:4]
        for di, date in enumerate(_season_dates(season)):
            pairs = [(1, 2), (3, 4)] if di % 2 == 0 else [(3, 1), (4, 2)]
            for home, away in pairs:
                n += 1
                gid = f'002{yy}{n:05d}'
                tip = pd.Timestamp(f'{date} 19:00')
                tips[gid] = tip
                margin = int(rng.integers(-25, 26))
                for team, opp, is_home in ((home, away, 1), (away, home, 0)):
                    roster_now = [p for p in players if _team_of(p, season, di) == team]
                    # three reports a game: tip-6h (usable at every T), tip-2h (usable from T-60), tip-30min (usable at T-5 only)
                    named = rng.choice(roster_now, size=min(3, len(roster_now)), replace=False)
                    status = {}
                    not_submitted = (gid.endswith('7') and team == home)
                    for ts_h, k in ((6.0, 1), (2.0, 2), (0.5, 3)):
                        for p in named[:k]:
                            st = ['Out', 'Questionable', 'Probable', 'Doubtful'][int(rng.integers(0, 4))] if p not in status else status[p]
                            if ts_h == 0.5 and p == named[0]:
                                st = 'Out'
                            status[p] = st
                            if not_submitted:
                                continue
                            team_on_report = np.nan if (p % 7 == 0) else team           # some rows carry no team (placed by the player's latest team)
                            rep.append({'gameId': gid, 'teamId': team_on_report, 'personId': p, 'report_ts': tip - pd.Timedelta(hours=ts_h), 'status': st,
                                        'reason': 'Injury/Illness - Left knee; Soreness', 'not_submitted': False})
                    for i, p in enumerate(roster_now):
                        st = status.get(p)
                        if st == 'Out' and rng.random() < 0.55:                          # an Out player is listed (inactive) about half the time
                            continue
                        played = st != 'Out' and rng.random() > 0.12
                        mins = float(rng.uniform(8, 38)) if played else 0.0
                        rec = {'game_id': gid, 'player_id': p, 'team_id': team, 'opp_team_id': opp, 'game_date': date, 'season': season, 'game_type': '002',
                               'is_home': is_home, 'lbl_dnp_reason': None if played else 'DNP - Coach', 'lbl_team_margin': margin if team == home else -margin,
                               'minutes': mins if played else np.nan, 'lbl_started': int(i < 5)}
                        for c in STAT_COLS:
                            if c == 'minutes':
                                continue
                            rec[c] = float(rng.poisson(4)) if played else np.nan
                        if played:
                            rec['usage_pct'] = float(rng.uniform(0.1, 0.3))
                        hrecs.append(rec)
    # the traded player on the trade date: the report of his NEW team's game names him (a same-day placement on another team)
    trade_date = _season_dates('2024-25')[6]
    g3 = [g for g, t in tips.items() if t.strftime('%Y-%m-%d') == trade_date]
    hist = pd.DataFrame(hrecs)
    g_new = hist[(hist['game_id'].isin(g3)) & (hist['team_id'] == 3)]['game_id'].iloc[0]
    rep = [r for r in rep if not (r['gameId'] == g_new and r['personId'] == 105)]
    for ts_h in (6.0, 2.0, 0.5):                               # on every report of that game, so he is placed on team 3 at every T
        rep.append({'gameId': g_new, 'teamId': 3, 'personId': 105, 'report_ts': tips[g_new] - pd.Timedelta(hours=ts_h), 'status': 'Questionable',
                    'reason': 'Injury/Illness - Left ankle; Sprain', 'not_submitted': False})
    hist['lbl_played'] = (hist['minutes'].fillna(0) > 0).astype(int)
    for src, tgt in rows_mod.TARGETS.items():
        hist[tgt] = hist[src]
    hist['y_pra'] = hist[['points', 'rebounds', 'assists']].sum(axis=1, min_count=3)
    hist = hist.sort_values(['player_id', 'game_date', 'game_id']).reset_index(drop=True)
    return hist, pd.DataFrame(rep), pd.Series(tips)


POS = {p: 'G' for p in range(100, 500)}


def _env(game_ids):
    g = sorted(set(game_ids))
    return pd.DataFrame([{'game_id': x, 'minutes_before': t, 'home_spread': -3.5, 'total': 220.0, 'home_win_prob': 0.6, 'n_books': 5, 'spread_dispersion': 0.5,
                          'env_source': 1} for x in g for t in GRID])


@pytest.fixture
def patched(monkeypatch):
    """build_at_t without the schedule, the odds stores or the SBR proxy: a fixed tip table and a flat environment."""
    state = {}

    def tips_fn(seasons, offset_min=10):
        return state['tips']
    monkeypatch.setattr(build, 'reference_tips_et', tips_fn)
    monkeypatch.setattr(build.market_env, 'snapshot_env', lambda props_db, grid=GRID: _env(state['tips'].index))
    monkeypatch.setattr(build, 'load_clean_odds', lambda p: None)
    monkeypatch.setattr(build.market_env, 'sbr_proxy', lambda co, grid=GRID: _env(['x']).iloc[0:0])
    return state


def base_and_labels(hist):
    base = history.add_history_features(hist, POS, '2019-20')
    labels = hist[['game_id', 'player_id', 'lbl_played']].copy()
    return base, labels


def at_t(state, hist, reports, tips, membership, only=None):
    state['tips'] = tips
    base, labels = base_and_labels(hist)
    return build.build_at_t(base, labels, 'unused.db', 'unused.db', GRID, reports=reports, membership=membership, hist=hist, pos=POS, only_game_ids=only)


def av_cols(df):
    return [c for c in df.columns if c.startswith(('av_', 'opp_av_'))]


@pytest.fixture(scope='module')
def lg():
    return league()


# ---------------------------------------------------------------- L5: equivalences ----------------------------------------------------------------
def test_l5_previous_game_members_equals_the_live_previous_game_rosters_on_every_date(lg):
    hist, _, _ = lg
    games = pg.game_table(hist)
    prev = pg.previous_game_members(hist, games)
    for date, g in games.groupby('game_date'):
        live = roster.previous_game_rosters(hist, date, str(g['season'].iloc[0]))
        want = pg.prev_frame_from_rosters(live, g)
        got = prev[prev['game_id'].isin(set(g['game_id']))]
        assert set(map(tuple, got[['game_id', 'team_id', 'player_id']].to_numpy().tolist())) == set(map(tuple, want.to_numpy().tolist())), date
    # the traded player leaves team 1's set once his latest team is 3; season openers have no previous game
    first = games.sort_values('game_date').groupby('season')['game_date'].min()
    assert prev.merge(games[games['game_date'].isin(first)], on='game_id').empty


def _synthetic_d(n_games=6, seed=3):
    rng = np.random.default_rng(seed)
    recs = []
    for g in range(n_games):
        for team, opp in ((1, 2), (2, 1)):
            for k in range(int(rng.integers(1, 9))):
                recs.append({'game_id': f'G{g}', 'player_id': team * 100 + k, 'team_id': team, 'opp_team_id': opp, 'inj_p_play': float(rng.choice([0.0, 0.02, 0.57, 0.93, rng.random()])),
                             'inj_status_code': int(rng.integers(-1, 6))})
    d = pd.DataFrame(recs)
    feat = d[['game_id', 'player_id']].copy()
    for c in inj.AT_T_FEAT_COLS[2:]:
        feat[c] = rng.uniform(0, 36, len(d)) if c == 'm_min_h15' else rng.uniform(0, 1, len(d))
    feat.loc[feat.sample(frac=0.1, random_state=1).index, 'm_min_h15'] = np.nan
    return d, feat


def test_l5_availability_over_members_with_the_box_set_is_teammate_availability_float_for_float():
    d, feat = _synthetic_d()
    old = inj.teammate_availability(d, feat)
    new = pg.availability_over_members(d[['game_id', 'player_id', 'team_id', 'opp_team_id']], d, feat)
    pd.testing.assert_frame_equal(new.reset_index(drop=True), old.reset_index(drop=True), check_exact=True)


def test_availability_over_members_counts_the_team_for_a_non_member_target_and_excludes_a_member_target():
    d, feat = _synthetic_d(2)
    members = d[d['player_id'] % 100 != 0]                                              # player k=0 of each team is a target but not a member
    out = pg.availability_over_members(d[['game_id', 'player_id', 'team_id', 'opp_team_id']], members, feat).set_index(['game_id', 'player_id'])
    for (g, t), grp in members.groupby(['game_id', 'team_id']):
        n_out = int((grp['inj_status_code'] == 5).sum())
        tgt = (g, t * 100)
        if tgt in out.index:
            assert out.loc[tgt, 'av_n_out'] == n_out                                     # whole team: nothing subtracted
        for r in grp.itertuples():
            assert out.loc[(g, r.player_id), 'av_n_out'] == n_out - int(r.inj_status_code == 5)


def test_l5_at_t_members_block_with_the_box_set_is_at_t_block(lg):
    hist, reports, tips = lg
    base, labels = base_and_labels(hist)
    chosen, games = inj.latest_report_rows(reports, tips, 60)
    status = inj.status_for_rows(base[['game_id', 'player_id', 'team_id']], chosen, games)
    keyed = status.merge(base[['game_id', 'player_id', 'season', 'm_bucket']], on=['game_id', 'player_id']).merge(labels, on=['game_id', 'player_id'])
    table = inj.play_rate_table(keyed, 60, '2019-20')
    old = inj.at_t_block(base, status, table)
    new = pg.at_t_members_block(base, status, table, base[pg.MEMBER_COLS], status, base)
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def _live_effective_v2_members(g, hist, date, rp, T):
    """The member set the live v2 path really uses with no quotes: rule_members pairs that get a placeholder row (props_d1_parity.build_live_frames)."""
    h = hist[hist['game_date'] < date]
    pairs = roster.rule_members('v2', g, hist, date, rp, rp.iloc[0:0][['game_id', 'player_id']])
    placed, _ = roster.resolve_extra_players(rp[['game_id', 'player_id', 'team_id']], g, h, date)
    rows = roster.placeholder_rows(['game_id', 'player_id', 'team_id', 'opp_team_id', 'game_date', 'season', 'game_type', 'is_home', 'lbl_played', 'lbl_started'],
                                   g, roster.previous_game_rosters(h, date, str(g['season'].iloc[0])), placed)
    return rows.merge(pairs, on=['game_id', 'player_id'], how='inner')[['game_id', 'player_id', 'team_id']]


@pytest.mark.parametrize('rule', ['v2nq', 'pregame_all'])
def test_l5_training_members_equal_the_live_rule_members_on_every_date_and_T(lg, rule):
    hist, reports, tips = lg
    games = pg.game_table(hist)
    prev = pg.previous_game_members(hist, games)
    checked_cross_game = False
    for x in GRID:
        chosen, _ = inj.latest_report_rows(reports, tips, x)
        rp_all = pg.report_players(chosen)
        train = pg.members_at_t(prev, rp_all, games, hist, rule)
        for date, g in games.groupby('game_date'):
            rp = rp_all[rp_all['game_id'].isin(set(g['game_id']))]
            live = roster.rule_member_teams(rule, g, hist, date, rp)
            t = train[train['game_id'].isin(set(g['game_id']))].reset_index(drop=True)
            pd.testing.assert_frame_equal(t, live.reset_index(drop=True), check_dtype=False)
            pairs = roster.rule_members(rule, g, hist, date, rp, rp.iloc[0:0])
            assert set(map(tuple, pairs.to_numpy().tolist())) == set(map(tuple, live[['game_id', 'player_id']].to_numpy().tolist()))
            if rule == 'v2nq':                                           # v2nq = the effective live v2 set without quotes (incl. same-day placement on a new team)
                eff = _live_effective_v2_members(g, hist, date, rp, x)
                assert set(map(tuple, eff.astype({'team_id': 'int64'}).to_numpy().tolist())) == set(map(tuple, live[['game_id', 'player_id', 'team_id']].to_numpy().tolist())), (date, x)
            if date == _season_dates('2024-25')[6]:
                assert not ((live['player_id'] == 105) & (live['team_id'] == 1)).any()      # named for team 3 today: off team 1
                checked_cross_game = True
    assert checked_cross_game


def test_report_players_matches_the_live_function(lg):
    from props.live import at_t as A
    _, reports, tips = lg
    for x in GRID:
        chosen, _ = inj.latest_report_rows(reports, tips, x)
        pd.testing.assert_frame_equal(pg.report_players(chosen), A.report_players(A.asof_reports(reports, tips, x), tips, x))


def test_membership_rules_by_hand():
    hist = pd.DataFrame({'game_id': ['p1'] * 4 + ['o1'], 'team_id': [1, 1, 1, 1, 1], 'player_id': [10, 11, 12, 13, 14],
                         'game_date': ['2025-11-01'] * 4 + ['2025-10-30'], 'season': ['2025-26'] * 5})
    games = pd.DataFrame({'game_id': ['g'], 'home_team_id': [1], 'away_team_id': [2], 'season': ['2025-26'], 'game_date': ['2025-11-03']})
    report = pd.DataFrame({'game_id': ['g', 'g', 'g', 'g'], 'player_id': [11, 12, 20, 30], 'team_id': [1, 1, 2, np.nan], 'status': ['Out', 'Questionable', 'Out', 'Probable']})
    v2nq = roster.rule_member_teams('v2nq', games, hist, '2025-11-03', report)
    assert set(v2nq['player_id']) == {10, 12, 13}                     # 11 Out; 20 Out; 30 has no team and no history this season: dropped; 14 not in the previous game
    allp = roster.rule_member_teams('pregame_all', games, hist, '2025-11-03', report).set_index('player_id')
    assert set(allp.index) == {10, 11, 12, 13, 20} and allp.loc[20, 'team_id'] == 2 and allp.loc[20, 'opp_team_id'] == 1
    opener = roster.rule_member_teams('pregame_all', games.assign(season='2026-27'), hist, '2025-11-03', report)
    assert set(opener['player_id']) == {11, 12, 20}                   # a season opener: the report players only (30 cannot be placed)


# ---------------------------------------------------------------- L4: placeholders ----------------------------------------------------------------
def _all_members(hist, reports, tips, rule):
    games = pg.game_table(hist)
    prev = pg.previous_game_members(hist, games)
    ms = []
    for x in GRID:
        chosen, _ = inj.latest_report_rows(reports, tips, x)
        ms.append(pg.members_at_t(prev, pg.report_players(chosen), games, hist, rule))
    return pd.concat(ms).drop_duplicates(['game_id', 'player_id']).merge(games[['game_id', 'game_date', 'season']], on='game_id')


@pytest.mark.parametrize('rule', ['v2nq', 'pregame_all'])
def test_l4_placeholders_leave_every_real_rows_weights_and_bucket_unchanged(lg, rule):
    hist, reports, tips = lg
    allm = _all_members(hist, reports, tips, rule)
    full = pg.with_placeholders(hist, allm)
    assert full is not None and (full['_ph'] == 0).sum() > 0
    with_ph = history.add_history_features(full.drop(columns=['_ph', '_k']), POS, '2019-20')
    real = with_ph[(full['_ph'] == 1).to_numpy()[(full['season'] >= '2019-20').to_numpy()]].reset_index(drop=True)
    ref = history.add_history_features(hist, POS, '2019-20')
    real = real.set_index(['game_id', 'player_id']).loc[ref.set_index(['game_id', 'player_id']).index].reset_index()
    for c in pg.WEIGHT_COLS + ['m_bucket']:
        pd.testing.assert_series_equal(real[c], ref[c], check_exact=True, check_names=False)
    # the stored base table never contains placeholders: build_base has no membership; role_*/rest_* WOULD change in a single pass, which is why it is separate
    assert np.array_equal(real['is_home'].to_numpy(float), ref['is_home'].to_numpy(float))        # only its dtype changes (the placeholders carry NaN)
    changed = [c for c in ref.columns if c not in ('game_id', 'player_id', 'is_home') and not real[c].equals(ref[c])]
    assert all(c.startswith(('role_', 'rest_')) for c in changed)


def test_l4_placeholder_weights_equal_the_live_history_path(lg):
    """A member with no listed row: the training placeholder (one global pass) equals live's placeholder on the history before the date."""
    hist, reports, tips = lg
    allm = _all_members(hist, reports, tips, 'pregame_all')
    ph = pg.placeholder_weights(hist, allm, POS)
    assert len(ph) > 0
    for date, grp in ph.merge(allm[['game_id', 'game_date']].drop_duplicates(), on='game_id').groupby('game_date'):
        h = hist[hist['game_date'] < date]
        need = allm[(allm['game_date'] == date)].merge(grp[['game_id', 'player_id']], on=['game_id', 'player_id'])
        rows = need.assign(game_type='002', is_home=np.nan, lbl_played=0, lbl_started=0)
        for c in hist.columns:
            if c not in rows.columns:
                rows[c] = np.nan
        full = pd.concat([h, rows[list(hist.columns)]], ignore_index=True).sort_values(['player_id', 'game_date', 'game_id']).reset_index(drop=True)
        live = history.add_history_features(full, POS, '2019-20').merge(need[['game_id', 'player_id']], on=['game_id', 'player_id'])
        got = grp.merge(live[['game_id', 'player_id'] + pg.WEIGHT_COLS + ['m_bucket']], on=['game_id', 'player_id'], suffixes=('', '_live'))
        assert len(got) == len(grp)
        assert np.array_equal(got['m_bucket'].to_numpy(), got['m_bucket_live'].to_numpy()), date
        for c in pg.WEIGHT_COLS:                                # the live history stops before the date, so its same-season prior sums differ: float noise only
            assert _max_diff(got[[c]], got[[c + '_live']].rename(columns={c + '_live': c})) <= ULP, (date, c)


# ---------------------------------------------------------------- L1-L3 through the real builder ----------------------------------------------------------------
def _target_game(hist, reports, tips):
    """A late-season game whose home team has an Out player who is LISTED (inactive) in the box score, plus another listed teammate with history."""
    chosen, _ = inj.latest_report_rows(reports, tips, 5)
    rp = pg.report_players(chosen)
    late = hist[hist['season'] == '2024-25']
    for gid, g in late.groupby('game_id'):
        outs = rp[(rp['game_id'] == gid) & (rp['status'] == 'Out')]
        listed_out = g[g['player_id'].isin(outs['player_id']) & (g['lbl_played'] == 0)]
        if len(listed_out) and g['game_date'].iloc[0] >= _season_dates('2024-25')[3]:
            return gid, int(listed_out['player_id'].iloc[0])
    raise AssertionError('the synthetic league has no listed Out player late in the season')


def _perturbations(hist, gid, j):
    g = hist[hist['game_id'] == gid]
    team = int(g.loc[g['player_id'] == j, 'team_id'].iloc[0])
    # (a) remove the listed did-not-play player j
    a = hist[~((hist['game_id'] == gid) & (hist['player_id'] == j))]
    # (b) add a fabricated listed player: a player with history on another team, listed (did not play) for j's team in this game
    k = int(hist.loc[(hist['team_id'] != team) & ~hist['player_id'].isin(g['player_id']), 'player_id'].iloc[0])
    row = g[g['player_id'] == j].iloc[[0]].assign(player_id=k)
    b = pd.concat([hist, row], ignore_index=True).sort_values(['player_id', 'game_date', 'game_id']).reset_index(drop=True)
    # (c) change the minutes of a player who played
    c = hist.copy()
    pl = c.index[(c['game_id'] == gid) & (c['lbl_played'] == 1)][0]
    c.loc[pl, ['minutes', 'points']] = [c.loc[pl, 'minutes'] + 7.0, 40.0]
    return {'remove_dnp': (a, {j}), 'add_listed': (b, {k}), 'change_minutes': (c, set())}


ULP = 1e-12          # history.py's league priors are cumsum - own season (_expanding_prior), so ANY change to a season's sums moves same-season rows by ~1e-16


def _max_diff(a, b):
    """Largest absolute difference of two aligned numeric frames; a value missing on one side only counts as infinite."""
    x, y = a.to_numpy(float), b.to_numpy(float)
    if (np.isnan(x) != np.isnan(y)).any():
        return np.inf
    d = np.abs(x - y)
    return float(np.nanmax(d)) if d.size and not np.isnan(d).all() else 0.0


def _same_frame(new, ref, tol):
    """Non-float columns equal; float columns within tol (missing on both sides agree)."""
    assert list(new.columns) == list(ref.columns) and len(new) == len(ref)
    for c in ref.columns:
        if ref[c].dtype.kind == 'f':
            assert _max_diff(new[[c]], ref[[c]]) <= tol, c
        else:
            pd.testing.assert_series_equal(new[c], ref[c], check_names=False)


def _rows_of(df, gid, skip):
    d = df[(df['game_id'] == gid) & ~df['player_id'].isin(skip)]
    return d.sort_values(['player_id', 'minutes_before']).reset_index(drop=True)


@pytest.mark.parametrize('membership', ['box', 'v2nq', 'pregame_all'])
def test_l1_roster_listing_invariance(lg, patched, membership):
    hist, reports, tips = lg
    gid, j = _target_game(hist, reports, tips)
    ref = at_t(patched, hist, reports, tips, membership, only=[gid])
    cols = av_cols(ref)
    changed = {}
    for name, (h2, skip) in _perturbations(hist, gid, j).items():
        new = at_t(patched, h2, reports, tips, membership, only=[gid])
        a, b = _rows_of(ref, gid, skip | {j}), _rows_of(new, gid, skip | {j})
        assert len(a) == len(b) and len(a) > 0
        exact = not a[cols].equals(b[cols])
        changed[name] = _max_diff(a[cols], b[cols]) > ULP
        if name != 'change_minutes':
            assert changed[name] == exact                      # removing or adding a did-not-play row changes no sum: exact, no float noise
    if membership == 'box':                                    # the documented leak: who is listed after the game moves the teammates' features
        assert changed['remove_dnp'] and changed['add_listed'], changed
    else:
        assert not any(changed.values()), changed
    assert not changed['change_minutes']                       # a player's own minutes in G never enter G's at-T features, under any membership


@pytest.mark.parametrize('membership', ['box', 'v2nq', 'pregame_all'])
def test_l2_a_report_stamped_inside_the_publish_lag_changes_nothing(lg, patched, membership):
    hist, reports, tips = lg
    gid, _ = _target_game(hist, reports, tips)
    ref = at_t(patched, hist, reports, tips, membership, only=[gid])
    g = hist[hist['game_id'] == gid]
    starter = int(g.loc[g['lbl_started'] == 1, 'player_id'].iloc[0])
    team = int(g.loc[g['player_id'] == starter, 'team_id'].iloc[0])
    late = []
    for x in GRID:                                             # stamped 10 minutes before T: later than T - PUBLISH_LAG_MIN
        late.append({'gameId': gid, 'teamId': team, 'personId': starter, 'report_ts': tips[gid] - pd.Timedelta(minutes=x + 10), 'status': 'Out',
                     'reason': 'Injury/Illness - Right knee; Sprain', 'not_submitted': False})
    new = at_t(patched, hist, pd.concat([reports, pd.DataFrame(late)], ignore_index=True), tips, membership, only=[gid])
    pd.testing.assert_frame_equal(new, ref, check_exact=True)


@pytest.mark.parametrize('membership', ['box', 'v2nq', 'pregame_all'])
def test_l3_outcomes_of_the_game_change_no_at_t_feature(lg, patched, membership):
    hist, reports, tips = lg
    gid, _ = _target_game(hist, reports, tips)
    ref = at_t(patched, hist, reports, tips, membership, only=[gid])
    h2 = hist.copy()
    m = (h2['game_id'] == gid) & (h2['lbl_played'] == 1)
    h2.loc[m, 'minutes'] = h2.loc[m, 'minutes'] * 0.5 + 3
    for c in [c for c in h2.columns if c.startswith('y_')] + ['points', 'rebounds', 'assists']:
        h2.loc[m, c] = h2.loc[m, c].fillna(0) + 5
    h2.loc[h2['game_id'] == gid, 'lbl_team_margin'] = 30
    h2.loc[h2['game_id'] == gid, 'lbl_started'] = 1 - h2.loc[h2['game_id'] == gid, 'lbl_started']
    new = at_t(patched, h2, reports, tips, membership, only=[gid])
    # exact except float noise from the priors (ULP); the same holds under 'box', so it is a property of history.py, not of the memberships
    _same_frame(new, ref, ULP)


def test_pregame_all_shows_out_teammates_and_v2nq_does_not(lg, patched):
    hist, reports, tips = lg
    out = {m: at_t(patched, hist, reports, tips, m) for m in ('box', 'v2nq', 'pregame_all')}
    share = {m: float((d['av_n_out'] > 0).mean()) for m, d in out.items()}
    assert share['v2nq'] == 0.0 and share['pregame_all'] > share['box'] > 0.0, share
    # the target rows and every non-av column are the same under every membership
    keep = [c for c in out['box'].columns if c not in av_cols(out['box'])]
    for m in ('v2nq', 'pregame_all'):
        pd.testing.assert_frame_equal(out[m][keep], out['box'][keep], check_exact=True)
        assert list(out[m].columns) == list(out['box'].columns)


# ---------------------------------------------------------------- default unchanged (frozen copies) ----------------------------------------------------------------
def _frozen_build_at_t(base, labels, db_path, props_db, grid=build.GRID, reports=None):
    """props/features/build.build_at_t as of commit 26a0319 (before WP6a), verbatim apart from the module prefixes."""
    seasons = sorted(base['season'].unique())
    tips = build.reference_tips_et(seasons)
    reports = build.load_linked_reports(db_path) if reports is None else reports
    lab = labels[['game_id', 'player_id', 'lbl_played']]
    snap = build.market_env.snapshot_env(props_db)
    proxy = build.market_env.sbr_proxy(build.load_clean_odds(Path(db_path)))
    pre_market = set(base.loc[base['season'] < build.MARKET_ERA_FIRST_SEASON, 'game_id'])
    env = build.market_env.combine(snap, proxy[proxy['game_id'].isin(pre_market)])
    out = []
    for x in grid:
        chosen, games = inj.latest_report_rows(reports, tips, x)
        status = inj.status_for_rows(base[['game_id', 'player_id', 'team_id']], chosen, games)
        keyed = status.merge(base[['game_id', 'player_id', 'season', 'm_bucket']], on=['game_id', 'player_id'], validate='one_to_one').merge(lab, on=['game_id', 'player_id'], validate='one_to_one')
        table = inj.play_rate_table(keyed, x, rows_mod.FIRST_DATASET_SEASON)
        env_x = build.market_env.team_view(base[['game_id', 'player_id', 'is_home']], env, x)
        t = inj.at_t_block(base, status, table).merge(env_x, on=['game_id', 'player_id'], validate='one_to_one')
        t.insert(2, 'minutes_before', x)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def test_default_build_at_t_is_unchanged(lg, patched):
    hist, reports, tips = lg
    patched['tips'] = tips
    base, labels = base_and_labels(hist)
    new = build.build_at_t(base, labels, 'unused.db', 'unused.db', reports=reports)
    old = _frozen_build_at_t(base, labels, 'unused.db', 'unused.db', reports=reports)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    sub = build.build_at_t(base, labels, 'unused.db', 'unused.db', reports=reports, only_game_ids=list(base['game_id'].unique()[:5]))
    pd.testing.assert_frame_equal(sub.reset_index(drop=True), old[old['game_id'].isin(set(sub['game_id']))].reset_index(drop=True), check_exact=True)


def _frozen_rule_members(rule, games, hist, date, report, quotes):
    """props/live/roster.rule_members as of commit 26a0319, verbatim apart from the module prefix."""
    h = hist[hist['game_date'] < date]
    if rule == 'registered':
        r = roster.current_team_only(roster.recent_rosters(h, date), roster.last_team(h, date))
        base = [(g.game_id, p) for g in games.itertuples() for team in (int(g.home_team_id), int(g.away_team_id)) for p in r.get(team, ())]
        return roster._pairs(pd.concat([pd.DataFrame(base, columns=['game_id', 'player_id']), report[['game_id', 'player_id']], quotes[['game_id', 'player_id']]], ignore_index=True))
    if rule == 'v2':
        prev = roster.previous_game_rosters(hist, date, str(games['season'].iloc[0]))
        base = pd.DataFrame([(g.game_id, p) for g in games.itertuples() for team in (int(g.home_team_id), int(g.away_team_id)) for p in prev.get(team, ())], columns=['game_id', 'player_id']).astype({'game_id': str, 'player_id': 'int64'})
        out = report[report['status'] == 'Out'][['game_id', 'player_id']]
        base = base.merge(out.assign(_out=1), on=['game_id', 'player_id'], how='left')
        base = base[base['_out'].isna()][['game_id', 'player_id']]
        return roster._pairs(pd.concat([base, report[report['status'] != 'Out'][['game_id', 'player_id']], quotes[['game_id', 'player_id']]], ignore_index=True))
    raise ValueError(f'unknown roster rule {rule}')


def test_existing_roster_rules_are_unchanged(lg):
    hist, reports, tips = lg
    games = pg.game_table(hist)
    rng = np.random.default_rng(0)
    for x, (date, g) in itertools.product(GRID, games.groupby('game_date')):
        chosen, _ = inj.latest_report_rows(reports, tips, x)
        rp = pg.report_players(chosen)
        rp = rp[rp['game_id'].isin(set(g['game_id']))]
        quotes = hist[hist['game_id'].isin(set(g['game_id']))][['game_id', 'player_id']].sample(frac=0.3, random_state=int(rng.integers(1e6)))
        for rule in ('registered', 'v2'):
            pd.testing.assert_frame_equal(roster.rule_members(rule, g, hist, date, rp, quotes), _frozen_rule_members(rule, g, hist, date, rp, quotes))
    with pytest.raises(ValueError):
        roster.rule_members('nope', games.iloc[:1], hist, '2024-11-01', rp, quotes)


# ---------------------------------------------------------------- scripts: defaults and guards ----------------------------------------------------------------
def _script(name):
    sys.path.insert(0, str(ROOT / 'scripts'))
    return importlib.import_module(name)


def test_build_script_default_call_is_unchanged(monkeypatch):
    bpf = _script('build_props_features')
    calls = {}

    def fake(*a, **k):
        calls['a'], calls['k'] = a, k
        raise SystemExit('stop')
    monkeypatch.setattr(bpf.build, 'build_all', fake)
    with pytest.raises(SystemExit):
        bpf.main([])
    assert calls['a'] == ('data/nba_stats.db', 'data/props_odds.db', str(build.OUT_DB)) and calls['k'] == {'max_date': None, 'refresh_reports': False}
    a = bpf.parse_args([])
    assert a.membership is None and (a.dictionary_dir or bpf.DEFAULT_DICTIONARY_DIR) == 'reports/props'
    bpf.research_guard(a)                                     # a default build is never refused


def test_build_script_research_guard(tmp_path):
    bpf = _script('build_props_features')
    cache = tmp_path / 'linked.pkl'
    cache.write_bytes(b'x')
    ok = ['--membership', 'v2nq', '--out-db', str(tmp_path / 'new.db'), '--dictionary-dir', str(tmp_path / 'dict'), '--linked-cache', str(cache)]
    bpf.research_guard(bpf.parse_args(ok))
    bad = {
        'v3 name': ['--out-db', 'data/props_features_v3.db'], 'default db': ['--out-db', str(build.OUT_DB)],
        'v3 elsewhere': ['--out-db', str(tmp_path / 'props_features_v3.db')], 'dictionary default': ['--dictionary-dir', 'reports/props'],
        'refresh': ['--refresh-reports'], 'no cache': ['--linked-cache', str(tmp_path / 'missing.pkl')]}
    for name, extra in bad.items():
        args = list(ok)
        for i in range(0, len(extra), 2) if extra[0] != '--refresh-reports' else []:
            j = args.index(extra[i])
            args[j + 1] = extra[i + 1]
        if extra[0] == '--refresh-reports':
            args.append('--refresh-reports')
        with pytest.raises(SystemExit):
            bpf.research_guard(bpf.parse_args(args))
    no_dict = [x for x in ok]
    k = no_dict.index('--dictionary-dir')
    del no_dict[k:k + 2]
    with pytest.raises(SystemExit):
        bpf.research_guard(bpf.parse_args(no_dict))
    exists = tmp_path / 'exists.db'
    exists.write_bytes(b'')
    args = list(ok)
    args[args.index('--out-db') + 1] = str(exists)
    with pytest.raises(SystemExit):
        bpf.research_guard(bpf.parse_args(args))
    with pytest.raises(SystemExit):                            # --membership box (the control build) is a research build too
        bpf.research_guard(bpf.parse_args(['--membership', 'box', '--out-db', 'data/props_features_v3.db', '--dictionary-dir', str(tmp_path)]))


def test_parity_script_defaults_are_unchanged():
    par = _script('props_d1_parity')
    a = par.parse_args(['--attempt', '3'])
    assert a.features_db == 'data/props_features_v3.db' and a.out is None and a.db == 'data/nba_stats.db' and a.linked_cache is None
    assert a.roster_rule == 'registered' and a.dates == 'reports/props/d1_parity_dates.json' and a.season == '2025-26'
    s11 = par.parse_args(['--attempt', '0', '--season', '2024-25', '--dates', 'reports/props/d1_roster_dev_dates.json', '--roster-rule', 'v2nq', '--no-model',
                          '--features-db', 'x.db', '--prices', 'p.pkl', '--out', 'o'])
    assert s11.roster_rule == 'v2nq' and s11.features_db == 'x.db' and s11.out == 'o'
