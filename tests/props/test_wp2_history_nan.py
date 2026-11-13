"""
WP2 at the level of history.add_history_features, on a synthetic box-score frame (no database, no model):
 - with no NaN on played rows the per-minute rate and conversion features equal the pre-WP2 formulas exactly (the stored features DB must not change);
 - a played game missing one tracking stat is dropped from that stat's rate and from nothing else;
 - the features never look at the row itself or later rows.
"""
import numpy as np
import pandas as pd
import pytest

from props.features import decay, history, rows

TRACK = list(rows.TRACK_COLS.values())
BASIC = [c for c in rows.BASIC_COLS.values() if c != 'minutes']
ALL_STATS = BASIC + TRACK
N_PLAYERS, GAMES_PER_SEASON = 6, 30
SEASONS = ('2018-19', '2019-20')


def make_hist(seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    gid = 0
    for si, season in enumerate(SEASONS):
        start = pd.Timestamp('2018-10-20') + pd.Timedelta(days=365 * si)
        for gi in range(GAMES_PER_SEASON):
            gid += 1
            for p in range(N_PLAYERS):
                played = rng.random() < 0.85
                minutes = rng.uniform(29, 40) if played else np.nan              # always in the same minutes bucket, so priors cannot differ by bucket
                r = {'game_id': f'{gid:010d}', 'player_id': 100 + p, 'team_id': 1 + p % 2, 'opp_team_id': 2 - p % 2,
                     'game_date': (start + pd.Timedelta(days=2 * gi)).strftime('%Y-%m-%d'), 'season': season, 'game_type': '002',
                     'is_home': gi % 2, 'lbl_played': int(played), 'lbl_started': int(p < 3), 'lbl_team_margin': rng.normal(0, 10),
                     'minutes': minutes}
                for s in ALL_STATS:
                    r[s] = rng.poisson(5 + 0.3 * p) * 1.0 if played else np.nan
                r['usage_pct'] = rng.uniform(0.1, 0.3) if played else np.nan
                recs.append(r)
    h = pd.DataFrame(recs)
    h['fgm'] = np.minimum(h['fgm'], h['fga'])
    return h.sort_values(['player_id', 'game_date', 'game_id']).reset_index(drop=True)


POS = {100 + p: 'GF' for p in range(N_PLAYERS)}


def feats(h):
    return history.add_history_features(h, POS, first_dataset_season='2019-20')


def old_league_priors(h, keys_df, stats, minutes, played):
    """Pre-WP2 history._league_priors: one shared minutes denominator, NaN counted as 0 on played rows."""
    tables = {}
    den = np.where(played, np.nan_to_num(minutes), 0.0)
    for s, vals in stats.items():
        d = keys_df.assign(num=np.where(played, np.nan_to_num(vals), 0.0), den=den)
        tables[s] = {
            'cell': history._expanding_prior(d.groupby(['season', 'pos', 'bucket'])[['num', 'den']].sum()),
            'bucket': history._expanding_prior(d.groupby(['season', 'bucket'])[['num', 'den']].sum()),
            'all': history._expanding_prior(d.groupby('season')[['num', 'den']].sum())}
    return history._Priors(tables)


def old_rate_and_pct_columns(h):
    """The r_* and c_* columns exactly as pre-WP2 history.add_history_features computed them."""
    g = h['player_id'].to_numpy()
    played = h['lbl_played'].to_numpy().astype(bool)
    minutes = h['minutes'].to_numpy(dtype=float)
    out = pd.DataFrame(index=h.index)
    mean15 = np.full(len(h), np.nan)
    s1, s0 = decay.decayed_before(minutes, g, played, decay.alpha_for(15)), decay.decayed_count_before(g, played, decay.alpha_for(15))
    mean15 = np.where(s0 > 0, s1 / np.where(s0 > 0, s0, 1), np.nan)
    pos = h['player_id'].map(POS).fillna('F').to_numpy()
    keys_df = pd.DataFrame({'season': h['season'], 'pos': pos, 'bucket': history._bucket(mean15)})
    stats = {s: h[s].to_numpy(dtype=float) for s in history.RATE_STATS if s in h.columns}
    stats['usage_x_min'] = h['usage_pct'].to_numpy(dtype=float) * minutes
    priors = old_league_priors(h, keys_df, stats, minutes, played)
    k = history.HYPER['k_minutes']
    for hl in history.HALF_LIVES:
        a = decay.alpha_for(hl)
        s_min = decay.decayed_before(minutes, g, played, a)
        for s, vals in stats.items():
            pr = priors.lookup(s, keys_df)
            name = 'r_usage_pct' if s == 'usage_x_min' else f'r_{s}'
            out[f'{name}_h{hl}'] = (decay.decayed_before(vals, g, played, a) + k * pr) / (s_min + k)
    ka = history.HYPER['k_attempts']
    for name, (made, att) in history.PCT_STATS.items():
        for hl in (15, 40):
            a = decay.alpha_for(hl)
            s_m, s_a = decay.decayed_before(stats[made], g, played, a), decay.decayed_before(stats[att], g, played, a)
            pm, pa = priors.lookup(made, keys_df), priors.lookup(att, keys_df)
            league_pct = np.where(pa > 0, pm / np.where(pa > 0, pa, 1), np.nan)
            out[f'c_{name}_h{hl}'] = (s_m + ka * league_pct) / (s_a + ka)
    out['season'] = h['season'].to_numpy()
    return out[out['season'] >= '2019-20'].drop(columns='season').reset_index(drop=True)


def test_no_nan_on_played_rows_leaves_every_rate_and_conversion_feature_unchanged():
    h = make_hist(0)
    new = feats(h)
    old = old_rate_and_pct_columns(h)
    assert list(old.columns) == [c for c in new.columns if c.startswith(('r_', 'c_'))]
    assert len(old) == len(new) > 100
    for c in old.columns:
        got, want = new[c].to_numpy(), old[c].to_numpy()
        assert np.array_equal(np.isnan(got), np.isnan(want)), c
        assert np.nanmax(np.abs(got - want)) <= 1e-12, c


def with_missing_tracking(h, player=102, nth_game=12, cols=('touches',)):
    """Same frame with `cols` set to NaN on one played row of season 2 (the NBA tracking sync gave up on that game). Returns (frame, row label)."""
    sel = h[(h['player_id'] == player) & (h['season'] == SEASONS[1]) & (h['lbl_played'] == 1)]
    j = sel.index[nth_game]
    h2 = h.copy()
    for c in cols:
        h2.loc[j, c] = np.nan
    return h2, j


def test_a_played_game_missing_one_tracking_stat_changes_only_that_stat_and_only_for_that_player():
    h = make_hist(1)
    base = feats(h)
    h2, j = with_missing_tracking(h, cols=('touches', 'passes'))
    got = feats(h2)
    key = ['game_id', 'player_id']
    assert (got[key].to_numpy() == base[key].to_numpy()).all()
    changed = [c for c in base.columns if base[c].dtype.kind == 'f' and not np.allclose(base[c], got[c], rtol=0, atol=0, equal_nan=True)]
    assert changed and all(c.startswith(('r_touches_', 'r_passes_')) for c in changed), changed         # no other stat, no minutes/role/conversion feature
    diff_rows = ~np.isclose(base['r_touches_h15'], got['r_touches_h15'], rtol=0, atol=0, equal_nan=True)
    assert set(base.loc[diff_rows, 'player_id']) == {int(h.loc[j, 'player_id'])}


@pytest.mark.parametrize('hl', history.HALF_LIVES)
def test_missing_stat_equals_the_game_removed_for_that_stat(hl):
    h = make_hist(2)
    h2, j = with_missing_tracking(h, player=103, nth_game=10, cols=('reb_chances',))
    nan_frame = feats(h2)
    gone = feats(h.drop(index=j).reset_index(drop=True))               # the game does not exist at all (for any stat)
    pid, season = int(h.loc[j, 'player_id']), SEASONS[1]
    date_j = h.loc[j, 'game_date']
    a = nan_frame[(nan_frame['player_id'] == pid) & (nan_frame['game_date'] > date_j)].set_index('game_id')
    b = gone[(gone['player_id'] == pid) & (gone['game_date'] > date_j)].set_index('game_id')
    assert len(a) >= 5 and a.index.equals(b.index)
    # the stat that is missing: the rate equals the one without the game (same minutes denominator, same prior from earlier seasons)
    assert a[f'r_reb_chances_h{hl}'].to_numpy() == pytest.approx(b[f'r_reb_chances_h{hl}'].to_numpy(), abs=1e-12)
    # the other stats of the same game still count that game: they differ from the frame in which it does not exist
    assert not np.allclose(a[f'r_points_h{hl}'], b[f'r_points_h{hl}'], rtol=0, atol=1e-9)
    # ... and they are exactly what the complete frame gives
    full = feats(h)
    c = full[(full['player_id'] == pid) & (full['game_date'] > date_j)].set_index('game_id')
    for s in ('points', 'touches', 'passes', 'contested_fga', 'reb_chances_off'):
        assert np.array_equal(a[f'r_{s}_h{hl}'].to_numpy(), c[f'r_{s}_h{hl}'].to_numpy()), s


def _leak_columns(f):
    return [c for c in f.columns if c.startswith(('r_', 'c_', 'm_', 'role_', 'rest_'))]


@pytest.mark.parametrize('with_nan', (False, True))
def test_features_at_row_i_ignore_row_i_and_later_rows_of_the_player(with_nan):
    h = make_hist(3)
    if with_nan:
        h, _ = with_missing_tracking(h, player=101, nth_game=6, cols=('touches', 'secondary_ast'))
    base = feats(h)
    pid, t = 101, 15
    mine = h[(h['player_id'] == pid) & (h['season'] == SEASONS[1])]
    cut = mine.index[t]                                                 # alter this row and every later row of the player (later in the same season)
    later = mine.index[t:]
    rng = np.random.default_rng(7)
    h2 = h.copy()
    for c in ALL_STATS + ['minutes', 'usage_pct', 'lbl_team_margin']:
        h2.loc[later, c] = rng.normal(500, 50, len(later))
    h2.loc[later, 'lbl_played'] = 1 - h2.loc[later, 'lbl_played']
    h2.loc[later, 'lbl_started'] = 1 - h2.loc[later, 'lbl_started']
    h2.loc[later[::3], 'touches'] = np.nan
    again = feats(h2)
    cols = _leak_columns(base)
    key = ['game_id', 'player_id']
    a = base.set_index(key)
    b = again.set_index(key).loc[a.index]
    first_altered = (pid, h.loc[cut, 'game_id'])
    # every row of the player up to and including the first altered row, and every row of every other player of those games, is unchanged
    mask = ~((a.index.get_level_values('player_id') == pid) & (a.index.get_level_values('game_id') >= first_altered[1]))
    # other players' RATE features can move through the league prior only in LATER seasons; the altered rows are in the last season, so none move
    assert mask.sum() > 100
    # (tolerance, not equality: the league prior is cumsum - own season, which moves in the last ulp when a later-season row changes; the decay itself is exact)
    pd.testing.assert_frame_equal(a.loc[mask, cols], b.loc[mask, cols], check_exact=False, rtol=1e-9, atol=1e-9)
    own = (a.index.get_level_values('player_id') == pid) & (a.index.get_level_values('game_id') == first_altered[1])
    pd.testing.assert_frame_equal(a.loc[own, cols], b.loc[own, cols], check_exact=False, rtol=1e-9, atol=1e-9)        # the altered row's own features
