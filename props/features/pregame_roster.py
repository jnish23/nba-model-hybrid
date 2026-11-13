"""
Pre-game teammate sets for the teammate-availability features (experiment 6a, reports/props/audit_2026-11/WP6_WP7_experiment_plan.md, section B).

The training rows of a game are the players listed in its box score, which is known only after the game. The target rows stay as they are (labels need
them); only the SET of teammates over which av_* and opp_av_* are aggregated changes. Memberships, for team X in game G on date D at decision time T:

  box          the players listed in G's box score (today's training behaviour; injury_at_t.at_t_block over the box-score rows).
  v2nq         players listed in X's previous game of the same season before D whose latest listed team before D (same season) is X, minus the players the
               usable report of G lists Out, plus the report's players with any other status (live rule v2 without its quote component).
  pregame_all  the same previous-game roster plus EVERY report player, Out included (an Out member enters with p_play about 0, so av_n_out > 0 is visible).

Shared definitions (one implementation for the training builder and, through props/live/roster.py, the live builder):
  * Usable report at T: injury_at_t.latest_report_rows (stamped at least PUBLISH_LAG_MIN before T, the latest such report of the game). A player named
    twice keeps his most severe status (as at_t.report_players).
  * A report player is placed on the team the report names; a player with no team on the report goes to his latest listed team this season before D when
    that team plays in G; otherwise he is dropped (roster.resolve_extra_players).
  * A previous-game member whom the usable reports of any game on D place on another team is removed from X (roster.placeholder_rows: a player named for a
    team is on that team on this date only).
  * First game of a season: no previous game, so the members are the report players only. No usable report at T: the members are the previous-game roster.
  * Weights (WEIGHTS of injury_at_t) and m_bucket of a member come from his pre-game history only: from his own base row when he is listed in G (or in another
    game on D), otherwise from a placeholder row run through history.add_history_features (placeholder_weights).
  * For every target row the aggregates are taken over the members of the target's team; the target's own contribution is subtracted only when he is a member
    of that team. The opponent aggregates are over the opponent's members.
"""
import numpy as np
import pandas as pd

from props.features import history, injury_at_t
from props.features.rows import FIRST_DATASET_SEASON

MEMBERSHIPS = ('box', 'v2nq', 'pregame_all')
PREGAME_RULES = ('v2nq', 'pregame_all')
WEIGHT_COLS = [c for c in injury_at_t.AT_T_FEAT_COLS if c not in ('game_id', 'player_id')]
MEMBER_COLS = ['game_id', 'player_id', 'team_id', 'opp_team_id']
INJ_COLS = ['game_id', 'player_id', 'inj_status', 'inj_status_code', 'inj_p_play', 'inj_reason_class', 'inj_report_age_min', 'inj_report_missing']


def _check_rule(rule):
    if rule not in PREGAME_RULES:
        raise ValueError(f'unknown pre-game membership {rule!r}; expected one of {PREGAME_RULES}')


def game_table(rows):
    """One row per game (game_id, game_date, season, home_team_id, away_team_id) from roster/base rows with team_id, opp_team_id, is_home."""
    r = rows[['game_id', 'team_id', 'opp_team_id', 'is_home', 'game_date', 'season']].drop_duplicates(['game_id'])
    home = np.where(r['is_home'].astype(int) == 1, r['team_id'], r['opp_team_id']).astype('int64')
    away = np.where(r['is_home'].astype(int) == 1, r['opp_team_id'], r['team_id']).astype('int64')
    return pd.DataFrame({'game_id': r['game_id'].astype(str).to_numpy(), 'game_date': r['game_date'].astype(str).str[:10].to_numpy(),
                         'season': r['season'].astype(str).to_numpy(), 'home_team_id': home, 'away_team_id': away}).reset_index(drop=True)


def _team_sides(games):
    g = games[['game_id', 'game_date', 'season', 'home_team_id', 'away_team_id']].astype({'game_id': str})
    home = g.rename(columns={'home_team_id': 'team_id', 'away_team_id': 'opp_team_id'})
    away = g.rename(columns={'away_team_id': 'team_id', 'home_team_id': 'opp_team_id'})
    out = pd.concat([home, away], ignore_index=True)
    out['game_date'] = out['game_date'].astype(str).str[:10]
    return out.astype({'team_id': 'int64', 'opp_team_id': 'int64'})


def latest_team_before(hist, keys):
    """
    keys: DataFrame with player_id, game_date, season. Returns keys plus `last_team`: the team of the player's most recent listed game of `season` strictly
    before game_date (ordered by date, then game id, as roster.last_team), NaN if none. Vectorised roster.last_team(hist, date, season).
    """
    right = hist[['player_id', 'season', 'game_date', 'game_id', 'team_id']].copy()
    right['_d'] = pd.to_datetime(right['game_date'].astype(str).str[:10])
    right = right.sort_values(['_d', 'game_id'], kind='mergesort').rename(columns={'team_id': 'last_team'})[['player_id', 'season', '_d', 'last_team']]
    right = right.astype({'player_id': 'int64', 'season': str})
    left = keys.reset_index(drop=True).copy()
    left['_i'] = np.arange(len(left))
    left['_d'] = pd.to_datetime(left['game_date'].astype(str).str[:10])
    left = left.astype({'player_id': 'int64', 'season': str}).sort_values('_d', kind='mergesort')
    m = pd.merge_asof(left, right, on='_d', by=['player_id', 'season'], allow_exact_matches=False, direction='backward')
    return m.sort_values('_i').drop(columns=['_i', '_d']).reset_index(drop=True)


def previous_game_members(hist, games=None):
    """
    The previous-game roster of every team-game (vectorised roster.previous_game_rosters + current_team_only): players listed in the team's most recent game
    of the same season strictly before the game's date, kept only if their latest listed team before that date (same season) is the team.
    games: the games to compute it for (game_table layout); default every game of `hist`. Returns game_id, team_id, player_id.
    """
    games = game_table(hist) if games is None else games
    sides = _team_sides(games)
    tg = hist[['game_id', 'team_id', 'game_date', 'season']].drop_duplicates(['game_id', 'team_id']).astype({'team_id': 'int64', 'season': str})
    tg = tg.assign(_d=pd.to_datetime(tg['game_date'].astype(str).str[:10])).sort_values(['_d', 'game_id'], kind='mergesort')
    tg = tg.rename(columns={'game_id': 'prev_game_id'})[['team_id', 'season', '_d', 'prev_game_id']]
    s = sides.assign(_i=np.arange(len(sides)), _d=pd.to_datetime(sides['game_date'])).astype({'season': str}).sort_values('_d', kind='mergesort')
    s = pd.merge_asof(s, tg, on='_d', by=['team_id', 'season'], allow_exact_matches=False, direction='backward').dropna(subset=['prev_game_id'])
    listed = hist[['game_id', 'team_id', 'player_id']].astype({'team_id': 'int64', 'player_id': 'int64'}).rename(columns={'game_id': 'prev_game_id'})
    cand = s[['game_id', 'team_id', 'game_date', 'season', 'prev_game_id']].merge(listed, on=['prev_game_id', 'team_id'], how='inner')
    cand = latest_team_before(hist, cand)
    cand = cand[cand['last_team'] == cand['team_id']]
    return cand[['game_id', 'team_id', 'player_id']].astype({'game_id': str}).sort_values(['game_id', 'team_id', 'player_id']).reset_index(drop=True)


def prev_frame_from_rosters(rosters, games):
    """{team_id: set(player)} (roster.previous_game_rosters for one date) -> the previous_game_members layout for those games."""
    recs = [(str(g.game_id), int(t), int(p)) for g in games.itertuples() for t in (int(g.home_team_id), int(g.away_team_id)) for p in rosters.get(t, ())]
    return pd.DataFrame(recs, columns=['game_id', 'team_id', 'player_id']).astype({'team_id': 'int64', 'player_id': 'int64'})


def report_players(chosen):
    """(game_id, player_id, team_id, status) of every player named on the chosen report of each game (latest_report_rows output); the most severe status if
    named twice. Same as props.live.at_t.report_players, applied to already-chosen rows."""
    c = chosen.dropna(subset=['personId'])
    sev = c['status'].map(injury_at_t.STATUS_CODE).fillna(0)
    out = pd.DataFrame({'game_id': c['gameId'].astype(str), 'player_id': c['personId'].astype('int64'), 'team_id': pd.to_numeric(c['teamId'], errors='coerce'),
                        'status': c['status'], '_sev': sev})
    return out.sort_values('_sev').drop_duplicates(['game_id', 'player_id'], keep='last').drop(columns='_sev').reset_index(drop=True)


def place_report_players(report, games, hist):
    """
    Report players placed on one of the game's teams: the report's team, else the player's latest listed team this season before the date (roster.resolve_extra_players).
    Returns (placed: game_id, player_id, team_id, status, game_date; n_unresolved).
    """
    g = games[['game_id', 'game_date', 'season', 'home_team_id', 'away_team_id']].astype({'game_id': str})
    r = report[['game_id', 'player_id', 'team_id', 'status']].astype({'game_id': str}).merge(g, on='game_id', how='inner')
    if r.empty:
        return r.assign(team_id=pd.Series(dtype='int64'))[['game_id', 'player_id', 'team_id', 'status', 'game_date']], 0
    r = r.drop_duplicates(['game_id', 'player_id'])
    lt = latest_team_before(hist, r[['player_id', 'game_date', 'season']])
    team = r['team_id'].to_numpy(dtype=float)
    team = np.where(np.isnan(team), lt['last_team'].to_numpy(dtype=float), team)
    ok = ~np.isnan(team) & ((team == r['home_team_id'].to_numpy()) | (team == r['away_team_id'].to_numpy()))
    placed = r[ok].assign(team_id=team[ok].astype('int64'))
    return placed[['game_id', 'player_id', 'team_id', 'status', 'game_date']].reset_index(drop=True), int((~ok).sum())


def members_at_t(prev, report, games, hist, rule):
    """
    Member set of every game in `games` at one decision time.
    prev: previous_game_members for these games (game_id, team_id, player_id). report: report_players of the usable report at T for these games (and only
    these: the date-wide placement below uses every game passed). hist: listed rows; only rows strictly before each game's date are used.
    Returns game_id, player_id, team_id, opp_team_id (one row per game and player).
    """
    _check_rule(rule)
    sides = _team_sides(games)[['game_id', 'game_date', 'team_id', 'opp_team_id']]
    placed, _ = place_report_players(report, games, hist)
    p = prev[['game_id', 'team_id', 'player_id']].astype({'game_id': str, 'team_id': 'int64', 'player_id': 'int64'}).merge(sides, on=['game_id', 'team_id'], how='inner')
    # a player the date's reports place on a team is on that team only, on this date
    pt = placed[['game_date', 'player_id', 'team_id']].drop_duplicates().rename(columns={'team_id': '_placed_team'})
    chk = p.merge(pt, on=['game_date', 'player_id'], how='left')
    chk['_has'] = chk['_placed_team'].notna()
    chk['_here'] = chk['_placed_team'] == chk['team_id']
    agg = chk.groupby(['game_id', 'team_id', 'player_id'], sort=False).agg(_has=('_has', 'any'), _here=('_here', 'any')).reset_index()
    keep = agg[~agg['_has'] | agg['_here']][['game_id', 'team_id', 'player_id']]            # kept: not placed anywhere today, or placed on this team
    p = p.merge(keep, on=['game_id', 'team_id', 'player_id'], how='inner')
    rep = report[['game_id', 'player_id', 'status']].astype({'game_id': str, 'player_id': 'int64'})
    if rule == 'v2nq':
        out_pairs = rep[rep['status'] == 'Out'][['game_id', 'player_id']].drop_duplicates()
        p = p.merge(out_pairs.assign(_out=1), on=['game_id', 'player_id'], how='left')
        p = p[p['_out'].isna()].drop(columns='_out')
        add = placed[placed['status'] != 'Out']
    else:
        add = placed
    add = add[['game_id', 'team_id', 'player_id']].astype({'game_id': str, 'team_id': 'int64', 'player_id': 'int64'})
    # a report-placed player belongs to the team he is placed on: drop his previous-game entry in this game when it is the other team
    base = p[['game_id', 'team_id', 'player_id']].merge(add[['game_id', 'player_id']].assign(_pl=1), on=['game_id', 'player_id'], how='left')
    base = base[base['_pl'].isna()].drop(columns='_pl')
    m = pd.concat([base, add], ignore_index=True).drop_duplicates(['game_id', 'player_id'])
    m = m.merge(sides[['game_id', 'team_id', 'opp_team_id']], on=['game_id', 'team_id'], how='inner', validate='many_to_one')
    return m[MEMBER_COLS].sort_values(['game_id', 'team_id', 'player_id']).reset_index(drop=True)


def placeholder_weights(hist, members, pos_by_player, first_dataset_season=FIRST_DATASET_SEASON):
    """
    Pre-game weight columns and m_bucket for members with no listed row in their game. One placeholder row (not played, no stats) per such (game, player) is
    appended to the history, ordered before any real row of the same player on the same date (so a player listed in another game that day gets exactly the
    state before that day, as live, whose history stops before the date), and history.add_history_features runs once. Only the placeholders' weight columns
    and m_bucket are returned; real rows keep the normal build (the placeholders do change real rows' role_* and rest_* columns, which is why they are not used).
    members: game_id, player_id, team_id, opp_team_id plus game_date, season. Returns game_id, player_id, season, m_bucket + WEIGHT_COLS.
    """
    cols = ['game_id', 'player_id', 'season', 'm_bucket'] + WEIGHT_COLS
    full = with_placeholders(hist, members)
    if full is None:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols}).astype({'game_id': str, 'player_id': 'int64', 'season': str, 'm_bucket': 'int64'})
    is_ph = (full['_ph'] == 0).to_numpy()
    first = min(first_dataset_season, str(full.loc[is_ph, 'season'].min()))
    out = history.add_history_features(full.drop(columns=['_ph', '_k']), pos_by_player, first_dataset_season=first)
    # add_history_features keeps the row order of its input (filtered by season); the placeholders are recovered by key (one per game and player)
    feats = out[['game_id', 'player_id', 'season', 'm_bucket'] + WEIGHT_COLS]
    keys = full.loc[is_ph, ['game_id', 'player_id']].astype({'game_id': str, 'player_id': 'int64'})
    res = keys.merge(feats.astype({'game_id': str, 'player_id': 'int64'}), on=['game_id', 'player_id'], how='left', validate='one_to_one')
    return res[cols].reset_index(drop=True)


def with_placeholders(hist, members):
    """The history plus one placeholder row per member not listed in his game (column _ph = 0; real rows 1), sorted by player, date, placeholders first, game.
    None when every member is listed."""
    listed = hist[['game_id', 'player_id']].astype({'game_id': str, 'player_id': 'int64'})
    need = members.astype({'game_id': str, 'player_id': 'int64'}).merge(listed.assign(_l=1), on=['game_id', 'player_id'], how='left')
    need = need[need['_l'].isna()].drop(columns='_l').drop_duplicates(['game_id', 'player_id'])
    if need.empty:
        return None
    ph = pd.DataFrame({'game_id': need['game_id'].to_numpy(), 'player_id': need['player_id'].to_numpy(), 'team_id': need['team_id'].to_numpy(),
                       'opp_team_id': need['opp_team_id'].to_numpy(), 'game_date': need['game_date'].astype(str).str[:10].to_numpy(),
                       'season': need['season'].astype(str).to_numpy(), 'game_type': need['game_id'].astype(str).str[:3].to_numpy(),
                       'is_home': np.nan, 'lbl_played': 0, 'lbl_started': 0})
    for c in hist.columns:
        if c not in ph.columns:
            ph[c] = np.nan
    ph = ph[list(hist.columns)]
    full = pd.concat([hist.assign(_ph=1), ph.assign(_ph=0)], ignore_index=True)
    full['_k'] = np.arange(len(full))
    return full.sort_values(['player_id', 'game_date', '_ph', 'game_id', '_k'], kind='mergesort').reset_index(drop=True)


def member_features(members, base, placeholders):
    """season, m_bucket and the weight columns of every member: his base row when he is listed in the game, else his placeholder row."""
    keys = members[['game_id', 'player_id']].astype({'game_id': str, 'player_id': 'int64'}).drop_duplicates()
    b = base[['game_id', 'player_id', 'season', 'm_bucket'] + WEIGHT_COLS].astype({'game_id': str, 'player_id': 'int64'})
    listed = keys.merge(b, on=['game_id', 'player_id'], how='inner')
    rest = keys.merge(listed[['game_id', 'player_id']].assign(_l=1), on=['game_id', 'player_id'], how='left')
    rest = rest[rest['_l'].isna()].drop(columns='_l').merge(placeholders.astype({'game_id': str, 'player_id': 'int64'}), on=['game_id', 'player_id'], how='inner')
    out = pd.concat([listed, rest], ignore_index=True)
    assert len(out) == len(keys), f'{len(keys) - len(out)} members have neither a base row nor a placeholder'
    return out


def availability_over_members(targets, members, feat, prefix='av'):
    """
    Generalised injury_at_t.teammate_availability. targets: game_id, player_id, team_id, opp_team_id (the rows that get features). members: game_id, player_id,
    team_id, opp_team_id, inj_p_play, inj_status_code (the teammate set). feat: weight columns of the members (AT_T_FEAT_COLS).
    Own-team aggregates are over the members of the target's team minus the target's own contribution when he is a member of that team; opponent aggregates
    are over the opponent's members. With members == targets this is teammate_availability, float for float.
    """
    x = members[['game_id', 'player_id', 'team_id', 'opp_team_id', 'inj_p_play', 'inj_status_code']].merge(
        feat, on=['game_id', 'player_id'], how='left', suffixes=('', '_f'), validate='many_to_one')
    q = 1.0 - x['inj_p_play'].to_numpy()
    for name, fn in injury_at_t.WEIGHTS.items():
        w = fn(x).fillna(0.0).clip(lower=0).to_numpy()
        x[f'_w_{name}'] = w
        x[f'_qw_{name}'] = q * w
    x['_out'] = (x['inj_status_code'] == injury_at_t.STATUS_CODE['Out']).astype(int)
    x['_q'] = ((x['inj_status_code'] == injury_at_t.STATUS_CODE['Questionable']) | (x['inj_status_code'] == injury_at_t.STATUS_CODE['Doubtful'])).astype(int)
    x['_starter_qw'] = np.where(x['m_min_h15'].fillna(0) >= 24, q * x['m_min_h15'].fillna(0), 0.0)
    keys = ['game_id', 'team_id']
    agg_cols = [c for c in x.columns if c.startswith(('_w_', '_qw_'))] + ['_out', '_q', '_starter_qw']
    tsum = x.groupby(keys)[agg_cols].sum()
    t = targets[['game_id', 'player_id', 'team_id', 'opp_team_id']].reset_index(drop=True)
    team = t[keys].merge(tsum.reset_index(), on=keys, how='left')[agg_cols].fillna(0.0)
    own = t[['game_id', 'player_id', 'team_id']].merge(x[['game_id', 'player_id', 'team_id'] + agg_cols], on=['game_id', 'player_id', 'team_id'], how='left',
                                                        validate='one_to_one')[agg_cols].fillna(0.0)
    res = t.copy()
    for name in injury_at_t.WEIGHTS:
        tot = team[f'_w_{name}']
        miss_excl_self = team[f'_qw_{name}'] - own[f'_qw_{name}']
        denom = tot - own[f'_w_{name}']
        res[f'{prefix}_miss_{name}'] = np.where(denom > 0, miss_excl_self / denom.where(denom > 0), np.nan)
    res[f'{prefix}_n_out'] = (team['_out'] - own['_out']).astype('int64')
    res[f'{prefix}_n_questionable'] = (team['_q'] - own['_q']).astype('int64')
    res[f'{prefix}_miss_starter_minutes'] = team['_starter_qw'] - own['_starter_qw']
    opp_out = pd.DataFrame(index=tsum.index)
    for name in injury_at_t.WEIGHTS:
        opp_out[f'opp_{prefix}_miss_{name}'] = np.where(tsum[f'_w_{name}'] > 0, tsum[f'_qw_{name}'] / tsum[f'_w_{name}'].where(tsum[f'_w_{name}'] > 0), np.nan)
    opp_out[f'opp_{prefix}_n_out'] = tsum['_out']
    opp_out[f'opp_{prefix}_miss_starter_minutes'] = tsum['_starter_qw']
    opp_out = opp_out.reset_index().rename(columns={'team_id': 'opp_team_id'})
    return res.merge(opp_out, on=['game_id', 'opp_team_id'], how='left', validate='many_to_one')


def at_t_members_block(base, status, table, members, member_status, member_feat):
    """
    injury_at_t.at_t_block with a separate teammate set. base/status: the target rows and their report status (as at_t_block). members: game_id, player_id,
    team_id, opp_team_id; member_status: status_for_rows of the members; member_feat: member_features output. Same columns as at_t_block.
    """
    d = injury_at_t.attach_p_play(status, base, table)
    dm = injury_at_t.attach_p_play(member_status, member_feat, table)
    dm = dm.drop(columns=['opp_team_id'], errors='ignore').merge(members[['game_id', 'player_id', 'opp_team_id']], on=['game_id', 'player_id'], validate='one_to_one')
    av = availability_over_members(base[['game_id', 'player_id', 'team_id', 'opp_team_id']], dm, member_feat[injury_at_t.AT_T_FEAT_COLS])
    return d[INJ_COLS].merge(av.drop(columns=['team_id', 'opp_team_id']), on=['game_id', 'player_id'], validate='one_to_one')


def prepare_members(rows, hist, pos_by_player, reports, tips, grid, rule, first_dataset_season=FIRST_DATASET_SEASON):
    """
    Member sets and member features of every game of `rows` (base rows) at every decision time of `grid`. hist: the full listed history (rows.load_history).
    Returns {minutes_before: (members, member_feat)} and a summary dict (counts, for the manifest and the report).
    """
    _check_rule(rule)
    games = game_table(rows)
    gids = set(games['game_id'])
    prev = previous_game_members(hist, games)
    mem, summary = {}, {'rule': rule, 'games': int(len(games))}
    for x in grid:
        chosen, _ = injury_at_t.latest_report_rows(reports, tips, x)
        rp = report_players(chosen)
        rp = rp[rp['game_id'].isin(gids)]
        _, n_unres = place_report_players(rp, games, hist)
        mem[x] = members_at_t(prev, rp, games, hist, rule)
        summary[f'members_T{x}'] = int(len(mem[x]))
        summary[f'unplaced_report_players_T{x}'] = n_unres
    allm = pd.concat(mem.values(), ignore_index=True).drop_duplicates(['game_id', 'player_id']).merge(games[['game_id', 'game_date', 'season']], on='game_id')
    ph = placeholder_weights(hist, allm, pos_by_player, first_dataset_season)
    summary['placeholders'] = int(len(ph))
    out = {}
    for x in grid:
        out[x] = (mem[x], member_features(mem[x], rows, ph))
    return out, summary
