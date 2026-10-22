"""
The live roster rule (reports/props/d1_parity_preregistration.md, section 2), fixed before the parity test.

For team X in game G on date D, the live roster is the union of:
  1. players who appeared in X's box score (played or listed) in any of X's last LOOKBACK_GAMES games before D;
  2. players on an injury report for G stamped at least the publish lag before T;
  3. players with a consensus quote for G at T (the traded set; always included).
Each becomes a PLACEHOLDER row in the history schema (no stats, not played) so the training history code produces their pre-game features.
"""
import logging

import numpy as np
import pandas as pd

LOOKBACK_GAMES = 15


def recent_rosters(hist, date, n=LOOKBACK_GAMES):
    """{team_id: set(player_id)} listed in each team's last n games strictly before `date`."""
    h = hist[hist['game_date'] < date]
    last = h[['team_id', 'game_id', 'game_date']].drop_duplicates().sort_values(['team_id', 'game_date', 'game_id']).groupby('team_id').tail(n)
    m = h[['team_id', 'game_id', 'player_id']].merge(last[['team_id', 'game_id']], on=['team_id', 'game_id'], validate='many_to_one')
    return {int(t): set(map(int, g['player_id'])) for t, g in m.groupby('team_id')}


def last_team(hist, date, season=None):
    """player_id -> team_id of his most recent listed game before `date` (within `season` when given: a player's team of last year is not his team now)."""
    h = hist[hist['game_date'] < date]
    if season is not None:
        h = h[h['season'] == season]
    return h.sort_values(['game_date', 'game_id']).groupby('player_id')['team_id'].last().astype(int).to_dict()


def current_team_only(rosters, lt):
    """A player traded inside the look-back window appears in two teams' recent rosters; he belongs only to the team of his most recent game."""
    return {team: {p for p in players if lt.get(p) == team} for team, players in rosters.items()}


def resolve_extra_players(extra, games, hist, date):
    """
    extra: DataFrame (game_id, player_id[, team_id]) of players from the injury reports (with the report's team) and the quotes. Returns ({(game_id): {team_id: set(player)}}, unresolved).
    A player is placed on the team the injury report gives him; otherwise on the team of his most recent listed game THIS season when that is one of the game's teams. A player with
    neither (a new signing or a player returning from last season, whose old team may be the opponent) is returned in `unresolved`, never guessed.
    """
    season = str(games['season'].iloc[0])
    lt = last_team(hist, date, season)
    teams = {r.game_id: (int(r.home_team_id), int(r.away_team_id)) for r in games.itertuples()}
    placed, unresolved = {}, []
    if 'team_id' not in extra.columns:
        extra = extra.assign(team_id=np.nan)
    extra = extra.assign(_has=extra['team_id'].notna()).sort_values('_has', ascending=False)
    for r in extra.drop_duplicates(['game_id', 'player_id']).itertuples():
        pid = int(r.player_id)
        t = int(r.team_id) if pd.notna(r.team_id) else lt.get(pid)
        if r.game_id in teams and t in teams[r.game_id]:
            placed.setdefault(r.game_id, {}).setdefault(t, set()).add(pid)
        else:
            unresolved.append((r.game_id, pid))
    if unresolved:
        logging.info(f'{len(unresolved)} quoted or reported players could not be placed on one of the game teams (last team unknown or different)')
    return placed, unresolved


def placeholder_rows(columns, games, rosters, extra_placed=None):
    """
    One placeholder per (game, team, player). games: game_id, game_date, season, home_team_id, away_team_id. rosters: {team_id: set(player)} from recent_rosters;
    extra_placed: output of resolve_extra_players. `columns` = the history frame's columns, so the rows can be appended to it unchanged.
    """
    extra_placed = extra_placed or {}
    recs = []
    for g in games.itertuples():
        for side, team, opp in (('home', int(g.home_team_id), int(g.away_team_id)), ('away', int(g.away_team_id), int(g.home_team_id))):
            placed_here = extra_placed.get(g.game_id, {})
            placed_elsewhere = set().union(*[v for t, v in placed_here.items() if t != team]) if placed_here else set()
            players = (set(rosters.get(team, set())) - placed_elsewhere) | placed_here.get(team, set())          # a player named for the other team by a report belongs there only
            for p in sorted(players):
                recs.append({'game_id': g.game_id, 'player_id': p, 'team_id': team, 'opp_team_id': opp, 'game_date': str(g.game_date)[:10], 'season': g.season,
                             'game_type': g.game_id[:3], 'is_home': 1 if side == 'home' else 0, 'lbl_played': 0, 'lbl_started': 0})
    df = pd.DataFrame(recs)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[list(columns)]
