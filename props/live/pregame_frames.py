"""
Production live frames under a pre-game teammate rule ('v2nq', 'pregame_all'; reports/props/audit_2026-11/A2_live_plan.md, step P1).

One function, `build_pregame_date`, serves the signal job (props.live.signal.build_frames) and the parity script (scripts/props_d1_parity.py), so the parity run exercises the production code.

Members (the set the av_* / opp_av_* aggregates run over) are exactly the rule's set (props.live.roster.rule_member_teams, which delegates to props.features.pregame_roster):
the team's previous-game roster plus the usable report's players. Quotes never change it. The rule's set needs the usable report of EVERY game of the date at the information time,
because a player a report places on another team leaves this team's set; `date_games` therefore carries all games of the date (schedule), not only the games with a due slot.

A quoted player always gets a target row. A quoted player who is not a member gets a base row (the live history path, as a placeholder) and an at-T row whose av_* / opp_av_* are
aggregated over the members of his team with no self-subtraction: this is what the training builder does for a box-score target who is not a member
(props.features.pregame_roster.availability_over_members, own contribution only when the target is a member of his team), so no retraining is needed.
His team: the team the usable report of his game names, else his latest listed team this season before the date when that team plays in the game, else he is skipped and the reason is recorded.

Each view has its own information time (`asof_minutes`: which reports are usable) and nominal slot (`slot`: labels the rows and picks the play-rate table), as in props.live.at_t.live_at_t.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from props.features import injury_at_t, pregame_roster
from props.live import at_t as A, roster
from props.live.base import build_live_base

SKIP_REASONS = ('no_team', 'report_team_not_in_game', 'last_team_not_in_game', 'member_of_other_game', 'no_base_row', 'team_conflict')
GAME_COLS = ['game_id', 'game_date', 'season', 'home_team_id', 'away_team_id']


@dataclass(frozen=True)
class PregameView:
    """One game at one decision slot. quoted: the player ids with a consensus quote for the game at this slot."""
    game_id: str
    slot: int
    asof_minutes: float
    quoted: Tuple[int, ...] = ()


def date_game_table(games, date, date_games=None):
    """The games of `date` for the rule's date-wide placement: `games` (the games with views) plus `date_games` (e.g. the schedule's games of the date), one row per game."""
    date = str(date)[:10]
    parts = [games.assign(game_date=date)[GAME_COLS]]
    if date_games is not None and len(date_games):
        parts.append(date_games.assign(game_date=date, season=games['season'].iloc[0])[GAME_COLS])
    return pd.concat(parts, ignore_index=True).astype({'game_id': str}).drop_duplicates('game_id').reset_index(drop=True)


def _usable(reports, tips, asof):
    """The report rows usable at information time `asof`, the chosen report of each game, and the per-game table (props.live.at_t.live_at_t does the same on a report table already cut)."""
    rep = A.asof_reports(reports, tips, asof)
    assert (pd.to_datetime(rep['report_ts']) <= rep['gameId'].map(tips) - pd.Timedelta(minutes=asof + injury_at_t.PUBLISH_LAG_MIN)).all(), 'a report stamped after T minus the publish lag is in use'
    chosen, games_t = injury_at_t.latest_report_rows(rep, tips, asof)
    return rep, chosen, games_t


def _resolve_team(pid, report_team, last_team, teams):
    """(team_id, None) or (None, reason): the report's team, else the latest listed team this season, as props.live.roster.resolve_extra_players."""
    if pd.notna(report_team):
        t = int(report_team)
        return (t, None) if t in teams else (None, 'report_team_not_in_game')
    t = last_team.get(pid)
    if t is None:
        return None, 'no_team'
    return (int(t), None) if int(t) in teams else (None, 'last_team_not_in_game')


def build_pregame_date(rule, hist, pos, date, games, views, reports, tips, tables, nba_db, date_games=None):
    """
    The live frames of one date under a pre-game rule. games: game_id, season, home_team_id, away_team_id of the games that have views (game_date is set to `date`).
    views: list of PregameView (the order matters only for a player who is placed on both teams of one game at different times: the later view's team wins, as in the parity script).
    reports: the linked report rows; tips: Series game_id -> naive ET reference tip; tables: {slot: play-rate table}. date_games: every game of the date (see the module docstring); default `games`.
    Returns (base_all, blocks, unresolved, skipped): the live base rows of the date, one at-T block per view (members and quoted non-members of the view's game),
    the players that could not be placed (as build_live_base) and a list of dicts (game_id, slot, player_id, reason) for the quoted players that got no row.
    """
    if rule not in roster.PREGAME_RULES:
        raise ValueError(f'build_pregame_date needs a pre-game rule {roster.PREGAME_RULES}, got {rule!r}')
    date = str(date)[:10]
    games = games.assign(game_date=date).astype({'game_id': str})
    dg = date_game_table(games, date, date_games)
    gids = set(dg['game_id'])
    teams = {r.game_id: {int(r.home_team_id), int(r.away_team_id)} for r in dg.itertuples()}
    h = hist[hist['game_date'] < date]
    last_team = roster.last_team(h, date, str(games['season'].iloc[0]))
    # 1. the usable report of every game of the date and the rule's members, once per information time
    per_asof = {}
    for v in views:
        m = float(v.asof_minutes)
        if m not in per_asof:
            rep, chosen, games_t = _usable(reports, tips, m)
            rp = A.report_players(rep, tips, m)
            rp = rp[rp['game_id'].isin(gids)]
            per_asof[m] = (chosen, games_t, rp, roster.rule_member_teams(rule, dg, hist, date, rp))
    # 2. the rows to build a base for: the members of every view (one placeholder per game and player, the later view's team wins) and the quoted players who are not members
    members_of = {}
    for v in views:
        _, _, _, mem_all = per_asof[float(v.asof_minutes)]
        members_of[(v.game_id, v.slot)] = mem_all[mem_all['game_id'] == v.game_id][pregame_roster.MEMBER_COLS].reset_index(drop=True)
    allm = pd.concat([members_of[(v.game_id, v.slot)] for v in views], ignore_index=True) if views else pd.DataFrame(columns=pregame_roster.MEMBER_COLS)
    allm = allm.drop_duplicates(['game_id', 'player_id'], keep='last')
    rosters = {int(t): set(map(int, grp['player_id'])) for t, grp in allm.groupby('team_id')}
    in_rosters = set(zip(allm['game_id'], allm['player_id'].astype(int)))
    member_elsewhere = {}                                    # player -> games of the date in which the rule makes him a member (a player is on one team on one date)
    for g, p in in_rosters:
        member_elsewhere.setdefault(p, set()).add(g)
    skipped, placed_nm, extra = [], {}, []
    for v in views:
        _, _, rp, _ = per_asof[float(v.asof_minutes)]
        member_ids = set(members_of[(v.game_id, v.slot)]['player_id'].astype(int))
        report_team = rp[rp['game_id'] == v.game_id].set_index('player_id')['team_id']
        for pid in sorted(set(map(int, v.quoted)) - member_ids):
            t, why = _resolve_team(pid, report_team.get(pid, np.nan), last_team, teams[v.game_id])
            if why is None and member_elsewhere.get(pid, {v.game_id}) - {v.game_id}:
                why = 'member_of_other_game'                  # the reports place him in another game of the date: one team per date
            if why is not None:
                skipped.append({'game_id': v.game_id, 'slot': int(v.slot), 'player_id': pid, 'reason': why})
                continue
            placed_nm[(v.game_id, v.slot, pid)] = t
            if (v.game_id, pid) not in in_rosters:
                extra.append((v.game_id, pid, t))
    extra = pd.DataFrame(extra, columns=['game_id', 'player_id', 'team_id']).drop_duplicates(['game_id', 'player_id']).astype({'game_id': str, 'player_id': 'int64', 'team_id': float})
    base_all, unresolved = build_live_base(hist, pos, games, extra, date, nba_db, rosters=rosters)
    # 3. the at-T block of every view: targets = members + quoted non-members, teammates = members
    blocks = []
    for v in views:
        chosen, games_t, _, _ = per_asof[float(v.asof_minutes)]
        mem = members_of[(v.game_id, v.slot)]
        base_m = base_all.drop(columns=['team_id', 'opp_team_id']).merge(mem, on=['game_id', 'player_id'], how='inner', validate='one_to_one')
        assert len(base_m) == len(mem), 'a member has no live base row'
        nm = [(pid, t) for (g, s, pid), t in placed_nm.items() if g == v.game_id and s == v.slot]
        keep = []
        for pid, t in nm:
            row = base_all[(base_all['game_id'] == v.game_id) & (base_all['player_id'] == pid)]
            why = 'no_base_row' if row.empty else ('team_conflict' if int(row['team_id'].iloc[0]) != t else None)
            if why is None:
                keep.append(row)
            else:
                skipped.append({'game_id': v.game_id, 'slot': int(v.slot), 'player_id': pid, 'reason': why})
        base_t = pd.concat([base_m] + keep, ignore_index=True) if keep else base_m
        member_status = injury_at_t.status_for_rows(base_m[['game_id', 'player_id', 'team_id']], chosen, games_t)
        status = injury_at_t.status_for_rows(base_t[['game_id', 'player_id', 'team_id']], chosen, games_t)
        block = pregame_roster.at_t_members_block(base_t, status, tables[v.slot], base_m[pregame_roster.MEMBER_COLS], member_status, base_m)
        block.insert(2, 'minutes_before', v.slot)
        blocks.append(block)
    return base_all, blocks, unresolved, skipped


def skipped_by_slot(skipped):
    """{(game_id, slot): {player_id: reason}} from the `skipped` list (the form frame.attrs['skipped_players'] carries to props.live.signal.evaluate_item)."""
    out = {}
    for r in skipped:
        out.setdefault((r['game_id'], r['slot']), {})[int(r['player_id'])] = r['reason']
    return out
