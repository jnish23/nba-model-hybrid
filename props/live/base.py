"""
The T-independent feature block for upcoming games, from the training functions: history features on the history before the date plus placeholder roster rows, then the
team and opponent context from props.live.context. Identity with the stored training rows is what the parity test (reports/props/d1_parity_preregistration.md) checks.
"""
import pandas as pd

from props.features import context, history
from props.features.rows import FIRST_DATASET_SEASON
from props.live import roster
from props.live.context import live_team_context


def build_live_base(hist, pos_by_player, games, extra_players, date, db_path, lookback=roster.LOOKBACK_GAMES, rosters=None):
    """
    hist: the box-score history (rows.load_history); only rows before `date` are used (filtered here, explicitly). games: game_id, game_date, season, home_team_id, away_team_id of
    the games of `date`. rosters: {team: players} to build placeholders for (default: the registered look-back rosters). extra_players: DataFrame (game_id, player_id) from injury reports and quotes. Returns (base rows of the live roster, unresolved players).
    """
    date = str(date)[:10]
    h = hist[hist['game_date'] < date]
    if rosters is None:
        rosters = roster.current_team_only(roster.recent_rosters(h, date, lookback), roster.last_team(h, date))
    placed, unresolved = roster.resolve_extra_players(extra_players, games, h, date)
    live = roster.placeholder_rows(h.columns, games, rosters, placed)
    full = pd.concat([h, live], ignore_index=True).sort_values(['player_id', 'game_date', 'game_id']).reset_index(drop=True)
    base = history.add_history_features(full, pos_by_player, FIRST_DATASET_SEASON)
    base = base.merge(live[['game_id', 'player_id']], on=['game_id', 'player_id'], how='inner', validate='one_to_one')
    base = context.add_context(base, live_team_context(db_path, games))
    return base, unresolved
