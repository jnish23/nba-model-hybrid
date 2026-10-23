"""
Team and opponent context for games that are not in the box-score tables yet: team form from the team builder's latest-features entry point (checked equal to the batch
values), travel from the travel builder's live entry point, zone profiles from the same code as training with placeholder rows, then the shared assembly (props.features.context).
"""
import logging

import pandas as pd

from core.travel_builder import TravelFeatureBuilder
from props.features import context as ctx_mod
from team.team_features_v2 import TeamFeatureBuilder


def live_team_context(db_path, games):
    """
    games: one date's games with game_id, game_date ('YYYY-MM-DD'), season, home_team_id, away_team_id. Returns the same table as props.features.context.team_context for
    (game_id, team_id) of these games, built only from history before game_date.
    """
    games = games.reset_index(drop=True)
    dates = games['game_date'].astype(str).str[:10].unique()
    if len(dates) != 1:
        raise ValueError(f'live_team_context expects one date, got {list(dates)}')
    date = dates[0]
    game_ids = {}
    for side in ('home', 'away'):
        game_ids.update({int(t): g for t, g in zip(games[f'{side}_team_id'], games['game_id'])})
    tf_map = TeamFeatureBuilder(str(db_path)).get_latest_team_features(date, game_ids=game_ids)
    rows, extra = [], []
    for g in games.itertuples(index=False):
        for tid in (int(g.home_team_id), int(g.away_team_id)):
            if tid not in tf_map:
                logging.error(f'no team-form features for team {tid} (game {g.game_id}); its context will be missing')
                continue
            rows.append({**tf_map[tid], 'game_id': g.game_id, 'team_id': tid})
            extra.append({'game_id': g.game_id, 'team_id': tid, 'game_date': date, 'season': g.season})
    tf = pd.DataFrame(rows)
    tg = pd.DataFrame({'gameId': games['game_id'].to_numpy(), 'gameDate': date, 'teamId_home': games['home_team_id'].astype('int64').to_numpy(),
                       'teamId_away': games['away_team_id'].astype('int64').to_numpy()})
    tr = TravelFeatureBuilder(str(db_path)).get_travel_features(tg).rename(columns={'gameId': 'game_id', 'teamId': 'team_id'})
    tr['game_id'] = tr['game_id'].astype(str).str.zfill(10)
    return ctx_mod.assemble_context(db_path, tf, tr, zone_extra=pd.DataFrame(extra))
