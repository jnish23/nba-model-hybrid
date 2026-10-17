"""Helpers for building model input rows at inference time."""
import logging

import numpy as np
import pandas as pd


class MissingFeaturesError(RuntimeError):
    """A model's required features are mostly absent from the live row (wrong config or broken builder)."""


def align_features(row_df, expected_cols, model_name='model', max_missing_frac=0.10):
    """
    Return row_df[expected_cols]. Features the live builder did not produce are filled with NaN
    (LightGBM's own missing-value path - the same thing training sees for an unavailable stat,
    unlike 0.0, which is a real value for most features) and reported at ERROR level.
    If more than `max_missing_frac` of the model's features are absent the row is not scored at
    all: that means a feature-name drift or a mismatched model/config pair, not a stray gap.
    """
    expected_cols = list(expected_cols)
    missing = [c for c in expected_cols if c not in row_df.columns]
    if missing:
        frac = len(missing) / max(len(expected_cols), 1)
        logging.error(f"{model_name}: {len(missing)}/{len(expected_cols)} model features were NOT built live "
                      f"and are set to NaN: {missing}")
        if frac > max_missing_frac:
            raise MissingFeaturesError(
                f"{model_name}: {len(missing)}/{len(expected_cols)} features missing ({frac:.0%} > "
                f"{max_missing_frac:.0%}); refusing to score. Missing: {missing[:10]}")
    out = row_df.reindex(columns=expected_cols)
    return out.astype(float) if len(out) else out


# ----------------------------------------------------------------------------------------------
# Live row assembly: one row per game, with exactly the columns the training set has.

ODDS_COLS = ('odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_away_spread', 'odds_total')


def build_live_game_frame(db_path, games, injury_features=None, roster_before=None):
    """
    Build the model-input frame for upcoming games.

    games: one row per game with gameId, gameDate (a single date), teamId_home, teamId_away and
        optionally the odds_* model inputs (opening lines; see models/live_odds.py).
    injury_features: (gameId, teamId, inj_*) frame from injury_report_features.build_injury_features;
        None = no report available, which is encoded as in training (inj_missing=1, zeros).
    roster_before: date for the roster features (defaults to the game date).

    Every block is produced by the same code as the training-set build: team form
    (TeamFeatureBuilder), travel (TravelFeatureBuilder), roster BPM (PlayerFeatureBuilder), injury
    features, then GameFeatureBuilder.add_roster_features / add_derived_features.
    """
    # Imported here so that importing this module (e.g. for align_features) stays cheap
    from models.game_features_v2 import GameFeatureBuilder
    from models.injury_report_features import INJ_COLS, add_injury_features
    from models.player_features import PlayerFeatureBuilder
    from models.team_features_v2 import TeamFeatureBuilder
    from models.travel_builder import TravelFeatureBuilder
    import sqlite3

    games = games.reset_index(drop=True).copy()
    dates = pd.to_datetime(games['gameDate']).dt.strftime('%Y-%m-%d').unique()
    if len(dates) != 1:
        raise ValueError(f"build_live_game_frame expects games from a single date, got {list(dates)}")
    date = dates[0]
    games['gameId'] = games['gameId'].astype(str)
    games['teamId_home'] = games['teamId_home'].astype('int64')
    games['teamId_away'] = games['teamId_away'].astype('int64')

    # --- team form, from the game log before `date` (one query for all teams)
    game_ids = {}
    for side in ('home', 'away'):
        game_ids.update(dict(zip(games[f'teamId_{side}'], games['gameId'])))
    team_feats = TeamFeatureBuilder(db_path).get_latest_team_features(date, game_ids=game_ids)

    # --- travel
    travel = TravelFeatureBuilder(db_path).get_travel_features(games.assign(gameDate=date))
    travel = travel.drop(columns=['team']).set_index(['gameId', 'teamId'])

    blocks = {}
    for side in ('home', 'away'):
        records = []
        for g in games.itertuples(index=False):
            tid = getattr(g, f'teamId_{side}')
            rec = dict(team_feats.get(tid, {}))
            if not rec:
                logging.error(f"No team-form features for team {tid} (game {g.gameId}); they will be NaN")
            key = (g.gameId, tid)
            if key in travel.index:
                rec.update(travel.loc[key].to_dict())
            else:
                logging.error(f"No travel features for team {tid} (game {g.gameId}); they will be NaN")
            records.append(rec)
        blocks[side] = pd.DataFrame(records).add_suffix(f'_{side}')
    frame = pd.concat([games, blocks['home'], blocks['away']], axis=1)

    # --- roster BPM, as of the morning of the game
    seasons = _db_seasons(db_path)
    roster = PlayerFeatureBuilder(db_path).next_game_roster_features(seasons, before=roster_before or date)
    p_feats = pd.concat([
        pd.DataFrame({'gameId': frame['gameId'], 'teamId': frame[f'teamId_{side}']}).merge(
            roster.reset_index().rename(columns={'index': 'teamId'}), on='teamId', how='left')
        for side in ('home', 'away')], ignore_index=True)
    if p_feats[list(GameFeatureBuilder.ROSTER_COLS)].isna().any().any():
        logging.error("Roster BPM missing for some teams; those features will be NaN")
    frame = GameFeatureBuilder.add_roster_features(frame, p_feats)

    # --- injury report features (training encodes 'no report' as inj_missing=1 and zeros)
    if injury_features is None or len(injury_features) == 0:
        inj = pd.concat([pd.DataFrame({'gameId': frame['gameId'], 'teamId': frame[f'teamId_{side}']})
                         for side in ('home', 'away')], ignore_index=True)
        for c in INJ_COLS:
            inj[c] = 1 if c == 'inj_missing' else 0.0
    else:
        inj = injury_features.copy()
        inj['gameId'] = inj['gameId'].astype(str)
    frame = add_injury_features(frame, inj)

    frame = GameFeatureBuilder.add_derived_features(frame)
    frame.columns = [str(c).replace(' ', '_') for c in frame.columns]
    return frame


def _db_seasons(db_path):
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        return sorted(pd.read_sql("SELECT DISTINCT seasonYear FROM team_game_full", conn)['seasonYear'])
