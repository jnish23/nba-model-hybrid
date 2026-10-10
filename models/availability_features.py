"""
Player-availability features from realized participation (who actually played).

No point-in-time injury reports exist in the DB before 2026-01 (nba_injury_logs is ESPN snapshots,
2026-01-15 onward), so for backtesting, availability is taken from player_basic_stats: a rotation
player with no row for a game did not play. That is mostly known before tip-off (official "Out"
designations), but NOT necessarily when the opening line is posted — so these features are an
upper bound on what injury information is worth at bet time. Compare against the closing line
(which has nearly all availability news) to judge that.

Per team-game (player values use pre-game ratings only):
  value_i            = (e_minutes_i / 48) * (bpm_i - REPLACEMENT_BPM)   ~ points per 100 poss. contributed
  avail_lost         sum of value over rotation players who did not play
  avail_lost_new     ...of those, players who DID play the team's previous game (fresh news)
  avail_lost_old     ...of those, players already out last game (long-term; mostly priced in)
  avail_returning    sum of value over rotation players back after missing >= 1 team game
  n_out / n_out_new  counts

Rotation = played for this team within its last ROTATION_LOOKBACK games, expected minutes >=
MIN_ROTATION_MINUTES, and has not appeared for another team since (trades).
"""
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from models.bpm_builder import BPMBuilder

REPLACEMENT_BPM = -2.0
ROTATION_LOOKBACK = 10      # team games
RETURN_LOOKBACK = 25        # team games: a player back after a longer layoff still counts as returning
MIN_ROTATION_MINUTES = 10.0


def _team_schedule(conn, seasons):
    q = f"""SELECT DISTINCT gameId, teamId, gameDate, seasonYear FROM team_game_full
            WHERE seasonYear IN ({','.join(repr(s) for s in seasons)})"""
    sched = pd.read_sql(q, conn)
    sched['gameDate'] = pd.to_datetime(sched['gameDate'])
    sched = sched.sort_values(['teamId', 'gameDate', 'gameId'])
    sched['k'] = sched.groupby(['teamId', 'seasonYear']).cumcount()
    return sched


def build_player_game_candidates(db_path, seasons, upcoming=None):
    """
    One row per (team game, player who appeared for that team earlier in the season), with the
    player's pre-game value, games since his last appearance for the team (`gap`), whether he
    played this game, and rotation flags. Traded-away players are removed.

    `upcoming`: optional DataFrame (gameId, teamId, gameDate, seasonYear) of games not yet in
    team_game_full (live inference). They are appended to the schedule, so they go through exactly
    the same logic as historical games (`played` is False for them; it is not used by the
    report-based features).
    """
    bb = BPMBuilder(db_path)
    bb.load_data(seasons)
    apps = bb.calculate_rolling_bpm()[['gameId', 'gameDate', 'personId', 'teamId', 'bpm', 'e_minutes']]
    apps = apps.dropna(subset=['bpm', 'e_minutes'])
    apps['gameDate'] = pd.to_datetime(apps['gameDate'])

    with sqlite3.connect(db_path) as conn:
        sched = _team_schedule(conn, seasons)
    if upcoming is not None and len(upcoming):
        up = upcoming[['gameId', 'teamId', 'gameDate', 'seasonYear']].copy()
        up['gameDate'] = pd.to_datetime(up['gameDate'])
        up = up[~up['gameId'].isin(sched['gameId'])]
        sched = pd.concat([sched.drop(columns='k'), up], ignore_index=True)
        sched = sched.sort_values(['teamId', 'gameDate', 'gameId'])
        sched['k'] = sched.groupby(['teamId', 'seasonYear']).cumcount()

    apps = apps.merge(sched[['gameId', 'teamId', 'k', 'seasonYear']], on=['gameId', 'teamId'], how='inner')
    apps['value'] = (apps['e_minutes'] / 48.0) * (apps['bpm'] - REPLACEMENT_BPM)

    # Candidates: every player who appeared for the team that season x every team game
    roster = apps[['teamId', 'seasonYear', 'personId']].drop_duplicates()
    cand = sched.merge(roster, on=['teamId', 'seasonYear'])

    # Last appearance for THIS team strictly before game k
    hist = apps[['teamId', 'personId', 'k', 'e_minutes', 'value', 'seasonYear']].rename(columns={'k': 'last_k'})
    cand = cand.sort_values('k')
    hist = hist.sort_values('last_k')
    cand = pd.merge_asof(cand, hist, left_on='k', right_on='last_k', by=['teamId', 'personId', 'seasonYear'],
                         allow_exact_matches=False, direction='backward')
    cand = cand.dropna(subset=['last_k'])

    # Played this game?
    played = apps[['gameId', 'teamId', 'personId']].assign(played=True)
    cand = cand.merge(played, on=['gameId', 'teamId', 'personId'], how='left')
    cand['played'] = cand['played'].notna()

    # Traded away: most recent appearance anywhere before this date was for a different team
    last_any = apps[['personId', 'gameDate', 'teamId']].rename(columns={'teamId': 'last_team', 'gameDate': 'last_date'})
    cand = cand.sort_values('gameDate')
    last_any = last_any.sort_values('last_date')
    cand = pd.merge_asof(cand, last_any, left_on='gameDate', right_on='last_date', by='personId',
                         allow_exact_matches=False, direction='backward')
    cand = cand[cand['last_team'] == cand['teamId']]

    cand['gap'] = cand['k'] - cand['last_k']  # 1 = played the team's previous game
    cand['rotation'] = (cand['e_minutes'] >= MIN_ROTATION_MINUTES) & (cand['gap'] <= RETURN_LOOKBACK)
    return cand.drop(columns=['last_date', 'last_team']), sched


def build_availability_features(db_path, seasons):
    """Return one row per (gameId, teamId) with realized-availability features."""
    cand, sched = build_player_game_candidates(db_path, seasons)
    gap = cand['gap']
    rotation = cand['e_minutes'] >= MIN_ROTATION_MINUTES
    out = rotation & ~cand['played'] & (gap <= ROTATION_LOOKBACK)
    out_new = out & (gap == 1)
    returning = rotation & cand['played'] & (gap >= 2) & (gap <= RETURN_LOOKBACK)

    cand['lost'] = np.where(out, cand['value'], 0.0)
    cand['lost_new'] = np.where(out_new, cand['value'], 0.0)
    cand['ret'] = np.where(returning, cand['value'], 0.0)
    cand['is_out'] = out
    cand['is_out_new'] = out_new

    feats = cand.groupby(['gameId', 'teamId']).agg(
        avail_lost=('lost', 'sum'), avail_lost_new=('lost_new', 'sum'), avail_returning=('ret', 'sum'),
        n_out=('is_out', 'sum'), n_out_new=('is_out_new', 'sum')).reset_index()
    feats['avail_lost_old'] = feats['avail_lost'] - feats['avail_lost_new']

    feats = sched[['gameId', 'teamId', 'gameDate', 'seasonYear', 'k']].merge(feats, on=['gameId', 'teamId'], how='left')
    fill = ['avail_lost', 'avail_lost_new', 'avail_lost_old', 'avail_returning', 'n_out', 'n_out_new']
    feats[fill] = feats[fill].fillna(0.0)
    logging.info(f"Availability features: {len(feats)} team-games, "
                 f"{(feats['n_out_new'] > 0).mean():.1%} with a fresh absence")
    return feats


def to_game_level(feats, games):
    """Attach home/away availability features and home-minus-away diffs to a games frame
    with gameId, teamId_home, teamId_away."""
    cols = ['avail_lost', 'avail_lost_new', 'avail_lost_old', 'avail_returning', 'n_out', 'n_out_new']
    f = feats[['gameId', 'teamId'] + cols]
    out = games.merge(f.rename(columns={'teamId': 'teamId_home', **{c: f'{c}_home' for c in cols}}),
                      on=['gameId', 'teamId_home'], how='left')
    out = out.merge(f.rename(columns={'teamId': 'teamId_away', **{c: f'{c}_away' for c in cols}}),
                    on=['gameId', 'teamId_away'], how='left')
    for c in cols:
        out[f'diff_{c}'] = out[f'{c}_home'] - out[f'{c}_away']
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    feats = build_availability_features(Path('data/nba_stats.db'), seasons)
    out_path = Path('data/processed/availability_features.csv')
    feats.to_csv(out_path, index=False)
    print(feats.describe().T.round(3).to_string())
    print(f"Saved {len(feats)} rows to {out_path}")
