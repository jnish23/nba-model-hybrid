"""
Noon-to-latest injury-report status downgrades (E9). A player listed Probable at the noon report and Out in the report
published shortly before tip is news the noon-snapshot features (inj_*) cannot contain. This module quantifies it per
(gameId, teamId):

    inj_downgrade_lost  sum over rotation players of value * max(0, P(miss | latest status) - P(miss | noon status))
    inj_downgrade_n     rotation players whose P(miss) rose by at least 0.25
    inj_upgrade_lost    the same for improvements (Out -> Questionable, ...)
    inj_late_newer      1 if a later report than the noon one exists for the game

It uses only reports published at or before the late cutoff (tip - 30 min, or `as_of` live) and the same point-in-time
candidates as the production features; the realised outcome (`played`) is never used. It is a flag for live logging - it
is NOT a model input and no backtest result is claimed for it.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from models import injury_report_features as irf

DOWNGRADE_STEP = 0.25


def _p_miss(rows, play_rates):
    p_play = play_rates['mean'].to_dict()
    r = rows.dropna(subset=['personId', 'status']).drop_duplicates(['gameId', 'teamId', 'personId']).copy()
    r['p_miss'] = 1 - r['status'].map(p_play).fillna(1.0)
    return r[['gameId', 'teamId', 'personId', 'p_miss']]


def compute_downgrades(rows_early, rows_late, cand, play_rates, late_newer=None):
    """Pure function over point-in-time inputs. rows_*: report rows of the chosen snapshot per game; cand: candidate players."""
    e = _p_miss(rows_early, play_rates).rename(columns={'p_miss': 'p_e'})
    l = _p_miss(rows_late, play_rates).rename(columns={'p_miss': 'p_l'})
    m = cand[cand['rotation']][['gameId', 'teamId', 'personId', 'value']]
    m = m.merge(e, on=['gameId', 'teamId', 'personId'], how='left').merge(l, on=['gameId', 'teamId', 'personId'], how='left')
    m[['p_e', 'p_l']] = m[['p_e', 'p_l']].fillna(0.0)               # not listed = expected to play
    delta = m['p_l'] - m['p_e']
    m['down'] = m['value'] * delta.clip(lower=0)
    m['up'] = m['value'] * (-delta).clip(lower=0)
    m['down_n'] = (delta >= DOWNGRADE_STEP).astype(int)
    f = m.groupby(['gameId', 'teamId']).agg(inj_downgrade_lost=('down', 'sum'), inj_downgrade_n=('down_n', 'sum'),
                                            inj_upgrade_lost=('up', 'sum')).reset_index()
    if late_newer is not None:
        f = f.merge(late_newer, on='gameId', how='left')
        f['inj_late_newer'] = f['inj_late_newer'].fillna(0).astype(int)
    return f


def build_downgrade_features(db_path, seasons, early='game_day_noon', late='tip_minus_30m', as_of=None, upcoming=None, reports=None):
    reports = irf.parse_all_reports(report_dir=Path('nonexistent')) if reports is None else reports
    if as_of is None:
        on_grid = {f: irf.report_slot(f) in irf.GRID_SLOTS for f in reports['source_file'].unique()}
        reports = reports[reports['source_file'].map(on_grid).astype(bool)]
    linked = irf.link_reports(reports, db_path, upcoming=upcoming)
    schedule = irf.load_schedule(db_path, upcoming=upcoming)
    from models.availability_features import build_player_game_candidates
    with sqlite3.connect(db_path) as conn:
        db_seasons = pd.read_sql("SELECT DISTINCT seasonYear FROM team_game_full", conn)['seasonYear'].tolist()
    history = sorted(s for s in set(db_seasons) | set(seasons) if irf.HISTORY_START_SEASON <= s <= max(seasons))
    cand, _ = build_player_game_candidates(db_path, history, upcoming=upcoming)
    cand = cand[cand['seasonYear'].isin(seasons)]

    def cutoff(name):
        base = irf.SNAPSHOT_CUTOFFS[name]
        return base if as_of is None else (lambda g: base(g).clip(upper=pd.Timestamp(as_of)))
    rows_e, cov_e = irf.snapshot(linked, cutoff(early), schedule)
    rows_l, cov_l = irf.snapshot(linked, cutoff(late), schedule)
    newer = cov_l[['gameId', 'report_ts']].merge(cov_e[['gameId', 'report_ts']], on='gameId', how='left', suffixes=('_l', '_e')).drop_duplicates('gameId')
    newer['inj_late_newer'] = (newer['report_ts_l'] > newer['report_ts_e']).astype(int)
    rate_table = irf.load_status_play_rates(db_path, reports)
    out = []
    for season in seasons:
        rates = irf.rates_for_season(rate_table, season)
        c = cand[cand['seasonYear'] == season]
        ids = set(c['gameId'])
        out.append(compute_downgrades(rows_e[rows_e['gameId'].isin(ids)], rows_l[rows_l['gameId'].isin(ids)], c, rates,
                                      late_newer=newer[['gameId', 'inj_late_newer']]))
    f = pd.concat(out, ignore_index=True)
    f['teamId'] = f['teamId'].astype('int64')
    return f, rows_e, rows_l
