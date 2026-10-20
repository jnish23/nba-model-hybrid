"""Injury-value variants for E7 (bench-depth replacement level) and E8 (fresh vs chronic absences).

They plug into models.injury_report_features.build_injury_features by patching `reported_features` (and INJ_COLS) for the
duration of one call, so the production code path and the baseline stay untouched. Inputs of these functions are the same
point-in-time objects the production function receives: the report snapshot rows, the coverage frame and the candidate
frame (cand['played'] is a post-game outcome and must never influence a feature - tests shuffle it).
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from models import injury_report_features as irf
from models.availability_features import REPLACEMENT_BPM

ROOT = Path(__file__).resolve().parent.parent
_POS = {}

BENCH_RANKS = (6, 10)          # healthy rotation players ranked 6th..10th by expected minutes
FRESH_MAX_GAP = 3              # played at least one of the team's last 3 games
CHRONIC_MIN_GAP = 6            # missed >= 5 consecutive team games
DECAY_HALF_LIFE = 5.0          # games; fixed constant (not tuned): e_minutes * 0.5 ** ((gap - 1) / 5)


def position_groups(db_path):
    key = str(db_path)
    if key not in _POS:
        with sqlite3.connect(db_path) as conn:
            info = pd.read_sql("SELECT PERSON_ID AS personId, POSITION FROM player_info", conn)
        first = info['POSITION'].fillna('Forward').str.split('-').str[0].str.strip()
        _POS[key] = dict(zip(info['personId'], first.map({'Guard': 'G', 'Forward': 'F', 'Center': 'C'}).fillna('F')))
    return _POS[key]


def _prepare(rows, cand, play_rates):
    p_play = play_rates['mean'].to_dict()
    r = rows.dropna(subset=['personId', 'status']).copy()
    r['p_miss'] = 1 - r['status'].map(p_play).fillna(1.0)
    r = r.drop_duplicates(['gameId', 'teamId', 'personId'])
    m = cand.merge(r[['gameId', 'teamId', 'personId', 'status', 'p_miss']], on=['gameId', 'teamId', 'personId'], how='left')
    m['reported'] = m['status'].notna()
    m['p_miss'] = m['p_miss'].fillna(0.0)
    m['bpm'] = np.where(m['e_minutes'] > 0, 48.0 * m['value'] / m['e_minutes'] + REPLACEMENT_BPM, np.nan)
    return m


def bench_bpm(m, db_path):
    """Team's own healthy-bench BPM per (gameId, teamId, position group); fallback team bench, then the constant."""
    groups = position_groups(db_path)
    rot = m[m['rotation'] & (m['p_miss'] <= 0.5)].copy()           # healthy: not listed, or listed with a high chance to play
    rot['grp'] = rot['personId'].map(groups).fillna('F')
    rot = rot.sort_values(['gameId', 'teamId', 'e_minutes'], ascending=[True, True, False])
    rot['rank'] = rot.groupby(['gameId', 'teamId']).cumcount() + 1
    bench = rot[(rot['rank'] >= BENCH_RANKS[0]) & (rot['rank'] <= BENCH_RANKS[1])].copy()
    bench['wb'] = bench['e_minutes'] * bench['bpm']
    g = bench.groupby(['gameId', 'teamId', 'grp']).agg(wb=('wb', 'sum'), w=('e_minutes', 'sum')).reset_index()
    g['bench_grp'] = g['wb'] / g['w']
    t = bench.groupby(['gameId', 'teamId']).agg(wb=('wb', 'sum'), w=('e_minutes', 'sum')).reset_index()
    t['bench_team'] = t['wb'] / t['w']
    return g[['gameId', 'teamId', 'grp', 'bench_grp']], t[['gameId', 'teamId', 'bench_team']]


def _finish(m, value_col, extra=None):
    m = m[m['reported'] & m['rotation']].copy()
    m['lost'] = m[value_col] * m['p_miss']
    m['lost_new'] = np.where(m['gap'] == 1, m['lost'], 0.0)
    m['is_out'] = m['status'] == 'Out'
    agg = dict(rep_lost=('lost', 'sum'), rep_lost_new=('lost_new', 'sum'), rep_n_out=('is_out', 'sum'))
    if extra:
        for name, cond_col in extra.items():
            m[name] = np.where(m[cond_col], m['lost'], 0.0) if cond_col in m else 0.0
            agg[name] = (name, 'sum')
    return m.groupby(['gameId', 'teamId']).agg(**agg).reset_index()


def _assemble(feats, coverage, cand, extra_cols=()):
    base = cand[['gameId', 'teamId']].drop_duplicates().merge(coverage, on=['gameId', 'teamId'], how='left')
    base = base.merge(feats, on=['gameId', 'teamId'], how='left')
    base['rep_missing'] = base['report_ts'].isna() | base['not_submitted'].eq(True)
    for c in ('rep_lost', 'rep_lost_new', 'rep_n_out', *extra_cols):
        base[c] = base[c].fillna(0.0)
    return base


def make_e7(db_path):
    def reported_features_e7(rows, coverage, cand, play_rates):
        m = _prepare(rows, cand, play_rates)
        g, t = bench_bpm(m, db_path)
        groups = position_groups(db_path)
        m['grp'] = m['personId'].map(groups).fillna('F')
        m = m.merge(g, on=['gameId', 'teamId', 'grp'], how='left').merge(t, on=['gameId', 'teamId'], how='left')
        repl = m['bench_grp'].fillna(m['bench_team']).fillna(REPLACEMENT_BPM)
        m['value_e7'] = (m['e_minutes'] / 48.0) * np.clip(m['bpm'] - repl, 0.0, None)
        feats = _finish(m, 'value_e7')
        return _assemble(feats, coverage, cand)
    return reported_features_e7


def make_e8(db_path):
    extra_names = ('inj_fresh_lost', 'inj_chronic_lost', 'inj_chronic_lost_dec', 'inj_mid_lost')

    def reported_features_e8(rows, coverage, cand, play_rates):
        m = _prepare(rows, cand, play_rates)
        m = m[m['reported'] & m['rotation']].copy()
        m['lost'] = m['value'] * m['p_miss']
        m['lost_new'] = np.where(m['gap'] == 1, m['lost'], 0.0)
        m['is_out'] = m['status'] == 'Out'
        fresh = m['gap'] <= FRESH_MAX_GAP
        chronic = m['gap'] >= CHRONIC_MIN_GAP
        m['inj_fresh_lost'] = np.where(fresh, m['lost'], 0.0)
        m['inj_chronic_lost'] = np.where(chronic, m['lost'], 0.0)
        m['inj_chronic_lost_dec'] = np.where(chronic, m['lost'] * 0.5 ** ((m['gap'] - 1) / DECAY_HALF_LIFE), 0.0)
        m['inj_mid_lost'] = np.where(~fresh & ~chronic, m['lost'], 0.0)
        feats = m.groupby(['gameId', 'teamId']).agg(
            rep_lost=('lost', 'sum'), rep_lost_new=('lost_new', 'sum'), rep_n_out=('is_out', 'sum'),
            **{c: (c, 'sum') for c in extra_names}).reset_index()
        return _assemble(feats, coverage, cand, extra_names)
    reported_features_e8.extra_cols = list(extra_names)
    return reported_features_e8


@contextmanager
def patched(fn, extra_cols=()):
    with mock.patch.object(irf, 'reported_features', fn), mock.patch.object(irf, 'INJ_COLS', list(irf.INJ_COLS) + list(extra_cols)):
        yield


def build_variant_features(db_path, seasons, kind):
    """Per (gameId, teamId) inj_* features computed with the E7 / E8 value definition."""
    fn = make_e7(db_path) if kind == 'e7' else make_e8(db_path)
    extra = getattr(fn, 'extra_cols', [])
    with patched(fn, extra):
        f = irf.build_injury_features(db_path, seasons, snapshot_name='game_day_noon')
    for c in extra:
        f[c] = f[c].fillna(0.0)
    return f


def attach(games, inj, cols):
    """Attach `cols` of inj as *_home/_away and diff_* to a games frame (replacing existing columns of the same name)."""
    out = games.drop(columns=[c for k in cols for c in (f'{k}_home', f'{k}_away', f'diff_{k}') if c in games.columns]).copy()
    out['_gid'] = out['gameId'].astype(str).str.zfill(10)
    f = inj.copy()
    f['_gid'] = f['gameId'].astype(str).str.zfill(10)
    for side in ('home', 'away'):
        sf = f[['_gid', 'teamId'] + cols].rename(columns={'teamId': f'teamId_{side}', **{c: f'{c}_{side}' for c in cols}})
        sf[f'teamId_{side}'] = sf[f'teamId_{side}'].astype('int64')
        out = out.merge(sf, on=['_gid', f'teamId_{side}'], how='left')
    for c in cols:
        out[f'diff_{c}'] = out[f'{c}_home'] - out[f'{c}_away']
    return out.drop(columns='_gid')
