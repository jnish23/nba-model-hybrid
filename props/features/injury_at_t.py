"""
Injury status and teammate availability AS OF a decision time T (the T-dependent part of the dataset).

T = reference tip - X minutes (X = 240, 60, 5), the reference tip being the NBA scheduled tip + 10 minutes (props/schedule.py). For every game the
latest official injury report published at or before T that lists the game is used; `inj_report_age_min` says how old it is (the archive is
hourly, so the age differs by season, and production will see fresher reports than most training rows did).

Status -> probability of playing is NOT a constant: P(played | status, X, trailing-minutes bucket) is estimated from EARLIER seasons only and shrunk
toward the spec's rates (Out 0, Doubtful 0.02, Questionable 0.57, Probable 0.93). A player not on the report gets P(played | unlisted, bucket):
a rotation player nobody mentions is almost surely playing, a deep-bench player often is not.

Teammate availability (decision D3: weighted-missing-share): for each player, the expected share of the team's minutes, usage, rebound chances,
assist creation and scoring that sits with teammates who are unavailable, sum over teammates j != i of (1 - p_j) * w_j divided by sum over all of w_j,
with w_j taken from j's own trailing (pre-game) rates. The same aggregates are computed for the opponent.
"""
import numpy as np
import pandas as pd

STATUS_CODE = {'no_report': -1, 'None': 0, 'Available': 1, 'Probable': 2, 'Questionable': 3, 'Doubtful': 4, 'Out': 5}
SPEC_P_PLAY = {'Out': 0.0, 'Doubtful': 0.02, 'Questionable': 0.57, 'Probable': 0.93, 'Available': 0.95, 'None': 0.85, 'no_report': 0.85}
K_RATE = 30.0                                         # pseudo-observations for the shrinkage toward the spec rates
REASON_CLASSES = ['injury', 'illness', 'rest', 'g_league', 'personal', 'other']


def reason_class(reason):
    """Reports print 'Injury/Illness - <body part>; <detail>': the 'Injury/Illness' label is a category heading, not information, so it is dropped first."""
    s = str(reason).lower()
    if s.startswith('injury/illness') and ' - ' in s:
        s = s.split(' - ', 1)[1]
    if 'g league' in s or 'two-way' in s or 'assignment' in s:
        return 'g_league'
    if 'rest' in s:
        return 'rest'
    if any(w in s for w in ('illness', 'covid', 'health and safety', 'flu', 'sick', 'concussion protocol')):
        return 'illness'
    if any(w in s for w in ('personal', 'bereavement', 'suspension', 'not with team', 'trade', 'coach')):
        return 'personal'
    if any(w in s for w in ('injury', 'management', 'soreness', 'sprain', 'strain', 'contusion', 'fracture', 'tendin', 'surgery', 'knee', 'ankle',
                            'hamstring', 'back', 'foot', 'calf', 'shoulder', 'wrist', 'groin', 'hip', 'thigh', 'elbow', 'finger', 'hand', 'toe',
                            'achilles', 'quad', 'rib', 'neck', 'eye', 'nose', 'face', 'head')):
        return 'injury'
    return 'other'


PUBLISH_LAG_MIN = 15          # a report is stamped with its scheduled slot, not the time it was posted: it counts as known only this many minutes after its stamp


def latest_report_rows(reports, tip_ref_et, minutes_before, publish_lag_min=PUBLISH_LAG_MIN):
    """
    reports: linked report rows (gameId, teamId, personId, report_ts [ET], status, reason, not_submitted).
    tip_ref_et: Series game_id -> naive ET reference tip. Returns (rows of the chosen report per game, per-game table with report_ts, age).
    A report is usable at the decision time T only if its stamp is at least publish_lag_min before T (the stamp is the scheduled slot; the PDF can appear minutes
    later). The age is still measured from T.
    """
    r = reports.dropna(subset=['gameId']).copy()
    r['gameId'] = r['gameId'].astype(str).str.zfill(10)
    r['report_ts'] = pd.to_datetime(r['report_ts'])
    cutoff = tip_ref_et - pd.Timedelta(minutes=minutes_before)
    r['cutoff'] = r['gameId'].map(cutoff)
    r = r[r['cutoff'].notna() & (r['report_ts'] <= r['cutoff'] - pd.Timedelta(minutes=publish_lag_min))]
    r['chosen_ts'] = r['gameId'].map(r.groupby('gameId')['report_ts'].max())
    r = r[r['report_ts'] == r['chosen_ts']]
    games = r.groupby('gameId').agg(report_ts=('report_ts', 'first'), cutoff=('cutoff', 'first'))
    games.index.name = None
    games['report_age_min'] = (games['cutoff'] - games['report_ts']).dt.total_seconds() / 60.0
    return r, games


def status_for_rows(rows, chosen_rows, games):
    """
    rows: roster rows (game_id, player_id, team_id, ...). Returns a frame aligned to rows with inj_status (string), inj_reason_class,
    inj_report_age_min and inj_report_missing.
    """
    out = rows[['game_id', 'player_id', 'team_id']].copy()
    c = chosen_rows.dropna(subset=['personId']).copy()
    c['player_id'] = c['personId'].astype('int64')
    c = c.rename(columns={'gameId': 'game_id'})
    # one status per player per game: the most severe if a player appears twice on the same report
    c['sev'] = c['status'].map(STATUS_CODE).fillna(0)
    c = c.sort_values('sev').drop_duplicates(['game_id', 'player_id'], keep='last')
    out = out.merge(c[['game_id', 'player_id', 'status', 'reason']], on=['game_id', 'player_id'], how='left', validate='many_to_one')
    has_report = out['game_id'].isin(games.index)
    out['inj_status'] = np.where(~has_report, 'no_report', out['status'].fillna('None'))
    out['inj_reason_class'] = np.where(out['reason'].notna(), out['reason'].map(reason_class), 'none')
    out['inj_report_missing'] = (~has_report).astype(int)
    out['inj_report_age_min'] = out['game_id'].map(games['report_age_min'])
    return out.drop(columns=['status', 'reason'])


def play_rate_table(history_rows, minutes_before, first_season):
    """
    history_rows: roster rows with season, inj_status, m_bucket and lbl_played for the SAME X. Returns a Series indexed (season, status, bucket):
    P(played) estimated from seasons strictly before `season`, shrunk toward SPEC_P_PLAY[status].
    """
    h = history_rows[['season', 'inj_status', 'm_bucket', 'lbl_played']]
    g = h.groupby(['season', 'inj_status', 'm_bucket'])['lbl_played'].agg(['sum', 'size'])
    g = g.sort_index()
    cum = g.groupby(level=[1, 2]).cumsum()
    before = cum - g                                           # earlier seasons of the same (status, bucket) cell
    idx = before.index
    prior = pd.Series([SPEC_P_PLAY[s] for s in idx.get_level_values(1)], index=idx)
    p = (before['sum'] + K_RATE * prior) / (before['size'] + K_RATE)
    return p.rename('p_play')


def attach_p_play(status_df, rows_feat, table):
    """Adds inj_p_play and inj_status_code to the status frame, using the table for the row's season, status and minutes bucket."""
    d = status_df.merge(rows_feat[['game_id', 'player_id', 'season', 'm_bucket']], on=['game_id', 'player_id'], how='left', validate='many_to_one')
    idx = pd.MultiIndex.from_frame(d[['season', 'inj_status', 'm_bucket']])
    p = table.reindex(idx).to_numpy()
    spec = d['inj_status'].map(SPEC_P_PLAY).to_numpy()
    d['inj_p_play'] = np.where(np.isnan(p), spec, p)          # unseen cell (e.g. the first dataset season): the spec rate
    d['inj_status_code'] = d['inj_status'].map(STATUS_CODE).astype(int)
    return d


WEIGHTS = {  # name -> function of the pre-game history features giving the teammate's trailing "amount" of that thing per game
    'minutes': lambda f: f['m_min_h15'],
    'usage': lambda f: f['r_usage_pct_h15'] * f['m_min_h15'],
    'reb': lambda f: f['r_reb_chances_h15'] * f['m_min_h15'],
    'ast': lambda f: (f['r_assists_h15'] + f['r_secondary_ast_h15']) * f['m_min_h15'],
    'pts': lambda f: f['r_points_h15'] * f['m_min_h15'],
}


def teammate_availability(d, feat, prefix='av'):
    """
    d: frame with game_id, player_id, team_id, opp_team_id, inj_p_play, inj_status_code (one row per roster player).
    feat: the history features for the same rows (m_min_h15, r_*_h15). Returns d's keys plus {prefix}_* (own team, excluding the player) and
    opp_{prefix}_* (the opponent's whole roster) columns.
    """
    x = d[['game_id', 'player_id', 'team_id', 'opp_team_id', 'inj_p_play', 'inj_status_code']].merge(
        feat, on=['game_id', 'player_id'], how='left', suffixes=('', '_f'), validate='many_to_one')
    q = 1.0 - x['inj_p_play'].to_numpy()
    cols = {}
    for name, fn in WEIGHTS.items():
        w = fn(x).fillna(0.0).clip(lower=0).to_numpy()
        x[f'_w_{name}'] = w
        x[f'_qw_{name}'] = q * w
    x['_out'] = (x['inj_status_code'] == STATUS_CODE['Out']).astype(int)
    x['_q'] = ((x['inj_status_code'] == STATUS_CODE['Questionable']) | (x['inj_status_code'] == STATUS_CODE['Doubtful'])).astype(int)
    x['_starter_qw'] = np.where(x['m_min_h15'].fillna(0) >= 24, q * x['m_min_h15'].fillna(0), 0.0)
    keys = ['game_id', 'team_id']
    agg_cols = [c for c in x.columns if c.startswith(('_w_', '_qw_'))] + ['_out', '_q', '_starter_qw']
    team = x.groupby(keys)[agg_cols].transform('sum')
    res = x[['game_id', 'player_id', 'team_id', 'opp_team_id']].copy()
    for name in WEIGHTS:
        tot = team[f'_w_{name}']
        miss_excl_self = team[f'_qw_{name}'] - x[f'_qw_{name}']
        denom = tot - x[f'_w_{name}']                                 # the rest of the team's total, excluding the player
        res[f'{prefix}_miss_{name}'] = np.where(denom > 0, miss_excl_self / denom.where(denom > 0), np.nan)
    res[f'{prefix}_n_out'] = team['_out'] - x['_out']
    res[f'{prefix}_n_questionable'] = team['_q'] - x['_q']
    res[f'{prefix}_miss_starter_minutes'] = team['_starter_qw'] - x['_starter_qw']
    # the opponent's whole-roster aggregates (nobody is excluded)
    opp = x.groupby(keys)[agg_cols].sum()
    opp_out = pd.DataFrame(index=opp.index)
    for name in WEIGHTS:
        opp_out[f'opp_{prefix}_miss_{name}'] = np.where(opp[f'_w_{name}'] > 0, opp[f'_qw_{name}'] / opp[f'_w_{name}'].where(opp[f'_w_{name}'] > 0), np.nan)
    opp_out[f'opp_{prefix}_n_out'] = opp['_out']
    opp_out[f'opp_{prefix}_miss_starter_minutes'] = opp['_starter_qw']
    opp_out = opp_out.reset_index().rename(columns={'team_id': 'opp_team_id'})
    return res.merge(opp_out, on=['game_id', 'opp_team_id'], how='left', validate='many_to_one')
