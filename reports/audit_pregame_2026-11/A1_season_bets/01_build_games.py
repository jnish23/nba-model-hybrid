"""Step 1: one row per 2026-27 game with
   * the logged prediction (predictions_log_v2), the parsed picks table of data/raw/daily_inference.log (Nov 1+),
   * the opening line the model actually read (prediction_explanations 'movement' piece, see RESULT.md),
   * snapshot lines (odds_snapshots): opening proxy, 11 am, run-time, close; cross-book medians and per-book values,
   * the book the logged line came from (matched against the run-time snapshot).
Writes games.csv, log_rows.csv, snap_series.csv (cross-book median series per game and time)."""
import re
import numpy as np
import pandas as pd
from common import *

BOOK_PRIORITY = ['draftkings', 'fanduel', 'betmgm', 'bet365', 'caesars']

c = con()
P = pd.read_sql("select * from predictions_log_v2 where game_date>='2026-10-20'", c)
assert not P.game_id.duplicated().any()
P['run_et'] = pd.to_datetime(P.prediction_date) + pd.Timedelta(hours=3)      # server clock is Pacific (log: 08:19 PT = 11:19 ET cutoff)
P['margin'] = P.home_score - P.away_score
P['total'] = P.home_score + P.away_score

# ------------------------------------------------------------------ 1. parse the log
text = LOG.read_text(encoding='utf-8', errors='replace').splitlines()
runs, cur_hdr = [], None
for i, ln in enumerate(text):
    m = re.match(r'===== \w+ (\d\d)/(\d\d)/(\d{4}) +(\d+:\d\d)', ln)
    if m:
        cur_hdr = (f"{m.group(3)}-{m.group(1)}-{m.group(2)}", i)
    if ln.strip() == '--- PREDICTIONS ---':
        runs.append((cur_hdr[0], i))
mt = P.set_index('game_id').matchup.to_dict()
rows = []
for day, i0 in runs:
    j = i0 + 2                                   # skip header line
    n_rows = 0
    while j < len(text) and re.match(r'^\s*00226\d{5}\s', text[j]):
        ln = text[j]
        gid = re.search(r'(00226\d{5})', ln).group(1)
        k = ln.index(mt[gid]) + len(mt[gid])
        tok = ln[k:].split()
        head = ln[:ln.index(mt[gid])].split()
        names = ['ml_prob', 'spread_pred', 'total_pred', 'odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_total', 'home_score',
                 'away_score', 'status', 'ml_bet', 'ml_edge', 'spread_bet', 'spread_edge', 'total_bet', 'total_edge',
                 'inj_downgrade_lost_home', 'inj_downgrade_lost_away']
        d = dict(zip(names, tok[:len(names)]))
        d['recommendation'] = ' '.join(tok[len(names):])
        d.update(game_id=gid, log_day=day, log_run_header_line=i0 + 1, game_date_log=head[1])
        rows.append(d)
        j += 1
        n_rows += 1
L = pd.DataFrame(rows)
for col in ['ml_prob', 'spread_pred', 'total_pred', 'odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_total', 'ml_edge',
            'spread_edge', 'total_edge', 'inj_downgrade_lost_home', 'inj_downgrade_lost_away']:
    L[col] = pd.to_numeric(L[col], errors='coerce')
for col in ['ml_bet', 'spread_bet', 'total_bet']:
    L[col] = L[col].replace('None', np.nan)
L.to_csv(OUT / 'log_rows.csv', index=False)
print('log: runs', len(runs), 'rows', len(L), 'days', L.log_day.nunique(), 'dup game ids', L.game_id.duplicated().sum())

# ------------------------------------------------------------------ 2. opening line the model read (explanations)
E = pd.read_sql("select game_id, prediction_date, market, feature, value, contribution from prediction_explanations "
                "where kind='piece' and feature in ('line','movement') and prediction_date>='2026-10-20'", c)
E['game_id'] = E.game_id.str.zfill(10)
E = E.sort_values('prediction_date').drop_duplicates(['game_id', 'market', 'feature'], keep='last')
W = E.pivot_table(index='game_id', columns=['market', 'feature'], values='value')
W.columns = [f'{a}_{b}' for a, b in W.columns]
W = W.reset_index()
X = pd.DataFrame({'game_id': W.game_id,
                  'ex_open_home_spread': -W['spread_movement'],      # movement piece value = -current + (current - open) = -open
                  'ex_cur_home_spread': -W['spread_line'],
                  'ex_open_total': W['total_movement'],              # = current + (open - current) = open
                  'ex_cur_total': W['total_line'],
                  'ex_p_open_home_ml': W['moneyline_movement'],      # implied prob of the opening home price (vig included)
                  'ex_p_cur_home_ml': W['moneyline_line']})
G = P.merge(X, on='game_id', how='left')

# ------------------------------------------------------------------ 3. snapshots
S = pd.read_sql("select * from odds_snapshots", c)
S['fetched'] = pd.to_datetime(S.fetched_at_et)
S['tip_et'] = pd.to_datetime(S.game_start, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
S['p_home_book'] = np.where(S.home_ml.notna() & S.away_ml.notna(), devig2(S.home_ml.fillna(-110), S.away_ml.fillna(-110)), np.nan)
key = ['game_date', 'home_team', 'away_team']


def med(df, col):
    v = df[col].dropna()
    return v.median() if len(v) else np.nan


def line_at(df):
    """cross-book summary of a set of per-book rows (one time)"""
    return dict(spread=med(df, 'home_spread'), total=med(df, 'total'), p_home=med(df, 'p_home_book'), n_books=df.book.nunique(),
                n_spread=df.home_spread.notna().sum(), n_total=df.total.notna().sum(), n_ml=df.p_home_book.notna().sum())


snap_rows, series_rows = [], []
for (gd, ht, at), g in S.groupby(key):
    tip = g.tip_et.iloc[0]
    cur = g[(g.line_type == 'current') & (g.in_game == 0)]
    opn = g[g.line_type == 'opening']
    rec = dict(game_date=gd, home_team=ht, away_team=at, tip_et=tip)
    # time series of cross-book medians (pre-game only)
    for t, d in cur.groupby('fetched'):
        r = line_at(d); r.update(game_date=gd, home_team=ht, away_team=at, fetched=t, mins_to_tip=(tip - t).total_seconds() / 60)
        series_rows.append(r)
    # opening proxy: per book the earliest 'opening' row that carries each field; then cross-book median
    if len(opn):
        o = {}
        for col, name in [('home_spread', 'spread'), ('total', 'total'), ('p_home_book', 'p_home')]:
            vals = []
            for b, db_ in opn.groupby('book'):
                v = db_.sort_values('fetched').dropna(subset=[col])
                if len(v):
                    vals.append(v[col].iloc[0])
            o[name] = np.median(vals) if vals else np.nan
            o['n_' + name] = len(vals)
        rec.update({'snap_open_' + k: v for k, v in o.items()})
        rec['snap_open_first_fetch'] = opn.fetched.min()
        rec['snap_open_mins_to_tip'] = (tip - opn.fetched.min()).total_seconds() / 60
    if len(cur):
        last_t = cur.fetched.max()
        rec['snap_close_fetch'] = last_t
        rec['snap_close_mins_to_tip'] = (tip - last_t).total_seconds() / 60
        for k, v in line_at(cur[cur.fetched == last_t]).items():
            rec['snap_close_' + k] = v
        for b, bd in cur[cur.fetched == last_t].groupby('book'):
            r = bd.iloc[0]
            rec[f'close_{b}_spread'], rec[f'close_{b}_total'], rec[f'close_{b}_p_home'] = r.home_spread, r.total, r.p_home_book
        # 11 am snapshot (the fetch at 11:00 ET on the game day, or the closest one at/before 11:10 if 11:00 is absent)
        day = pd.Timestamp(gd)
        t11 = cur[(cur.fetched >= day + pd.Timedelta(hours=10, minutes=45)) & (cur.fetched <= day + pd.Timedelta(hours=11, minutes=10))]
        if len(t11):
            tt = t11.fetched.iloc[(t11.fetched - (day + pd.Timedelta(hours=11))).abs().argmin()]
            rec['snap_11_fetch'] = tt
            for k, v in line_at(cur[cur.fetched == tt]).items():
                rec['snap_11_' + k] = v
    snap_rows.append(rec)
SG = pd.DataFrame(snap_rows)
SER = pd.DataFrame(series_rows)
SER.to_csv(OUT / 'snap_series.csv', index=False)
G = G.merge(SG, on=key, how='left')

# run-time snapshot (last pre-game snapshot at or before the model run) and the book that matches the logged line
run_rows = []
Sc = S[(S.line_type == 'current') & (S.in_game == 0)]
for r in G.itertuples():
    g = Sc[(Sc.game_date == r.game_date) & (Sc.home_team == r.home_team) & (Sc.away_team == r.away_team) & (Sc.fetched <= r.run_et)]
    out = dict(game_id=r.game_id)
    if len(g):
        t = g.fetched.max()
        d = g[g.fetched == t]
        out['snap_run_fetch'] = t
        out['snap_run_gap_min'] = (r.run_et - t).total_seconds() / 60
        for k, v in line_at(d).items():
            out['snap_run_' + k] = v
        best, bn = None, -1
        for b in BOOK_PRIORITY:
            bd = d[d.book == b]
            if bd.empty:
                continue
            x = bd.iloc[0]
            n = int(np.isclose(x.home_spread, r.odds_home_spread, equal_nan=False)) + int(np.isclose(x.total, r.odds_total)) + \
                int(np.isclose(x.home_ml, r.odds_home_ml)) + int(np.isclose(x.away_ml, r.odds_away_ml))
            if n > bn:
                best, bn = b, n
                bx = x
        out['bet_book'], out['bet_book_match'] = best, bn
        if best is not None:
            for f in ['home_spread', 'home_spread_odds', 'away_spread_odds', 'total', 'over_odds', 'under_odds', 'home_ml', 'away_ml']:
                out['bk_' + f] = bx[f]
            out['bk_p_home'] = bx.p_home_book
            # per-book closing values for the matched book
            cd = S[(S.game_date == r.game_date) & (S.home_team == r.home_team) & (S.away_team == r.away_team) & (S.book == best) &
                   (S.line_type == 'current') & (S.in_game == 0)].sort_values('fetched')
            if len(cd):
                z = cd.iloc[-1]
                out['bk_close_spread'], out['bk_close_total'], out['bk_close_p_home'] = z.home_spread, z.total, z.p_home_book
                out['bk_close_fetch'] = z.fetched
    run_rows.append(out)
G = G.merge(pd.DataFrame(run_rows), on='game_id', how='left')

# ------------------------------------------------------------------ 4. log vs database cross-check
G = G.merge(L.add_prefix('lg_').rename(columns={'lg_game_id': 'game_id'}), on='game_id', how='left')
G['in_log'] = G.lg_log_day.notna()
G['odds_source'] = G.game_id.map(pd.read_sql("select game_id, source from odds_source_log", c).drop_duplicates('game_id', keep='last').set_index('game_id').source)
G.to_csv(OUT / 'games.csv', index=False)
print(G.shape, 'in log', G.in_log.sum(), 'completed', G.home_score.notna().sum())
