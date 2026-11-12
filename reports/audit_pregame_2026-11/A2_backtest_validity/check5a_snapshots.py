"""Check 5a: for 2026-27, is nba_odds_opening really an opening line and nba_odds really a closing line?
Compare, game by game and book by book, with the timestamped odds_snapshots table (first 'opening' rows; last pre-game 'current' rows)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *   # noqa

BOOKS = ['betmgm', 'fanduel', 'caesars', 'bet365', 'draftkings', 'bet_rivers_ny']
ALIAS = {'LA Clippers': 'Los Angeles Clippers'}
with conn() as c:
    s = pd.read_sql('select * from odds_snapshots', c)
    op = pd.read_sql("select * from nba_odds_opening where gameDate>='2026-10-01'", c)
    cl = pd.read_sql("select * from nba_odds where gameDate>='2026-10-01'", c)
for d in (op, cl):
    d['gameDate'] = pd.to_datetime(d['gameDate']).dt.strftime('%Y-%m-%d')
    for col in ('home_team', 'away_team'):
        d[col] = d[col].replace(ALIAS)
for col in ('home_team', 'away_team'):
    s[col] = s[col].replace(ALIAS)
s['fetched'] = pd.to_datetime(s['fetched_at_et'])
s['tip_et'] = pd.to_datetime(s['game_start'], utc=True, errors='coerce').dt.tz_convert('America/New_York').dt.tz_localize(None)
key = ['game_date', 'home_team', 'away_team']

# ---------- timing of first appearance
first_open = s[s.line_type == 'opening'].groupby(key).agg(first_open=('fetched', 'min'), last_open=('fetched', 'max'),
                                                          tip=('tip_et', 'first'), n_open_rows=('book', 'size')).reset_index()
first_any = s.groupby(key).agg(first_seen=('fetched', 'min'), tip=('tip_et', 'first')).reset_index()
fo = first_any.merge(first_open[key + ['first_open', 'last_open', 'n_open_rows']], on=key, how='left')
fo['noon'] = pd.to_datetime(fo['game_date']) + pd.Timedelta(hours=12)
fo['hours_before_tip'] = (fo['tip'] - fo['first_open']).dt.total_seconds() / 3600
fo['hours_before_noon'] = (fo['noon'] - fo['first_open']).dt.total_seconds() / 3600
fo['seen_hours_before_noon'] = (fo['noon'] - fo['first_seen']).dt.total_seconds() / 3600
fo['open_span_h'] = (fo['last_open'] - fo['first_open']).dt.total_seconds() / 3600
print('2026-27 games in snapshots:', len(fo), ' with opening rows:', int(fo.first_open.notna().sum()))
print('first opening row relative to tip (hours before): ', fo.hours_before_tip.describe().round(2).to_dict())
print('first opening row relative to game-day noon (hours before; negative = AFTER noon):', fo.hours_before_noon.describe().round(2).to_dict())
print('share of games whose FIRST opening row appears at/after noon on game day:', round(float((fo.hours_before_noon < 0).mean()), 3),
      ' after tip-30m:', round(float((fo.hours_before_tip < 0.5).mean()), 3))
print('opening rows re-stamped over time (hours between first and last "opening" row of a game): ', fo.open_span_h.describe().round(2).to_dict())
fo.to_csv(OUT / 'check5a_opening_first_seen.csv', index=False)

# ---------- opening: snapshot 'opening' (first fetch) vs nba_odds_opening, per book
so = s[s.line_type == 'opening'].sort_values('fetched')
so_first = so.drop_duplicates(key + ['book'], keep='first')
so_last = so.drop_duplicates(key + ['book'], keep='last')
rows = []
for b in BOOKS:
    for fld in ('home_spread', 'total', 'home_ml', 'away_ml'):
        col = f'{fld}_{b}'
        if col not in op.columns:
            continue
        m = op[['gameDate', 'home_team', 'away_team', col]].rename(columns={'gameDate': 'game_date'}).merge(
            so_first[key + ['book', fld]].query('book==@b').rename(columns={fld: 'snap_first'}), on=key, how='inner').merge(
            so_last[key + ['book', fld]].query('book==@b').rename(columns={fld: 'snap_last'}), on=key + ['book'], how='inner')
        m[col] = pd.to_numeric(m[col], errors='coerce')
        both = m[col].notna() & m['snap_first'].notna()
        rows.append({'table': 'nba_odds_opening', 'book': b, 'field': fld, 'n_both': int(both.sum()),
                     'eq_first_opening_snap': float((m.loc[both, col] == m.loc[both, 'snap_first']).mean()) if both.any() else np.nan,
                     'eq_last_opening_snap': float((m.loc[both, col] == m.loc[both, 'snap_last']).mean()) if both.any() else np.nan,
                     'mean_abs_diff_first': float((m.loc[both, col] - m.loc[both, 'snap_first']).abs().mean()) if both.any() else np.nan})
# ---------- closing: nba_odds vs last pre-game 'current' snapshot (in_game=0)
cur = s[(s.line_type == 'current') & (s.in_game == 0)].sort_values('fetched')
cur_last = cur.drop_duplicates(key + ['book'], keep='last')
cur_first = cur.drop_duplicates(key + ['book'], keep='first')
cur_all = s[(s.line_type == 'current')].sort_values('fetched').drop_duplicates(key + ['book'], keep='last')
for b in BOOKS:
    for fld in ('home_spread', 'total', 'home_ml', 'away_ml'):
        col = f'{fld}_{b}'
        if col not in cl.columns:
            continue
        m = cl[['gameDate', 'home_team', 'away_team', col]].rename(columns={'gameDate': 'game_date'}).merge(
            cur_last[key + ['book', fld, 'fetched']].query('book==@b').rename(columns={fld: 'snap_last_pre', 'fetched': 't_last'}), on=key, how='inner')
        m = m.merge(cur_first[key + ['book', fld]].query('book==@b').rename(columns={fld: 'snap_first_cur'}), on=key + ['book'], how='left')
        m = m.merge(cur_all[key + ['book', fld]].query('book==@b').rename(columns={fld: 'snap_last_any'}), on=key + ['book'], how='left')
        m[col] = pd.to_numeric(m[col], errors='coerce')
        both = m[col].notna() & m['snap_last_pre'].notna()
        rows.append({'table': 'nba_odds', 'book': b, 'field': fld, 'n_both': int(both.sum()),
                     'eq_last_pregame_snap': float((m.loc[both, col] == m.loc[both, 'snap_last_pre']).mean()) if both.any() else np.nan,
                     'eq_first_current_snap': float((m.loc[both, col] == m.loc[both, 'snap_first_cur']).mean()) if both.any() else np.nan,
                     'eq_last_any_snap_incl_ingame': float((m.loc[both, col] == m.loc[both, 'snap_last_any']).mean()) if both.any() else np.nan,
                     'mean_abs_diff_last_pre': float((m.loc[both, col] - m.loc[both, 'snap_last_pre']).abs().mean()) if both.any() else np.nan})
res = pd.DataFrame(rows)
res.to_csv(OUT / 'check5a_table_vs_snapshots.csv', index=False)
pd.set_option('display.width', 250)
print(res.groupby(['table', 'field'], dropna=False).agg(n=('n_both', 'sum'), eq_first_open=('eq_first_opening_snap', 'mean'),
                                                        eq_last_open=('eq_last_opening_snap', 'mean'),
                                                        eq_last_pre=('eq_last_pregame_snap', 'mean'),
                                                        eq_first_cur=('eq_first_current_snap', 'mean'),
                                                        eq_last_any=('eq_last_any_snap_incl_ingame', 'mean')).round(3).to_string())

# ---------- game-level consensus comparison (median across books) for spread/total
def med(df, fld, how):
    d = df.groupby(key)[fld].median().rename(how)
    return d
op_c = op.assign(opening_spread=op[[f'home_spread_{b}' for b in BOOKS]].apply(pd.to_numeric, errors='coerce').median(axis=1),
                 opening_total=op[[f'total_{b}' for b in BOOKS]].apply(pd.to_numeric, errors='coerce').median(axis=1)
                 )[['gameDate', 'home_team', 'away_team', 'opening_spread', 'opening_total']].rename(columns={'gameDate': 'game_date'})
cl_c = cl.assign(close_spread=cl[[f'home_spread_{b}' for b in BOOKS]].apply(pd.to_numeric, errors='coerce').median(axis=1),
                 close_total=cl[[f'total_{b}' for b in BOOKS]].apply(pd.to_numeric, errors='coerce').median(axis=1)
                 )[['gameDate', 'home_team', 'away_team', 'close_spread', 'close_total']].rename(columns={'gameDate': 'game_date'})
snap_open_first = med(so_first, 'home_spread', 'snap_open_spread').to_frame().join(med(so_first, 'total', 'snap_open_total')).reset_index()
snap_cur_last = med(cur_last, 'home_spread', 'snap_close_spread').to_frame().join(med(cur_last, 'total', 'snap_close_total')).reset_index()
snap_cur_first = med(cur_first, 'home_spread', 'snap_firstcur_spread').to_frame().join(med(cur_first, 'total', 'snap_firstcur_total')).reset_index()
G = op_c.merge(cl_c, on=key, how='outer').merge(snap_open_first, on=key, how='left').merge(snap_cur_last, on=key, how='left').merge(snap_cur_first, on=key, how='left')
for a, b_, nm in (('opening_spread', 'snap_open_spread', 'open spread table-snapshot'), ('opening_total', 'snap_open_total', 'open total table-snapshot'),
                  ('close_spread', 'snap_close_spread', 'close spread table-snapshot'), ('close_total', 'snap_close_total', 'close total table-snapshot'),
                  ('opening_spread', 'snap_close_spread', 'opening(table) vs last pre-game snapshot'),
                  ('close_spread', 'snap_open_spread', 'close(table) vs first opening snapshot')):
    d = G[a] - G[b_]
    print(f'{nm:50s} n={int(d.notna().sum()):4d}  exact={float((d.abs() < 1e-9).mean()):.3f}  mean|diff|={d.abs().mean():.3f}  max|diff|={d.abs().max():.2f}')
G.to_csv(OUT / 'check5a_game_level.csv', index=False)
# how many 2026-27 games have table 'close' identical to table 'opening' (a closing table that is just the opening line)
g2 = G.dropna(subset=['opening_spread', 'close_spread'])
print('games with nba_odds close == nba_odds_opening (spread):', int((g2.opening_spread == g2.close_spread).sum()), 'of', len(g2),
      ' total:', int((g2.opening_total == g2.close_total).sum()))
print('mean |close - open| spread: table %.3f ; snapshots %.3f' % ((g2.close_spread - g2.opening_spread).abs().mean(),
                                                                   (G.snap_close_spread - G.snap_open_spread).abs().mean()))
