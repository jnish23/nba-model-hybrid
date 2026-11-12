"""Lead's check: by ET clock hour on game day, (a) how much of the open-to-close line move is still to come, and
(b) how much of the game-day injury news is already on the official report. Read-only."""
import sqlite3
import numpy as np
import pandas as pd

DB = 'data/nba_stats.db'
HOURLY = 'C:/Users/jorda/nba-model-hybrid/data/processed/props_injury_hourly_parsed.csv'
pd.set_option('display.width', 220)

# ---------------------------------------------------------------- (a) lines
c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
s = pd.read_sql("select fetched_at_et, game_date, game_start, home_team, book, line_type, in_game, home_spread, total, home_ml, away_ml from odds_snapshots", c)
s['key'] = s.game_date + '|' + s.home_team
s['ts'] = pd.to_datetime(s.fetched_at_et)
s['tip'] = pd.to_datetime(s.game_start, utc=True).dt.tz_convert('US/Eastern').dt.tz_localize(None)


def fair(h, a):
    h, a = pd.to_numeric(h, errors='coerce'), pd.to_numeric(a, errors='coerce')
    ph = np.where(h > 0, 100 / (h + 100), -h / (-h + 100)); pa = np.where(a > 0, 100 / (a + 100), -a / (-a + 100))
    return ph / (ph + pa)


s['fair'] = fair(s.home_ml, s.away_ml)
cur = s[(s.line_type == 'current') & (s.in_game == 0) & (s.ts < s.tip)]
cons = cur.groupby(['key', 'ts', 'tip']).agg(spread=('home_spread', 'median'), total=('total', 'median'), fair=('fair', 'median')).reset_index()
close = cons.sort_values('ts').groupby('key').tail(1).set_index('key')[['spread', 'total', 'fair', 'ts']].add_prefix('close_')
opn = s[s.line_type == 'opening'].groupby('key').agg(open_spread=('home_spread', 'median'), open_total=('total', 'median'), open_fair=('fair', 'median'))
cons = cons.join(close, on='key').join(opn, on='key')
cons = cons[(cons.tip - cons.close_ts) <= pd.Timedelta(minutes=65)]             # games that have a real late snapshot
cons = cons[cons.tip.dt.hour >= 18]                                             # evening tips only, so the clock hour means the same thing
cons['same_day'] = cons.ts.dt.date == cons.tip.dt.date
cons['clock'] = np.where(cons.same_day, cons.ts.dt.hour + cons.ts.dt.minute / 60, np.nan)
rows = []
grid = [('open', None)] + [(f'{h:02d}:00 ET', h) for h in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18)]
for lab, h in grid:
    if h is None:
        g = cons.drop_duplicates('key').assign(spread=lambda d: d.open_spread, total=lambda d: d.open_total, fair=lambda d: d.open_fair)
    else:
        g = cons[cons.same_day & (cons.clock <= h)].sort_values('ts').groupby('key').tail(1)
        g = g[g.clock >= h - 1.01]
    r = {'time': lab, 'games': g.key.nunique()}
    for m in ('spread', 'total', 'fair'):
        left = (g[f'close_{m}'] - g[m]).dropna()
        tot = (g[f'close_{m}'] - g[f'open_{m}']).dropna()
        r[f'{m}_rms_left'] = float(np.sqrt((left ** 2).mean()))
        r[f'{m}_mean_abs_left'] = float(left.abs().mean())
        r[f'{m}_share_var_left'] = float((left ** 2).mean() / (tot ** 2).mean()) if len(tot) else np.nan
    rows.append(r)
print('(a) line movement still to come before the last pre-game snapshot (2026-27, evening tips)')
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------- (b) injuries
inj = pd.read_csv(HOURLY, usecols=['report_ts', 'game_date', 'game_time', 'matchup', 'team', 'player', 'status'])
inj['report_ts'] = pd.to_datetime(inj.report_ts)
inj['game_date'] = pd.to_datetime(inj.game_date)
print('\nhourly injury archive:', inj.report_ts.min(), 'to', inj.report_ts.max(), len(inj), 'rows')
inj = inj[inj.game_date >= '2024-10-01']
hh = inj.game_time.str.extract(r'(\d+):(\d+)').astype(float)
inj['tip'] = inj.game_date + pd.to_timedelta((hh[0] % 12) + 12, unit='h') + pd.to_timedelta(hh[1], unit='m')   # game_time is PM ET
inj = inj[(inj.tip.dt.hour >= 18) & inj.status.notna()]
inj = inj[inj.report_ts <= inj.tip - pd.Timedelta(minutes=30)]
inj['out'] = inj.status.isin(['Out', 'Doubtful']).astype(int)
k = ['game_date', 'matchup', 'team', 'player']
final = inj.sort_values('report_ts').groupby(k).tail(1)[k + ['out', 'status']].rename(columns={'out': 'final_out', 'status': 'final_status'})
# status the evening before (last report at or before 20:00 ET the previous day); absent = not listed
prev = inj[inj.report_ts <= inj.game_date - pd.Timedelta(hours=4)].sort_values('report_ts').groupby(k).tail(1)[k + ['out']].rename(columns={'out': 'prev_out'})
f = final.merge(prev, on=k, how='left')
f['prev_out'] = f.prev_out.fillna(0)
news = f[f.final_out != f.prev_out]                     # game-day news: a player whose out/doubtful state differs from the night before
print('player-games with a final report row:', len(f), '| game-day changes of out/doubtful state:', len(news),
      f'({(news.final_out == 1).sum()} became out, {(news.final_out == 0).sum()} cleared)')
out = []
for h in (9, 10, 11, 12, 13, 14, 15, 16, 17, 17.5, 18):
    cut = inj[inj.report_ts <= inj.game_date + pd.Timedelta(hours=h)].sort_values('report_ts').groupby(k).tail(1)[k + ['out']].rename(columns={'out': 'now_out'})
    m = news.merge(cut, on=k, how='left')
    m['now_out'] = m.now_out.fillna(m.prev_out)
    out.append({'by ET': h, 'share of game-day changes already on the report': float((m.now_out == m.final_out).mean()),
                'of players who end up out': float((m.now_out == m.final_out)[m.final_out == 1].mean()),
                'of players cleared': float((m.now_out == m.final_out)[m.final_out == 0].mean())})
print('(b) game-day injury news known by clock time (2024-25 onward, evening tips, final = last report 30+ min before tip)')
print(pd.DataFrame(out).round(3).to_string(index=False))
