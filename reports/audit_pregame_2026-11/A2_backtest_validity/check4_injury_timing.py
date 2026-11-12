"""Check 4: injury-report timing, re-derived independently from data/processed/injury_reports_parsed.csv.
For every game 2019-20..2026-27 find, with own code, the report the training rule should have used (latest content timestamp
<= min(game-day noon ET, tip - 30 min), on the canonical download grid), then
  (a) report how far before the cutoff and before tip each chosen report is, and that nothing after tip/cutoff is used,
  (b) compare my 'team has no usable report' flag with the training set's inj_missing_home / inj_missing_away,
  (c) test the tip-time heuristic against real start times (odds_snapshots.game_start, 2026-27) and show the implied tip-hour table,
  (d) check the status play-rate table is cumulative over earlier seasons only."""
import json
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *   # noqa

r = pd.read_csv(ROOT / 'data/processed/injury_reports_parsed.csv', low_memory=False)
r['report_ts'] = pd.to_datetime(r['report_ts'])
r['game_date'] = pd.to_datetime(r['game_date'])
r = r.dropna(subset=['report_ts', 'game_date', 'matchup'])


def slot(f):
    m = re.search(r'_(\d{2})(?:_(\d{2}))?(AM|PM)\.pdf$', str(f))
    if not m:
        return None
    return (int(m.group(1)) % 12 + (12 if m.group(3) == 'PM' else 0), int(m.group(2) or 0))


GRID = {(11, 0), (12, 0), (13, 0), (17, 0), (18, 0), (19, 0), (20, 0)}
files = pd.Series(r['source_file'].unique())
on_grid = {f: slot(f) in GRID for f in files}
print('files:', len(files), ' on canonical grid:', sum(on_grid.values()))
r_all = r
r = r[r['source_file'].map(on_grid)]
print('rows on grid', len(r), 'of', len(r_all))

# tip time (ET) per game_date x matchup from the printed 'HH:MM' (no AM/PM printed)
WEST = {'LAL', 'LAC', 'GSW', 'GS', 'SAC', 'POR', 'PHX', 'PHO', 'UTA', 'UTAH', 'DEN', 'SEA'}
g = r.drop_duplicates(['game_date', 'matchup'])[['game_date', 'matchup', 'game_time']].copy()
hm = g['game_time'].str.extract(r'(\d\d):(\d\d)').astype(float)
home = g['matchup'].str.split('@').str[1]
hr = hm[0]
am = hr.isin([10, 11]) & ~home.isin(WEST)
hour24 = np.where(am, hr, hr % 12 + 12)
g['tip'] = g['game_date'] + pd.to_timedelta(hour24, unit='h') + pd.to_timedelta(hm[1], unit='m')
g['tip_hour'] = hour24
g['cutoff'] = np.minimum((g['game_date'] + pd.Timedelta(hours=12)).to_numpy('datetime64[ns]'),
                         (g['tip'] - pd.Timedelta(minutes=30)).to_numpy('datetime64[ns]'))
g['cutoff'] = pd.to_datetime(g['cutoff'])

# (c) tip-hour heuristic validated against real start times
with conn() as c:
    sn = pd.read_sql("select distinct game_date, home_team_abbr, away_team_abbr, game_start from odds_snapshots", c)
sn['tip_real'] = pd.to_datetime(sn['game_start'], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
sn['matchup'] = sn['away_team_abbr'] + '@' + sn['home_team_abbr']
sn['game_date'] = pd.to_datetime(sn['game_date'])
cmp_ = g.merge(sn.drop_duplicates(['game_date', 'matchup']), on=['game_date', 'matchup'])
cmp_['err_min'] = (cmp_['tip'] - cmp_['tip_real']).dt.total_seconds() / 60
print('tip heuristic vs real start (2026-27 games present in both):', len(cmp_), ' exact:', int((cmp_.err_min == 0).sum()),
      ' wrong:', int((cmp_.err_min != 0).sum()))
print(cmp_[cmp_.err_min != 0][['game_date', 'matchup', 'game_time', 'tip', 'tip_real']].head(10).to_string())
print('implied tip-hour counts:'); print(g['tip_hour'].value_counts().sort_index().to_string())

# (a) chosen report per game
rr = r.merge(g[['game_date', 'matchup', 'tip', 'cutoff']], on=['game_date', 'matchup'])
ok = rr[rr['report_ts'] <= rr['cutoff']]
ch = ok.groupby(['game_date', 'matchup'])['report_ts'].max().rename('chosen_ts').reset_index()
gg = g.merge(ch, on=['game_date', 'matchup'], how='left')
gg['age_h'] = (gg['cutoff'] - gg['chosen_ts']).dt.total_seconds() / 3600
gg['before_tip_h'] = (gg['tip'] - gg['chosen_ts']).dt.total_seconds() / 3600
gg['noon_minus_ts_h'] = ((gg['game_date'] + pd.Timedelta(hours=12)) - gg['chosen_ts']).dt.total_seconds() / 3600
print('games with a usable report:', int(gg.chosen_ts.notna().sum()), 'of', len(gg))
x = gg.dropna(subset=['chosen_ts'])
print('chosen report minus noon: max (must be <= 0 after cutoff):', float((-x.noon_minus_ts_h).max()),
      ' chosen after tip-30m:', int((x.before_tip_h < 0.5 - 1e-9).sum()), ' chosen after tip:', int((x.before_tip_h < 0).sum()))
print('hours between chosen report and cutoff: median %.2f p90 %.2f max %.2f' % (x.age_h.median(), x.age_h.quantile(.9), x.age_h.max()))
print('chosen report timestamp, hour-of-day distribution:'); print(x.chosen_ts.dt.hour.value_counts().sort_index().to_string())
# what share of games would have used a LATER report had the cutoff been 6pm (to show the rule matters)
gg.to_csv(OUT / 'check4_injury_snapshot_choice.csv', index=False)

# (b) agreement with training inj_missing
t = load_train()
with conn() as c:
    tgf = pd.read_sql('select gameId, teamId, teamTricode, gameDate, MATCHUP from team_game_full', c)
tgf['gameId'] = tgf['gameId'].astype(str).str.zfill(10)
tri = tgf.drop_duplicates('teamId').set_index('teamTricode')['teamId'].to_dict()
for k, v in {'PHO': 'PHX', 'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS', 'UTAH': 'UTA', 'BK': 'BKN', 'CHA': 'CHA'}.items():
    if v in tri:
        tri.setdefault(k, tri[v])
gg['home_tri'] = gg['matchup'].str.split('@').str[1]
gg['away_tri'] = gg['matchup'].str.split('@').str[0]
gg['teamId_home'] = gg['home_tri'].map(tri)
gg['teamId_away'] = gg['away_tri'].map(tri)
print('report matchups with unknown tricode:', int(gg.teamId_home.isna().sum() + gg.teamId_away.isna().sum()))
gm = t[['gameId', 'gameDate', 'teamId_home', 'teamId_away', 'inj_missing_home', 'inj_missing_away', 'seasonYear']].merge(
    gg.rename(columns={'game_date': 'gameDate'})[['gameDate', 'teamId_home', 'teamId_away', 'chosen_ts', 'tip']],
    on=['gameDate', 'teamId_home', 'teamId_away'], how='left')
gm = gm[gm.seasonYear >= '2019-20']
gm['mine_no_report'] = gm['chosen_ts'].isna()
# the project's flag is 1 when no report OR team 'not yet submitted'; so inj_missing==0 must imply a usable report existed
for side in ('home', 'away'):
    f = gm[f'inj_missing_{side}']
    print(side, 'inj_missing=0 yet I find NO usable report before cutoff:', int(((f == 0) & gm.mine_no_report).sum()),
          '| inj_missing=1 and I find a report:', int(((f == 1) & ~gm.mine_no_report).sum()),
          '(these are teams marked not-yet-submitted or unlinked) | n =', len(gm))
print('by season: share inj_missing_home=1:'); print(gm.groupby('seasonYear')[['inj_missing_home', 'inj_missing_away']].mean().round(3).to_string())
print('games with no report at all before cutoff by season:'); print(gm.groupby('seasonYear')['mine_no_report'].mean().round(3).to_string())

# (d) play-rate table cumulative-ness
j = json.load(open(ROOT / 'models/saved/injury_status_play_rates.json'))
seasons = [k for k in j if k != 'all']
prev = {}
for s in seasons:
    n = {st: v['n'] for st, v in j[s].items()}
    inc = {st: n[st] - prev.get(st, 0) for st in n}
    print(s, 'n by status', n, 'increment over previous season (= previous season own sample):', inc)
    prev = n
print('all - last season (= last season own sample):', {st: j['all'][st]['n'] - prev[st] for st in prev})
