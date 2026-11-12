"""Check 4b: independent numeric recomputation of inj_n_out and inj_missing for a random sample of games (2021-22..2026-27, seed 42) from
the parsed injury reports + my own BPM/e_minutes (check3 code): which report is used (latest <= min(noon, tip-30m), canonical grid),
who is a 'rotation' player (expected minutes >= 10 at his last appearance for the team this season, <= 25 team games ago, not traded away),
and how many rotation players the report lists as Out. Compares with inj_n_out_home/away and inj_missing_* in the training set."""
import re
import sys
import unicodedata
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *   # noqa
from check3_bpm_recompute import load_players, player_game_bpm
import warnings; warnings.filterwarnings('ignore')

t = load_train()
ps = load_players()
pg = player_game_bpm(ps)[['gameId', 'teamId', 'personId', 'gameDate', 'seasonYear', 'e_min', 'bpm', 'minutes']]
pg['gameId'] = pg['gameId'].astype(str).str.zfill(10)
pg_names = ps[['personId']].drop_duplicates()
with conn() as c:
    names = pd.read_sql('select distinct personId, playerName, teamId, seasonYear from player_basic_stats', c)
    sched = pd.read_sql('select gameId, teamId, gameDate, seasonYear, teamTricode from team_game_full', c)
    teamnames = pd.read_sql('select distinct teamId, teamName from team_basic_stats', c)
sched['gameId'] = sched.gameId.astype(str).str.zfill(10); sched['gameDate'] = pd.to_datetime(sched.gameDate)
sched = sched.sort_values(['teamId', 'gameDate', 'gameId'])
sched['k'] = sched.groupby(['teamId', 'seasonYear']).cumcount()


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z]', '', s.lower())


def report_key(n):
    if not isinstance(n, str): return None
    if ',' in n:
        last, first = n.split(',', 1); return norm(first) + norm(last)
    return norm(n)


names['key'] = names.playerName.map(norm)
tn = {norm(n): i for n, i in zip(teamnames.teamName, teamnames.teamId)}
tn.update({'laclippers': tn.get('losangelesclippers'), 'charlottebobcats': tn.get('charlottehornets')})

# ---- reports on the canonical grid
r = pd.read_csv(ROOT / 'data/processed/injury_reports_parsed.csv', low_memory=False)
r['report_ts'] = pd.to_datetime(r.report_ts); r['game_date'] = pd.to_datetime(r.game_date)
r = r.dropna(subset=['report_ts', 'game_date', 'matchup'])
def slot(f):
    m = re.search(r'_(\d{2})(?:_(\d{2}))?(AM|PM)\.pdf$', str(f))
    return None if not m else (int(m.group(1)) % 12 + (12 if m.group(3) == 'PM' else 0), int(m.group(2) or 0))
GRID = {(11, 0), (12, 0), (13, 0), (17, 0), (18, 0), (19, 0), (20, 0)}
ok = {f: slot(f) in GRID for f in r.source_file.unique()}
r = r[r.source_file.map(ok)]
WEST = {'LAL', 'LAC', 'GSW', 'GS', 'SAC', 'POR', 'PHX', 'PHO', 'UTA', 'UTAH', 'DEN', 'SEA'}
g = r.drop_duplicates(['game_date', 'matchup'])[['game_date', 'matchup', 'game_time']].copy()
hm = g.game_time.str.extract(r'(\d\d):(\d\d)').astype(float); hr = hm[0]; home = g.matchup.str.split('@').str[1]
am = hr.isin([10, 11]) & ~home.isin(WEST); h24 = np.where(am, hr, hr % 12 + 12)
g['tip'] = g.game_date + pd.to_timedelta(h24, unit='h') + pd.to_timedelta(hm[1], unit='m')
g['cutoff'] = pd.to_datetime(np.minimum((g.game_date + pd.Timedelta(hours=12)).to_numpy('datetime64[ns]'), (g.tip - pd.Timedelta(minutes=30)).to_numpy('datetime64[ns]')))
rr = r.merge(g[['game_date', 'matchup', 'cutoff']], on=['game_date', 'matchup'])
rr = rr[rr.report_ts <= rr.cutoff]
chosen = rr.groupby(['game_date', 'matchup']).report_ts.transform('max')
rr = rr[rr.report_ts == chosen]
rr['tkey'] = rr.team.map(norm).map(tn)
rr['pkey'] = rr.player.map(report_key)

# tricode of home team to match report matchup
tri = sched.drop_duplicates('teamId').set_index('teamTricode').teamId.to_dict()
for k, v in {'PHO': 'PHX', 'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS', 'UTAH': 'UTA'}.items():
    if v in tri: tri.setdefault(k, tri[v])
rr['home_id'] = rr.matchup.str.split('@').str[1].map(tri)

# ---- sample
rng = np.random.default_rng(SEED)
cand_games = t[t.seasonYear >= '2021-22']
samp = cand_games.iloc[rng.choice(len(cand_games), 300, replace=False)]
pg_by_team = {k: v for k, v in pg.groupby('teamId')}
sk = sched.set_index(['gameId', 'teamId'])['k']
rows = []
for _, gm in samp.iterrows():
    for side in ('home', 'away'):
        tid = int(gm[f'teamId_{side}']); d = gm.gameDate; season = gm.seasonYear
        k_now = sk.loc[(gm.gameId, tid)]
        # report rows for this game and team
        rep = rr[(rr.game_date == d) & (rr.home_id == int(gm.teamId_home))]
        has_report = len(rep) > 0
        team_rows = rep[rep.tkey == tid]
        nys = bool(team_rows.not_submitted.astype(bool).any()) if len(team_rows) else False
        # rotation candidates
        tp = pg_by_team[tid]
        tp = tp[(tp.seasonYear == season) & (tp.gameDate < d)]
        tp = tp.merge(sched[['gameId', 'teamId', 'k']], on=['gameId', 'teamId'])
        last = tp.sort_values('k').groupby('personId').tail(1)
        last = last.assign(gap=k_now - last.k)
        rot = last[(last.e_min >= 10) & (last.gap <= 25)]
        # not traded away: most recent appearance anywhere before this date was for this team
        recent = pg[(pg.personId.isin(rot.personId)) & (pg.gameDate < d)].sort_values('gameDate').groupby('personId').tail(1).set_index('personId').teamId
        rot = rot[rot.personId.map(recent) == tid]
        pk = names[(names.teamId == tid) & (names.seasonYear == season)].drop_duplicates('personId').set_index('personId').key
        rot = rot.assign(key=rot.personId.map(pk))
        out_keys = set(team_rows[team_rows.status == 'Out'].pkey.dropna())
        n_out = int(rot.key.isin(out_keys).sum())
        rows.append({'gameId': gm.gameId, 'side': side, 'season': season, 'has_report': has_report, 'team_not_submitted': nys,
                     'my_missing': int((not has_report) or nys), 'my_n_out': n_out, 'ds_missing': gm[f'inj_missing_{side}'], 'ds_n_out': gm[f'inj_n_out_{side}'],
                     'n_rot': len(rot), 'listed_out_unmatched': int(len(out_keys - set(rot.key)))})
res = pd.DataFrame(rows)
res.to_csv(OUT / 'check4b_inj_n_out_sample.csv', index=False)
print('team-games compared:', len(res))
print('inj_missing agreement: %.3f ; mismatches %d' % ((res.my_missing == res.ds_missing).mean(), (res.my_missing != res.ds_missing).sum()))
nm = res[(res.ds_missing == 0)]
print('inj_n_out exact agreement where the report covers the team: %.3f of %d' % ((nm.my_n_out == nm.ds_n_out).mean(), len(nm)))
print('  mean |diff| %.3f ; mine > dataset in %d, mine < dataset in %d' % ((nm.my_n_out - nm.ds_n_out).abs().mean(), (nm.my_n_out > nm.ds_n_out).sum(), (nm.my_n_out < nm.ds_n_out).sum()))
print('  dataset n_out distribution:', nm.ds_n_out.value_counts().sort_index().to_dict())
print('  rows where they differ:'); print(nm[nm.my_n_out != nm.ds_n_out].head(15).to_string())
mm = res[res.my_missing != res.ds_missing]
print('missing mismatches:'); print(mm.head(10).to_string())
