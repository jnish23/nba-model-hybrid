"""Follow-up to check3: the one game whose roster_bpm differs by >1e-6, and where roster_bpm==0 at a season's first game."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from common import *
from check3_bpm_recompute import *
t = load_train(); ps = load_players(); pg = player_game_bpm(ps); roster = team_game_roster(pg)
for side in ('home','away'):
    for f in ('roster_bpm','roster_bpm_ewma'):
        r = t[['gameId','gameDate',f'teamId_{side}',f'{f}_{side}']].merge(roster.rename(columns={'teamId':f'teamId_{side}',f:'mine'})[['gameId',f'teamId_{side}','mine']], on=['gameId',f'teamId_{side}'])
        d=(r[f'{f}_{side}']-r['mine']).abs(); bad=r[d>1e-6]
        if len(bad): print(side,f); print(bad.assign(diff=d[d>1e-6]).to_string())
z = t[(t.roster_bpm_home==0)|(t.roster_bpm_away==0)]
print('rows with roster_bpm exactly 0:', len(z)); print(z.groupby('seasonYear').size())
