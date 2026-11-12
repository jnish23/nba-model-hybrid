"""Run from the worktree root. What the two repaired matchup pairs look like on the 2026-11-01 training set, and the sign question.
Reads only data/processed/training_set_v1_reduced_2026-11-01.csv; writes nothing."""
import numpy as np, pandas as pd
from team.game_features_v2 import GameFeatureBuilder
ts = pd.read_csv('data/processed/training_set_v1_reduced_2026-11-01.csv', low_memory=False)
print('rows', len(ts), 'matchup cols in file:', [c for c in ts.columns if c.startswith('matchup_') and c not in ('matchup_home', 'matchup_away')])
for h, a, n in GameFeatureBuilder.MATCHUP_PAIRS:
    print(f'{n}: home col in file={h in ts.columns}, away col in file={a in ts.columns}')
base = ts.drop(columns=[c for c in ts.columns if c.startswith('diff_') or (c.startswith('matchup_') and c not in ('matchup_home', 'matchup_away'))])
out = GameFeatureBuilder.add_derived_features(base)   # rebuilt from the base columns: the file already holds the third matchup column
margin = ts['points_home'] - ts['points_away']
total = ts['points_home'] + ts['points_away']
print('\ncorrelations with the home margin / the total (all rows with the inputs)')
for n in ('matchup_off_vs_def_rating_5g', 'matchup_off_vs_def_rating_20g', 'matchup_3pa_vs_opp_3pa_5g'):
    ok = out[n].notna()
    print(f'  {n:34s} n={ok.sum():6d} mean={out[n].mean():7.3f} sd={out[n].std():6.3f} corr(margin)={np.corrcoef(out.loc[ok, n], margin[ok])[0,1]:+.3f} corr(total)={np.corrcoef(out.loc[ok, n], total[ok])[0,1]:+.3f}')
# the modelling question: off_home - def_away (as built) versus off_home + def_away - league mean (expected rating of the home offence)
for s in ('5g', '20g'):
    oh, da = ts[f'off_offensiveRating_{s}_home'], ts[f'off_defensiveRating_{s}_away']
    ok = oh.notna() & da.notna()
    diff, summ = (oh - da)[ok], (oh + da - (oh.mean() + da.mean()) / 2 * 1)[ok]
    print(f'\n{s}: corr(home offence vs away defence, as built: off - def) with home points: {np.corrcoef(diff, ts.loc[ok, "points_home"])[0,1]:+.3f};'
          f' with the sum form off + def: {np.corrcoef(summ, ts.loc[ok, "points_home"])[0,1]:+.3f}')
