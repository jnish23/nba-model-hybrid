"""Check 1b: statistical leak probe over ALL feature columns of the 2026-11-01 training set.
(1) corr of every feature with the same game's margin, total and home win (Pearson and Spearman), compared with the opening lines
    (which are legitimately informative) and with the corr between the PREVIOUS game's outcome and the feature (should be larger for
    form features, which carry past results).
(2) exact-equality / near-equality of every feature with any same-game box-score value of either team (all numeric team_game_full
    columns, both sides) and with the game's margin / total / points."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *   # noqa

t = load_train()
meta = ['gameDate', 'seasonYear', 'gameId', 'teamId_home', 'teamId_away', 'target_home_win', 'target_point_diff', 'target_total_pts',
        'points_home', 'points_away']
num = t.select_dtypes('number').columns
feat = [c for c in num if c not in meta and not c.startswith('target_')]
print('numeric feature columns:', len(feat))
odds = [c for c in feat if c.startswith('odds_')]
y = {'margin': t['target_point_diff'], 'total': t['target_total_pts'], 'home_win': t['target_home_win']}
rows = []
for c in feat:
    x = t[c]
    r = {'feature': c, 'non_null': int(x.notna().sum())}
    for k, v in y.items():
        r[f'pearson_{k}'] = x.corr(v)
        r[f'spearman_{k}'] = x.corr(v, method='spearman')
    rows.append(r)
cr = pd.DataFrame(rows)
cr['absmax'] = cr[[c for c in cr.columns if c.startswith('pearson')]].abs().max(axis=1)
cr.sort_values('absmax', ascending=False).to_csv(OUT / 'check1b_outcome_correlations.csv', index=False)
print('odds columns:'); print(cr[cr.feature.isin(odds)].round(3).to_string())
nonodds = cr[~cr.feature.isin(odds)].sort_values('absmax', ascending=False)
print('top 15 non-odds features by max |pearson| with same-game outcome:')
print(nonodds.head(15)[['feature', 'pearson_margin', 'pearson_total', 'pearson_home_win']].round(3).to_string())
print('non-odds features with |corr|>0.5 with any outcome:', int((nonodds.absmax > 0.5).sum()), ' >0.4:', int((nonodds.absmax > 0.4).sum()))
ml = ['odds_home_spread', 'odds_total']
print('benchmark: odds_home_spread vs margin', t['odds_home_spread'].corr(t['target_point_diff']), ' odds_total vs total',
      t['odds_total'].corr(t['target_total_pts']))

# --- (2) equality with same-game box-score values
with conn() as c:
    tg = pd.read_sql('select * from team_game_full', c)
tg['gameId'] = tg['gameId'].astype(str).str.zfill(10)
numcols = [c for c in tg.select_dtypes('number').columns if c not in ('teamId',)]
side = {}
for s in ('home', 'away'):
    side[s] = t[['gameId', f'teamId_{s}']].merge(tg[['gameId', 'teamId'] + numcols].rename(columns={'teamId': f'teamId_{s}'}),
                                                  on=['gameId', f'teamId_{s}'], how='left')
    assert len(side[s]) == len(t)
ref = {}
for s in ('home', 'away'):
    for c in numcols:
        ref[f'{s}:{c}'] = side[s][c].to_numpy(float)
ref['game:margin'] = t['target_point_diff'].to_numpy(float)
ref['game:total'] = t['target_total_pts'].to_numpy(float)
ref['game:home_win'] = t['target_home_win'].to_numpy(float)
refn = list(ref)
R = np.column_stack([ref[k] for k in refn])
hits = []
for c in feat:
    x = t[c].to_numpy(float)
    ok = ~np.isnan(x)
    if ok.sum() < 100:
        continue
    xs = x[ok][:, None]
    Rs = R[ok]
    with np.errstate(invalid='ignore'):
        eq = (np.abs(Rs - xs) < 1e-9)
        share = eq.mean(axis=0)
    if np.all(np.isnan(share)):
        continue
    j = int(np.nanargmax(share))
    # also correlation (>0.98) with a same-game value
    with np.errstate(invalid='ignore', divide='ignore'):
        xc = xs[:, 0] - xs.mean()
        Rc = np.where(np.isnan(Rs), np.nan, Rs)
        cors = np.array([pd.Series(xs[:, 0]).corr(pd.Series(Rc[:, k])) for k in range(Rc.shape[1])])
    if np.all(np.isnan(cors)):
        continue
    k = int(np.nanargmax(np.abs(cors)))
    hits.append({'feature': c, 'best_equal_ref': refn[j], 'equal_share': float(share[j]), 'best_corr_ref': refn[k],
                 'corr': float(cors[k])})
h = pd.DataFrame(hits)
h.sort_values('equal_share', ascending=False).to_csv(OUT / 'check1b_samegame_equality.csv', index=False)
print('features compared against', len(refn), 'same-game values')
print('max equal share:'); print(h.sort_values('equal_share', ascending=False).head(10).to_string())
print('max |corr| with a same-game box-score value:'); print(h.reindex(h['corr'].abs().sort_values(ascending=False).index).head(15).to_string())
# the legit case: roster/travel/etc features can equal values at low share by coincidence (0 == 0). Report features with >50% equal share
print('features equal to some same-game value in >50% of rows:', h[h.equal_share > .5].to_dict('records'))
