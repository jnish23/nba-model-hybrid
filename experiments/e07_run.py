"""E7 runner: python -m experiments.e07_run   (bench-depth-aware replacement level for the injury value)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide, inj_variants as iv, metrics as M          # noqa: E402
from experiments.feature_exp import run_feature_experiment                  # noqa: E402
from experiments.lib import OUT, load_run                                   # noqa: E402
from experiments.report import compare, fmt_compare                         # noqa: E402
from team.data_paths import latest_training_set                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
VARIANT = ROOT / 'data' / 'processed' / 'experiments' / 'e07_training_set.csv'
NAME = 'E07_bench_replacement'
VALUE_COLS = ['inj_lost', 'inj_lost_new']


def build_csv():
    games = pd.read_csv(latest_training_set(), low_memory=False)
    games['teamId_home'], games['teamId_away'] = games['teamId_home'].astype('int64'), games['teamId_away'].astype('int64')
    seasons = sorted(s for s in games['seasonYear'].unique() if s >= '2019-20')
    inj = iv.build_variant_features(DB, seasons, 'e7')
    new = iv.attach(games, inj, VALUE_COLS)
    # games before the injury-report era keep the production values (they are NaN there in both versions)
    pre = ~games['seasonYear'].isin(seasons)
    for c in [f'{k}_{s}' for k in VALUE_COLS for s in ('home', 'away')] + [f'diff_{k}' for k in VALUE_COLS]:
        new.loc[pre, c] = games.loc[pre, c].values
    VARIANT.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(VARIANT, index=False)
    return games, new


def decision(cmp, test=None, base=None, **_):
    """KEEP iff, on games with >= 1 Out player, MC holds for spread MAE or total MAE, and NH holds on all games."""
    games = pd.read_csv(latest_training_set(), usecols=['gameId', 'inj_n_out_home', 'inj_n_out_away'], low_memory=False)
    games['gameId'] = games['gameId'].astype(str).str.zfill(10)
    has_out = set(games.loc[(games['inj_n_out_home'].fillna(0) + games['inj_n_out_away'].fillna(0)) >= 1, 'gameId'])
    sub = compare(NAME, exp_test=test, base_test=base, subset=has_out, save=False)
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / NAME / 'vs_baseline_out_games.json').write_text(json.dumps(sub, indent=1, default=float))
    mc = {'Spread MAE': decide.mc(sub['ae_spread']), 'Total MAE': decide.mc(sub['ae_total'])}
    nh = {k: decide.nh(cmp[c]) for k, c in (('ML log loss', 'll'), ('Spread MAE', 'ae_spread'), ('Total MAE', 'ae_total'))}
    decision.subset = (sub, len(has_out))
    return bool(any(mc.values()) and all(nh.values())), {f'games with >=1 Out: {k}': v for k, v in mc.items()}, nh


def extra(test, base, cmp):
    sub, n = decision.subset
    lines = [f'## Games with at least one Out player ({n} games in the training set; subset used for the keep rule)\n', fmt_compare(sub).to_markdown(index=False), '']
    # change in how often / how strongly injury features are used
    import lightgbm as lgb
    from experiments.lib import GRID, load_games, make_est, split_fold
    rows = []
    for name, csv in (('baseline', None), (NAME, VARIANT)):
        m, df = load_games(str(csv) if csv else None)
        feats = json.loads((OUT / name / 'features.json').read_text())
        info = json.loads((OUT / name / 'fold_info.json').read_text())
        for s in ('2023-24', '2024-25', '2025-26'):
            mt, calib, tst, n_fl = split_fold(df, s)
            for tgt, kind, col in (('spread', 'reg', 'target_point_diff'), ('total', 'reg', 'target_total_pts'), ('ml', 'clf', 'target_home_win')):
                f = feats[s][tgt]
                est = make_est(kind, GRID[info[s][tgt]['cfg']]).fit(mt[f], mt[col])
                imp = pd.Series(est.booster_.feature_importance('gain'), index=f)
                inj = imp[[c for c in f if 'inj_' in c]].sum() / imp.sum()
                rows.append({'run': name, 'season': s, 'target': tgt, 'inj features selected': sum('inj_' in c for c in f), 'inj share of total gain': round(float(inj), 4)})
    imp = pd.DataFrame(rows)
    piv = imp.pivot_table(index=['season', 'target'], columns='run', values=['inj features selected', 'inj share of total gain'])
    lines += ['## Feature-importance change (refit of each fold\'s chosen config; folds 2023-24..2025-26)\n', piv.round(4).to_markdown(), '']
    return lines


if __name__ == '__main__':
    build_csv()
    run_feature_experiment(NAME, VARIANT, 'bench-depth-aware replacement level in the injury value',
                           extra_sections=extra, decision_fn=decision)
