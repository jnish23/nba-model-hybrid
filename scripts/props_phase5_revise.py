"""
Estimate the revised rule's k on the two development seasons pooled (reports/props/phase5_revised_rule_preregistration.md) and write the frozen rule file. Run ONCE.

    python scripts/props_phase5_revise.py --out reports/props/phase5_revised_rule.frozen_rule.json
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.betting import rules  # noqa: E402
from props.models import market_bench as mb, registry  # noqa: E402


def load_labels(db):
    import sqlite3
    with sqlite3.connect(db) as conn:
        lab = pd.read_sql('SELECT game_id, player_id, lbl_played, lbl_minutes, y_points, y_rebounds, y_assists, y_threes, y_pra FROM player_game_labels', conn)
    lab['game_id'] = lab['game_id'].astype(str)
    return lab


STATS = ['rebounds', 'assists', 'threes', 'ra']
SEASONS = (('2023-24', 'v3_dev_p5'), ('2024-25', 'v3_seen_p5'))
PRIMARY = {'e': 0.03, 'time': 'earliest', 'min_books': 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--n-boot', type=int, default=2000)
    a = ap.parse_args()
    labels = load_labels(a.features_db)
    parts = []
    for season, version in SEASONS:
        cons, _ = pickle.load(open(f'data/props_betting/prices_{season}.pkl', 'rb'))
        idx, pm = registry.load_predictions(version, season)
        rows = idx.merge(labels, on=['game_id', 'player_id'], how='left', validate='many_to_one')
        parts.append(mb.compare(rows, pm, cons.rename(columns={'stat': 'market'})).assign(season=season))
    cmp = pd.concat(parts, ignore_index=True)
    pooled = mb.information_slope(cmp, n_boot=a.n_boot)
    per_season = pd.concat([mb.information_slope(g, n_boot=500).assign(season=s) for s, g in cmp.groupby('season')])
    k = {r['stat']: rules.clip_k(r['slope']) for _, r in pooled.iterrows() if r['stat'] in STATS}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    out = {'stats': STATS, 'k_by_stat': k, 'primary': PRIMARY, 'developed_on': [s for s, _ in SEASONS], 'tried_rules': 37, 'recalibration': 'none',
           'pooled_slopes': {r['stat']: [round(r['slope'], 4), round(r['slope_lo'], 4), round(r['slope_hi'], 4)] for _, r in pooled.iterrows()},
           'code_commit': head, 'features_db_sha256': registry.sha256_file(a.features_db), 'n_comparisons': int(len(cmp))}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(pooled.round(3).to_string(index=False)); print(per_season.round(3).to_string(index=False)); print('k:', {s: round(v, 3) for s, v in k.items()})


if __name__ == '__main__':
    main()
