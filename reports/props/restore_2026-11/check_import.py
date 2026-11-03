"""
Step 5 checks for the imported live record: row counts, a re-export into a scratch folder compared with data/props_dashboard_live, and the immutability triggers.

    .venv/Scripts/python.exe reports/props/restore_2026-11/check_import.py [--db data/props_live.db] [--out reports/props/restore_2026-11/export_check]
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from props.live import export  # noqa: E402

EXPORT = ROOT / 'data/props_dashboard_live'
OLD_CONFIG = ROOT / 'reports/props/restore_2026-11/live_config_v3_live_2026-27.old.json'
BASELINE = {'bets': 733, 'player_games': 461, 'roi_best_stands': 0.019716, 'roi_median_stands': -0.020376, 'roi_flat_stands': -0.015503}


def compare_bets(a, b):
    """Column-by-column differences between two bets frames on the same bet_ids. Returns {column: n differing rows (and an example)}."""
    a, b = a.set_index('bet_id').sort_index(), b.set_index('bet_id').sort_index()
    out = {}
    for c in sorted(set(a.columns) & set(b.columns)):
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            diff = ~((np.isnan(xv) & np.isnan(yv)) | (np.abs(xv - yv) <= 1e-9))
        else:
            diff = ~((x.isna() & y.isna()) | (x.astype(str) == y.astype(str))).to_numpy()
        if diff.any():
            i = np.flatnonzero(diff)[0]
            out[c] = {'n': int(diff.sum()), 'example': [a.index[i], str(x.iloc[i]), str(y.iloc[i])]}
    return out, sorted(set(a.columns) ^ set(b.columns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=str(ROOT / 'data/props_live.db'))
    ap.add_argument('--out', default=str(ROOT / 'reports/props/restore_2026-11/export_check'))
    a = ap.parse_args()
    assert Path(a.out).resolve() != EXPORT.resolve(), 'never re-export into the real live export folder'
    cfg = json.loads(OLD_CONFIG.read_text(encoding='utf-8'))
    report = {}

    with sqlite3.connect(f'file:{a.db}?mode=ro', uri=True) as conn:
        report['counts'] = {t: conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('prop_bets', 'prop_results')}
        report['game_id_lengths'] = dict(conn.execute('SELECT length(game_id), COUNT(*) FROM prop_bets GROUP BY 1').fetchall())
        report['player_id_types'] = dict(conn.execute('SELECT typeof(player_id), COUNT(*) FROM prop_bets GROUP BY 1').fetchall())

    export.export_live(a.db, a.out, cfg, 1000, None, None, None)
    out = Path(a.out)
    old_b = pd.read_csv(EXPORT / 'bets.csv', dtype={'game_id': str})
    new_b = pd.read_csv(out / 'bets.csv', dtype={'game_id': str})
    report['bet_ids_equal'] = set(old_b['bet_id']) == set(new_b['bet_id']) and len(old_b) == len(new_b)
    diffs, only = compare_bets(old_b, new_b)
    report['bets_column_differences'] = diffs
    report['columns_in_one_only'] = only
    unsettled = set(old_b.loc[old_b['status_stands'].isna(), 'bet_id'])
    if diffs:
        o, n = old_b[~old_b['bet_id'].isin(unsettled)], new_b[~new_b['bet_id'].isin(unsettled)]
        report['differences_on_settled_bets'] = compare_bets(o, n)[0]

    s_old = pd.read_csv(EXPORT / 'season_summary.csv').iloc[0]
    s_new = pd.read_csv(out / 'season_summary.csv').iloc[0]
    report['season_summary'] = {k: {'baseline': v, 'old_file': float(s_old[k]), 'new': float(s_new[k]), 'equal_to_6dp': round(float(s_new[k]), 6) == v}
                                for k, v in BASELINE.items()}
    for name in ('pmf.json', 'books.json'):
        report[f'{name}_equal'] = json.loads((EXPORT / name).read_text(encoding='utf-8')) == json.loads((out / name).read_text(encoding='utf-8'))

    conn = sqlite3.connect(a.db)
    try:
        conn.execute('BEGIN')
        conn.execute("UPDATE prop_bets SET line = line + 1 WHERE bet_id = (SELECT bet_id FROM prop_bets LIMIT 1)")
        report['update_trigger'] = 'NO ERROR (trigger missing!)'
    except sqlite3.DatabaseError as e:
        report['update_trigger'] = str(e)
    finally:
        conn.rollback()
    try:
        conn.execute('BEGIN')
        conn.execute("DELETE FROM prop_bets WHERE bet_id = (SELECT bet_id FROM prop_bets LIMIT 1)")
        report['delete_trigger'] = 'NO ERROR (trigger missing!)'
    except sqlite3.DatabaseError as e:
        report['delete_trigger'] = str(e)
    finally:
        conn.rollback()
    report['prop_bets_after_trigger_test'] = conn.execute('SELECT COUNT(*) FROM prop_bets').fetchone()[0]
    conn.close()
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
