"""
Step 3 checks for the rebuilt features database: tables and row counts (per table and season), size, max game_date, and the feature dictionary against the tracked one.

    .venv/Scripts/python.exe reports/props/restore_2026-11/step3_parity.py
Writes step3_parity.json and step3_dictionary_diff.csv in this folder.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DB = ROOT / 'data/props_features_v3.db'
OLD_BYTES = 684130304
TABLES = ('player_game_base', 'player_game_labels', 'player_game_at_T')


def main():
    out = {'db_bytes': DB.stat().st_size, 'old_db_bytes': OLD_BYTES}
    with sqlite3.connect(f'file:{DB}?mode=ro', uri=True) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        out['tables'] = sorted(names)
        assert set(TABLES) <= names, f'missing tables: {set(TABLES) - names}'
        out['rows'] = {t: conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in TABLES}
        seasons = pd.read_sql('SELECT game_id, season, game_date FROM player_game_base', conn)
        out['base_rows_by_season'] = seasons['season'].value_counts().sort_index().to_dict()
        out['max_game_date'] = {'player_game_base': str(seasons['game_date'].max())}
        gs = seasons[['game_id', 'season']].drop_duplicates('game_id')
        for t in ('player_game_labels', 'player_game_at_T'):
            g = pd.read_sql(f'SELECT game_id FROM {t}', conn).merge(gs, on='game_id', how='left')
            out[f'{t}_rows_by_season'] = g['season'].fillna('unmatched').value_counts().sort_index().to_dict()
        out['at_T_rows_by_minutes_before'] = dict(conn.execute('SELECT minutes_before, COUNT(*) FROM player_game_at_T GROUP BY 1').fetchall())
    out['max_game_date_on_or_before_2026-06-30'] = out['max_game_date']['player_game_base'][:10] <= '2026-06-30'

    old = pd.read_csv(ROOT / 'reports/props/phase2_feature_dictionary.csv')
    new = pd.read_csv(HERE / 'phase2_feature_dictionary.csv')
    ko, kn = set(zip(old['table'], old['feature'])), set(zip(new['table'], new['feature']))
    out['features_old'], out['features_new'] = len(ko), len(kn)
    out['feature_set_identical'] = ko == kn
    out['only_old'], out['only_new'] = sorted(ko - kn), sorted(kn - ko)
    m = old.merge(new, on=['table', 'feature'], suffixes=('_old', '_new'))
    m['d_non_null'] = m['non_null_share_new'] - m['non_null_share_old']
    m['first_season_changed'] = m['first_season_with_data_old'].astype(str) != m['first_season_with_data_new'].astype(str)
    for c in ('group', 'source', 'dtype', 'description'):
        out[f'{c}_changed'] = m.loc[m[f'{c}_old'].astype(str) != m[f'{c}_new'].astype(str), 'feature'].tolist()
    changed = m[(m['d_non_null'].abs() > 0) | m['first_season_changed']]
    cols = ['table', 'feature', 'group_old', 'non_null_share_old', 'non_null_share_new', 'd_non_null', 'first_season_with_data_old', 'first_season_with_data_new',
            'non_null_by_season_old', 'non_null_by_season_new']
    changed[cols].sort_values('d_non_null', key=abs, ascending=False).to_csv(HERE / 'step3_dictionary_diff.csv', index=False)
    out['n_changed_non_null_or_first_season'] = int(len(changed))
    out['changed_by_group'] = changed['group_old'].value_counts().to_dict()
    big = changed[(changed['d_non_null'].abs() > 0.01) | changed['first_season_changed']]
    out['moved_more_than_0.01_or_first_season'] = big[['table', 'feature', 'group_old', 'non_null_share_old', 'non_null_share_new', 'first_season_with_data_old',
                                                       'first_season_with_data_new']].to_dict('records')
    (HERE / 'step3_parity.json').write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(json.dumps(out, indent=1, default=str))
    return 0 if out['feature_set_identical'] else 1


if __name__ == '__main__':
    sys.exit(main())
