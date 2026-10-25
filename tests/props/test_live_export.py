import json

import numpy as np
import pandas as pd

from dashboard_props import logic
from props.live import export, store

CONFIG = {'rule_file': 'reports/props/phase5_revised_rule.frozen_rule.json', 'rule_sha256': 'abc', 'features_db_sha256': 'def', 'model_version': 'v', 'model_season': '2026-27', 'roster_rule': 'v2'}


def _bet(pid, **kw):
    row = {'bet_id': f'S|g1|{pid}|assists', 'season': '2026-27', 'game_id': 'g1', 'game_date': '2026-10-23', 'tip_ref_utc': '2026-10-23T01:40:00Z', 'placed_at_utc': '2026-10-23T00:40:00Z',
           'created_utc': '2026-10-23T00:41:00Z', 'minutes_before': 60, 'matchup': 'A @ B', 'home_team': 'B', 'away_team': 'A', 'player_id': pid, 'player_name': f'P{pid}', 'stat': 'assists',
           'side': 'over', 'line': 4.5, 'book': 'fanduel', 'dec_best': 2.0, 'dec_median': 1.9, 'dec_flat': 1.909, 'ev_best': 0.05, 'ev_median': 0.03, 'ev_flat': 0.02, 'p_over_model': 0.6,
           'p_over_star': 0.55, 'p_over_market': 0.5, 'push': 0.0, 'n_at_line': 5, 'n_quotes': 5, 'expected_minutes': 30.0, 'snapshot_minutes': 58.0, 'pmf_json': '[0.1, 0.2, 0.7]',
           'books_json': '{"fanduel": 2.0}', 'model_version': 'v', 'code_commit': 'abc123'}
    row.update(kw)
    return pd.DataFrame([row])


def test_export_writes_the_backtest_schema_with_blank_results_for_pending_bets(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    store.write_slot(conn, 'g1', 60, 'done', 'now', 'r1', bets=pd.concat([_bet(1), _bet(2)]))
    with conn:
        store._insert(conn, 'prop_results', pd.DataFrame([{'bet_id': 'S|g1|1|assists', 'settled_utc': 'x', 'y': 6.0, 'played': 1, 'minutes_actual': 30.0, 'status_stands': 'win',
                                                          'status_flagged': 'win', 'status_all': 'win', 'profit_stands_best': 1.0, 'clv_prob': 0.04, 'run_id': 'r2'}]))
    b = export.export_live(tmp_path / 'live.db', tmp_path / 'out', CONFIG, n_boot=50)
    csv = pd.read_csv(tmp_path / 'out' / 'bets.csv', dtype={'game_id': str})
    assert len(csv) == 2 and csv['status_stands'].notna().sum() == 1 and csv['played'].tolist() == [1, 0]
    meta = json.loads((tmp_path / 'out' / 'meta.json').read_text())
    assert meta['mode'] == 'live' and meta['pending'] == 1 and 'waived' in meta['disclosure']
    assert json.loads((tmp_path / 'out' / 'pmf.json').read_text())['S|g1|1|assists'] == [0.1, 0.2, 0.7]
    # the dashboard's own logic runs on the pending rows (no stake) and on the settled one
    wp = logic.with_profit(csv, 'median', 'stands')
    assert wp['stake'].sum() == 1.0 and np.isclose(wp['profit'].sum(), 0.9)
    shown, hidden = logic.board(csv, 'median', 0.0, ['assists'], ['over'], 15)
    assert len(shown) == 2
