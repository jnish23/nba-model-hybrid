"""
Dashboard tables from the live store, in the schema of the backtest export (props/betting/export.py) so the dashboard reads either. Pending bets have blank results; the live summary
carries the metrics that can be computed on a short record and leaves the rest blank. Nothing is computed that the stored bets do not support.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard_props import logic
from props.live import settle as live_settle

DISCLOSURE = ('Paper record of the frozen rule with the live model and roster rule v2. The live inputs do not reproduce the training inputs (teammate availability cannot be rebuilt '
              'before the game): in the parity test on 40 held-out 2025-26 dates the live-feature bets earned about 1.7 points less than the training-feature bets (+3.9% vs +5.6% at the '
              'selected price), A1/A4/A5/A6 were waived by the owner, and October coverage is about 60%. See reports/props/d1_parity_report.md.')
SUMMARY_COLS = ['season', 'kind', 'bets', 'player_games', 'roi_best_stands', 'roi_median_stands', 'roi_flat_stands', 'clv_real_pts', 'clv_placebo_mean_pts', 'clv_excess_pts',
                'clv_excess_lower_pts', 'placebo_roi', 'slope4', 'slope4_lo', 'slope4_hi']
RESULT_COLS = ['minutes_actual', 'y', 'played', 'status_stands', 'status_flagged', 'status_all', 'clv_prob', 'line_move_favour', 'close_line', 'line_240', 'line_60', 'line_5',
               'mkt_240', 'mkt_60', 'mkt_5']
HISTORY_COLS = ['line_240', 'line_60', 'line_5', 'mkt_240', 'mkt_60', 'mkt_5']


def export_live(live_db, out_dir, config, n_boot=1000, checks=None, current=None, results_now=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f'file:{live_db}?mode=ro', uri=True) as conn:
        bets = pd.read_sql('SELECT * FROM prop_bets ORDER BY created_utc', conn)
        res = pd.read_sql('SELECT * FROM prop_results', conn)
        sig = pd.read_sql('SELECT game_id, player_id, stat, slot_minutes, line, p_market FROM prop_signals', conn)
        health = pd.read_sql('SELECT * FROM pipeline_health ORDER BY started_utc DESC LIMIT 200', conn)
        slots = pd.read_sql('SELECT * FROM slot_log ORDER BY written_utc DESC LIMIT 500', conn)
    pmf = {r.bet_id: json.loads(r.pmf_json) for r in bets.itertuples() if r.pmf_json}
    books = {r.bet_id: json.loads(r.books_json) for r in bets.itertuples() if r.books_json}
    hist = live_settle.signal_context(sig)[1] if len(sig) else pd.DataFrame(columns=['game_id', 'player_id', 'stat'])
    b = bets.drop(columns=['pmf_json', 'books_json']).merge(res.drop(columns=['run_id']), on='bet_id', how='left', validate='one_to_one')
    fresh = b[['game_id', 'player_id', 'stat']].merge(hist, on=['game_id', 'player_id', 'stat'], how='left')
    for c in HISTORY_COLS:                                          # the recorded signals fill the line history of a bet whose game has not been settled yet
        new = fresh[c].to_numpy() if c in fresh.columns else np.full(len(b), np.nan)
        b[c] = b[c].where(b[c].notna(), pd.Series(new, index=b.index)) if c in b.columns else new
    b['data_kind'] = 'live paper record'
    b['played'] = b['played'].fillna(0).astype(int)
    keep = ['bet_id', 'season', 'data_kind', 'game_id', 'game_date', 'tip_ref_utc', 'placed_at_utc', 'minutes_before', 'matchup', 'home_team', 'away_team', 'player_id', 'player_name', 'stat',
            'side', 'line', 'book', 'dec_best', 'dec_median', 'dec_flat', 'ev_best', 'ev_median', 'ev_flat', 'p_over_model', 'p_over_star', 'p_over_market', 'push', 'n_at_line', 'n_quotes',
            'expected_minutes', 'created_utc', 'model_version', 'snapshot_minutes'] + RESULT_COLS
    b = b[[c for c in keep if c in b.columns]]
    b.to_csv(out / 'bets.csv', index=False)
    (out / 'pmf.json').write_text(json.dumps(pmf, separators=(',', ':')), encoding='utf-8')
    (out / 'books.json').write_text(json.dumps(books, separators=(',', ':')), encoding='utf-8')
    settled = b[b['status_stands'].notna()] if 'status_stands' in b.columns else b.iloc[0:0]
    row = {'season': config['model_season'], 'kind': 'live paper record (settled bets only)', 'bets': 0, 'player_games': 0}
    if len(settled):
        for mode in ('best', 'median', 'flat'):
            row[f'roi_{mode}_stands'] = logic.roi_summary(settled, mode, 'stands', n_boot)['roi']
        s = logic.roi_summary(settled, 'median', 'stands', 10)
        row.update({'bets': s['bets'], 'player_games': s['player_games'], 'clv_real_pts': float(settled['clv_prob'].mean() * 100) if settled['clv_prob'].notna().any() else np.nan})
    pd.DataFrame([row]).reindex(columns=SUMMARY_COLS).to_csv(out / 'season_summary.csv', index=False)
    pd.DataFrame(columns=['season', 'mean_p_model', 'hit_rate']).to_csv(out / 'reliability.csv', index=False)
    pd.DataFrame(columns=['season', 'month', 'stat', 'ratio']).to_csv(out / 'bias_monthly.csv', index=False)
    health.to_csv(out / 'pipeline_health.csv', index=False)
    slots.to_csv(out / 'slot_log.csv', index=False)
    if checks is not None:
        checks.to_csv(out / 'checks.csv', index=False)
    if current is not None:
        current.to_csv(out / 'current.csv', index=False)
    if results_now is not None:
        results_now.to_csv(out / 'results_now.csv', index=False)
    rule = json.loads(Path(config['rule_file']).read_text(encoding='utf-8'))
    meta = {'mode': 'live', 'rule': rule, 'rule_sha256': config['rule_sha256'], 'code_commit': bets['code_commit'].iloc[-1] if len(bets) else '', 'features_db_sha256': config['features_db_sha256'],
            'model_version': config['model_version'], 'roster_rule': config['roster_rule'], 'built_at_utc': pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
            'record_started_utc': bets['created_utc'].min() if len(bets) else None, 'seasons': {config['model_season']: 'live paper record'}, 'disclosure': DISCLOSURE,
            'pending': int(len(b) - len(settled)), 'pricing': {'median': 'median over the bettable books at the consensus line', 'best': 'best quote among the bettable books', 'flat': 'flat -110'}}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return b
