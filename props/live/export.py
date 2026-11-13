"""
Dashboard tables from the live store, in the schema of the backtest export (props/betting/export.py) so the dashboard reads either. Pending bets have blank results; the live summary
carries the metrics that can be computed on a short record and leaves the rest blank. Nothing is computed that the stored bets do not support.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from core import props_board_logic as logic
from props.live import settle as live_settle

DISCLOSURE = ('Paper record of the frozen rule with the live model and roster rule v2. The live inputs do not reproduce the training inputs (teammate availability cannot be rebuilt '
              'before the game): in the parity test on 40 held-out 2025-26 dates the live-feature bets earned about 1.7 points less than the training-feature bets (+3.9% vs +5.6% at the '
              'selected price), A1/A4/A5/A6 were waived by the owner, and October coverage is about 60%. See reports/props/d1_parity_report.md.')
PREGAME_ROSTER_RULES = ('v2nq', 'pregame_all')                  # props.live.roster.PREGAME_RULES: the teammate set is known before the game and training builds it with the same code
PARITY_REPORT = 'reports/props/audit_2026-11/6a_roster/report.md'
SEGMENTS_PATH = Path(__file__).resolve().parents[2] / 'config' / 'props_forward_segments.json'


def disclosure(config, rule=None):
    """
    The warning text of the live dashboard, a function of the pinned configuration. Roster rule v2 (and any rule that is not a pre-game one): the original text, unchanged. A pre-game roster
    rule (v2nq, pregame_all): the paper record, model version, the form of the rule file (`rule`: the loaded rule file; its `candidate` name, or "frozen rule"), the roster rule and the parity
    report, the waivers LISTED IN THE CONFIGURATION (not a fixed list), and that the record before `effective_from` came from earlier configurations.
    """
    roster = config.get('roster_rule', 'v2')
    if roster not in PREGAME_ROSTER_RULES:
        return DISCLOSURE
    form = f"candidate rule {rule['candidate']}" if isinstance(rule, dict) and rule.get('candidate') else 'frozen rule'
    waivers = [str(w) for w in (config.get('waivers') or [])]
    waived = ('The owner waived: ' + '; '.join(waivers) + '.') if waivers else 'No waiver is recorded for this configuration.'
    since = config.get('effective_from')
    earlier = (f'The record before {since} was produced by earlier configurations (see the segment table on the Tracker page).' if since
               else 'The record before this configuration took effect was produced by earlier configurations (see the segment table on the Tracker page).')
    return (f"Model version {config.get('model_version', 'unknown')}, {form}, roster rule {roster}. Training and live build the teammate features with the same code "
            f'(the teammates are the ones known before the game; parity report {PARITY_REPORT}). {waived} {earlier}')


def load_segments(path=None):
    """The forward-test segment definitions (config/props_forward_segments.json) as a list, or None when the file is missing or unreadable; the exporter copies them into meta.json."""
    p = Path(path) if path is not None else SEGMENTS_PATH
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, list) and data else None


SUMMARY_COLS = ['season', 'kind', 'bets', 'player_games', 'roi_best_stands', 'roi_median_stands', 'roi_flat_stands', 'clv_real_pts', 'clv_placebo_mean_pts', 'clv_excess_pts',
                'clv_excess_lower_pts', 'placebo_roi', 'slope4', 'slope4_lo', 'slope4_hi', 'model_version']
RESULT_COLS = ['minutes_actual', 'y', 'played', 'status_stands', 'status_flagged', 'status_all', 'clv_prob', 'line_move_favour', 'close_line', 'line_240', 'line_60', 'line_5',
               'mkt_240', 'mkt_60', 'mkt_5']
HISTORY_COLS = ['line_240', 'line_60', 'line_5', 'mkt_240', 'mkt_60', 'mkt_5']


def summary_row(settled, season, kind, model_version, n_boot):
    """One season_summary row (settled bets only) for `model_version` ('all' for the combined record); metrics that need a longer record stay blank."""
    row = {'season': season, 'kind': kind, 'bets': 0, 'player_games': 0, 'model_version': model_version}
    if len(settled):
        for mode in ('best', 'median', 'flat'):
            row[f'roi_{mode}_stands'] = logic.roi_summary(settled, mode, 'stands', n_boot)['roi']
        s = logic.roi_summary(settled, 'median', 'stands', 10)
        row.update({'bets': s['bets'], 'player_games': s['player_games'], 'clv_real_pts': float(settled['clv_prob'].mean() * 100) if settled['clv_prob'].notna().any() else np.nan})
    return row


def model_versions_meta(bets):
    """
    meta.json `model_versions`: one dict per model_version in the stored bets, oldest first: version, first_game_date, last_game_date, bets (all bets of the version, settled or not) and
    provenance. Provenance is 'restored from dashboard export 2026-11 (not re-derivable)' when every bet of the version has a NULL snapshot_ts_utc (the rows restored after the 2026-10-31
    data loss carry no snapshot time, quote age or code commit; their results came from the export), else 'recorded live'. A NULL model_version is reported as 'unknown'.
    """
    if bets is None or not len(bets):
        return []
    v = bets['model_version'].fillna('unknown').astype(str) if 'model_version' in bets.columns else pd.Series('unknown', index=bets.index)
    snap = bets['snapshot_ts_utc'] if 'snapshot_ts_utc' in bets.columns else pd.Series(np.nan, index=bets.index)
    out = []
    for ver, g in bets.assign(_v=v, _snap=snap).groupby('_v', sort=False):
        d = g['game_date'].astype(str)
        out.append({'version': ver, 'first_game_date': d.min(), 'last_game_date': d.max(), 'bets': int(len(g)),
                    'provenance': logic.RESTORED_PROVENANCE if g['_snap'].isna().all() else logic.LIVE_PROVENANCE})
    return sorted(out, key=lambda r: (r['first_game_date'], r['version']))


def export_live(live_db, out_dir, config, n_boot=1000, checks=None, current=None, results_now=None, segments=None):
    """`segments`: the forward-test segment definitions copied into meta.json (default: config/props_forward_segments.json; none if that file is missing, and the dashboard then shows no segment table)."""
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
            'expected_minutes', 'created_utc', 'model_version', 'rule_sha256', 'roster_rule', 'snapshot_minutes'] + RESULT_COLS
    b = b[[c for c in keep if c in b.columns]]
    b.to_csv(out / 'bets.csv', index=False)
    (out / 'pmf.json').write_text(json.dumps(pmf, separators=(',', ':')), encoding='utf-8')
    (out / 'books.json').write_text(json.dumps(books, separators=(',', ':')), encoding='utf-8')
    settled = b[b['status_stands'].notna()] if 'status_stands' in b.columns else b.iloc[0:0]
    rows = [summary_row(settled, config['model_season'], 'live paper record (settled bets only)', 'all', n_boot)]       # the combined row stays first (readers take iloc[0])
    vmeta = model_versions_meta(bets)
    vcol = b['model_version'].fillna('unknown') if 'model_version' in b.columns else pd.Series('unknown', index=b.index)
    for ver in vmeta:
        rows.append(summary_row(settled[vcol.loc[settled.index] == ver['version']], config['model_season'], f"live paper record (settled bets only), model {ver['version']}", ver['version'], n_boot))
    pd.DataFrame(rows).reindex(columns=SUMMARY_COLS).to_csv(out / 'season_summary.csv', index=False)
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
    meta = {'mode': 'live', 'rule': rule, 'rule_sha256': config['rule_sha256'], 'code_commit': (bets['code_commit'].dropna().iloc[-1] if 'code_commit' in bets.columns and bets['code_commit'].notna().any() else ''), 'features_db_sha256': config['features_db_sha256'],
            'model_version': config['model_version'], 'model_versions': vmeta, 'roster_rule': config['roster_rule'], 'built_at_utc': pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
            'record_started_utc': bets['created_utc'].min() if len(bets) else None, 'seasons': {config['model_season']: 'live paper record'}, 'disclosure': disclosure(config, rule),
            'pending': int(len(b) - len(settled)), 'pricing': {'median': 'median over the bettable books at the consensus line', 'best': 'best quote among the bettable books', 'flat': 'flat -110'}}
    segs = load_segments() if segments is None else segments
    if segs:
        meta['segments'] = segs                                       # the published app reads the definitions from here, so the bundle needs no extra file
    (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return b
