"""Health checks of the live pipeline, shown on the dashboard's health page. Each check returns (name, status, detail) with status ok / warn / fail."""
import sqlite3
from pathlib import Path

import pandas as pd


def _age_min(ts, now):
    return (now - pd.Timestamp(ts, tz='UTC')).total_seconds() / 60.0 if ts else float('inf')


def checks(live_conn, odds_db, report_dir, now, games_soon):
    """
    games_soon: whether any game tips within the next 6 hours (the pipeline is only expected to be busy then). Checks the collector, the injury reports, the signal job, missed slots and
    bets waiting too long for a result.
    """
    out = []
    with sqlite3.connect(f'file:{odds_db}?mode=ro', uri=True, timeout=60) as c:
        last_snap = c.execute("SELECT MAX(snapshot_ts_utc) FROM snapshot_log WHERE mode IN ('slow','fast','close')").fetchone()[0]
    age = _age_min(last_snap, now)
    out.append(('odds collector', 'ok' if age <= 45 or not games_soon else 'fail', f'latest live snapshot {age:.0f} minutes ago'))
    pdfs = sorted(Path(report_dir).rglob('*.pdf'), key=lambda p: p.stat().st_mtime)
    rep_age = (now.timestamp() - pdfs[-1].stat().st_mtime) / 60.0 if pdfs else float('inf')
    out.append(('injury reports', 'ok' if rep_age <= 120 or not games_soon else 'warn', f'newest report file {rep_age:.0f} minutes old'))
    row = live_conn.execute("SELECT started_utc, status FROM pipeline_health WHERE job = 'signal' ORDER BY started_utc DESC LIMIT 1").fetchone()
    sig_age = _age_min(row[0], now) if row else float('inf')
    status = 'ok' if (row and sig_age <= 25 and row[1] in ('ok', 'idle')) or not games_soon else ('fail' if not row or sig_age > 25 else 'warn')
    out.append(('signal job', status, f"last run {sig_age:.0f} minutes ago ({row[1] if row else 'never'})"))
    since = (now - pd.Timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
    n = live_conn.execute('SELECT status, count(*) FROM slot_log WHERE written_utc >= ? GROUP BY status', (since,)).fetchall()
    d = dict(n)
    total = sum(d.values())
    missed = d.get('missed', 0) + d.get('no_quotes', 0)
    out.append(('decision slots (7 days)', 'ok' if total == 0 or missed / total <= 0.2 else 'warn', f'{d.get("done", 0)} done, {missed} missed or without quotes of {total}'))
    old = live_conn.execute("SELECT count(*) FROM prop_bets b LEFT JOIN prop_results r USING (bet_id) WHERE r.bet_id IS NULL AND b.created_utc < ?",
                            ((now - pd.Timedelta(hours=36)).strftime('%Y-%m-%dT%H:%M:%SZ'),)).fetchone()[0]
    out.append(('settlement', 'ok' if old == 0 else 'warn', f'{old} bets older than 36 hours are not settled'))
    return pd.DataFrame(out, columns=['check', 'status', 'detail'])
