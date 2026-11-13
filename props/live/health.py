"""Health checks of the live pipeline, shown on the dashboard's health page. Each check returns (name, status, detail) with status ok / warn / fail."""
import json
import sqlite3
from pathlib import Path

import pandas as pd


def _age_min(ts, now):
    return (now - pd.Timestamp(ts, tz='UTC')).total_seconds() / 60.0 if ts else float('inf')


SLOTS = (240, 60, 5)
MISSED_SHARE_MAX = 0.2           # per slot: (missed + no_quotes) / all logged slots of that time
CLOSING_AGE_MEDIAN_MAX = 15.0    # minutes before tip: the snapshot served as T-5 is expected about 10 minutes before tip


def slot_checks(live_conn, since):
    """
    Two rows for the dashboard: the decision slots of the window by nominal time, and the age of the snapshot behind the T-5 slots.
    'decision slots (7 days)' warns when ANY slot time (240, 60, 5) has a missed-or-no-quotes share above MISSED_SHARE_MAX, so one structurally failing slot cannot hide behind the others;
    the detail names every slot with its done / missed / no_quotes counts. 'closing snapshot age' warns when the median snapshot_minutes of the done T-5 slots exceeds CLOSING_AGE_MEDIAN_MAX.
    """
    rows = live_conn.execute('SELECT slot_minutes, status, detail FROM slot_log WHERE written_utc >= ?', (since,)).fetchall()
    by = {}
    for slot, status, _ in rows:
        by.setdefault(int(slot), {'done': 0, 'missed': 0, 'no_quotes': 0, 'other': 0})[status if status in ('done', 'missed', 'no_quotes') else 'other'] += 1
    parts, bad = [], []
    for slot in sorted(set(by) | set(SLOTS), reverse=True):
        c = by.get(slot)
        if c is None:
            parts.append(f'T-{slot}: no slots')
            continue
        total = sum(c.values())
        lost = c['missed'] + c['no_quotes']
        parts.append(f'T-{slot}: {c["done"]} done, {c["missed"]} missed, {c["no_quotes"]} no_quotes of {total}')
        if lost / total > MISSED_SHARE_MAX:
            bad.append(f'T-{slot} {lost / total:.0%} missed or without quotes')
    detail = '; '.join(parts)
    out = [('decision slots (7 days)', 'warn' if bad else 'ok', (f'WARN {", ".join(bad)} (limit {MISSED_SHARE_MAX:.0%}): ' if bad else '') + detail)]
    ages = []
    for slot, status, det in rows:
        if int(slot) == 5 and status == 'done':
            try:
                ages.append(float(json.loads(det)['snapshot_minutes']))
            except (TypeError, ValueError, KeyError):
                continue
    if ages:
        med = float(pd.Series(ages).median())
        status = 'warn' if med > CLOSING_AGE_MEDIAN_MAX else 'ok'
        out.append(('closing snapshot age', status, f'T-5 slots done: {len(ages)}; snapshot taken a median of {med:.1f} and at most {max(ages):.1f} minutes before tip (expected about 10, limit {CLOSING_AGE_MEDIAN_MAX:.0f})'))
    else:
        out.append(('closing snapshot age', 'ok', 'no T-5 slot done in the window'))
    return out


def checks(live_conn, odds_db, report_dir, now, games_soon, nba_db=None, season=None):
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
    out.extend(slot_checks(live_conn, since))
    old = live_conn.execute("SELECT count(*) FROM prop_bets b LEFT JOIN prop_results r USING (bet_id) WHERE r.bet_id IS NULL AND b.created_utc < ?",
                            ((now - pd.Timedelta(hours=36)).strftime('%Y-%m-%dT%H:%M:%SZ'),)).fetchone()[0]
    out.append(('settlement', 'ok' if old == 0 else 'warn', f'{old} bets older than 36 hours are not settled'))
    if nba_db and season:                               # box-score tables that lack a game the others have (features silently wrong or, for tracking stats, that game dropped from the rates)
        try:
            from props.live import freshness
            gaps = freshness.missing_stat_tables(nba_db, season)
            shown = ', '.join(f'{g} {t}' for g, t in gaps[:6]) + (f' and {len(gaps) - 6} more' if len(gaps) > 6 else '')
            out.append(('box-score tables', 'warn' if gaps else 'ok', f'{len(gaps)} game-table gaps: {shown}' if gaps else 'every game is in all four tables'))
        except Exception as e:                           # noqa: BLE001 - a health row must never stop the export
            out.append(('box-score tables', 'warn', f'check failed: {e!r}'[:200]))
    return pd.DataFrame(out, columns=['check', 'status', 'detail'])
