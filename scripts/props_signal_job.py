"""
The live signal job (D1.3). Scheduled every 10 minutes; for every game whose decision slot (T-240, T-60, T-5 before the reference tip) has arrived and has no log row, it builds the
point-in-time features, predicts with the pinned live model, applies the frozen rule and writes the slot's signals and its first-qualification bets in one transaction.

    python scripts/props_signal_job.py                  # live
    python scripts/props_signal_job.py --dry-run        # evaluate and print, write nothing
    python scripts/props_signal_job.py --now 2026-10-23T22:00:00Z --dry-run

Refuses to run (exit 2, health row 'refused') if the rule file, the model bundle or the features database differ from reports/props/live_config.json.
"""
import argparse
import json
import logging
import pickle
import sqlite3
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props import schedule as S  # noqa: E402
from props.features import build, history, rows as rows_mod  # noqa: E402
from props.live import at_t as A, config, quotes, reports as live_reports_mod, signal, store  # noqa: E402
from props.models import registry  # noqa: E402

SLOTS = (240, 60, 5)
HISTORY_LAG_HOURS = 5.0            # games that began more than this long ago must be in the box-score tables
GAME_PREFIXES = ('002', '004')


def utc_now():
    return pd.Timestamp.now(tz='UTC')


def iso(ts):
    return pd.Timestamp(ts).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


def events_in_window(odds_db, now, hours_back=1.0, hours_ahead=7.0):
    lo, hi = iso(now - pd.Timedelta(hours=hours_back)), iso(now + pd.Timedelta(hours=hours_ahead))
    with sqlite3.connect(f'file:{odds_db}?mode=ro', uri=True, timeout=120) as conn:
        ev = pd.read_sql('SELECT event_id_vendor, home_team, away_team, commence_time_utc FROM events WHERE commence_time_utc BETWEEN ? AND ?', conn, params=[lo, hi])
        if ev.empty:
            return ev
        first = pd.read_sql(f"SELECT event_id_vendor, MIN(commence_time_utc) AS first_tip FROM prop_snapshots WHERE snapshot_ts_utc >= ? AND event_id_vendor IN ({','.join('?' * len(ev))}) "
                            'GROUP BY event_id_vendor', conn, params=[iso(now - pd.Timedelta(days=3)), *ev['event_id_vendor']])
    ev = ev.merge(first, on='event_id_vendor', how='left')
    ev['commence_time_utc'] = ev['first_tip'].fillna(ev['commence_time_utc'])      # the reference clock uses the vendor's earliest estimate, as in the backtest
    return ev.drop(columns='first_tip')


def stale_history(hist, sched, now):
    """Games that began more than HISTORY_LAG_HOURS ago and are missing from the box-score history: the features of later games would be wrong."""
    s = sched[sched['game_id'].str[:3].isin(GAME_PREFIXES)]
    due = s[pd.to_datetime(s['nba_tip_utc'], utc=True) < now - pd.Timedelta(hours=HISTORY_LAG_HOURS)]
    return sorted(set(due['game_id']) - set(hist['game_id'].astype(str)))


def play_tables_cached(features_db, season):
    path = Path(f'data/processed/live_play_tables_{season}.pkl')
    key = (Path(features_db).stat().st_size, season)
    if path.exists():
        k, t = pickle.load(open(path, 'rb'))
        if k == key:
            return t
    t = A.play_tables(features_db, build.GRID, live_season=season)
    pickle.dump((key, t), open(path, 'wb'))
    return t


def due_items(conn, odds_db, sched, matched, now, season, log):
    items = []
    for g in matched.itertuples(index=False):
        if not isinstance(g.game_id, str):
            continue
        srow = sched[sched['game_id'] == g.game_id].iloc[0].copy()
        srow['tip_ref_utc'], srow['season'], srow['game_date'] = g.tip_ref_utc, season, srow['et_game_date']
        now_m = quotes.minutes_before_tip(iso(now), g.tip_ref_utc)
        snaps = quotes.event_snapshots(odds_db, g.event_id_vendor, since_utc=iso(now - pd.Timedelta(hours=8)))
        sm = pd.Series({ts: quotes.minutes_before_tip(ts, g.tip_ref_utc) for ts in snaps['snapshot_ts_utc']}, dtype=float)
        for T in SLOTS:
            if store.slot_done(conn, g.game_id, T):
                continue
            status, ts, m = quotes.choose_snapshot(sm, T, now_m)
            if status == 'wait':
                continue
            if status == 'missed':
                log.append((g.game_id, T, 'missed', {'minutes_to_tip': round(now_m, 1), 'snapshots': len(sm)}, None, None))
                continue
            cons, bets, raw = quotes.slot_quotes(odds_db, g.event_id_vendor, g.game_id, ts, T)
            if cons is None or cons.empty:
                log.append((g.game_id, T, 'no_quotes', {'snapshot': ts}, None, None))
                continue
            names = dict(zip(raw['player_id'].astype(int), raw['player_name_raw']))
            items.append(signal.SlotItem(game=srow, slot=T, snapshot_ts_utc=ts, snapshot_minutes=m, event_id=g.event_id_vendor, cons=cons, bets=bets, names=names))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=store.DEFAULT_PATH)
    ap.add_argument('--odds-db', default='data/props_odds.db')
    ap.add_argument('--nba-db', default='data/nba_stats.db')
    ap.add_argument('--config', default=config.CONFIG_PATH)
    ap.add_argument('--now', default=None, help='UTC instant to act as (testing); default: the clock')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    now = pd.Timestamp(a.now, tz='UTC') if a.now else utc_now()
    run_id = now.strftime('%Y%m%dT%H%M%SZ')
    conn = store.connect(a.db)
    started = iso(utc_now())
    job = 'signal'
    try:
        cfg = config.verify(config.load(a.config))
    except (config.ConfigRefused, FileNotFoundError) as e:
        logging.error(f'refused: {e}')
        store.health(conn, run_id, job, started, iso(utc_now()), 'refused', 0, {'reason': str(e)})
        return 2
    try:
        season = cfg['model_season']
        sched = S.load_schedule(season)
        sched = sched[sched['game_id'].str[:3].isin(GAME_PREFIXES)].reset_index(drop=True)
        ev = events_in_window(a.odds_db, now)
        matched = S.match_games(ev, sched, S.team_id_map(a.nba_db)) if len(ev) else ev.assign(game_id=None, tip_ref_utc=None)
        log = []
        items = due_items(conn, a.odds_db, sched, matched, now, season, log)
        if not items and not log:
            store.health(conn, run_id, job, started, iso(utc_now()), 'idle', 0, {'events': len(ev)})
            logging.info(f'nothing due ({len(ev)} events in the window)')
            return 0
        written = {'signals': 0, 'bets': 0, 'slots': 0}
        if items:
            t0 = time.time()
            hist = rows_mod.load_history(a.nba_db)
            hist['game_id'] = hist['game_id'].astype(str)
            missing = stale_history(hist, sched, now)
            if missing:
                msg = f'{len(missing)} games that began more than {HISTORY_LAG_HOURS:.0f} hours ago are not in the box-score history (first: {missing[:3]}); slots are not worked'
                logging.error(msg)
                store.health(conn, run_id, job, started, iso(utc_now()), 'history_stale', 0, {'missing': missing[:20]})
                return 3
            ctx = signal.Context(
                hist=hist, pos=history.position_groups(a.nba_db), reports=live_reports_mod.live_reports(a.nba_db, sched, season, str((now - pd.Timedelta(days=2)).date())),
                tips=build.reference_tips_et([season]), tables=play_tables_cached(cfg['features_db'], season), bundle=registry.load_bundle(cfg['model_version'], cfg['model_season']),
                rule=json.loads(Path(cfg['rule_file']).read_text(encoding='utf-8')), nba_db=a.nba_db, n_sims=cfg['n_sims'], model_version=cfg['model_version'],
                provenance={'model_sha256': cfg['model_bundle_sha256'], 'features_db_sha256': cfg['features_db_sha256'], 'rule_sha256': cfg['rule_sha256'],
                            'roster_rule': cfg['roster_rule'], 'code_commit': config.code_commit()}, roster_rule=cfg['roster_rule'])
            logging.info(f'context ready in {time.time() - t0:.0f}s; {len(items)} slots due')
            by_date = {}
            for it in sorted(items, key=lambda i: (i.game['game_date'], -i.slot)):
                by_date.setdefault(it.game['game_date'], []).append(it)
            for date, group in by_date.items():
                frame, unresolved = signal.build_frames(ctx, date, group)
                for it in group:
                    existing = store.existing_bet_keys(conn, [it.game['game_id']])
                    sig, new, detail = signal.evaluate_item(ctx, it, frame, existing)
                    detail.update({'snapshot_ts_utc': it.snapshot_ts_utc, 'snapshot_minutes': round(it.snapshot_minutes, 1), 'bets': int(len(new)), 'signals': int(len(sig))})
                    print(f"{it.game['game_id']} T-{it.slot}: {len(sig)} markets evaluated, {len(new)} new bets" + (' [dry run]' if a.dry_run else ''))
                    if len(new):
                        print(new[['player_name', 'stat', 'side', 'line', 'book', 'dec_best', 'dec_median', 'ev_best', 'ev_median']].round(3).to_string(index=False))
                    if not a.dry_run:
                        ns, nb = store.write_slot(conn, it.game['game_id'], it.slot, 'done', iso(utc_now()), run_id, sig, new, detail)
                        written['signals'] += ns
                        written['bets'] += nb
                        written['slots'] += 1
        if not a.dry_run:
            for gid, T, status, detail, _, _ in log:
                store.write_slot(conn, gid, T, status, iso(utc_now()), run_id, detail=detail)
            store.health(conn, run_id, job, started, iso(utc_now()), 'ok', written['bets'], {**written, 'missed_or_no_quotes': len(log)})
        return 0
    except Exception as e:                       # noqa: BLE001 - the job must leave a health row whatever happens
        logging.error(traceback.format_exc())
        store.health(conn, run_id, job, started, iso(utc_now()), 'error', 0, {'error': repr(e)[:500]})
        return 1


if __name__ == '__main__':
    sys.exit(main())
