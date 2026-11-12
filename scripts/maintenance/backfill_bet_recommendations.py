"""
Fill bet_recommendations_log for the games that were recommended before the table existed (2026-10-20 to 2026-11-11), so the dashboard's season record is the
record of the bets the live run recommended and not a recomputation.

  python scripts/maintenance/backfill_bet_recommendations.py                    # DRY RUN (default): prints what it would insert, writes nothing
  python scripts/maintenance/backfill_bet_recommendations.py --apply            # inserts into the database (data/nba_stats.db, or --db)

Inputs: reports/audit_pregame_2026-11/A1_season_bets/bets.csv (the reconstruction of the season's bets: the 24 settled / pending bets of Nov 1-11 are READ from the
printed picks table of data/raw/daily_inference.log, the Oct 20-31 ones are recomputed with team/betting.py), predictions_log_v2 (current line and the model's numbers),
prediction_explanations (the opening line the model read, from the "line movement" pieces; the 14 games of Oct 20-21 have none and use the odds_snapshots opening lines
as a proxy for spread and total, nothing for the moneyline) and calibration_meta<deployed>.json (spread sigma).

One row per game (the one run the table has for it), spread, total and moneyline each with its bet side (or none), edge, current line and the bet side's price, own view and
line move, in the columns core/prediction_logger.BET_LOG_COLUMNS defines. Provenance is NOT in that table: it goes in the small table bet_recommendations_provenance
(prov_ml / prov_spread / prov_total = 'printed' when the decision is read from the daily log's printed picks, 'recomputed' otherwise; open_source; a note with the
price and sigma assumptions), which the dashboard joins on (game_id, run_ts_utc) and shows ("N bets logged, M recomputed").

Stand-ins, all written into each row's note: the total sigma is the one the reconstruction identified from the printed edges (the model's per-game sigma is not stored);
a spread or total price that was not recoverable is NULL (the dashboard then settles it at -110 and marks it); both moneyline prices are the logged ones.

Safety: dry run by default; never updates or deletes; a game that already has a log row written by a real inference run (a row with no provenance entry) is skipped and
reported, never overwritten; running it twice inserts nothing the second time (rows with a provenance entry are skipped).
"""
import argparse
import datetime as dt
import json
import logging
import sqlite3
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from core.prediction_logger import BET_LOG_NAMES, BET_LOG_TABLE, ensure_bet_log_schema
from dashboard.bet_record import PROVENANCE_TABLE, canonical_ids, deployed_meta_path
from team.betting import DEFAULT_PROB_EDGE, calculate_implied_prob, get_ml_bet, get_spread_bet, get_total_bet

A1_DIR = ROOT / 'reports' / 'audit_pregame_2026-11' / 'A1_season_bets'
DEFAULT_FROM, DEFAULT_TO = '2026-10-20', '2026-11-11'
PRINTED_FROM = '2026-11-01'              # daily_inference.log starts on this day: decisions from here on are read from its printed picks table
SERVER_TZ = ZoneInfo('America/Los_Angeles')      # predictions_log_v2.prediction_date is the scheduler machine's local (Pacific) time
PROVENANCE_CREATE_SQL = (f"CREATE TABLE IF NOT EXISTS {PROVENANCE_TABLE} (game_id TEXT, run_ts_utc TEXT, prov_ml TEXT, prov_spread TEXT, prov_total TEXT, "
                         "open_source TEXT, note TEXT, written_utc TEXT, source_file TEXT, PRIMARY KEY (game_id, run_ts_utc))")


def run_ts_utc(prediction_date):
    """The run time of a predictions_log_v2 row in the log's format (UTC, 'YYYY-MM-DDTHH:MM:SS.ffffffZ')."""
    t = pd.Timestamp(prediction_date)
    t = t.tz_localize(SERVER_TZ) if t.tzinfo is None else t
    return t.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def opening_lines(conn, game_ids):
    """
    {game_id: dict(open_spread, open_total, open_p_home_imp, source)}: the opening line the model read, from prediction_explanations (latest explanation of each game):
    the 'movement' piece value is -open home spread, open total, and the implied probability of the opening home price (vig included). Games without explanations:
    source 'none' (the caller may fill spread / total from the odds_snapshots opening rows).
    """
    out = {g: dict(open_spread=np.nan, open_total=np.nan, open_p_home_imp=np.nan, source='none') for g in game_ids}
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='prediction_explanations'").fetchone():
        return out
    e = pd.read_sql("SELECT game_id, prediction_date, market, feature, value FROM prediction_explanations WHERE kind='piece' AND feature='movement'", conn)
    e['game_id'] = canonical_ids(e['game_id']).to_numpy()
    e = e[e['game_id'].isin(set(game_ids))].sort_values('prediction_date').drop_duplicates(['game_id', 'market'], keep='last')
    for r in e.itertuples():
        o = out[r.game_id]
        if r.market == 'spread':
            o['open_spread'] = -r.value
        elif r.market == 'total':
            o['open_total'] = r.value
        elif r.market == 'moneyline':
            o['open_p_home_imp'] = r.value
        o['source'] = 'explanations'
    return out


def snapshot_opening(conn, games):
    """{game_id: (open_spread, open_total)}: cross-book median of each book's first 'opening' odds_snapshots row of the game (a proxy: it matches the true input exactly in
    68% / 49% of games, mean absolute difference 0.4 / 0.7 points; reconstruction A1). Empty when the table is missing."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='odds_snapshots'").fetchone():
        return {}
    dates = sorted(games['game_date'].astype(str).str[:10].unique())
    s = pd.read_sql(f"SELECT fetched_at_et, game_date, home_team, away_team, book, home_spread, total FROM odds_snapshots WHERE line_type='opening' "
                    f"AND game_date IN ({', '.join('?' * len(dates))})", conn, params=dates)
    res = {}
    for r in games.itertuples():
        g = s[(s['game_date'] == str(r.game_date)[:10]) & (s['home_team'] == r.home_team) & (s['away_team'] == r.away_team)].sort_values('fetched_at_et')
        sp = [b['home_spread'].dropna().iloc[0] for _, b in g.groupby('book') if b['home_spread'].notna().any()]
        tt = [b['total'].dropna().iloc[0] for _, b in g.groupby('book') if b['total'].notna().any()]
        res[r.game_id] = (float(np.median(sp)) if sp else np.nan, float(np.median(tt)) if tt else np.nan)
    return res


def _fair_home(home_ml, away_ml):
    ph, pa = calculate_implied_prob(home_ml), calculate_implied_prob(away_ml)
    return ph / (ph + pa)


def build_rows(games, bets, opening, snap_open, spread_std, total_std, printed_from=PRINTED_FROM):
    """
    (log_rows, provenance_rows): one bet_recommendations_log row and one provenance row per game of `games` (predictions_log_v2 rows, game_id canonical).
    `bets` = A1's bets.csv (columns game_id, market, side, edge, price, price_source, source).
    """
    b = bets.copy()
    b['game_id'] = canonical_ids(b['game_id']).to_numpy()
    b['market'] = b['market'].str.lower()
    bet_of = {(r.game_id, r.market): r for r in b.itertuples()}
    now = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    log_rows, prov_rows = [], []
    for g in games.itertuples():
        gid = g.game_id
        printed = str(g.game_date)[:10] >= printed_from
        o = opening.get(gid, {})
        open_sp, open_tt, open_src = o.get('open_spread', np.nan), o.get('open_total', np.nan), o.get('source', 'none')
        if open_src == 'none' and gid in snap_open:                     # games without explanations: the odds_snapshots opening proxy for spread and total
            open_sp, open_tt = snap_open[gid]
            open_src = 'snapshot proxy' if not (np.isnan(open_sp) and np.isnan(open_tt)) else 'none'
        row = {c: None for c in BET_LOG_NAMES}
        row.update(game_id=gid, game_date=str(g.game_date)[:10], run_ts_utc=run_ts_utc(g.prediction_date), model_set_hash=g.model_set_hash,
                   odds_home_spread=None if np.isnan(open_sp) else float(open_sp), odds_total=None if np.isnan(open_tt) else float(open_tt),
                   curr_odds_home_ml=g.odds_home_ml, curr_odds_away_ml=g.odds_away_ml, curr_odds_home_spread=g.odds_home_spread, curr_odds_total=g.odds_total,
                   ml_prob=g.ml_prob, spread_pred=g.spread_pred, total_pred=g.total_pred, spread_std=spread_std, total_std=total_std)
        notes = []
        # ---- moneyline (both prices are the logged ones; no sigma involved)
        ml_bet = bet_of.get((gid, 'ml'))
        if pd.notna(g.ml_prob) and pd.notna(g.odds_home_ml) and pd.notna(g.odds_away_ml):
            side, edge = get_ml_bet({'ml_prob': g.ml_prob, 'odds_home_ml': g.odds_home_ml, 'odds_away_ml': g.odds_away_ml}, DEFAULT_PROB_EDGE)
            row['ml_bet'], row['ml_edge'] = (ml_bet.side if ml_bet is not None else None), (float(ml_bet.edge) if ml_bet is not None else float(edge))
            p_open_imp = o.get('open_p_home_imp', np.nan)
            if pd.notna(p_open_imp):           # constant-overround assumption for the away price at the open (the away opening price is not stored)
                pc_h, pc_a = calculate_implied_prob(g.odds_home_ml), calculate_implied_prob(g.odds_away_ml)
                open_fair = p_open_imp / (p_open_imp + pc_a + (pc_h - p_open_imp))
                row['ml_own_view'] = float(g.ml_prob - open_fair)
                row['ml_line_move'] = float(open_fair - _fair_home(g.odds_home_ml, g.odds_away_ml))
        # ---- spread
        sp_bet = bet_of.get((gid, 'spread'))
        if pd.notna(g.spread_pred) and pd.notna(g.odds_home_spread):
            _, edge = get_spread_bet(g.spread_pred, g.odds_home_spread, spread_std, DEFAULT_PROB_EDGE)
            row['spread_bet'], row['spread_edge'] = (sp_bet.side if sp_bet is not None else None), (float(sp_bet.edge) if sp_bet is not None else float(edge))
            if sp_bet is not None and pd.notna(sp_bet.price):
                row['curr_price_home_spread_odds' if sp_bet.side == 'home' else 'curr_price_away_spread_odds'] = float(sp_bet.price)
                notes.append(f"spread price: {sp_bet.price_source}")
            elif sp_bet is not None:
                notes.append("spread price: not recovered")
            if not np.isnan(open_sp):
                row['spread_own_view'] = float(g.spread_pred + open_sp)
                row['spread_line_move'] = float(open_sp - g.odds_home_spread)
        # ---- total
        tt_bet = bet_of.get((gid, 'total'))
        if pd.notna(g.total_pred) and pd.notna(g.odds_total):
            _, edge = get_total_bet(g.total_pred, g.odds_total, total_std, DEFAULT_PROB_EDGE)
            row['total_bet'], row['total_edge'] = (tt_bet.side if tt_bet is not None else None), (float(tt_bet.edge) if tt_bet is not None else float(edge))
            if tt_bet is not None and pd.notna(tt_bet.price):
                row['curr_price_over_odds' if tt_bet.side == 'over' else 'curr_price_under_odds'] = float(tt_bet.price)
                notes.append(f"total price: {tt_bet.price_source}")
            elif tt_bet is not None:
                notes.append("total price: not recovered")
            if not np.isnan(open_tt):
                row['total_own_view'] = float(g.total_pred - open_tt)
                row['total_line_move'] = float(open_tt - g.odds_total)
        row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        note = "; ".join(notes + [f"opening line: {open_src}", f"total sigma {total_std:.2f} is a stand-in (per-game sigma not stored)",
                                   "non-bet markets: edge recomputed at -110", "source: A1 season-bets reconstruction"])
        prov = 'printed' if printed else 'recomputed'
        log_rows.append(row)
        prov_rows.append(dict(game_id=gid, run_ts_utc=row['run_ts_utc'], prov_ml=prov, prov_spread=prov, prov_total=prov, open_source=open_src, note=note,
                              written_utc=now, source_file='A1_season_bets/bets.csv'))
    return log_rows, prov_rows


def classify_existing(conn, game_ids):
    """(already_backfilled, real_run): game ids that have a provenance row (a previous backfill) and game ids that have a log row with NO provenance row (a real run)."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if BET_LOG_TABLE not in tables:
        return set(), set()
    marks = ', '.join('?' * len(game_ids))
    logged = pd.read_sql(f"SELECT game_id, run_ts_utc FROM {BET_LOG_TABLE} WHERE game_id IN ({marks})", conn, params=list(game_ids))
    if PROVENANCE_TABLE in tables:
        prov = pd.read_sql(f"SELECT game_id, run_ts_utc FROM {PROVENANCE_TABLE}", conn)
        keys = set(zip(prov['game_id'], prov['run_ts_utc']))
    else:
        keys = set()
    back, real = set(), set()
    for r in logged.itertuples():
        (back if (r.game_id, r.run_ts_utc) in keys else real).add(r.game_id)
    return back - real, real


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=str(ROOT / 'data' / 'nba_stats.db'))
    ap.add_argument('--bets', default=str(A1_DIR / 'bets.csv'))
    ap.add_argument('--recon', default=str(A1_DIR / 'recon.json'), help="A1's recon.json: its implied_total_sigma_median is the total sigma stand-in")
    ap.add_argument('--total-sigma', type=float, default=None, help='total sigma stand-in (default: recon.json implied_total_sigma_median, else the model set meta value)')
    ap.add_argument('--date-from', default=DEFAULT_FROM)
    ap.add_argument('--date-to', default=DEFAULT_TO)
    ap.add_argument('--apply', action='store_true', help='write to the database (default: dry run)')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    meta_path = deployed_meta_path(ROOT)
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path else {}
    spread_std = float(meta['spread_std'])
    total_std = args.total_sigma
    if total_std is None:
        try:
            total_std = float(json.loads(Path(args.recon).read_text(encoding='utf-8'))['implied_total_sigma_median'])
        except Exception:
            total_std = float(meta['total_std'])
    bets = pd.read_csv(args.bets, dtype={'game_id': str})

    with sqlite3.connect(args.db) as conn:
        games = pd.read_sql("SELECT * FROM predictions_log_v2 WHERE game_date >= ? AND game_date <= ? ORDER BY game_date, game_id", conn, params=[args.date_from, args.date_to])
        games['game_id'] = canonical_ids(games['game_id']).to_numpy()
        opening = opening_lines(conn, list(games['game_id']))
        snap_open = snapshot_opening(conn, games[[g for g in ('game_id', 'game_date', 'home_team', 'away_team')]])
        log_rows, prov_rows = build_rows(games, bets, opening, snap_open, spread_std, total_std)
        back, real = classify_existing(conn, list(games['game_id']))
        todo = [(r, p) for r, p in zip(log_rows, prov_rows) if r['game_id'] not in back and r['game_id'] not in real]

        print(f"database: {args.db}\ngames {args.date_from}..{args.date_to} in predictions_log_v2: {len(games)}; bets in {Path(args.bets).name}: {len(bets)}")
        print(f"already backfilled (skipped): {len(back)}; written by a real inference run (never overwritten): {len(real)}; to insert: {len(todo)}")
        for m in ('ml', 'spread', 'total'):
            sides = pd.Series([r[f'{m}_bet'] for r, _ in todo]).value_counts().to_dict()
            print(f"  {m}: bets {sum(v for k, v in sides.items())} {sides}")
        print(f"  provenance: {pd.Series([p['prov_ml'] for _, p in todo]).value_counts().to_dict()}; opening line source: {pd.Series([p['open_source'] for _, p in todo]).value_counts().to_dict()}")
        print(f"  spread sigma {spread_std:.3f}, total sigma stand-in {total_std:.3f}")
        if todo:
            show = pd.DataFrame([r for r, _ in todo[:5]])[['game_id', 'game_date', 'run_ts_utc', 'ml_bet', 'ml_edge', 'spread_bet', 'spread_edge', 'total_bet', 'total_edge']]
            print("first rows:\n" + show.to_string(index=False))
        if not args.apply:
            print("DRY RUN: nothing written. Re-run with --apply to insert.")
            return 0
        if not todo:
            print("nothing to insert")
            return 0
        ensure_bet_log_schema(conn)      # creates the table, or adds the columns an older copy lacks (bet_suppressed_reason), before the insert by BET_LOG_NAMES
        conn.execute(PROVENANCE_CREATE_SQL)
        cols, marks = ', '.join(BET_LOG_NAMES), ', '.join('?' * len(BET_LOG_NAMES))
        conn.executemany(f"INSERT INTO {BET_LOG_TABLE} ({cols}) VALUES ({marks})", [[r[c] for c in BET_LOG_NAMES] for r, _ in todo])
        pcols = ['game_id', 'run_ts_utc', 'prov_ml', 'prov_spread', 'prov_total', 'open_source', 'note', 'written_utc', 'source_file']
        conn.executemany(f"INSERT INTO {PROVENANCE_TABLE} ({', '.join(pcols)}) VALUES ({', '.join('?' * len(pcols))})", [[p[c] for c in pcols] for _, p in todo])
        conn.commit()
        print(f"inserted {len(todo)} rows into {BET_LOG_TABLE} and {PROVENANCE_TABLE}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
