"""
Data-quality report for the props odds store (Phase 1 gate). Works on any mix of live and historical snapshots and is built to
scale: events are processed in chunks and only additive tables (counts, sums) are kept, so a full 3-season backfill does not
need to fit in memory.

Sections: overview, coverage (season x book, market, player tier vs the box score), snapshot timing relative to tip, missing
closes, quote quality flags and ladders by book, line movement by interval, player-ID matching.
"""
import sqlite3

import numpy as np
import pandas as pd

from core.season_utils import season_for_date
from props import analysis, tiers

MOVE_EDGES = [0, 5, 15, 30, 60, 90, 120, 180, 240, 10_000]       # minutes to tip
CLOSE_MAX_MIN = 12.0                                             # a live "close" must be at most this far from tip
MIN_PLAYED_MIN = 15.0                                            # box-score players counted in tier coverage


def bin_label(m):
    i = np.searchsorted(MOVE_EDGES, m, side='right') - 1
    i = np.clip(i, 0, len(MOVE_EDGES) - 2)
    lo, hi = MOVE_EDGES[i], MOVE_EDGES[i + 1]
    return f'T-{lo}..{hi}' if hi < 10_000 else f'T-{lo}+'


def md_table(df, floatfmt=3):
    if df is None or len(df) == 0:
        return '_none_\n'
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: '' if pd.isna(v) else f'{v:.{floatfmt}f}'.rstrip('0').rstrip('.') if abs(v) < 1e6 else f'{v:,.0f}')
    cols = [str(c) for c in d.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    lines += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in d.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def _chunks(ids, size):
    for i in range(0, len(ids), size):
        yield ids[i:i + size]


def _sum_tables(parts):
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts)
    return t.groupby(level=list(range(t.index.nlevels))).sum()


def chunk_tables(conn, event_ids, events, log, form, nba_conn, method='multiplicative'):
    ph = ','.join('?' * len(event_ids))
    p = pd.read_sql(f"SELECT * FROM prop_snapshots WHERE event_id_vendor IN ({ph})", conn, params=list(event_ids))
    valid = log.loc[log['in_game'] == 0, ['snapshot_ts_utc', 'event_id_vendor']]      # drop in-game snapshots
    p = p.merge(valid, on=['snapshot_ts_utc', 'event_id_vendor'], how='inner')
    if p.empty:
        return {}
    ev = events.set_index('event_id_vendor')
    p['season'] = p['event_id_vendor'].map(ev['season'])
    p['et_game_date'] = p['event_id_vendor'].map(ev['et_game_date'])
    f = analysis.flag_main_lines(analysis.add_devig(p, method))
    out = {}
    pe = f.drop_duplicates(['event_id_vendor', 'book', 'player_name_raw', 'market'])
    out['cov_book'] = pd.concat([
        f.groupby(['season', 'book']).size().rename('rows'),
        f.groupby(['season', 'book'])['event_id_vendor'].nunique().rename('events'),
        pe.groupby(['season', 'book']).size().rename('player_markets')], axis=1)
    out['cov_market'] = pd.concat([
        f.groupby(['season', 'market']).size().rename('rows'),
        f.drop_duplicates(['event_id_vendor', 'player_name_raw', 'market']).groupby(['season', 'market']).size()
        .rename('player_markets')], axis=1)
    g = f.groupby('book')
    out['flags'] = pd.concat([
        g.size().rename('rows'), g['one_sided'].sum().rename('one_sided'), g['no_margin'].sum().rename('no_margin'),
        g['wide'].sum().rename('wide'), g['stale'].sum().rename('stale'), g['is_ladder'].sum().rename('ladder_rows'),
        g['is_main_line'].sum().rename('main_rows')], axis=1)
    # --- line movement between consecutive snapshots of the same event (consensus of >= 2 books)
    c = analysis.consensus(f, min_books=2)
    c = c.merge(log[['snapshot_ts_utc', 'event_id_vendor', 'minutes_to_tip']], on=['snapshot_ts_utc', 'event_id_vendor'])
    c = c.sort_values(['event_id_vendor', 'player_name_raw', 'market', 'minutes_to_tip'], ascending=[True, True, True, False])
    k = ['event_id_vendor', 'player_name_raw', 'market']
    c['prev_line'] = c.groupby(k)['consensus_line'].shift()
    c['prev_min'] = c.groupby(k)['minutes_to_tip'].shift()
    c = c.dropna(subset=['prev_line'])
    if len(c):
        c['d'] = (c['consensus_line'] - c['prev_line']).abs()
        c['window'] = [bin_label((a + b) / 2) for a, b in zip(c['prev_min'], c['minutes_to_tip'])]   # by interval midpoint
        c['hours'] = (c['prev_min'] - c['minutes_to_tip']) / 60.0
        c['moved'] = (c['d'] > 0).astype(int)
        out['move'] = c.groupby(['market', 'window']).agg(n=('d', 'size'), moved=('moved', 'sum'), sum_abs=('d', 'sum'),
                                                          hours=('hours', 'sum'))
    # --- selection: who played (box score) vs who was ever priced
    if nba_conn is not None:
        evg = events[events['event_id_vendor'].isin(event_ids) & events['game_id'].notna()]
        if len(evg):
            gph = ','.join('?' * len(evg))
            box = pd.read_sql(f"SELECT gameId AS game_id, personId AS player_id, minutes FROM player_basic_stats "
                              f"WHERE gameId IN ({gph}) AND minutes >= {MIN_PLAYED_MIN}", nba_conn,
                              params=list(evg['game_id']))
            box['game_id'] = box['game_id'].astype(str).str.zfill(10)
            box = box.merge(evg[['event_id_vendor', 'game_id', 'et_game_date']], on='game_id')
            priced = set(zip(p['event_id_vendor'], p['player_id'].dropna().astype(int)))
            box['priced'] = [(e, int(pl)) in priced for e, pl in zip(box['event_id_vendor'], box['player_id'])]
            box = tiers.assign_tiers(box, date_col='et_game_date', form=form)
            out['tier'] = box.groupby('tier').agg(players=('priced', 'size'), priced=('priced', 'sum'))
    return out


def build_report(props_db, nba_db=None, chunk_events=150, method='multiplicative'):
    conn = sqlite3.connect(props_db)
    events = pd.read_sql("SELECT * FROM events", conn)
    if events.empty:
        return '# Props odds data quality\n\nNo events in the store yet.\n'
    events['season'] = [season_for_date(d) if d else None for d in events['et_game_date']]
    log = pd.read_sql("SELECT * FROM snapshot_log", conn)
    log['kind'] = np.where(log['mode'].str.startswith('hist_'), 'historical', 'live')
    ref_note = ''
    if nba_db:
        try:
            from props import schedule
            seasons = sorted({s for s in events['season'].dropna()})
            sched = pd.concat([schedule.load_schedule(s) for s in seasons], ignore_index=True)
            schedule.annotate(conn, sched, schedule.team_id_map(nba_db))
        except Exception as e:                                    # offline: fall back to whatever annotation exists
            ref_note = f'_Could not refresh the NBA schedule ({type(e).__name__}); using the existing reference-clock annotation._\n'
    has_ref = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'snapshot_ref'").fetchone() is not None
    if has_ref:
        ref = pd.read_sql("SELECT snapshot_ts_utc, event_id_vendor, minutes_to_ref, in_game FROM snapshot_ref", conn)
        log = log.merge(ref, on=['snapshot_ts_utc', 'event_id_vendor'], how='left')
        log['minutes_to_tip'] = log['minutes_to_ref'].fillna(log['minutes_to_tip'])
    log['in_game'] = log['in_game'].fillna(0).astype(int) if 'in_game' in log else 0
    evref = pd.read_sql("SELECT * FROM event_ref", conn) if has_ref else pd.DataFrame()
    nba_conn = sqlite3.connect(f'file:{nba_db}?mode=ro', uri=True) if nba_db else None
    form = tiers.trailing_form(nba_db) if nba_db else None
    ids = list(events['event_id_vendor'])
    parts = [chunk_tables(conn, c, events, log, form, nba_conn, method) for c in _chunks(ids, chunk_events)]
    T = {k: _sum_tables([p.get(k) for p in parts]) for k in ('cov_book', 'cov_market', 'flags', 'move', 'tier')}

    md = ['# Props odds data quality\n']
    # 1. overview
    n_rows = conn.execute("SELECT COUNT(*), COUNT(player_id) FROM prop_snapshots").fetchone()
    ov = pd.DataFrame({
        'events': [len(events)], 'with_game_id': [int(events['game_id'].notna().sum())],
        'snapshots': [len(log)], 'prop_rows': [n_rows[0]],
        'player_id_matched': [n_rows[1] / max(n_rows[0], 1)],
        'first_game_date': [events['et_game_date'].min()], 'last_game_date': [events['et_game_date'].max()]})
    md += ['## 1. Overview\n', md_table(ov)]
    if ref_note:
        md += [ref_note]
    ks = log.groupby('kind').agg(snapshots=('mode', 'size'), events=('event_id_vendor', 'nunique'),
                                 snapshot_credits=('credit_cost', 'sum'))
    md += [md_table(ks.reset_index())]
    # 2. coverage
    cb = T['cov_book'].reset_index()
    ev_per_season = events.groupby('season').size().rename('season_events')
    cb = cb.merge(ev_per_season, on='season')
    cb['share_of_events'] = cb['events'] / cb['season_events']
    md += ['## 2. Coverage\n', '### By season and book (share_of_events = events with at least one quote / events in the store)\n',
           md_table(cb[['season', 'book', 'events', 'share_of_events', 'player_markets', 'rows']]),
           '### By season and market\n', md_table(T['cov_market'].reset_index())]
    if len(T['tier']):
        t = T['tier'].reset_index()
        t['share_priced'] = t['priced'] / t['players']
        md += [f'### Selection: box-score players with >= {MIN_PLAYED_MIN:g} minutes, by trailing-form tier, and the share '
               'that had a line in any snapshot (games with a resolved game id only)\n', md_table(t),
               '_Tiers: star = trailing 20-game mean points >= 20; starter = mean minutes >= 24; rotation = the rest '
               '(games strictly before the date). Unpriced players are the missing-line selection the backtests must respect._\n']
    # 3. timing
    h = log[(log.kind == 'historical') & (log.in_game == 0)].copy()
    if len(h):
        h['requested'] = h['mode'].str.replace('hist_T-', '', regex=False).astype(int)
        h['early_by'] = h['minutes_to_tip'] - h['requested']
        tm = h.groupby('requested').agg(snapshots=('mode', 'size'), median_actual_min=('minutes_to_tip', 'median'),
                                        median_early_by=('early_by', 'median'), max_early_by=('early_by', 'max'),
                                        empty=('n_prop_rows', lambda s: int((s == 0).sum()))).reset_index()
        md += ['## 3. Snapshot timing\n', '### Historical: requested minutes before tip vs the snapshot actually returned '
               '(history is on a 5-minute grid and returns the latest snapshot at or before the requested instant)\n',
               md_table(tm)]
    lv = log[(log.kind == 'live') & (log.in_game == 0)].sort_values(['event_id_vendor', 'minutes_to_tip'], ascending=[True, False])
    if len(lv):
        gaps = lv.groupby('event_id_vendor')['minutes_to_tip'].apply(lambda s: s.diff().abs().max())
        per = lv.groupby('event_id_vendor').agg(n=('mode', 'size'), first=('minutes_to_tip', 'max'),
                                                last=('minutes_to_tip', 'min'))
        per['max_gap'] = gaps
        md += ['### Live: per-game snapshot count, first and last minutes to tip, largest gap\n',
               md_table(per.describe().loc[['mean', 'min', '50%', 'max']].reset_index().rename(columns={'index': 'stat'}))]
    # 3b. clock integrity
    n_in = int((log['in_game'] == 1).sum())
    rev = int((evref['vendor_tip_revisions'] > 0).sum()) if len(evref) else 0
    big = 0
    if len(evref):
        gap = (pd.to_datetime(evref['vendor_tip_first_utc'], utc=True) - pd.to_datetime(evref['tip_ref_utc'], utc=True)
               ).dt.total_seconds().abs() / 60
        big = int((gap > 15).sum())
    unmatched_games = int(len(events) - evref['game_id'].notna().sum()) if len(evref) else len(events)
    md += ['## 3b. Clock integrity (reference tip = NBA scheduled tip + 0 to 10 minutes)\n',
           f"- snapshots at or after the reference tip (in-game), excluded everywhere below: **{n_in}** of {len(log)}",
           f"- games whose vendor tip time changed between snapshots: **{rev}** of {len(events)}",
           f"- games whose first vendor tip time differs from the reference by more than 15 minutes: **{big}**",
           f"- games not matched to the NBA schedule: **{unmatched_games}**\n"]
    # 4. missing closes
    last = log.groupby('event_id_vendor')['minutes_to_tip'].min()
    live_ev = set(lv['event_id_vendor']) if len(lv) else set()
    miss_live = [e for e in live_ev if last[e] > CLOSE_MAX_MIN]
    hist_ev = set(h['event_id_vendor']) if len(h) else set()
    has5 = set(h[h['requested'] <= 10]['event_id_vendor']) if len(h) else set()
    empty_close = set(h[(h['requested'] <= 10) & (h['n_prop_rows'] == 0)]['event_id_vendor']) if len(h) else set()
    md += ['## 4. Missing closes\n',
           f"- live games without a snapshot within {CLOSE_MAX_MIN:g} minutes of tip: **{len(miss_live)}** of {len(live_ev)}",
           f"- historical games without a closing-side (<= T-10) snapshot: **{len(hist_ev - has5)}** of {len(hist_ev)}",
           f"- historical games whose closing-side snapshot came back empty: **{len(empty_close)}**\n"]
    # 5. quote quality
    fl = T['flags'].reset_index()
    for c in ('one_sided', 'no_margin', 'wide', 'stale', 'ladder_rows', 'main_rows'):
        fl[c + '_share'] = fl[c] / fl['rows']
    md += ['## 5. Quote quality by book (shares of rows)\n',
           md_table(fl[['book', 'rows', 'one_sided_share', 'no_margin_share', 'wide_share', 'stale_share',
                        'ladder_rows_share', 'main_rows']]),
           '_Ladder rows = quotes from a book that posted more than one line for that player and market in the same snapshot; '
           'analysis uses the main line (closest to 50%) and keeps the rest._\n']
    # 6. movement
    if len(T['move']):
        mv = T['move'].reset_index()
        mv['share_moved'] = mv['moved'] / mv['n']
        mv['mean_abs_move'] = mv['sum_abs'] / mv['n']
        mv['abs_move_per_hour'] = mv['sum_abs'] / mv['hours']
        mv['_o'] = mv['window'].str.extract(r'^T-(\d+)')[0].astype(float)
        mv = mv.sort_values(['market', '_o'], ascending=[True, False])
        md += ['## 6. Consensus line movement (>= 2 books; consecutive snapshots of the same game, binned by the interval '
               'midpoint in minutes before tip; abs_move_per_hour = total |change| / total hours)\n',
               md_table(mv[['market', 'window', 'n', 'share_moved', 'mean_abs_move', 'abs_move_per_hour']])]
    # 7. matching
    un = pd.read_sql("SELECT player_name_raw, team_context, reason, n_rows FROM unmatched_names ORDER BY n_rows DESC", conn)
    md += ['## 7. Player-ID matching\n', f"- matched rows: {n_rows[1]:,} of {n_rows[0]:,} "
           f"({n_rows[1] / max(n_rows[0], 1):.2%}); unmatched names: {len(un)}\n", md_table(un.head(25))]
    return '\n'.join(md)
