"""
Historical backfill planner and runner.

Clock: every snapshot instant is (NBA scheduled tip + 0 to 10 min, see schedule.ref_offset_min) - offset, from props/schedule.py. The vendor's historical tip time is not
used for timing (it was an hour wrong for one pilot game). The vendor is asked for the snapshot at or before that instant; history is
on a 5-minute grid, so the snapshot returned is 0 to 5 minutes earlier than requested.

Cost (The Odds API): 10 credits x markets x regions per event-snapshot (markets counted when data comes back, so this is an upper
bound) + 1 credit per ET game date for the events listing used to find the vendor's event ids.

plan() is pure: it prints what would be fetched and the upper-bound cost, subtracting what is already in the store (resumable).
run() refuses to start unless the caller passes back the exact planned credit figure.
"""
import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from core.odds_vendor.base import BudgetExceeded
from core.odds_vendor.theoddsapi import OddsApiError
from core.timeutil import to_utc_iso
from props import collect, schedule

GAME_ID_PREFIXES = ('002', '003', '004', '005')       # regular season, play-in, playoffs, NBA Cup knockout rounds
DEFAULT_OFFSETS = (240, 120, 60, 30, 5)
LISTING_LEAD_MIN = 60                                  # events listing taken this long before the day's first tip


@dataclass
class Plan:
    games: pd.DataFrame                  # game_id, et_game_date, nba_tip_utc, tip_ref_utc, season
    offsets: tuple
    n_markets: int
    n_regions: int
    todo: dict = field(default_factory=dict)       # game_id -> offsets still to fetch
    dates_todo: list = field(default_factory=list)

    @property
    def snapshot_count(self):
        return sum(len(v) for v in self.todo.values())

    @property
    def credits(self):
        return 10 * self.n_markets * self.n_regions * self.snapshot_count + len(self.dates_todo)


def select_games(schedule_df, start, end, prefixes=GAME_ID_PREFIXES):
    g = schedule_df[schedule_df['game_id'].str[:3].isin(prefixes)].copy()
    g = g[(g['et_game_date'] >= start) & (g['et_game_date'] <= end)]
    g['tip_ref_utc'] = [schedule.tip_ref_utc(t) for t in g['nba_tip_utc']]
    return g.sort_values('nba_tip_utc').reset_index(drop=True)


def done_snapshots(conn):
    """{(game_id, offset)} already stored (mode hist_T-<offset>). Game ids come from events.game_id or event_ref.game_id."""
    ref = _has_table(conn, 'event_ref')
    gid = 'COALESCE(r.game_id, e.game_id)' if ref else 'e.game_id'
    join = 'LEFT JOIN event_ref r USING (event_id_vendor)' if ref else ''
    rows = conn.execute(f"SELECT {gid}, l.mode FROM snapshot_log l JOIN events e USING (event_id_vendor) {join} "
                        f"WHERE l.mode LIKE 'hist_T-%'").fetchall()
    return {(g, int(m.replace('hist_T-', ''))) for g, m in rows if g}


def _has_table(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def plan(schedule_df, start, end, offsets=DEFAULT_OFFSETS, n_markets=len(collect.ALL_MARKET_KEYS), n_regions=1,
         done=frozenset(), limit_games=None, prefixes=GAME_ID_PREFIXES):
    games = select_games(schedule_df, start, end, prefixes)
    if limit_games:
        games = games.head(limit_games)
    todo = {}
    for gid in games['game_id']:
        left = tuple(o for o in offsets if (gid, o) not in done)
        if left:
            todo[gid] = left
    dates = sorted(games.loc[games['game_id'].isin(todo), 'et_game_date'].unique())
    return Plan(games=games, offsets=tuple(offsets), n_markets=n_markets, n_regions=n_regions, todo=todo, dates_todo=dates)


def summarize(p: Plan):
    g = p.games.copy()
    g['season'] = [_season(d) for d in g['et_game_date']]
    g['todo'] = g['game_id'].map(lambda x: len(p.todo.get(x, ())))
    s = g.groupby('season').agg(games=('game_id', 'size'), snapshots_todo=('todo', 'sum'))
    s['credits_upper_bound'] = 10 * p.n_markets * p.n_regions * s['snapshots_todo']
    return s


def _season(d):
    from core.season_utils import season_for_date
    return season_for_date(d)


def run(vendor, conn, p: Plan, team_ids, markets=None, regions='us', bookmakers=None, sleep_s=0.1, progress_every=1,
        max_consecutive_errors=10):
    """Fetch the plan. Safe to rerun. Returns a dict of counters; BudgetExceeded propagates after committing what was stored.
    Stops with RuntimeError after max_consecutive_errors API errors in a row (bad key, exhausted quota, vendor outage)."""
    markets = markets or collect.ALL_MARKET_KEYS
    consecutive = 0
    stats = {'snapshots': 0, 'empty': 0, 'errors': 0, 'unmatched_events': 0, 'credits': 0.0}
    by_date = p.games[p.games['game_id'].isin(p.todo)].groupby('et_game_date')
    for n, (date, day) in enumerate(by_date, 1):
        first_tip = pd.Timestamp(day['nba_tip_utc'].min())
        listing_ts = to_utc_iso(first_tip - pd.Timedelta(minutes=LISTING_LEAD_MIN))
        try:
            listing = vendor.historical_events(listing_ts)
        except OddsApiError as e:
            logging.error(f'{date}: events listing failed ({e}); skipping the date')
            stats['errors'] += 1
            continue
        ev = pd.DataFrame(listing.payload or [])
        if ev.empty:
            logging.warning(f'{date}: the vendor listed no events at {listing_ts}')
            stats['unmatched_events'] += len(day)
            continue
        ev = ev.rename(columns={'id': 'event_id_vendor', 'commence_time': 'commence_time_utc'})
        matched = schedule.match_games(ev[['event_id_vendor', 'home_team', 'away_team', 'commence_time_utc']],
                                       day, team_ids)
        # reference tip per game: NBA scheduled tip + the vendor's own offset when plausible (see schedule.ref_offset_min)
        by_game = {r.game_id: (r.event_id_vendor, r.tip_ref_utc) for r in matched.itertuples(index=False)
                   if isinstance(r.game_id, str)}
        for r in day.itertuples(index=False):
            eid, tip_ref = by_game.get(r.game_id, (None, None))
            if eid is None:
                logging.warning(f'{date}: no vendor event for {r.game_id} ({r.away_team} @ {r.home_team})')
                stats['unmatched_events'] += 1
                continue
            for off in p.todo[r.game_id]:
                instant = to_utc_iso(pd.Timestamp(tip_ref) - pd.Timedelta(minutes=off))
                try:
                    resp = vendor.historical_event_odds(eid, instant, markets, regions=regions, bookmakers=bookmakers)
                except OddsApiError as e:
                    logging.error(f'{r.game_id} T-{off}: {e}')
                    stats['errors'] += 1
                    consecutive += 1
                    if consecutive >= max_consecutive_errors:
                        raise RuntimeError(f'{consecutive} API errors in a row (last: {e}); stopping. Stored snapshots are kept; '
                                           f'rerun to resume.') from e
                    continue
                consecutive = 0
                if not resp.payload:
                    stats['empty'] += 1
                    continue
                collect.store_event_odds(conn, resp, mode=f'hist_T-{off}')
                stats['snapshots'] += 1
                stats['credits'] += resp.cost or 0
                time.sleep(sleep_s)
        if n % progress_every == 0:
            logging.info(f"{date}: done ({n}/{by_date.ngroups} dates; {stats['snapshots']} snapshots, "
                         f"{stats['credits']:g} credits, {stats['errors']} errors)")
    return stats
