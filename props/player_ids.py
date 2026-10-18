"""
Vendor player names -> nba_api PLAYER_ID (personId), and vendor events -> NBA game ids.

Deterministic, in this order (the first rule that gives exactly one player wins, and the rule is recorded):
  override        props/player_overrides.csv (name_raw -> player_id), reviewed by hand
  team_exact      normalised name equals a player whose latest team before the game date is one of the two teams
  team_variant    same, after dropping suffixes (Jr., III) and punctuation differences on both sides
  unique_name     the normalised name belongs to exactly one player in the whole league history (rookie / new signing
                  with no game for his team yet); lower confidence, recorded as such
  fuzzy_team      difflib ratio >= 0.93 against the two teams' players, one candidate only
Anything else is left unmatched with a reason. Unmatched names are logged and counted, never dropped from the odds
store (their player_id stays NULL).
"""
import csv
import difflib
import logging
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

OVERRIDES_PATH = Path(__file__).resolve().parent / 'player_overrides.csv'
SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
FUZZY_MIN = 0.93
NEXT_GAME_DAYS = 21


def normalize(name):
    """'P.J. Washington Jr.' -> 'pj washington jr'; accents folded, apostrophes and periods removed, hyphens -> space."""
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    s = s.lower().replace("'", '').replace('.', '').replace('-', ' ')
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z ]', '', s)).strip()


def strip_suffix(norm):
    parts = norm.split()
    return ' '.join(p for i, p in enumerate(parts) if not (i > 0 and p in SUFFIXES))


def load_overrides(path=OVERRIDES_PATH):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline='', encoding='utf-8') as f:
        return {normalize(r['name_raw']): int(r['player_id']) for r in csv.DictReader(f) if r.get('player_id')}


class PlayerMatcher:
    def __init__(self, nba_db_path, overrides_path=OVERRIDES_PATH):
        with sqlite3.connect(f'file:{nba_db_path}?mode=ro', uri=True) as conn:
            games = pd.read_sql("SELECT personId, playerName, teamId, gameDate FROM player_basic_stats", conn)
            info = pd.read_sql("SELECT PERSON_ID AS personId, DISPLAY_FIRST_LAST AS playerName FROM player_info", conn)
            teams = pd.read_sql("SELECT DISTINCT teamId, teamName, teamTricode FROM team_basic_stats", conn)
        games['gameDate'] = pd.to_datetime(games['gameDate']).dt.strftime('%Y-%m-%d')
        self.games = games.sort_values('gameDate')[['personId', 'playerName', 'teamId', 'gameDate']]
        # (personId, date) -> latest team before a date is resolved with searchsorted on per-player arrays
        self._hist = {pid: (g['gameDate'].to_numpy(), g['teamId'].to_numpy())
                      for pid, g in self.games.groupby('personId')}
        names = pd.concat([games[['personId', 'playerName']], info]).drop_duplicates()
        self._by_norm = {}
        for pid, nm in names.itertuples(index=False):
            self._by_norm.setdefault(normalize(nm), set()).add(int(pid))
        self._by_variant = {}
        for n, ids in self._by_norm.items():
            self._by_variant.setdefault(strip_suffix(n), set()).update(ids)
        self._pid_names = {int(pid): normalize(nm) for pid, nm in names.itertuples(index=False)}
        self.teams = {}
        for tid, name, tri in teams.itertuples(index=False):
            self.teams[normalize(name)] = int(tid)
            self.teams[normalize(tri)] = int(tid)
        self.teams.setdefault('los angeles clippers', self.teams.get('la clippers'))
        self.overrides = load_overrides(overrides_path)

    def team_id(self, team_name):
        return self.teams.get(normalize(team_name))

    def team_on(self, player_id, date):
        """The team of the player's most recent game strictly before `date` (None if he has no game yet)."""
        hist = self._hist.get(int(player_id))
        if hist is None:
            return None
        dates, tids = hist
        i = dates.searchsorted(str(date), side='left')       # games before date
        return int(tids[i - 1]) if i > 0 else None

    def team_next(self, player_id, date, within_days=NEXT_GAME_DAYS):
        """The team of the player's first game on/after `date`, if that game is within `within_days` days (debuts, new signings).
        Used only to link a vendor name to a player; it never feeds a model feature."""
        hist = self._hist.get(int(player_id))
        if hist is None:
            return None
        dates, tids = hist
        i = dates.searchsorted(str(date), side='left')
        if i >= len(dates):
            return None
        gap = (pd.Timestamp(dates[i]) - pd.Timestamp(str(date))).days
        return int(tids[i]) if gap <= within_days else None

    def _on_teams(self, ids, date, team_ids):
        return sorted(p for p in ids if self.team_on(p, date) in team_ids)

    def match(self, name_raw, date, team_ids):
        """Returns (player_id or None, method_or_reason). team_ids: the two NBA team ids of the event."""
        n = normalize(name_raw)
        if n in self.overrides:
            return self.overrides[n], 'override'
        team_ids = {t for t in team_ids if t is not None}
        hit = self._on_teams(self._by_norm.get(n, ()), date, team_ids)
        if len(hit) == 1:
            return hit[0], 'team_exact'
        hit = self._on_teams(self._by_variant.get(strip_suffix(n), ()), date, team_ids)
        if len(hit) == 1:
            return hit[0], 'team_variant'
        if len(hit) > 1:
            return None, 'ambiguous_on_teams'
        league = self._by_variant.get(strip_suffix(n), set())
        if len(league) == 1:
            return next(iter(league)), 'unique_name'
        if len(league) > 1:
            return None, 'ambiguous_league'
        cands = [p for p in self._pid_names if self.team_on(p, date) in team_ids or self.team_next(p, date) in team_ids]
        return self._roster_rules(n, cands, name_raw)

    def _roster_rules(self, n, cands, name_raw):
        """Looser rules, applied only to the two teams' players as of the game date; each needs exactly one candidate."""
        vt = strip_suffix(n).split()
        toks = {p: strip_suffix(self._pid_names[p]).split() for p in cands}

        def last(p):
            return toks[p][-1] if toks[p] else ''

        def first(p):
            return toks[p][0] if toks[p] else ''

        def related(a, b):          # a is a prefix of b or the reverse, at least 2 letters ('cam' / 'cameron')
            return min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a))

        rules = [
            ('team_token_set', lambda p: set(toks[p]) == set(vt)),                       # reversed order, 'Last, First'
            ('team_prefix_last', lambda p: last(p) in vt and any(t != last(p) and related(t, first(p)) for t in vt)),
            ('team_initial_last', lambda p: last(p) in vt and bool(first(p)) and vt[0][:1] == first(p)[:1]),
        ]
        ambiguous = False
        for method, test in rules:
            hit = [p for p in cands if test(p)]
            if len(hit) == 1:
                return hit[0], method
            ambiguous |= len(hit) > 1
        scored = sorted(((difflib.SequenceMatcher(None, ' '.join(vt), ' '.join(toks[p])).ratio(), p) for p in cands),
                        reverse=True)
        close = [p for r, p in scored if r >= FUZZY_MIN]
        if len(close) == 1:
            logging.warning(f"fuzzy player match: '{name_raw}' -> {self._pid_names[close[0]]} ({close[0]})")
            return close[0], 'fuzzy_team'
        hit = [p for p in cands if last(p) in vt]                                        # nickname for a first name
        if len(hit) == 1:
            return hit[0], 'team_last_name_only'
        return None, 'ambiguous' if (ambiguous or len(close) > 1 or len(hit) > 1) else 'no_candidate'


def resolve_game_id(nba_db_path, et_game_date, home_team_id):
    """NBA gameId for a played game (team_basic_stats has one row per team per game; the home row says 'X vs. Y')."""
    with sqlite3.connect(f'file:{nba_db_path}?mode=ro', uri=True) as conn:
        row = conn.execute("SELECT gameId FROM team_basic_stats WHERE teamId = ? AND substr(gameDate, 1, 10) = ? "
                           "AND MATCHUP LIKE '% vs. %' LIMIT 1", (int(home_team_id), et_game_date)).fetchone()
    return str(row[0]).zfill(10) if row else None


def ensure_indexes(conn):
    """Partial indexes over the unresolved rows only, so resolution scales with what is pending, not with the table size."""
    conn.execute("CREATE INDEX IF NOT EXISTS ix_props_pid_null ON prop_snapshots (event_id_vendor, player_name_raw) "
                 "WHERE player_id IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_props_gid_null ON prop_snapshots (event_id_vendor) WHERE game_id IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_team_gid_null ON team_lines (event_id_vendor) WHERE game_id IS NULL")


def has_pending(conn):
    """True if any quote still lacks a player id or any event/quote lacks a game id (cheap: uses the partial indexes)."""
    ensure_indexes(conn)
    q = ("SELECT 1 FROM prop_snapshots WHERE player_id IS NULL LIMIT 1",
         "SELECT 1 FROM prop_snapshots WHERE game_id IS NULL LIMIT 1",
         "SELECT 1 FROM team_lines WHERE game_id IS NULL LIMIT 1",
         "SELECT 1 FROM events WHERE game_id IS NULL LIMIT 1")
    return any(conn.execute(s).fetchone() for s in q)


def resolve_store(store_conn, matcher, nba_db_path, now_iso=None, resolve_games=True):
    """
    Fill events.game_id / team ids and prop_snapshots.player_id (and team_lines.game_id) for rows still NULL.
    Returns a summary dict; unmatched names are written to unmatched_names (with counts), never dropped.
    """
    from core.timeutil import et_game_date, now_utc_iso
    now_iso = now_iso or now_utc_iso()
    ensure_indexes(store_conn)
    store_conn.execute("""CREATE TABLE IF NOT EXISTS player_match_audit (
        player_name_raw TEXT NOT NULL, player_id INTEGER NOT NULL, method TEXT, team_context TEXT, matched_utc TEXT,
        PRIMARY KEY (player_name_raw, player_id, team_context))""")
    ev = pd.read_sql("SELECT * FROM events", store_conn)
    for r in ev.itertuples(index=False):
        h, a = matcher.team_id(r.home_team), matcher.team_id(r.away_team)
        d = r.et_game_date or et_game_date(r.commence_time_utc)
        gid = r.game_id or (resolve_game_id(nba_db_path, d, h) if (resolve_games and h) else None)
        store_conn.execute("UPDATE events SET et_game_date=?, home_team_id=?, away_team_id=?, game_id=? "
                           "WHERE event_id_vendor=?", (d, h, a, gid, r.event_id_vendor))
    store_conn.execute("UPDATE team_lines SET game_id = (SELECT game_id FROM events e WHERE e.event_id_vendor = "
                       "team_lines.event_id_vendor) WHERE game_id IS NULL")
    store_conn.execute("UPDATE prop_snapshots SET game_id = (SELECT game_id FROM events e WHERE e.event_id_vendor = "
                       "prop_snapshots.event_id_vendor) WHERE game_id IS NULL")
    todo = pd.read_sql("""SELECT p.player_name_raw, p.event_id_vendor, COUNT(*) n FROM prop_snapshots p
                          WHERE p.player_id IS NULL GROUP BY 1, 2""", store_conn)
    evi = ev.set_index('event_id_vendor')
    matched = {}
    unmatched = {}
    for name, eid, n in todo.itertuples(index=False):
        e = evi.loc[eid]
        h, a = matcher.team_id(e['home_team']), matcher.team_id(e['away_team'])
        d = e['et_game_date'] or et_game_date(e['commence_time_utc'])
        pid, how = matcher.match(name, d, (h, a))
        if pid is not None:
            store_conn.execute("UPDATE prop_snapshots SET player_id=? WHERE player_name_raw=? AND event_id_vendor=? "
                               "AND player_id IS NULL", (int(pid), name, eid))
            matched[how] = matched.get(how, 0) + n
            if how not in ('team_exact', 'team_variant', 'override'):        # looser rules: keep them reviewable
                store_conn.execute("INSERT OR REPLACE INTO player_match_audit VALUES (?,?,?,?,?)",
                                   (name, int(pid), how, f"{e['away_team']} @ {e['home_team']}", now_iso))
        else:
            ctx = f"{e['away_team']} @ {e['home_team']}"
            unmatched[(name, ctx)] = (how, n)
    # the table mirrors what is unmatched NOW: drop names that have since been resolved (override added, roster data caught up)
    keep = set(unmatched)
    for name, ctx in store_conn.execute('SELECT player_name_raw, team_context FROM unmatched_names').fetchall():
        if (name, ctx) not in keep:
            store_conn.execute('DELETE FROM unmatched_names WHERE player_name_raw=? AND team_context=?', (name, ctx))
    for (name, ctx), (how, n) in unmatched.items():
        store_conn.execute("""INSERT INTO unmatched_names VALUES (?,?,?,?,?,?)
                              ON CONFLICT(player_name_raw, team_context) DO UPDATE SET reason=excluded.reason,
                              n_rows=excluded.n_rows, last_seen_utc=excluded.last_seen_utc""",
                           (name, ctx, how, n, now_iso, now_iso))
    store_conn.commit()
    total_m, total_u = sum(matched.values()), sum(n for _, n in unmatched.values())
    if unmatched:
        logging.warning(f"{len(unmatched)} unmatched player-event names ({total_u} rows), e.g. "
                        f"{[k[0] for k in list(unmatched)[:5]]}")
    return {'matched_rows': total_m, 'unmatched_rows': total_u, 'by_method': matched,
            'match_rate': total_m / max(total_m + total_u, 1)}
