import pandas as pd
import numpy as np
import sqlite3
import logging
import datetime
import sys
import os
import time
from pathlib import Path
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.prediction_logger import PredictionLogger, SKIPPED_TABLE, log_bet_recommendations, log_skipped_games
from core.odds_rerun import unpriced_games
from core.odds_fallback import SOURCE, fetch_team_odds, find_game
from team.live_odds import MARKETS, bet_log_row, split_open_and_current
from team.live_features import align_features, build_live_game_frame
from team.betting import (get_ml_bet, get_spread_bet, get_total_bet, DEFAULT_SPREAD_STD,
                            DEFAULT_TOTAL_STD)
from team.model_registry import DEPLOYED_SUFFIX, MODEL_DIR, load_bundle
from team.market_models import predict_markets
from team.explain import build_reference, explain_markets, load_oos_history, save_explanations
from team.data_paths import PROCESSED_DIR, latest_training_set
from core import injury_report_features as irf
import core.scrape_injury_reports as scrape_injury_reports
from team.game_features_v2 import GameFeatureBuilder
from core.update_database import NBADatabaseSync
from nba_api.stats.endpoints import scoreboardv3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from core.season_utils import season_for_date



# ---------------------------------------------------------------- exit codes (scripts/run_daily_inference.bat documents what it does for each)
EXIT_OK = 0
EXIT_CRASH = 1             # an unexpected exception (a programming or data-format error); the traceback is in the log
EXIT_TOO_EARLY = 3         # the injury-cutoff guard refused: started hours before the report the model needs exists
EXIT_NO_INPUTS = 4         # games are scheduled but nothing could be predicted: odds source or NBA schedule unavailable, or every game skipped
EXIT_BETS_SUPPRESSED = 5   # predictions made and logged, but no bet recommended: the injury report was missing or stale

# ---------------------------------------------------------------- Eastern time
def now_et():
    """US/Eastern wall-clock time (naive), looked up in core.scrape_injury_reports at call time so tests can replace it in either module."""
    return scrape_injury_reports.now_et()


# Every date decision of the job (which slate is "today", which season, the injury cutoff) is made in US/Eastern time with now_et(),
# never with the machine's local clock (Pacific on the production machine; 21:00-24:00 Pacific is already tomorrow in Eastern).
# The ONE deliberate exception is the `prediction_date` column of predictions_log_v2 (and its CSV mirror, and the explanations'
# prediction_date): it stays machine-local, in its old format, because the published dashboard reads it. Use
# bet_recommendations_log.run_ts_utc (UTC, 'Z') as the unambiguous run timestamp.


def current_season_et():
    return season_for_date(now_et().date())


current_season = current_season_et()


class ScheduleFetchFailure(Exception):
    """The NBA scoreboard could not be read (as opposed to being read and listing no games)."""


class OddsSourceFailure(Exception):
    """Games are scheduled but SBR gave nothing usable after the retries."""


# Retries for the free sources (NBA scoreboard, SBR): 3 attempts, a short pause between them.
SOURCE_ATTEMPTS = 3
SOURCE_BACKOFF_SECONDS = (10, 30)


def call_with_retries(fn, what, attempts=SOURCE_ATTEMPTS, backoff_seconds=SOURCE_BACKOFF_SECONDS, sleep_fn=None):
    """
    fn() up to `attempts` times; any exception is a failed attempt, logged at WARNING; sleep_fn(backoff_seconds[i]) between attempts
    (time.sleep unless a test injects one). Returns fn()'s value, or re-raises the last exception. Only for FETCH calls: the
    processing of what was fetched is never inside a retry, so a programming error there is not retried or hidden.
    """
    sleep_fn = sleep_fn or time.sleep
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            logging.warning(f"{what}: attempt {i + 1}/{attempts} failed ({type(e).__name__}: {e})")
            if i == attempts - 1:
                raise
        sleep_fn(backoff_seconds[min(i, len(backoff_seconds) - 1)])


def _read_nba_schedule(game_date):
    """The scoreboard for game_date as a frame (gameId, gameDate, gameStatus, gameTimeUTC, home/away tricodes); EMPTY if it lists no games. Raises on any failure."""
    games = scoreboardv3.ScoreboardV3(game_date=game_date).get_data_frames()[1]
    if games.empty:                                      # a successful response with no games: an off day
        return pd.DataFrame()
    # UTC time might be next day for late games: use the requested (Eastern) game_date.
    games['gameDate'] = game_date
    games['away_team_abbr'] = games['gameCode'].str[-6:-3]
    games['home_team_abbr'] = games['gameCode'].str[-3:]
    return games[['gameId', 'gameDate', 'gameStatus', 'gameTimeUTC', 'home_team_abbr', 'away_team_abbr']]


def get_nba_schedule(game_date=None):
    """
    Get today's games (Eastern date by default) with official NBA gameIds.
    Returns DataFrame with gameId, home_team, away_team, gameDate. Lenient: a failed fetch is logged and returns an EMPTY frame, the
    same as an off day. The daily run uses fetch_nba_schedule, which tells the two apart; this one serves the injury-cutoff guard.
    """
    if game_date is None:
        game_date = now_et().strftime('%Y-%m-%d')
    try:
        return _read_nba_schedule(game_date)
    except Exception as e:
        logging.error(f"Failed to fetch NBA schedule: {e}")
        return pd.DataFrame()


def fetch_nba_schedule(game_date=None, sleep_fn=None):
    """
    Strict version of get_nba_schedule: an empty frame means the NBA lists no games (a clean off day); a fetch that fails on all
    SOURCE_ATTEMPTS tries raises ScheduleFetchFailure.
    """
    if game_date is None:
        game_date = now_et().strftime('%Y-%m-%d')
    try:
        return call_with_retries(lambda: _read_nba_schedule(game_date), f"NBA schedule for {game_date}", sleep_fn=sleep_fn)
    except Exception as e:
        raise ScheduleFetchFailure(f"{type(e).__name__}: {e}") from e


# ---------------------------------------------------------------- injury-cutoff guard
# The model is trained on the injury report as of the snapshot irf.SNAPSHOT_CLOCK[GameFeatureBuilder.INJURY_SNAPSHOT]
# (game-day noon ET, capped at tip - 30 min per game: irf.SNAPSHOT_CUTOFFS). get_injury_features reads the report as of
# min(now, that cutoff), so a run before the cutoff silently reads an older report. The guard makes the run wait for it.

# How long after its nominal time a report can be downloaded. Evidence (2026-10-20..2026-11-11): the half-hourly snapshot job
# saved 371 of 456 report files within 0.14-1.63 minutes (median 0.2) of the slot time in the file name, and the PDF's header
# time equals the slot time in this era; the other files were bulk-restored later or rewritten by the daily run, not unavailable.
# The noon slot itself was saved 0.1-0.4 min after 12:00 on each of the 11 days 2026-11-01..11. Five minutes is three times the
# slowest prompt download seen.
REPORT_PUBLICATION_ALLOWANCE = pd.Timedelta(minutes=5)
# A run earlier than this before the cutoff is a scheduling error, not something to sleep through. 90, not 120: the scheduled task's
# execution time limit is 2 hours and the run itself (sync, features, predictions, publish) needs time after the wait.
DEFAULT_MAX_WAIT_MINUTES = 90
WAIT_STEP_SECONDS = 30
# The report used may be this much older than the snapshot's nominal time and still be "the training report": through 2024-25
# the hourly files carry a header time 30 minutes after their slot, so training's noon report was typically the one published
# 11:30. One grid step (60 min) would let the 11:00 report through at noon, which is exactly the failure being guarded.
STALE_SNAPSHOT_TOLERANCE = pd.Timedelta(minutes=30)


def tip_times_et(game_time_utc):
    """gameTimeUTC strings from the NBA scoreboard -> naive US/Eastern Timestamps (NaT where missing or unparsable)."""
    t = pd.to_datetime(pd.Series(list(game_time_utc)), utc=True, errors='coerce')
    return t.dt.tz_convert('America/New_York').dt.tz_localize(None)


def snapshot_nominal_cutoff(game_date, tips_et=None, snapshot=GameFeatureBuilder.INJURY_SNAPSHOT):
    """
    The latest per-game training cutoff of the slate (ET): irf.SNAPSHOT_CUTOFFS[snapshot], i.e. game date + clock, capped at tip - 30 min
    per game. A game with an unknown tip uses the clock time. None when the snapshot has no clock and no tip is known.
    """
    tips = list(tips_et) if tips_et is not None and len(tips_et) else [pd.NaT]
    g = pd.DataFrame({'game_date': pd.Timestamp(game_date).normalize(), 'tip_ts': pd.to_datetime(pd.Series(tips))})
    cutoffs = pd.Series(irf.SNAPSHOT_CUTOFFS[snapshot](g)).dropna()
    return None if cutoffs.empty else cutoffs.max()


def earliest_run_time(game_date, tips_et=None, allowance=REPORT_PUBLICATION_ALLOWANCE, snapshot=GameFeatureBuilder.INJURY_SNAPSHOT):
    """
    Earliest ET time at which the report the training snapshot reads can be fetched for today's slate: the latest game's cutoff plus the
    publication allowance. With normal evening tips that is noon + allowance; when every game's own cutoff (tip - 30 min) is earlier than
    the clock time, it is the latest of those cutoffs + allowance, so the run never waits past what the slate needs. None: no constraint.
    """
    nominal = snapshot_nominal_cutoff(game_date, tips_et, snapshot)
    return None if nominal is None else nominal + allowance


def injury_cutoff_guard(wait=True, max_wait_minutes=DEFAULT_MAX_WAIT_MINUTES, schedule_fn=None, now_fn=None, sleep_fn=None,
                        allowance=REPORT_PUBLICATION_ALLOWANCE, step_seconds=WAIT_STEP_SECONDS):
    """
    Run-time guard for a normal daily run. Returns 'ready' (nothing to wait for), 'waited', 'ignored' (too early but wait=False: the old
    behaviour, with a warning) or 'too_early' (the wait would exceed max_wait_minutes: nothing should be predicted or logged).
    The slate (tip times) comes from the NBA scoreboard; if it cannot be fetched the clock rule (noon + allowance) is used.
    """
    now_fn = now_fn or now_et
    sleep_fn = sleep_fn or time.sleep
    schedule_fn = schedule_fn or get_nba_schedule
    now = pd.Timestamp(now_fn())
    game_date = now.normalize()
    tips = None
    if wait:         # without the wait the slate is not needed
        try:
            sched = schedule_fn(game_date.strftime('%Y-%m-%d'))
            if sched is not None and len(sched) and 'gameTimeUTC' in sched.columns:
                tips = tip_times_et(sched['gameTimeUTC'])
        except Exception as e:
            logging.warning(f"Injury cutoff guard: could not fetch today's slate ({e})")
        if tips is None:
            logging.info("Injury cutoff guard: today's slate is not available; using the clock rule "
                         f"({irf.SNAPSHOT_CLOCK[GameFeatureBuilder.INJURY_SNAPSHOT]} after midnight ET + publication allowance)")
    ready_at = earliest_run_time(game_date, tips, allowance)
    if ready_at is None or now >= ready_at:
        logging.info(f"Injury cutoff guard: {now:%H:%M} ET; the {GameFeatureBuilder.INJURY_SNAPSHOT} report is readable "
                     f"(from {'n/a' if ready_at is None else format(ready_at, '%H:%M')} ET). Proceeding.")
        return 'ready'
    wait_minutes = (ready_at - now) / pd.Timedelta(minutes=1)
    if not wait:
        logging.warning(f"Injury cutoff guard DISABLED (--no-wait-for-injury-cutoff): it is {now:%H:%M} ET, the {GameFeatureBuilder.INJURY_SNAPSHOT} "
                        f"report is readable from about {ready_at:%H:%M} ET. Proceeding now: the injury features will come from an older "
                        f"report than the training snapshot's.")
        return 'ignored'
    if wait_minutes > max_wait_minutes:
        logging.error(f"RUN TOO EARLY: it is {now:%Y-%m-%d %H:%M} ET and the {GameFeatureBuilder.INJURY_SNAPSHOT} injury report the model was trained on "
                      f"is readable from {ready_at:%H:%M} ET ({wait_minutes:.0f} min away, more than the {max_wait_minutes:g} min maximum). "
                      f"Nothing was predicted or logged. Check the task's fire time (it must be after {ready_at:%H:%M} ET), or use "
                      f"--max-wait-minutes / --no-wait-for-injury-cutoff.")
        return 'too_early'
    logging.info(f"Injury cutoff guard: it is {now:%H:%M} ET; waiting {wait_minutes:.1f} min until {ready_at:%H:%M} ET for the "
                 f"{GameFeatureBuilder.INJURY_SNAPSHOT} injury report to be published.")
    while True:
        remaining = (ready_at - pd.Timestamp(now_fn())) / pd.Timedelta(seconds=1)
        if remaining <= 0:
            break
        sleep_fn(min(step_seconds, remaining))
    logging.info(f"Injury cutoff guard: wait over, it is {pd.Timestamp(now_fn()):%H:%M} ET. Proceeding.")
    return 'waited'


def expected_grid_slot(nominal):
    """The latest report slot on the training grid (irf.GRID_SLOTS) at or before the snapshot time, or None."""
    day = pd.Timestamp(nominal).normalize()
    slots = sorted(day + pd.Timedelta(hours=h, minutes=m) for h, m in irf.GRID_SLOTS)
    before = [s for s in slots if s <= pd.Timestamp(nominal)]
    return before[-1] if before else None


def snapshot_report_problem(latest, nominal, tolerance=STALE_SNAPSHOT_TOLERANCE):
    """
    None if the newest report at/before the cutoff (`latest`, its published time) is the one the training snapshot would read: published no
    more than `tolerance` before the nominal snapshot time. Otherwise the number of minutes it is behind. `latest` None (no report at all)
    is reported separately by check_report_freshness.
    """
    if latest is None or nominal is None:
        return None
    behind = (pd.Timestamp(nominal) - pd.Timestamp(latest)) / pd.Timedelta(minutes=1)
    return behind if behind > tolerance / pd.Timedelta(minutes=1) else None


# ---------------------------------------------------------------- team names
# SBR's abbreviations for teams whose tricode differs from the NBA's (applied to the schedule-merge keys, and accepted as team names).
SBR_ABBREVIATIONS = {'BK': 'BKN', 'CHAR': 'CHA', 'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'PHO': 'PHX', 'SA': 'SAS'}
# Explicit spellings besides the full names and tricodes in team_basic_stats -> tricode. Everything the old fuzzy lookup could reach
# by exact match is here: the LA spellings, the SBR abbreviations, every nickname (the old "last word of the team name" keys) and the
# old Charlotte name. A string that is not here (after normalising case, punctuation and spaces) is NOT guessed: the game is skipped
# as unmapped_team.
TEAM_NAME_ALIASES = {
    'LA Clippers': 'LAC', 'L.A. Clippers': 'LAC', 'LA Lakers': 'LAL', 'L.A. Lakers': 'LAL',
    'Hawks': 'ATL', 'Celtics': 'BOS', 'Nets': 'BKN', 'Hornets': 'CHA', 'Bobcats': 'CHA', 'Bulls': 'CHI', 'Cavaliers': 'CLE',
    'Mavericks': 'DAL', 'Nuggets': 'DEN', 'Pistons': 'DET', 'Warriors': 'GSW', 'Rockets': 'HOU', 'Pacers': 'IND', 'Clippers': 'LAC',
    'Lakers': 'LAL', 'Grizzlies': 'MEM', 'Heat': 'MIA', 'Bucks': 'MIL', 'Timberwolves': 'MIN', 'Pelicans': 'NOP', 'Knicks': 'NYK',
    'Thunder': 'OKC', 'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHX', 'Trail Blazers': 'POR', 'Blazers': 'POR', 'Kings': 'SAC',
    'Spurs': 'SAS', 'Raptors': 'TOR', 'Jazz': 'UTA', 'Wizards': 'WAS',
    **SBR_ABBREVIATIONS,
}


def normalize_team_name(name):
    """Lower case with everything but letters and digits removed: 'L.A.  Clippers' and 'la clippers' are the same string."""
    import re
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def build_team_lookup(team_info, aliases=TEAM_NAME_ALIASES):
    """
    {normalised name: teamId} over the full team names and tricodes in `team_info` (teamId, teamName, teamTricode: team_basic_stats,
    which holds both 'LA Clippers' / 'Los Angeles Clippers' and 'Charlotte Bobcats' / 'Hornets') plus the explicit aliases.
    Raises ValueError if one normalised name would stand for two teams.
    """
    lookup, by_tricode = {}, {}

    def add(key, team_id):
        k = normalize_team_name(key)
        if k and lookup.setdefault(k, team_id) != team_id:
            raise ValueError(f"team name {key!r} (normalised {k!r}) maps to two team ids: {lookup[k]} and {team_id}")

    for r in team_info.itertuples(index=False):
        team_id = int(r.teamId)
        by_tricode[r.teamTricode] = team_id
        add(r.teamName, team_id)
        add(r.teamTricode, team_id)
    for alias, tricode in aliases.items():
        if tricode in by_tricode:
            add(alias, by_tricode[tricode])
    return lookup


def map_team_id(name, lookup):
    """The team id for an SBR team name by exact lookup on the normalised name, or None. Never a partial or fuzzy match."""
    if not isinstance(name, str):
        return None
    return lookup.get(normalize_team_name(name))


# ---------------------------------------------------------------- SBR fetch
def fetch_sbr_boards(game_date, sleep_fn=None, attempts=SOURCE_ATTEMPTS, backoff_seconds=SOURCE_BACKOFF_SECONDS, scoreboard_cls=None):
    """
    (current-line games, opening-line games) SBR lists for game_date. sbrscrape swallows HTTP errors (core/sbr.py), so an outage
    mostly shows up as an EMPTY board: an empty current board, or a fetch that raises, is retried (`attempts` tries, sleep_fn(backoff)
    between them) and then raises OddsSourceFailure. An empty opening board is retried too but accepted on the last try: the games
    are then skipped one by one as missing_opening_line. Free calls only; The Odds API is not involved.
    """
    if scoreboard_cls is None:
        from core.sbr import Scoreboard as scoreboard_cls
    sleep_fn = sleep_fn or time.sleep
    reason = 'no attempt made'
    for i in range(attempts):
        try:
            curr = list(scoreboard_cls(sport='NBA', date=game_date, current_line=True).games or [])
            if not curr:
                reason = 'SBR listed no games on the current-line board'
            else:
                opened = list(scoreboard_cls(sport='NBA', date=game_date, current_line=False).games or [])
                if opened or i == attempts - 1:
                    return curr, opened
                reason = 'SBR listed no games on the opening-line board'
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        logging.warning(f"SBR fetch for {game_date}: attempt {i + 1}/{attempts} failed ({reason})")
        if i < attempts - 1:
            sleep_fn(backoff_seconds[min(i, len(backoff_seconds) - 1)])
    raise OddsSourceFailure(reason)


class InferenceEngine:
    def __init__(self, db_path, suffix=DEPLOYED_SUFFIX, model_dir=MODEL_DIR, target_date=None, only_unpriced=False):
        self.db_path = db_path
        self.only_unpriced = only_unpriced  # re-run: predict only the games that have not started and still lack a line
        self.target_date = target_date  # Optional: override datetime.now() for backfilling
        self.injury_cutoff = None       # set by get_injury_features: the report cutoff used and the report file fetched (bet log)
        self.injury_report_path = None

        # One explicit, validated model set (team/model_registry.py): all pickles, calibrator and
        # the feature config share `suffix`, and the config is checked against each pickle.
        bundle = load_bundle(suffix, model_dir)
        self.bundle = bundle
        self.model_ml = bundle.ml
        self.calibrator_ml = bundle.calibrator
        self.model_spread = bundle.spread
        self.model_total = bundle.total
        self.selected_features = bundle.features
        self.calib_meta = bundle.calib_meta

        # Initialize Logger
        self.logger = PredictionLogger(db_path) 
        logging.info(f"Inference Engine initialized with model set '{suffix}'.")
        
    def _skip(self, reason, game_date, home_team, away_team, game_id=None, detail=''):
        """Record a game this run does not predict (written to inference_skipped_games by run())."""
        rec = {'game_date': game_date, 'game_id': game_id, 'home_team': home_team, 'away_team': away_team, 'reason': reason, 'detail': detail}
        self.skipped_games.append(rec)
        return rec

    def team_lookup(self):
        """{normalised team name: teamId} from team_basic_stats plus TEAM_NAME_ALIASES (build_team_lookup)."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                # Columns are likely camelCase now: teamId, teamName, teamTricode
                team_info = pd.read_sql("SELECT DISTINCT teamId, teamName, teamTricode FROM team_basic_stats", conn)
            except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
                logging.warning(f"team_basic_stats has no camelCase columns ({e}); trying the upper-case names")
                # Fallback try original uppercase just in case
                team_info = pd.read_sql("SELECT DISTINCT TEAM_ID as teamId, TEAM_NAME as teamName, TEAM_ABBREVIATION as teamTricode FROM team_basic_stats", conn)
        return build_team_lookup(team_info)

    def get_todays_games(self):
        """
        Today's slate (Eastern date): the NBA schedule left-merged with the games SBR lists. Returns an EMPTY frame only when the NBA
        schedule lists no games (a clean off day). Otherwise one row per scheduled game; a game without odds has NaN team ids and
        is dropped by run(). Raises ScheduleFetchFailure (scoreboard unreadable) or OddsSourceFailure (games are scheduled but SBR
        gave nothing after the retries); any other exception is a bug or a data-format change and is NOT caught here.
        Every game that will not be predicted is recorded in self.skipped_games with its reason.
        """
        logging.info("Fetching Today's Odds & Schedule...")
        # Use target_date if provided (for backfilling), otherwise today in US/Eastern time
        today = self.target_date if self.target_date else now_et().strftime('%Y-%m-%d')
        self.skipped_games = []
        sleep_fn = getattr(self, 'sleep_fn', None)

        nba_schedule = fetch_nba_schedule(today, sleep_fn=sleep_fn)
        if nba_schedule.empty:
            logging.info(f"No games on the NBA schedule for {today}.")
            return pd.DataFrame()
        # Direct SBR fetch: current lines for bets, opening lines for the model (team/live_odds.py)
        logging.info(f"Fetching from SBR for {today}...")
        curr_games, open_games = fetch_sbr_boards(today, sleep_fn=sleep_fn)

        open_map = {}
        for g in open_games:
            open_map[f"{g.get('away_team')} @ {g.get('home_team')}"] = g

        # Games SBR lists without a usable line are filled from The Odds API (core/odds_fallback.py); one bulk call covers them all
        api_games = None
        games = []
        for g in curr_games:
            key = f"{g.get('away_team')} @ {g.get('home_team')}"
            lines = split_open_and_current(g, open_map.get(key))
            source = 'sbr'
            if not any(pd.notna(lines[f'{prefix}{m}']) for m in MARKETS for prefix in ('odds_', 'curr_odds_')):
                # SBR has nothing usable for the game (no books, or none passing the cross-book checks): ask the API
                if api_games is None:
                    api_games = fetch_team_odds()
                backup = find_game(api_games, g.get('away_team'), g.get('home_team'))
                if backup is not None:
                    lines = split_open_and_current(backup, backup)    # the API only has the current line: it stands in for the open too
                    source = SOURCE
                    logging.info(f"No usable SBR lines for {key}; using The Odds API")
                else:
                    source = 'none'
            elif key not in open_map:
                logging.warning(f"No opening line on SBR for {key}; the game is skipped (the current line is not a model input)")
            row = {
                'gameDate': today,
                'away_team': g.get('away_team'),
                'home_team': g.get('home_team'),
                'away_team_abbr': g.get('away_team_abbr'),
                'home_team_abbr': g.get('home_team_abbr'),
                'odds_source': source,
                **lines,
            }
            games.append(row)

        df = pd.DataFrame(games)
        df['away_team_abbr'] = df['away_team_abbr'].replace(SBR_ABBREVIATIONS)
        df['home_team_abbr'] = df['home_team_abbr'].replace(SBR_ABBREVIATIONS)

        # Map team ids by exact lookup on the normalised name: an unknown name is never guessed
        lookup = self.team_lookup()
        logging.info(f"SBR Away Teams: {df['away_team'].unique()}")
        logging.info(f"SBR Home Teams: {df['home_team'].unique()}")
        df['teamId_home'] = df['home_team'].apply(lambda n: map_team_id(n, lookup))
        df['teamId_away'] = df['away_team'].apply(lambda n: map_team_id(n, lookup))

        dropped = {}      # (home abbr, away abbr) -> the record of an SBR game skipped before the merge
        unmapped = df['teamId_home'].isna() | df['teamId_away'].isna()
        for r in df[unmapped].itertuples():
            bad = [n for n, i in ((r.home_team, r.teamId_home), (r.away_team, r.teamId_away)) if pd.isna(i)]
            names = ', '.join(repr(n) for n in bad)
            logging.error(f"UNMAPPED TEAM: {names} is not an NBA team name this job knows (SBR game {r.away_team} @ {r.home_team}); the game is "
                          f"skipped. If it is a real team spelling, add it to TEAM_NAME_ALIASES in scripts/run_inference_v2.py.")
            dropped[(r.home_team_abbr, r.away_team_abbr)] = self._skip(
                'unmapped_team', today, r.home_team, r.away_team, detail=f"unknown team name(s): {names}")
        incomplete = (df['odds_open_missing'] > 0) & ~unmapped
        if incomplete.any():
            logging.warning("Skipping games without an opening line: "
                            + ", ".join(f"{a} @ {h}" for a, h in zip(df.loc[incomplete, 'away_team'], df.loc[incomplete, 'home_team'])))
        for r in df[incomplete].itertuples():
            dropped[(r.home_team_abbr, r.away_team_abbr)] = self._skip(
                'missing_opening_line', today, r.home_team, r.away_team,
                detail=f"{int(r.odds_open_missing)} market(s) have a current line but no opening line; the current line is not a model input")
        df = df[~unmapped & ~incomplete].reset_index(drop=True)

        # validate: a duplicated SBR row (or schedule row) must fail loudly, not double a game
        merged = nba_schedule.merge(
            df,
            on=['home_team_abbr', 'away_team_abbr'],
            how='left',
            suffixes=('', '_x'),
            validate='one_to_one'
        )
        merged = merged.drop(columns=['gameDate_x'])

        schedule_keys = set(zip(nba_schedule['home_team_abbr'], nba_schedule['away_team_abbr']))
        off_schedule = [(r.home_team_abbr, r.away_team_abbr, r.home_team, r.away_team) for r in df.itertuples()
                        if (r.home_team_abbr, r.away_team_abbr) not in schedule_keys]
        leftovers = ', '.join(f"{a} @ {h}" for _, _, h, a in off_schedule) or 'none'
        for r in merged[merged['odds_source'].isna()].itertuples():
            key = (r.home_team_abbr, r.away_team_abbr)
            if key in dropped:                      # SBR listed it but it was skipped above: that record gets the NBA game id
                dropped[key]['game_id'] = str(r.gameId)
                continue
            logging.warning(f"NBA game {r.gameId} {r.away_team_abbr} @ {r.home_team_abbr} is not in the odds source; skipped (no_odds)")
            self._skip('no_odds', today, r.home_team_abbr, r.away_team_abbr, game_id=str(r.gameId),
                       detail=f"on the NBA schedule (gameStatus {r.gameStatus}) but SBR has no row for it after the merge on the team "
                              f"abbreviations; SBR rows not matched to the schedule: {leftovers}")
        for hab, aab, home, away in off_schedule:
            logging.warning(f"SBR lists {away} @ {home} ({aab} @ {hab}) but the NBA schedule for {today} does not; skipped (not_on_schedule)")
            self._skip('not_on_schedule', today, home, away, detail=f"listed by SBR as {aab} @ {hab}; not on the NBA scoreboard for {today}")

        self.log_odds_sources(merged)

        return merged

    def get_injury_features(self, games_df):
        """
        inj_* features for today's games from the official NBA injury report, built with the same
        function and snapshot as the training set (core/injury_report_features.py). The cutoff
        is min(training snapshot, now), so a run before the snapshot time uses the latest report;
        main() waits for the snapshot time first (injury_cutoff_guard), and the report used is checked
        against the snapshot here (an ERROR is logged if it is older than the snapshot expects).
        """
        self.injury_problem = None      # (code, detail) when the report the model needs is missing or stale: run() then recommends no bets
        games_df = games_df.dropna(subset=['teamId_home', 'teamId_away'])  # games SBR did not list
        game_date = pd.Timestamp(games_df['gameDate'].iloc[0]).normalize()
        season = season_for_date(game_date)
        # Fetch the report the training snapshot would use: latest at/before the snapshot time
        # (or now, if earlier). Training caps the cutoff at tip - 30 min; for normal evening tips
        # the clock time is earlier, so the cap only matters for early tips (handled in training).
        now = now_et()
        # The slate's nominal cutoff: game date + clock, capped at tip - 30 min per game (same function as training). For a normal
        # slate that is the clock time; only a slate of nothing but early tips has an earlier one.
        tips = tip_times_et(games_df['gameTimeUTC']) if 'gameTimeUTC' in games_df.columns else None
        nominal = snapshot_nominal_cutoff(game_date, tips)
        cutoff = min(now, nominal) if nominal is not None else now
        # saves into data/raw/injury_reports; 3 attempts with a short backoff (a failed download is not silently final)
        report_path = scrape_injury_reports.live_with_retries(as_of=cutoff, sleep_fn=getattr(self, 'sleep_fn', None))
        self.injury_cutoff, self.injury_report_path = cutoff, report_path
        if report_path is None:
            logging.error("Injury report download FAILED; using only the reports already on disk")
        logging.info(f"Injury report for snapshot {GameFeatureBuilder.INJURY_SNAPSHOT} (cutoff {cutoff:%Y-%m-%d %H:%M} ET): {report_path}")
        reports = irf.parse_all_reports()      # re-scans the report folder: picks up the PDF just fetched
        latest, age, stale = irf.check_report_freshness(reports, game_date, cutoff)
        # Bets are withheld (run() returns exit code 5) when the report the model needs is missing or stale for the slate. Decided per
        # slate, not per game, because that is what the code can know: one report file covers every game of the date, the freshness
        # check and the snapshot check are per date, and the cutoff is the slate's. A team merely NOT LISTED on a fresh report is NOT
        # a failure (inj_missing=1, the same as in training); only a missing report, or one too old, is.
        if stale:
            self.injury_problem = ('injury_report_missing' if latest is None else 'injury_report_stale',
                                   f"newest report for {game_date:%Y-%m-%d} is "
                                   f"{'missing' if latest is None else f'{age:.1f} h older than the {cutoff:%H:%M} ET cutoff ({latest})'}")
            logging.error(f"STALE INJURY DATA: newest report for {game_date:%Y-%m-%d} is "
                          f"{'missing' if latest is None else f'{age:.1f} h older than the {cutoff:%H:%M} ET cutoff ({latest})'}; "
                          f"teams without a covering report are flagged inj_missing=1")
        else:
            logging.info(f"Newest injury report for {game_date:%Y-%m-%d}: {latest} ({age:.1f} h before the cutoff)")
        if latest is not None:
            day = reports[pd.to_datetime(reports['game_date']).dt.normalize() == game_date]
            used = day.loc[pd.to_datetime(day['report_ts']) == latest, 'source_file']
            used_file = str(used.iloc[0]) if len(used) else None
            if used_file and (report_path is None or Path(str(report_path)).name != used_file):
                self.injury_report_path = used_file      # the file the features really come from (bet log), not the download
            behind = snapshot_report_problem(latest, nominal)
            if behind is not None:
                if self.injury_problem is None:
                    self.injury_problem = ('injury_snapshot_mismatch', f"the report used is {behind:.0f} min older than the snapshot time")
                slot = expected_grid_slot(nominal)
                logging.error(f"INJURY SNAPSHOT MISMATCH: the report used is {used_file} (published {latest:%H:%M} ET), {behind:.0f} min older than "
                              f"the {GameFeatureBuilder.INJURY_SNAPSHOT} time {nominal:%H:%M} ET; the training snapshot reads the report published at "
                              f"{'n/a' if slot is None else format(slot, '%H:%M')} ET. The run started at {now:%H:%M} ET: too early, or that report was "
                              f"not downloadable. The inj_* features differ from training and the inj_downgrade_lost_* flags are meaningless.")
        upcoming = pd.concat([
            pd.DataFrame({'gameId': games_df['gameId'].astype(str), 'teamId': games_df['teamId_home'].astype(int)}),
            pd.DataFrame({'gameId': games_df['gameId'].astype(str), 'teamId': games_df['teamId_away'].astype(int)}),
        ], ignore_index=True)
        upcoming['gameDate'] = game_date.strftime('%Y-%m-%d')
        upcoming['seasonYear'] = season
        inj = irf.build_injury_features(self.db_path, [season], snapshot_name=GameFeatureBuilder.INJURY_SNAPSHOT,
                                        as_of=now, upcoming=upcoming, reports=reports)
        inj = inj[inj['gameId'].isin(upcoming['gameId'])]
        logging.info(f"Injury features for {len(inj)} team-games; missing report for {int(inj['inj_missing'].sum())}")
        # Informational only (never a model input): noon-to-latest status downgrades, and both injury sources side by side
        self.downgrades = {}
        try:
            from core.injury_downgrades import build_downgrade_features
            d, _, _ = build_downgrade_features(self.db_path, [season], as_of=now, upcoming=upcoming, reports=reports)
            d = d[d['gameId'].isin(upcoming['gameId'])]
            self.downgrades = {(r.gameId, int(r.teamId)): r.inj_downgrade_lost for r in d.itertuples()}
        except Exception as e:
            logging.warning(f"Downgrade flags unavailable: {e}")
        from core.injury_source_compare import log_side_by_side
        log_side_by_side(self.db_path, game_date, reports=reports, as_of=now)
        return inj.set_index(['gameId', 'teamId'])

    def verify_feature_alignment(self, row_df, model_name, expected_cols):
        """Select the model's columns in order; missing features become NaN and are logged loudly."""
        return align_features(row_df, expected_cols, model_name)

    def explain_games(self, frame, games_df):
        """
        Why each spread/total differs from the current line (team/explain.py), stored in prediction_explanations next to the
        predictions. An explanation problem is logged and never stops the daily run.
        """
        try:
            feats = sorted({f for cols in self.selected_features.values() for f in cols})
            ref = None
            try:
                train = pd.read_csv(latest_training_set(), usecols=lambda c: c in feats or c == 'seasonYear')
                ref = build_reference(train[train['seasonYear'].isin(sorted(train['seasonYear'].unique())[-3:])], feats)
            except Exception as e:
                logging.warning(f"No league-wide reference for the explanations (percentiles left empty): searched "
                                f"{Path(PROCESSED_DIR).resolve()} for training_set_v1_reduced_YYYY-MM-DD.csv "
                                f"(working directory {Path.cwd()}): {type(e).__name__}: {e}")
            lines = games_df.assign(gameId=games_df['gameId'].astype(str)).set_index('gameId').reindex(frame['gameId'].astype(str))
            current = pd.DataFrame({'spread_line': pd.to_numeric(lines['curr_odds_home_spread'], errors='coerce').to_numpy(),
                                    'total_line': pd.to_numeric(lines['curr_odds_total'], errors='coerce').to_numpy(),
                                    'home_ml': pd.to_numeric(lines['curr_odds_home_ml'], errors='coerce').to_numpy()})
            expl = explain_markets(self.bundle, frame, current, align=self.verify_feature_alignment, reference=ref,
                                   history=load_oos_history(), prediction_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))   # machine-local on purpose, like predictions_log_v2.prediction_date
            n = save_explanations(self.db_path, expl)
            logging.info(f"Stored {n} explanation rows for {expl['game_id'].nunique()} games")
        except Exception as e:
            logging.error(f"Explanations failed (predictions are unaffected): {e}", exc_info=True)

    def log_odds_sources(self, games):
        """Which source priced each game today (sbr / theoddsapi / none), kept in odds_source_log so the backup's use can be reviewed."""
        try:
            rows = [(str(r.gameId), str(r.gameDate), r.odds_source, datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
                    for r in games.itertuples() if isinstance(getattr(r, 'odds_source', None), str)]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS odds_source_log (game_id TEXT, game_date TEXT, source TEXT, ts_utc TEXT, "
                             "PRIMARY KEY (game_id, ts_utc))")
                conn.executemany("INSERT OR IGNORE INTO odds_source_log VALUES (?,?,?,?)", rows)
            used = [r[0] for r in rows if r[2] == SOURCE]
            if used:
                logging.info(f"{len(used)} game(s) priced by The Odds API instead of SBR")
        except Exception as e:                       # bookkeeping only: never stops the run
            logging.warning(f"Could not log the odds sources: {e}")

    def keep_unpriced(self, games_df):
        """
        Games not yet started (gameStatus 1) whose LOGGED prediction lacks the moneyline, spread or total: the ones a re-run can improve.
        It goes by the log, not the lines fetched now: a game the backup priced a minute ago still has its old no-odds row to replace.
        """
        logged = set(unpriced_games(self.db_path, games_df['gameDate'].iloc[0])['game_id'])
        not_started = pd.to_numeric(games_df['gameStatus'], errors='coerce') == 1
        keep = games_df[games_df['gameId'].astype(str).isin(logged) & not_started].reset_index(drop=True)
        logging.info(f"Re-run: {len(keep)} of {len(games_df)} games were logged without a line and have not started: "
                     + ", ".join(f"{a} @ {h}" for a, h in zip(keep['away_team'], keep['home_team'])))
        return keep

    def run(self):
        """
        Returns the exit code for main(): EXIT_OK (0) after a normal run, an off day or a re-run with nothing to do; EXIT_NO_INPUTS (4)
        when games are scheduled but none could be predicted (odds source or schedule unavailable, or every game skipped);
        EXIT_BETS_SUPPRESSED (5) when the games were predicted and logged but the injury report was missing or stale, so no bet
        was recommended. An unexpected exception (a bug, a changed data format) is NOT caught here: it propagates to main().
        """
        logging.info("Starting Inference Run...")
        run_ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')   # one stamp for every row of this run
        self.skipped_games = []
        self.injury_problem = None

        # 1. Get Schedule
        try:
            games_df = self.get_todays_games()
        except ScheduleFetchFailure as e:
            logging.error(f"NBA SCHEDULE FAILURE: the NBA scoreboard could not be read after {SOURCE_ATTEMPTS} attempts ({e}). "
                          f"Nothing was predicted: this is NOT an off day.")
            return EXIT_NO_INPUTS
        except OddsSourceFailure as e:
            logging.error(f"ODDS SOURCE FAILURE: games are on the NBA schedule but SBR gave no usable odds after {SOURCE_ATTEMPTS} attempts "
                          f"({e}). Nothing was predicted: this is NOT an off day.")
            return EXIT_NO_INPUTS
        if games_df.empty:
            logging.info("No games found today.")                  # the NBA schedule lists none: a clean off day
            return EXIT_OK
        has_odds = ~games_df[['teamId_home', 'teamId_away']].isna().any(axis=1)     # the games the skip record already explains
        games_df = games_df[has_odds].reset_index(drop=True)
        log_skipped_games(self.db_path, self.skipped_games, run_ts)
        if games_df.empty:
            logging.error(f"ALL GAMES SKIPPED: none of today's {len(self.skipped_games)} listed games could be predicted ("
                          + ", ".join(sorted({r['reason'] for r in self.skipped_games})) + f"); see {SKIPPED_TABLE}.")
            return EXIT_NO_INPUTS
        if self.skipped_games:
            logging.warning(f"{len(self.skipped_games)} game(s) skipped, {len(games_df)} to predict: "
                            + "; ".join(f"{r['away_team']} @ {r['home_team']} ({r['reason']})" for r in self.skipped_games))
        if self.only_unpriced:
            games_df = self.keep_unpriced(games_df)
            if games_df.empty:
                logging.info("Re-run: every game that has not started already has its lines; nothing to predict.")
                return EXIT_OK

        # 2. Build the model inputs with the same code as the training set (team/live_features.py):
        #    team form, travel, roster BPM, injury report, differences and matchups
        logging.info("Building Live Features...")
        try:
            inj_feats = self.get_injury_features(games_df).reset_index()
        except Exception as e:
            # Same values training uses for a team with no report: inj_missing=1, zeros elsewhere
            logging.error(f"Injury features unavailable, treating as missing report: {e}")
            inj_feats = None
            if self.injury_problem is None:
                self.injury_problem = ('injury_features_failed', f"{type(e).__name__}: {e}")
        frame = build_live_game_frame(self.db_path, games_df, injury_features=inj_feats)

        # 3. Predict (model columns come from the validated feature config)
        cols_ml = self.selected_features['ml']
        cols_spread = self.selected_features['spread']
        cols_total = self.selected_features['total']
        logging.info(f"Models expect {len(cols_ml)} / {len(cols_spread)} / {len(cols_total)} features (ML / Spread / Total).")
        # models/market_models.predict_markets: handles both model-set generations (isotonic + plain regressors, or the
        # market-aware set: residual spread model, margin-stack ML calibrator, per-game total sigma)
        out = predict_markets(self.bundle, frame, align=self.verify_feature_alignment)
        probs, pred_spreads, pred_totals = out['ml_prob'].to_numpy(), out['spread_pred'].to_numpy(), out['total_pred'].to_numpy()
        spread_stds = out['spread_std'].fillna(DEFAULT_SPREAD_STD).to_numpy()
        total_stds = out['total_std'].fillna(DEFAULT_TOTAL_STD).to_numpy()
        self.explain_games(frame, games_df)
        # A missing or stale injury report: the predictions and edges are computed and logged as usual, but no bet is recommended.
        suppressed = self.injury_problem[0] if self.injury_problem else None
        if suppressed:
            logging.error(f"BETS SUPPRESSED ({suppressed}): {self.injury_problem[1]}. The {len(games_df)} game(s) are predicted and logged, "
                          f"the edges are recorded, but NO bet is recommended (bet_suppressed_reason in bet_recommendations_log). "
                          f"Exit code {EXIT_BETS_SUPPRESSED}.")
        bet_rows = []
        preds = []
        for i, g in games_df.iterrows():
            prob, pred_spread, pred_total = float(probs[i]), float(pred_spreads[i]), float(pred_totals[i])
            spread_std, total_std = float(spread_stds[i]), float(total_stds[i])

            # Tradable (current) lines for edges, EV and the prediction log
            line = float(g['curr_odds_home_spread']) if pd.notna(g.get('curr_odds_home_spread')) else np.nan
            total_line = float(g['curr_odds_total']) if pd.notna(g.get('curr_odds_total')) else np.nan
            ml_home = float(g['curr_odds_home_ml']) if pd.notna(g.get('curr_odds_home_ml')) else np.nan
            ml_away = float(g['curr_odds_away_ml']) if pd.notna(g.get('curr_odds_away_ml')) else np.nan

            # Log Prediction
            try:
                log_entry = {
                    'gameId': g['gameId'],
                    'gameDate': g['gameDate'],
                    'home_team': g['home_team'],
                    'away_team': g['away_team'],
                    'pred_home_win_prob': prob, # Calibrated Prob
                    'pred_spread': pred_spread,
                    'pred_total': pred_total,
                    'line_spread': line,
                    'line_total': total_line,
                    'line_ml_home': ml_home, # Raw American Odds
                    'line_ml_away': ml_away,
                    'model_version': 'v1.0.0_hybrid',
                    'model_set_hash': self.bundle.model_set_hash,       # ties the row to the exact files / training CSV
                    'training_set_hash': self.bundle.training_set_hash,
                }
                self.logger.log_prediction(log_entry)
            except Exception as e:
                logging.error(f"Failed to log prediction: {e}")

            # Recommendations: ONE rule for all markets (team/betting.py, shared with the backtests and
            # the dashboard): bet when the model's win probability beats the price's break-even
            # probability by more than DEFAULT_PROB_EDGE. Spread/total win probability is
            # Phi(point edge / residual std) with the std saved with the model set.
            ml_bet, ml_edge = get_ml_bet({'ml_prob': prob, 'odds_home_ml': ml_home, 'odds_away_ml': ml_away})
            def price(name):
                v = g.get(f'curr_price_{name}')
                return float(v) if v is not None and pd.notna(v) else None
            spread_bet, spread_edge = get_spread_bet(pred_spread, line, spread_std,
                                                     home_price=price('home_spread_odds'), away_price=price('away_spread_odds'))
            total_bet, total_edge = get_total_bet(pred_total, total_line, total_std,
                                                  over_price=price('over_odds'), under_price=price('under_odds'))
            if suppressed:                                    # the edge stays as computed; only the side is withheld
                ml_bet = spread_bet = total_bet = None
            try:     # the full record of what this run read and judged (core/prediction_logger.bet_recommendations_log)
                bet_rows.append(bet_log_row(
                    g, out.iloc[i].to_dict(),
                    {'ml': (ml_bet, ml_edge), 'spread': (spread_bet, spread_edge), 'total': (total_bet, total_edge)},
                    run_ts, self.bundle.model_set_hash, self.injury_cutoff, self.injury_report_path,
                    bet_suppressed_reason=suppressed))
            except Exception as e:
                logging.error(f"Could not build the bet-log row for game {g.get('gameId')}: {e}")
            recommendation = "; ".join(
                f"{market} {side.upper()} ({edge:+.1%})"
                for market, side, edge in (('ML', ml_bet, ml_edge), ('SPREAD', spread_bet, spread_edge),
                                           ('TOTAL', total_bet, total_edge))
                if side)

            preds.append({
                'game_id': g['gameId'],
                'game_date': g['gameDate'],
                # machine-local time on purpose, format unchanged (the published dashboard reads it); run_ts_utc in
                # bet_recommendations_log is the unambiguous timestamp
                'prediction_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'home_team': g['home_team'],
                'away_team': g['away_team'],
                'matchup': f"{g['away_team']} @ {g['home_team']}",
                'ml_prob': prob,
                'spread_pred': round(pred_spread, 1),
                'total_pred': round(pred_total, 1),
                'odds_home_ml': ml_home,
                'odds_away_ml': ml_away,
                'odds_home_spread': line,
                'odds_total': total_line,
                'home_score': np.nan, # Pending
                'away_score': np.nan, # Pending
                'status': 'PENDING',
                'ml_bet': ml_bet,
                'ml_edge': round(ml_edge, 4),
                'spread_bet': spread_bet,
                'spread_edge': round(spread_edge, 4),
                'total_bet': total_bet,
                'total_edge': round(total_edge, 4),
                'inj_downgrade_lost_home': getattr(self, 'downgrades', {}).get((str(g['gameId']), int(g['teamId_home'])), 0.0),
                'inj_downgrade_lost_away': getattr(self, 'downgrades', {}).get((str(g['gameId']), int(g['teamId_away'])), 0.0),
                'recommendation': recommendation # Keeping as extra column for CSV utility
            })
            
        log_bet_recommendations(self.db_path, bet_rows)
        res = pd.DataFrame(preds)
        print("\n--- PREDICTIONS ---")
        print(res.to_string(index=False))
        if self.only_unpriced and Path('todays_picks.csv').exists():
            # a re-run covers a few games: refresh their rows and keep the rest of the day's file
            earlier = pd.read_csv('todays_picks.csv', dtype={'game_id': str})
            res = pd.concat([earlier[~earlier['game_id'].isin(res['game_id'].astype(str))], res], ignore_index=True)
        res.to_csv('todays_picks.csv', index=False)
        return EXIT_BETS_SUPPRESSED if suppressed else EXIT_OK


def main(argv=None):
    """
    Returns the process exit code (scripts/run_daily_inference.bat says what it does for each): EXIT_OK 0, EXIT_CRASH 1 (unexpected
    exception, traceback in the log), EXIT_TOO_EARLY 3 (injury-cutoff guard refused), EXIT_NO_INPUTS 4 (games scheduled, none predicted),
    EXIT_BETS_SUPPRESSED 5 (predicted and logged, no bet recommended: injury report missing or stale).
    """
    import argparse
    parser = argparse.ArgumentParser(description="Predict today's NBA games with the deployed model set")
    parser.add_argument('--suffix', default=DEPLOYED_SUFFIX,
                        help=f"model set suffix in models/saved (default: deployed set {DEPLOYED_SUFFIX})")
    parser.add_argument('--date', default=None, help='YYYY-MM-DD to backfill instead of today (ET)')
    parser.add_argument('--only-unpriced', action='store_true',
                        help="re-run: skip the database sync and predict only today's not-yet-started games that still lack a line "
                             "(scripts/run_odds_rerun.bat, scheduled by scripts/schedule_odds_rerun.py)")
    parser.add_argument('--no-wait-for-injury-cutoff', action='store_true',
                        help="do not wait for the injury report the model was trained on (noon ET on game day): run at once with an older "
                             "report, as before the guard existed (logs a WARNING). Daily runs only; --date and --only-unpriced never wait")
    parser.add_argument('--max-wait-minutes', type=float, default=DEFAULT_MAX_WAIT_MINUTES,
                        help=f"longest wait for the injury report (default {DEFAULT_MAX_WAIT_MINUTES}); a run earlier than that exits with code "
                             f"{EXIT_TOO_EARLY} without predicting or logging anything")
    args = parser.parse_args(argv)

    seasons_to_check = [current_season_et()]       # decided in Eastern time, when the run starts
    db_path = Path.cwd() / 'data' / 'nba_stats.db'

    if not args.only_unpriced:                 # a re-run uses the morning's sync
        syncer = NBADatabaseSync(db_path)
        try:
            syncer.sync_database(seasons_to_check)
        except Exception as e:
            # Predict anyway, but make the staleness visible in the log
            with sqlite3.connect(db_path) as conn:
                last = conn.execute("SELECT MAX(gameDate) FROM team_game_full").fetchone()[0]
            logging.error(f"DATABASE SYNC FAILED ({e}). Predicting with data through {last}; "
                          f"features may be stale.")

    if not args.only_unpriced and not args.date:    # a normal daily run: wait for the report the model was trained on (after the sync, so the box scores are fresh)
        if injury_cutoff_guard(wait=not args.no_wait_for_injury_cutoff, max_wait_minutes=args.max_wait_minutes) == 'too_early':
            return EXIT_TOO_EARLY

    engine = InferenceEngine(db_path, suffix=args.suffix, target_date=args.date, only_unpriced=args.only_unpriced)

    try:
        code = engine.run()
    except Exception:
        logging.exception("INFERENCE RUN CRASHED: an unexpected error (a bug or a changed data format), not a data outage; nothing may have been predicted")
        return EXIT_CRASH
    return EXIT_OK if code is None else code


if __name__ == "__main__":
    sys.exit(main())
