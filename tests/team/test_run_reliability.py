"""Reliability of the daily run (B3): an odds-source outage is not an off day (exit 4), team names are looked up exactly, every game the
run does not predict leaves a record (inference_skipped_games), dates are decided in Eastern time, and a missing or stale injury report
withholds the bets (exit 5). No network, no real sleeping: the NBA scoreboard, SBR, The Odds API, the model and the injury download
are stubs."""
import datetime
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

rm = pytest.importorskip('run_inference_v2')
pytest.importorskip('core.sbr')
import core.sbr as sbr_module                      # noqa: E402

TEAMS = [(1610612737, 'Atlanta Hawks', 'ATL'), (1610612738, 'Boston Celtics', 'BOS'), (1610612739, 'Cleveland Cavaliers', 'CLE'),
         (1610612740, 'New Orleans Pelicans', 'NOP'), (1610612741, 'Chicago Bulls', 'CHI'), (1610612742, 'Dallas Mavericks', 'DAL'),
         (1610612743, 'Denver Nuggets', 'DEN'), (1610612744, 'Golden State Warriors', 'GSW'), (1610612745, 'Houston Rockets', 'HOU'),
         (1610612746, 'LA Clippers', 'LAC'), (1610612746, 'Los Angeles Clippers', 'LAC'), (1610612747, 'Los Angeles Lakers', 'LAL'),
         (1610612748, 'Miami Heat', 'MIA'), (1610612749, 'Milwaukee Bucks', 'MIL'), (1610612750, 'Minnesota Timberwolves', 'MIN'),
         (1610612751, 'Brooklyn Nets', 'BKN'), (1610612752, 'New York Knicks', 'NYK'), (1610612753, 'Orlando Magic', 'ORL'),
         (1610612754, 'Indiana Pacers', 'IND'), (1610612755, 'Philadelphia 76ers', 'PHI'), (1610612756, 'Phoenix Suns', 'PHX'),
         (1610612757, 'Portland Trail Blazers', 'POR'), (1610612758, 'Sacramento Kings', 'SAC'), (1610612759, 'San Antonio Spurs', 'SAS'),
         (1610612760, 'Oklahoma City Thunder', 'OKC'), (1610612761, 'Toronto Raptors', 'TOR'), (1610612762, 'Utah Jazz', 'UTA'),
         (1610612763, 'Memphis Grizzlies', 'MEM'), (1610612764, 'Washington Wizards', 'WAS'), (1610612765, 'Detroit Pistons', 'DET'),
         (1610612766, 'Charlotte Bobcats', 'CHA'), (1610612766, 'Charlotte Hornets', 'CHA')]
TEAM_INFO = pd.DataFrame(TEAMS, columns=['teamId', 'teamName', 'teamTricode'])
ID = {tri: tid for tid, _, tri in TEAMS}


class Sleeper:
    """Injected sleep: records the pauses, never waits."""
    def __init__(self):
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)


# ---------------------------------------------------------------------------------------------------------- stubs for the sources
def sbr_game(away, home, away_abbr, home_abbr, *, spread=-4.0, total=221.5, hml=-150, aml=130, lines=True):
    book = 'draftkings'
    if not lines:
        return {'away_team': away, 'home_team': home, 'away_team_abbr': away_abbr, 'home_team_abbr': home_abbr}
    return {'away_team': away, 'home_team': home, 'away_team_abbr': away_abbr, 'home_team_abbr': home_abbr,
            'home_ml': {book: hml}, 'away_ml': {book: aml}, 'home_spread': {book: spread}, 'away_spread': {book: -spread},
            'total': {book: total}, 'home_spread_odds': {book: -110}, 'away_spread_odds': {book: -110},
            'over_odds': {book: -110}, 'under_odds': {book: -110}}


def open_only(game, **drop):
    """The opening-line version of an SBR game with some markets removed (no opening line for them)."""
    g = dict(game)
    for k in drop:
        g[k] = {}
    return g


class World:
    """Fake NBA scoreboard + SBR. `schedule` rows are (gameId, away tricode, home tricode); boards are lists of SBR game dicts or an
    Exception / list per attempt. Counts the calls so a test can see the retries (and that The Odds API was never asked)."""
    def __init__(self, monkeypatch, schedule, curr=None, opened=None, schedule_error=None):
        self.schedule, self.schedule_error = schedule, schedule_error
        self.curr_attempts = curr if isinstance(curr, list) and curr and isinstance(curr[0], (list, Exception)) else [curr or []]
        self.open_attempts = opened if isinstance(opened, list) and opened and isinstance(opened[0], (list, Exception)) else [opened or []]
        self.calls = SimpleNamespace(schedule=[], sbr=[], odds_api=0)
        world = self

        class FakeScoreboard:
            def __init__(self, game_date=None, **kw):
                world.calls.schedule.append(game_date)
                if isinstance(world.schedule_error, list):
                    err = world.schedule_error.pop(0) if world.schedule_error else None
                else:
                    err = world.schedule_error
                if err is not None:
                    raise err

            def get_data_frames(self):
                rows = [{'gameId': gid, 'gameCode': f'20261111/{a}{h}', 'gameStatus': 1, 'gameTimeUTC': '2026-11-12T00:30:00Z'}
                        for gid, a, h in world.schedule]
                return [pd.DataFrame(), pd.DataFrame(rows, columns=['gameId', 'gameCode', 'gameStatus', 'gameTimeUTC'])]

        class FakeSBR:
            def __init__(self, sport=None, date=None, current_line=True):
                attempts = world.curr_attempts if current_line else world.open_attempts
                n = sum(1 for c in world.calls.sbr if c[1] == current_line)
                world.calls.sbr.append((date, current_line))
                item = attempts[min(n, len(attempts) - 1)]
                if isinstance(item, Exception):
                    raise item
                self.games = item

        monkeypatch.setattr(rm.scoreboardv3, 'ScoreboardV3', FakeScoreboard)
        monkeypatch.setattr(sbr_module, 'Scoreboard', FakeSBR)

        def no_odds_api(*a, **k):
            world.calls.odds_api += 1
            return {}
        monkeypatch.setattr(rm, 'fetch_team_odds', no_odds_api)


@pytest.fixture()
def engine_factory(tmp_path, monkeypatch):
    """A real InferenceEngine.run() / get_todays_games with the model, the features, the explanations and the logger stubbed."""
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / 'x.db')
    with sqlite3.connect(db) as conn:
        TEAM_INFO.to_sql('team_basic_stats', conn, index=False)
    sleeper = Sleeper()
    logged = []

    def make(only_unpriced=False):
        eng = object.__new__(rm.InferenceEngine)
        eng.db_path, eng.only_unpriced, eng.target_date = db, only_unpriced, None
        eng.injury_cutoff, eng.injury_report_path = None, None
        eng.sleep_fn = sleeper
        eng.bundle = SimpleNamespace(model_set_hash='hash-xyz', training_set_hash='t', ml=None, spread=None, total=None,
                                     features={'ml': [], 'spread': [], 'total': []})
        eng.selected_features = eng.bundle.features
        eng.logger = SimpleNamespace(log_prediction=lambda entry: logged.append(entry))
        eng.get_injury_features = lambda g: pd.DataFrame({'gameId': [], 'teamId': []}).set_index(['gameId', 'teamId'])
        eng.explain_games = lambda *a, **k: None
        return eng

    def predict(bundle, frame, align=None):
        n = len(frame)
        return pd.DataFrame({'ml_prob_raw': [0.58] * n, 'ml_prob': [0.60] * n, 'spread_pred': [5.0] * n, 'total_pred': [230.0] * n,
                             'spread_std': [13.0] * n, 'total_std': [18.0] * n})
    monkeypatch.setattr(rm, 'build_live_game_frame', lambda db_path, games_df, injury_features=None: games_df.copy())
    monkeypatch.setattr(rm, 'predict_markets', predict)
    return SimpleNamespace(make=make, db=db, sleeper=sleeper, logged=logged)


def table(db, name):
    with sqlite3.connect(db) as conn:
        return pd.read_sql(f"SELECT * FROM {name}", conn)


G_BOS_NYK = sbr_game('Boston Celtics', 'New York Knicks', 'BOS', 'NY')
G_LAL_LAC = sbr_game('Los Angeles Lakers', 'LA Clippers', 'LAL', 'LAC')
SCHEDULE = [('0022600001', 'BOS', 'NYK'), ('0022600002', 'LAL', 'LAC')]


# ---------------------------------------------------------------------------------------------------------- 1. outage is not an off day
def test_no_games_on_the_nba_schedule_is_a_clean_exit_zero(engine_factory, monkeypatch, caplog):
    w = World(monkeypatch, schedule=[], curr=[])
    eng = engine_factory.make()
    with caplog.at_level(logging.INFO):
        assert eng.run() == rm.EXIT_OK == 0
    assert w.calls.sbr == [] and w.calls.odds_api == 0                      # SBR is not even asked on an off day
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert engine_factory.sleeper.slept == [] and 'No games found today' in caplog.text


def test_sbr_returning_nothing_is_retried_then_odds_source_failure_exit_4(engine_factory, monkeypatch, caplog):
    w = World(monkeypatch, schedule=SCHEDULE, curr=[])                       # sbrscrape swallows HTTP errors: an outage is an empty board
    eng = engine_factory.make()
    with caplog.at_level(logging.INFO):
        assert eng.run() == rm.EXIT_NO_INPUTS == 4
    assert [c[1] for c in w.calls.sbr] == [True, True, True]                  # 3 attempts, current board each time
    assert engine_factory.sleeper.slept == [10, 30]                           # short backoff, injectable, no pause after the last attempt
    errs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errs) == 1 and errs[0].startswith('ODDS SOURCE FAILURE') and 'NOT an off day' in errs[0]
    assert w.calls.odds_api == 0                                              # the paid fallback is never used for a total SBR failure
    assert engine_factory.logged == [] and not Path('todays_picks.csv').exists()     # nothing predicted, nothing written


def test_sbr_raising_every_time_is_also_a_failure_and_names_the_exception(engine_factory, monkeypatch, caplog):
    w = World(monkeypatch, schedule=SCHEDULE, curr=[ConnectionError('boom'), ConnectionError('boom'), ConnectionError('boom')])
    eng = engine_factory.make()
    assert eng.run() == 4
    assert len(w.calls.sbr) == 3 and 'ConnectionError: boom' in caplog.text and 'ODDS SOURCE FAILURE' in caplog.text


def test_sbr_that_recovers_on_the_second_attempt_gives_a_normal_run(engine_factory, monkeypatch, caplog):
    w = World(monkeypatch, schedule=SCHEDULE, curr=[ConnectionError('blip'), [G_BOS_NYK, G_LAL_LAC]], opened=[G_BOS_NYK, G_LAL_LAC])
    eng = engine_factory.make()
    with caplog.at_level(logging.INFO):
        assert eng.run() == 0
    assert engine_factory.sleeper.slept == [10] and len(engine_factory.logged) == 2
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert w.calls.odds_api == 0


def test_an_empty_opening_board_is_retried_but_the_games_are_then_skipped_not_a_failure(engine_factory, monkeypatch):
    World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_LAL_LAC], opened=[])
    eng = engine_factory.make()
    assert eng.run() == 4                                                    # every game lacks its opening line: all skipped
    assert set(table(engine_factory.db, 'inference_skipped_games')['reason']) == {'missing_opening_line'}


def test_games_without_any_sbr_line_still_use_the_odds_api_fallback_once(engine_factory, monkeypatch):
    w = World(monkeypatch, schedule=SCHEDULE, curr=[sbr_game('Boston Celtics', 'New York Knicks', 'BOS', 'NY', lines=False),
                                                    sbr_game('Los Angeles Lakers', 'LA Clippers', 'LAL', 'LAC', lines=False)],
              opened=[sbr_game('Boston Celtics', 'New York Knicks', 'BOS', 'NY', lines=False)])
    eng = engine_factory.make()
    assert eng.run() == 0
    assert w.calls.odds_api == 1                                             # one bulk call covers both games, as before
    assert len(engine_factory.logged) == 2                                   # predicted without odds, as before (the re-run picks them up)


def test_a_failed_schedule_fetch_is_retried_and_then_exits_non_zero_not_as_an_off_day(engine_factory, monkeypatch, caplog):
    w = World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK], schedule_error=TimeoutError('read timed out'))
    eng = engine_factory.make()
    with caplog.at_level(logging.INFO):
        assert eng.run() == 4
    assert len(w.calls.schedule) == 3 and engine_factory.sleeper.slept == [10, 30] and w.calls.sbr == []
    errs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errs and errs[-1].startswith('NBA SCHEDULE FAILURE') and 'TimeoutError' in errs[-1] and 'No games found today' not in caplog.text


def test_a_schedule_fetch_that_recovers_is_a_normal_run(engine_factory, monkeypatch):
    w = World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_LAL_LAC], opened=[G_BOS_NYK, G_LAL_LAC],
              schedule_error=[TimeoutError('blip'), None])
    assert engine_factory.make().run() == 0 and len(w.calls.schedule) == 2


def test_the_lenient_schedule_function_still_returns_an_empty_frame_for_the_guard(monkeypatch):
    World(monkeypatch, schedule=SCHEDULE, curr=[], schedule_error=TimeoutError('x'))
    assert rm.get_nba_schedule('2026-11-11').empty                           # the injury-cutoff guard keeps its old, lenient behaviour
    with pytest.raises(rm.ScheduleFetchFailure):
        rm.fetch_nba_schedule('2026-11-11', sleep_fn=Sleeper())


def test_a_programming_error_inside_the_odds_block_is_not_an_off_day(engine_factory, monkeypatch):
    World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_LAL_LAC], opened=[G_BOS_NYK, G_LAL_LAC])
    monkeypatch.setattr(rm, 'split_open_and_current', lambda *a, **k: {}['nope'])         # a KeyError from our own code
    eng = engine_factory.make()
    with pytest.raises(KeyError):
        eng.run()
    assert engine_factory.sleeper.slept == []                                # and it was not retried or turned into an SBR failure


def test_main_returns_the_engines_exit_code_and_logs_a_traceback_for_a_crash(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    codes = {}

    class FakeSync:
        def __init__(self, db): pass
        def sync_database(self, seasons): pass

    class FakeEngine:
        def __init__(self, *a, **k): pass
        def run(self):
            c = codes['next']
            if isinstance(c, Exception):
                raise c
            return c

    monkeypatch.setattr(rm, 'NBADatabaseSync', FakeSync)
    monkeypatch.setattr(rm, 'InferenceEngine', FakeEngine)
    for want in (None, 0, 4, 5):
        codes['next'] = want
        assert rm.main(['--date', '2026-11-10']) == (want or 0)
    codes['next'] = KeyError('boom')
    with caplog.at_level(logging.ERROR):
        assert rm.main(['--date', '2026-11-10']) == rm.EXIT_CRASH == 1
    assert 'INFERENCE RUN CRASHED' in caplog.text and 'Traceback' in caplog.text and "KeyError: 'boom'" in caplog.text


def test_call_with_retries_pauses_between_attempts_only():
    sl, calls = Sleeper(), []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError('down')
        return 'ok'
    assert rm.call_with_retries(flaky, 'x', sleep_fn=sl) == 'ok' and sl.slept == [10, 30]
    sl2 = Sleeper()
    with pytest.raises(OSError):
        rm.call_with_retries(lambda: (_ for _ in ()).throw(OSError('down')), 'x', sleep_fn=sl2)
    assert sl2.slept == [10, 30]


def test_exit_codes_are_distinct():
    assert (rm.EXIT_OK, rm.EXIT_CRASH, rm.EXIT_TOO_EARLY, rm.EXIT_NO_INPUTS, rm.EXIT_BETS_SUPPRESSED) == (0, 1, 3, 4, 5)


# ---------------------------------------------------------------------------------------------------------- 2. team names
LOOKUP = rm.build_team_lookup(TEAM_INFO)


@pytest.mark.parametrize('name, tri', [
    ('Los Angeles Clippers', 'LAC'), ('LA Clippers', 'LAC'), ('L.A. Clippers', 'LAC'), ('la clippers', 'LAC'), ('LA  CLIPPERS', 'LAC'),
    ('Los Angeles Lakers', 'LAL'), ('LA Lakers', 'LAL'), ('L.A. Lakers', 'LAL'), ('Clippers', 'LAC'), ('Lakers', 'LAL'),
    ('Charlotte Hornets', 'CHA'), ('Charlotte Bobcats', 'CHA'), ('Hornets', 'CHA'), ('Portland Trail Blazers', 'POR'), ('Blazers', 'POR'),
    ('philadelphia 76ers', 'PHI'), ('  Golden   State Warriors ', 'GSW'), ('GOLDEN STATE WARRIORS', 'GSW'), ('NY', 'NYK'), ('PHO', 'PHX'),
    ('BK', 'BKN'), ('CHAR', 'CHA'), ('GS', 'GSW'), ('NO', 'NOP'), ('SA', 'SAS'), ('WAS', 'WAS'),
])
def test_known_spellings_map_to_their_team(name, tri):
    assert rm.map_team_id(name, LOOKUP) == ID[tri]


@pytest.mark.parametrize('name', ['Portland', 'Los Angeles', 'LA', 'San Antonio', 'Utah', 'Charlotte', 'NY Knicks', 'Hawks of Atlanta',
                                  'Atlantis Hawks', 'Seattle SuperSonics', '', None, np.nan, 17])
def test_an_unknown_or_partial_name_is_never_guessed(name):
    # the old lookup mapped 'Los Angeles' to the Lakers, 'Portland' / 'Utah' / 'San Antonio' by substring: now they are unmapped
    assert rm.map_team_id(name, LOOKUP) is None


def test_every_tricode_full_name_and_sbr_abbreviation_is_covered():
    for tid, name, tri in TEAMS:
        assert rm.map_team_id(name, LOOKUP) == tid and rm.map_team_id(tri, LOOKUP) == tid
    assert set(rm.SBR_ABBREVIATIONS.values()) <= set(ID) and all(rm.map_team_id(a, LOOKUP) for a in rm.SBR_ABBREVIATIONS)
    assert len({tid for tid in LOOKUP.values()}) == 30


def test_a_normalised_name_that_stands_for_two_teams_fails_loudly():
    with pytest.raises(ValueError, match='two team ids'):
        rm.build_team_lookup(TEAM_INFO, aliases={'Hawks': 'ATL', 'Celtics': 'ATL', 'hawks!': 'BOS'})


def test_no_substring_match_is_left_in_the_source():
    import inspect
    src = inspect.getsource(rm.InferenceEngine.get_todays_games) + inspect.getsource(rm.map_team_id)
    assert 'k in name' not in src and 'name in k' not in src and "split(' ')[-1]" not in src


def _old_map_id(team_info):
    """The pre-B3 lookup, verbatim: names, tricodes and last words, manual fixes, then the first substring hit."""
    name_map = {}
    for _, t in team_info.iterrows():
        name_map[t['teamName']] = t['teamId']
        name_map[t['teamTricode']] = t['teamId']
        name_map[t['teamName'].split(' ')[-1]] = t['teamId']
    name_map['LA Clippers'] = 1610612746
    name_map['L.A. Clippers'] = 1610612746
    name_map['LA Lakers'] = 1610612747
    name_map['L.A. Lakers'] = 1610612747
    name_map['Clippers'] = 1610612746
    name_map['Lakers'] = 1610612747

    def map_id(name):
        if not isinstance(name, str): return None
        if name in name_map: return name_map[name]
        for k, v in name_map.items():
            if k in name or name in k: return v
        return None
    return map_id


@pytest.mark.db
def test_every_team_name_in_the_odds_tables_maps_to_the_same_id_as_the_old_code():
    conn = sqlite3.connect(Path(__file__).resolve().parents[2] / 'data' / 'nba_stats.db')
    team_info = pd.read_sql("SELECT DISTINCT teamId, teamName, teamTricode FROM team_basic_stats", conn)
    names = set()
    for table_name in ('nba_odds', 'nba_odds_opening', 'odds_snapshots'):
        for col in ('home_team', 'away_team'):
            names |= {n for n, in conn.execute(f"SELECT DISTINCT {col} FROM {table_name}") if n is not None}
    assert len(names) >= 30
    old, lookup = _old_map_id(team_info), rm.build_team_lookup(team_info)
    different = {n: (old(n), rm.map_team_id(n, lookup)) for n in names if old(n) != rm.map_team_id(n, lookup)}
    assert different == {}, different
    assert all(rm.map_team_id(n, lookup) is not None for n in names)         # none of them is left unmapped
    # the abbreviation columns: every value is a tricode once the SBR abbreviations are applied
    abbrs = set()
    for table_name in ('nba_odds', 'nba_odds_opening', 'odds_snapshots'):
        for col in ('home_team_abbr', 'away_team_abbr'):
            abbrs |= {a for a, in conn.execute(f"SELECT DISTINCT {col} FROM {table_name}") if a is not None}
    assert {rm.SBR_ABBREVIATIONS.get(a, a) for a in abbrs} <= set(team_info['teamTricode'])


# ---------------------------------------------------------------------------------------------------------- 3. skipped games
def _mixed_world(monkeypatch):
    schedule = SCHEDULE + [('0022600003', 'MIA', 'ORL'), ('0022600004', 'DEN', 'UTA')]
    miami = sbr_game('Miami Heat', 'Orlando Magic', 'MIA', 'ORL')
    curr = [G_BOS_NYK, G_LAL_LAC, miami,
            sbr_game('Atlantis Hawks', 'Chicago Bulls', 'ATL', 'CHI'),                 # an unknown team name
            sbr_game('Houston Rockets', 'Dallas Mavericks', 'HOU', 'DAL')]             # not on the NBA schedule; DEN @ UTA is not listed by SBR
    opened = [G_BOS_NYK, G_LAL_LAC, open_only(miami, home_ml={}, away_ml={}, home_spread={}, away_spread={}, total={}),
              sbr_game('Atlantis Hawks', 'Chicago Bulls', 'ATL', 'CHI'), sbr_game('Houston Rockets', 'Dallas Mavericks', 'HOU', 'DAL')]
    return World(monkeypatch, schedule=schedule, curr=curr, opened=opened)


def test_every_dropped_game_leaves_a_record_and_the_run_still_exits_0(engine_factory, monkeypatch, caplog):
    _mixed_world(monkeypatch)
    eng = engine_factory.make()
    with caplog.at_level(logging.INFO):
        assert eng.run() == 0                                                  # some skipped, at least one predicted
    assert [e['gameId'] for e in engine_factory.logged] == ['0022600001', '0022600002']
    sk = table(engine_factory.db, 'inference_skipped_games')
    by_reason = {r.reason: r for r in sk.itertuples()}
    assert sorted(by_reason) == ['missing_opening_line', 'no_odds', 'not_on_schedule', 'unmapped_team'] and len(sk) == 4
    assert (by_reason['missing_opening_line'].game_id, by_reason['missing_opening_line'].home_team) == ('0022600003', 'Orlando Magic')
    assert (by_reason['no_odds'].game_id, by_reason['no_odds'].home_team, by_reason['no_odds'].away_team) == ('0022600004', 'UTA', 'DEN')
    assert by_reason['not_on_schedule'].game_id is None and by_reason['not_on_schedule'].home_team == 'Dallas Mavericks'
    assert by_reason['unmapped_team'].game_id is None and "'Atlantis Hawks'" in by_reason['unmapped_team'].detail
    assert sk.run_ts_utc.nunique() == 1 and sk.game_date.nunique() == 1
    errs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("UNMAPPED TEAM: 'Atlantis Hawks'" in m for m in errs)           # the ERROR names the string
    picks = pd.read_csv('todays_picks.csv', dtype={'game_id': str})
    assert list(picks['game_id']) == ['0022600001', '0022600002']
    bets = table(engine_factory.db, 'bet_recommendations_log')
    assert bets.run_ts_utc.nunique() == 1 and set(bets.run_ts_utc) == set(sk.run_ts_utc)      # one run, one stamp, in both tables


def test_the_skipped_games_table_is_append_only_across_runs(engine_factory, monkeypatch):
    _mixed_world(monkeypatch)
    engine_factory.make().run()
    engine_factory.make().run()
    sk = table(engine_factory.db, 'inference_skipped_games')
    assert len(sk) == 8 and sk.run_ts_utc.nunique() == 2


def test_when_every_scheduled_game_is_skipped_the_exit_code_is_4(engine_factory, monkeypatch, caplog):
    World(monkeypatch, schedule=SCHEDULE, curr=[sbr_game('Atlantis Hawks', 'Chicago Bulls', 'ATL', 'CHI')],
          opened=[sbr_game('Atlantis Hawks', 'Chicago Bulls', 'ATL', 'CHI')])
    eng = engine_factory.make()
    assert eng.run() == 4
    sk = table(engine_factory.db, 'inference_skipped_games')
    assert sorted(sk.reason) == ['no_odds', 'no_odds', 'unmapped_team']
    assert 'ALL GAMES SKIPPED' in caplog.text and engine_factory.logged == []


def test_a_duplicated_sbr_row_fails_loudly_instead_of_doubling_a_game(engine_factory, monkeypatch):
    World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_BOS_NYK, G_LAL_LAC], opened=[G_BOS_NYK, G_LAL_LAC])
    with pytest.raises(pd.errors.MergeError):
        engine_factory.make().run()


def test_an_sbr_abbreviation_the_schedule_does_not_use_is_reported_as_no_odds_with_the_leftover_rows(engine_factory, monkeypatch):
    World(monkeypatch, schedule=SCHEDULE[:1], curr=[sbr_game('Boston Celtics', 'New York Knicks', 'BOS', 'NYX')],
          opened=[sbr_game('Boston Celtics', 'New York Knicks', 'BOS', 'NYX')])
    assert engine_factory.make().run() == 4
    sk = table(engine_factory.db, 'inference_skipped_games')
    assert sorted(sk.reason) == ['no_odds', 'not_on_schedule']
    assert 'Boston Celtics @ New York Knicks' in sk.loc[sk.reason == 'no_odds', 'detail'].iloc[0]      # the detail names the unmatched SBR row


def test_a_normal_slate_writes_no_skip_rows_and_changes_nothing(engine_factory, monkeypatch):
    World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_LAL_LAC], opened=[G_BOS_NYK, G_LAL_LAC])
    assert engine_factory.make().run() == 0
    with sqlite3.connect(engine_factory.db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'inference_skipped_games' not in tables or table(engine_factory.db, 'inference_skipped_games').empty
    picks = pd.read_csv('todays_picks.csv', dtype={'game_id': str})
    assert list(picks['game_id']) == ['0022600001', '0022600002'] and (picks['total_bet'] == 'over').all()
    assert (picks['recommendation'].str.contains('TOTAL OVER')).all()
    assert table(engine_factory.db, 'bet_recommendations_log')['bet_suppressed_reason'].isna().all()


# ---------------------------------------------------------------------------------------------------------- 4. Eastern time
class LocalClock:
    """The machine's local wall clock (Pacific) and the Eastern one, as two views of one instant."""
    def __init__(self, local, eastern):
        self.local, self.eastern = pd.Timestamp(local), pd.Timestamp(eastern)

    def install(self, monkeypatch):
        local = self.local.to_pydatetime()

        class FakeDateTime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return local if tz is None else datetime.datetime.now(tz)

        class FakeDate(datetime.date):
            @classmethod
            def today(cls):
                return local.date()
        monkeypatch.setattr(rm, 'datetime', SimpleNamespace(datetime=FakeDateTime, date=FakeDate, timezone=datetime.timezone,
                                                            timedelta=datetime.timedelta))
        monkeypatch.setattr(rm, 'now_et', lambda: self.eastern.to_pydatetime())


@pytest.mark.parametrize('local, eastern, et_date', [
    ('2026-11-10 23:30', '2026-11-11 02:30', '2026-11-11'),     # 23:30 Pacific is already tomorrow in Eastern time
    ('2026-11-11 21:30', '2026-11-12 00:30', '2026-11-12'),     # 00:30 ET: the machine still says the 11th
    ('2026-11-11 09:15', '2026-11-11 12:15', '2026-11-11'),     # the scheduled run: both zones agree on the date
])
def test_the_slate_date_is_the_eastern_date_not_the_machines(engine_factory, monkeypatch, local, eastern, et_date):
    LocalClock(local, eastern).install(monkeypatch)
    w = World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_LAL_LAC], opened=[G_BOS_NYK, G_LAL_LAC])
    assert rm.get_nba_schedule() is not None
    assert w.calls.schedule == [et_date]                                    # the default date of get_nba_schedule
    w.calls.schedule.clear()
    assert engine_factory.make().run() == 0
    assert set(w.calls.schedule) == {et_date} and {c[0] for c in w.calls.sbr} == {et_date}   # schedule and SBR both asked for the ET date
    picks = pd.read_csv('todays_picks.csv', dtype={'game_id': str})
    assert set(picks['game_date']) == {et_date}
    # prediction_date is deliberately still machine-local, same format (the published dashboard reads it)
    assert set(picks['prediction_date']) == {f"{pd.Timestamp(local):%Y-%m-%d %H:%M:%S}"}
    assert table(engine_factory.db, 'bet_recommendations_log')['run_ts_utc'].str.endswith('Z').all()
    assert (table(engine_factory.db, 'bet_recommendations_log')['game_date'] == et_date).all()


def test_the_season_is_decided_in_eastern_time(monkeypatch):
    # 2026-09-30 21:30 Pacific is 2026-10-01 00:30 Eastern: the new season has started in ET, not on the machine's clock
    LocalClock('2026-09-30 21:30', '2026-10-01 00:30').install(monkeypatch)
    assert rm.current_season_et() == '2026-27' and rm.season_for_date(rm.datetime.date.today()) == '2025-26'


def test_main_syncs_the_season_of_the_eastern_date(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    LocalClock('2026-09-30 21:30', '2026-10-01 00:30').install(monkeypatch)
    synced = []

    class FakeSync:
        def __init__(self, db): pass
        def sync_database(self, seasons): synced.append(seasons)

    class FakeEngine:
        def __init__(self, *a, **k): pass
        def run(self): return 0
    monkeypatch.setattr(rm, 'NBADatabaseSync', FakeSync)
    monkeypatch.setattr(rm, 'InferenceEngine', FakeEngine)
    assert rm.main(['--no-wait-for-injury-cutoff']) == 0 and synced == [['2026-27']]


def test_the_injury_cutoff_guard_reads_the_eastern_day(monkeypatch):
    # 23:30 Pacific on the 10th: the guard must plan for the ET game day (the 11th), whose report is not readable until 12:05 ET
    LocalClock('2026-11-10 23:30', '2026-11-11 02:30').install(monkeypatch)
    asked = []
    result = rm.injury_cutoff_guard(schedule_fn=lambda d: asked.append(d) or pd.DataFrame({'gameTimeUTC': ['2026-11-12T00:30:00Z']}),
                                    sleep_fn=Sleeper())
    assert asked == ['2026-11-11'] and result == 'too_early'


def test_no_naive_local_date_is_left_in_the_job_except_prediction_date():
    import inspect
    src = inspect.getsource(rm)
    assert 'date.today()' not in src
    local_now = [ln.strip() for ln in src.splitlines() if 'datetime.datetime.now()' in ln]
    assert local_now and all('prediction_date' in ln for ln in local_now), local_now      # only the documented prediction_date stamps


# ---------------------------------------------------------------------------------------------------------- 5. injury report failed or stale
def _reports(*stamps, game_date='2026-11-11'):
    rows = [{'source_file': f"Injury-Report_{pd.Timestamp(s):%Y-%m-%d_%I_%M%p}.pdf", 'report_ts': pd.Timestamp(s),
             'game_date': pd.Timestamp(game_date)} for s in stamps]
    return pd.DataFrame(rows, columns=['source_file', 'report_ts', 'game_date'])


def _injury_engine(monkeypatch, tmp_path, now, reports, fetched, inj_missing=0):
    import core.scrape_injury_reports as sir
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sir, 'now_et', lambda: pd.Timestamp(now).to_pydatetime())
    monkeypatch.setattr(sir, 'live', lambda as_of=None: fetched)
    monkeypatch.setattr(sir, 'time', SimpleNamespace(sleep=lambda s: None))
    monkeypatch.setattr(rm.irf, 'parse_all_reports', lambda *a, **k: reports)
    monkeypatch.setattr(rm.irf, 'build_injury_features',
                        lambda *a, **k: pd.DataFrame({'gameId': ['1', '1'], 'teamId': [10, 20], 'inj_lost': 0.0, 'inj_lost_new': 0.0,
                                                      'inj_n_out': 0.0, 'inj_missing': inj_missing}))
    import core.injury_downgrades as dg, core.injury_source_compare as sc
    monkeypatch.setattr(dg, 'build_downgrade_features', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('stub')))
    monkeypatch.setattr(sc, 'log_side_by_side', lambda *a, **k: None)
    eng = object.__new__(rm.InferenceEngine)
    eng.db_path, eng.injury_cutoff, eng.injury_report_path = 'x.db', None, None
    games = pd.DataFrame({'gameId': ['1'], 'gameDate': ['2026-11-11'], 'teamId_home': [10], 'teamId_away': [20],
                          'gameTimeUTC': ['2026-11-12T00:30:00Z']})
    return eng, games


NOON_PATH = 'data/raw/injury_reports/2026/Injury-Report_2026-11-11_12_00PM.pdf'


@pytest.mark.parametrize('label, now, stamps, fetched, code', [
    ('fresh noon report downloaded', '2026-11-11 12:06', ['11:00', '11:30', '12:00'], NOON_PATH, None),
    ('download failed but the noon file is already on disk (the snapshot job saved it)', '2026-11-11 12:06', ['11:30', '12:00'], None, None),
    ('download failed, the 11:30 file is within the tolerance', '2026-11-11 12:06', ['11:00', '11:30'], None, None),
    ('download failed, only the 11:00 report on disk: a whole grid step behind', '2026-11-11 12:06', ['10:00', '11:00'], None,
     'injury_snapshot_mismatch'),
    ('no report at all for the date', '2026-11-11 12:06', [], None, 'injury_report_missing'),
    ('only a report from before dawn: more than 6 hours old', '2026-11-11 12:06', ['05:00'], None, 'injury_report_stale'),
])
def test_when_the_report_the_model_needs_is_missing_or_stale(monkeypatch, tmp_path, label, now, stamps, fetched, code):
    reports = _reports(*[f'2026-11-11 {s}' for s in stamps])
    eng, games = _injury_engine(monkeypatch, tmp_path, now, reports, fetched)
    eng.get_injury_features(games)
    assert (eng.injury_problem[0] if eng.injury_problem else None) == code, label


def test_a_team_missing_from_a_fresh_report_is_not_a_failure(monkeypatch, tmp_path):
    eng, games = _injury_engine(monkeypatch, tmp_path, '2026-11-11 12:06', _reports('2026-11-11 12:00'), NOON_PATH, inj_missing=1)
    inj = eng.get_injury_features(games)
    assert inj['inj_missing'].sum() == 2 and eng.injury_problem is None        # inj_missing=1 is what training encodes: normal


def test_the_download_is_retried_three_times_with_a_backoff(monkeypatch, tmp_path):
    import core.scrape_injury_reports as sir
    eng, games = _injury_engine(monkeypatch, tmp_path, '2026-11-11 12:06', _reports('2026-11-11 12:00'), None)
    attempts, slept = [], []
    monkeypatch.setattr(sir, 'live', lambda as_of=None: attempts.append(as_of) or (NOON_PATH if len(attempts) == 3 else None))
    monkeypatch.setattr(sir, 'time', SimpleNamespace(sleep=slept.append))
    eng.get_injury_features(games)
    assert len(attempts) == 3 and slept == [10, 30] and eng.injury_report_path == NOON_PATH


def _bets_engine(engine_factory, problem):
    eng = engine_factory.make()

    def features(games_df):
        eng.injury_problem = problem
        eng.injury_cutoff, eng.injury_report_path = pd.Timestamp('2026-11-11 12:00'), 'data/raw/injury_reports/r.pdf'
        return pd.DataFrame({'gameId': [], 'teamId': []}).set_index(['gameId', 'teamId'])
    eng.get_injury_features = features
    return eng


@pytest.mark.parametrize('code', ['injury_report_missing', 'injury_report_stale', 'injury_snapshot_mismatch'])
def test_a_bad_injury_report_withholds_every_bet_but_keeps_the_predictions_and_edges(engine_factory, monkeypatch, caplog, code):
    World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_LAL_LAC], opened=[G_BOS_NYK, G_LAL_LAC])
    eng = _bets_engine(engine_factory, (code, 'the detail'))
    with caplog.at_level(logging.INFO):
        assert eng.run() == rm.EXIT_BETS_SUPPRESSED == 5
    assert len(engine_factory.logged) == 2                                    # predictions are still logged, exactly as on a normal day
    assert engine_factory.logged[0]['pred_total'] == 230.0
    picks = pd.read_csv('todays_picks.csv', dtype={'game_id': str})
    assert picks[['ml_bet', 'spread_bet', 'total_bet']].isna().all().all() and picks['recommendation'].isna().all()
    assert (picks['total_edge'] > 0.05).all()                                  # the edge is still recorded
    bets = table(engine_factory.db, 'bet_recommendations_log')
    assert bets[['ml_bet', 'spread_bet', 'total_bet']].isna().all().all() and (bets['total_edge'] > 0.05).all()
    assert (bets['bet_suppressed_reason'] == code).all()
    errs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any(m.startswith(f'BETS SUPPRESSED ({code})') for m in errs)


def test_the_same_slate_with_a_good_report_is_unchanged(engine_factory, monkeypatch):
    World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_LAL_LAC], opened=[G_BOS_NYK, G_LAL_LAC])
    eng = _bets_engine(engine_factory, None)
    assert eng.run() == 0
    picks = pd.read_csv('todays_picks.csv', dtype={'game_id': str})
    assert (picks['total_bet'] == 'over').all() and picks['recommendation'].str.startswith('TOTAL OVER').all()
    bets = table(engine_factory.db, 'bet_recommendations_log')
    assert (bets['total_bet'] == 'over').all() and bets['bet_suppressed_reason'].isna().all()


def test_failed_injury_features_withhold_the_bets_too(engine_factory, monkeypatch, caplog):
    World(monkeypatch, schedule=SCHEDULE, curr=[G_BOS_NYK, G_LAL_LAC], opened=[G_BOS_NYK, G_LAL_LAC])
    eng = engine_factory.make()
    eng.get_injury_features = lambda g: (_ for _ in ()).throw(RuntimeError('parser broke'))
    assert eng.run() == 5
    bets = table(engine_factory.db, 'bet_recommendations_log')
    assert (bets['bet_suppressed_reason'] == 'injury_features_failed').all() and bets['total_bet'].isna().all()
    assert 'RuntimeError: parser broke' in caplog.text


def test_suppression_is_all_or_nothing_for_the_slate_and_does_not_hide_skips(engine_factory, monkeypatch):
    _mixed_world(monkeypatch)
    eng = _bets_engine(engine_factory, ('injury_report_stale', 'old'))
    assert eng.run() == 5                                                     # 5, not 4: games were predicted
    assert len(table(engine_factory.db, 'inference_skipped_games')) == 4


# ---------------------------------------------------------------------------------------------------------- the scheduler wrapper (text only: never executed)
def test_the_bat_documents_every_exit_code_and_the_fire_time():
    bat = (Path(__file__).resolve().parents[2] / 'scripts' / 'run_daily_inference.bat').read_text(encoding='utf-8')
    for code in ('0 :', '1 :', '3 :', '4 :', '5 :'):
        assert f'REM   {code}' in bat, code
    assert '09:15 Pacific = 12:15 ET' in bat and 'from 09:05 Pacific' in bat and 'up to 90 minutes' in bat and '09:05 Pacific = 12:05' not in bat
    lines = [ln.strip() for ln in bat.splitlines()]
    rerun = next(i for i, ln in enumerate(lines) if 'schedule_odds_rerun.py' in ln and not ln.startswith('REM'))
    goto = next(i for i, ln in enumerate(lines) if ln == 'if "%INFER_RC%"=="4" goto PUBLISH')
    label = lines.index(':PUBLISH')
    publish = next(i for i, ln in enumerate(lines) if 'export_dashboard_data.py --publish' in ln)
    assert goto < rerun < label < publish                       # code 4 jumps over the re-run scheduler to the publish
    assert lines.index('if "%INFER_RC%"=="3" (') < goto         # code 3 still exits first, before both
    assert 'exit /b %INFER_RC%' in bat and bat.rstrip().endswith('exit /b %EXITCODE%')
