"""The daily run must not read an injury report older than the training snapshot (noon ET on game day, capped at tip - 30 min).
run_inference_v2.injury_cutoff_guard waits for it (fake clock and fake sleep here: no real waiting, no network), refuses a run that is
hours too early, and get_injury_features logs an ERROR when the report it ends up using is older than the snapshot expects."""
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

rm = pytest.importorskip('run_inference_v2')

ALLOWANCE = rm.REPORT_PUBLICATION_ALLOWANCE
EVENING = ['2026-11-12T00:30:00Z']          # 7:30 PM ET on 2026-11-11 (EST)


class Clock:
    """A fake ET wall clock: sleep(s) moves it forward and nothing really waits."""
    def __init__(self, t):
        self.t = pd.Timestamp(t)
        self.slept = []

    def now(self):
        return self.t.to_pydatetime()

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += pd.Timedelta(seconds=seconds)


def _guard(clock, tips_utc=EVENING, **kw):
    sched = lambda d: pd.DataFrame({'gameTimeUTC': tips_utc})
    return rm.injury_cutoff_guard(schedule_fn=sched, now_fn=clock.now, sleep_fn=clock.sleep, **kw)


def test_constants_are_what_the_notes_say():
    # 90, not 120: the scheduled task's execution time limit is 2 hours and the run needs time after the wait
    assert ALLOWANCE == pd.Timedelta(minutes=5) and rm.DEFAULT_MAX_WAIT_MINUTES == 90
    assert rm.irf.SNAPSHOT_CLOCK[rm.GameFeatureBuilder.INJURY_SNAPSHOT] == pd.Timedelta(hours=12)


def test_default_maximum_wait_is_90_minutes_exactly():
    assert _guard(Clock('2026-11-11 10:35')) == 'waited'          # 90 min before 12:05: waits
    assert _guard(Clock('2026-11-11 10:34')) == 'too_early'       # 91 min: refuses


def test_run_at_11_15_waits_until_the_allowance_after_noon_then_proceeds(caplog):
    caplog.set_level(logging.INFO)
    clock = Clock('2026-11-11 11:15')
    assert _guard(clock) == 'waited'
    assert clock.t == pd.Timestamp('2026-11-11 12:05')
    assert sum(clock.slept) == 50 * 60 and max(clock.slept) <= rm.WAIT_STEP_SECONDS and len(clock.slept) > 1    # short steps
    msgs = [r.getMessage() for r in caplog.records if 'cutoff guard' in r.getMessage()]
    assert len(msgs) == 2 and 'waiting 50.0 min until 12:05' in msgs[0] and 'wait over' in msgs[1]       # once at the start, once at the end


@pytest.mark.parametrize('now', ['2026-11-11 12:05', '2026-11-11 12:20', '2026-11-11 19:00'])
def test_run_after_the_allowance_proceeds_at_once(now):
    clock = Clock(f'{now}')
    assert _guard(clock) == 'ready' and clock.slept == []


def test_run_at_08_00_with_the_default_maximum_is_too_early(caplog):
    caplog.set_level(logging.INFO)
    clock = Clock('2026-11-11 08:00')
    assert _guard(clock) == 'too_early' and clock.slept == []
    err = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(err) == 1 and 'RUN TOO EARLY' in err[0].getMessage() and '12:05' in err[0].getMessage()


def test_max_wait_minutes_is_configurable():
    assert _guard(Clock('2026-11-11 08:00'), max_wait_minutes=300) == 'waited'
    assert _guard(Clock('2026-11-11 11:15'), max_wait_minutes=40) == 'too_early'          # 50 min away
    assert _guard(Clock('2026-11-11 11:15'), max_wait_minutes=50) == 'waited'


def test_no_wait_flag_proceeds_with_a_warning_and_does_not_sleep(caplog):
    caplog.set_level(logging.INFO)
    clock = Clock('2026-11-11 11:15')
    assert _guard(clock, wait=False) == 'ignored' and clock.slept == []
    assert [r for r in caplog.records if r.levelno == logging.WARNING and 'DISABLED' in r.getMessage()]
    clock = Clock('2026-11-11 05:00')                        # even hours early: the old behaviour, no refusal
    assert _guard(clock, wait=False) == 'ignored'


def test_slate_of_only_early_tips_waits_only_for_the_latest_early_cutoff():
    # tips 11:00 and 11:30 ET (a morning game and a noon-ish one) -> cutoffs 10:30 and 11:00, both before noon
    tips = ['2026-11-11T16:00:00Z', '2026-11-11T16:30:00Z']
    assert rm.snapshot_nominal_cutoff(pd.Timestamp('2026-11-11'), rm.tip_times_et(tips)) == pd.Timestamp('2026-11-11 11:00')
    clock = Clock('2026-11-11 10:40')
    assert _guard(clock, tips_utc=tips) == 'waited' and clock.t == pd.Timestamp('2026-11-11 11:05')   # not 12:05
    clock = Clock('2026-11-11 11:15')
    assert _guard(clock, tips_utc=tips) == 'ready' and clock.slept == []


def test_one_evening_game_keeps_the_noon_rule_even_with_early_games_on_the_slate():
    tips = ['2026-11-11T16:00:00Z', '2026-11-12T00:30:00Z']          # 11:00 ET and 7:30 PM ET
    assert rm.earliest_run_time(pd.Timestamp('2026-11-11'), rm.tip_times_et(tips)) == pd.Timestamp('2026-11-11 12:05')


def test_a_tip_just_after_noon_caps_the_cutoff_at_tip_minus_30():
    tips = rm.tip_times_et(['2026-11-11T17:10:00Z'])                # 12:10 ET -> cutoff 11:40 (< noon)
    assert rm.snapshot_nominal_cutoff(pd.Timestamp('2026-11-11'), tips) == pd.Timestamp('2026-11-11 11:40')
    tips = rm.tip_times_et(['2026-11-11T17:40:00Z'])                # 12:40 ET -> noon applies
    assert rm.snapshot_nominal_cutoff(pd.Timestamp('2026-11-11'), tips) == pd.Timestamp('2026-11-11 12:00')


def test_the_slate_cutoff_is_the_trainings_own_function():
    g = pd.DataFrame({'game_date': pd.Timestamp('2026-11-11'), 'tip_ts': pd.to_datetime(['2026-11-11 11:40', '2026-11-11 19:30'])})
    assert rm.irf.SNAPSHOT_CUTOFFS['game_day_noon'](g).max() == rm.snapshot_nominal_cutoff(
        pd.Timestamp('2026-11-11'), pd.to_datetime(['2026-11-11 11:40', '2026-11-11 19:30']))


@pytest.mark.parametrize('day, tip_utc', [
    ('2026-11-01', '2026-11-02T00:30:00Z'),     # clocks went back that night: 7:30 PM EST
    ('2027-03-14', '2027-03-14T23:30:00Z'),     # clocks went forward that night: 7:30 PM EDT
    ('2026-11-11', '2026-11-12T00:30:00Z'),     # an ordinary day, for comparison
])
def test_daylight_saving_dates_behave_like_any_other_day(day, tip_utc):
    assert rm.tip_times_et([tip_utc]).iloc[0] == pd.Timestamp(f'{day} 19:30')
    noon_ready = pd.Timestamp(f'{day} 12:05')
    clock = Clock(f'{day} 11:15')
    assert _guard(clock, tips_utc=[tip_utc]) == 'waited' and clock.t == noon_ready and sum(clock.slept) == 50 * 60
    assert _guard(Clock(f'{day} 12:20'), tips_utc=[tip_utc]) == 'ready'
    assert _guard(Clock(f'{day} 08:00'), tips_utc=[tip_utc]) == 'too_early'


def test_an_early_tip_on_a_daylight_saving_date_is_converted_with_that_days_offset():
    # 2027-03-14, 12:00 PM EDT = 16:00Z; the same instant on 2026-11-01 (EST) would be 17:00Z
    assert rm.tip_times_et(['2027-03-14T16:00:00Z']).iloc[0] == pd.Timestamp('2027-03-14 12:00')
    assert rm.tip_times_et(['2026-11-01T17:00:00Z']).iloc[0] == pd.Timestamp('2026-11-01 12:00')
    assert rm.snapshot_nominal_cutoff(pd.Timestamp('2027-03-14'), rm.tip_times_et(['2027-03-14T16:00:00Z'])) == pd.Timestamp('2027-03-14 11:30')


def test_without_a_slate_the_clock_rule_is_used(caplog):
    caplog.set_level(logging.INFO)

    def boom(d):
        raise RuntimeError('scoreboard down')
    for sched in (boom, lambda d: pd.DataFrame(), lambda d: pd.DataFrame({'other': [1]})):
        clock = Clock('2026-11-11 11:15')
        assert rm.injury_cutoff_guard(schedule_fn=sched, now_fn=clock.now, sleep_fn=clock.sleep) == 'waited'
        assert clock.t == pd.Timestamp('2026-11-11 12:05')
    assert any('clock rule' in r.getMessage() for r in caplog.records)


def test_unparsable_tip_times_fall_back_to_the_clock():
    assert rm.earliest_run_time(pd.Timestamp('2026-11-11'), rm.tip_times_et(['not a time', None])) == pd.Timestamp('2026-11-11 12:05')


# ------------------------------------------------------------------ main(): which runs are guarded

@pytest.fixture()
def harness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clock = Clock('2026-11-11 08:00')
    state = SimpleNamespace(clock=clock, engines=[], synced=[], runs=[])

    class FakeSync:
        def __init__(self, db): pass
        def sync_database(self, seasons): state.synced.append(seasons)

    class FakeEngine:
        def __init__(self, db_path, suffix=None, target_date=None, only_unpriced=False):
            state.engines.append((target_date, only_unpriced))
        def run(self): state.runs.append(clock.t)

    monkeypatch.setattr(rm, 'NBADatabaseSync', FakeSync)
    monkeypatch.setattr(rm, 'InferenceEngine', FakeEngine)
    monkeypatch.setattr(rm, 'now_et', clock.now)
    monkeypatch.setattr(rm, 'time', SimpleNamespace(sleep=clock.sleep))
    monkeypatch.setattr(rm, 'get_nba_schedule', lambda d=None: pd.DataFrame({'gameTimeUTC': EVENING}))
    return state


def test_main_08_00_exits_non_zero_and_predicts_nothing(harness, tmp_path):
    assert rm.main([]) == rm.EXIT_TOO_EARLY != 0
    assert harness.engines == [] and harness.runs == [] and harness.clock.slept == []
    assert harness.synced == [[rm.current_season]]                    # the database sync still ran first
    assert not (tmp_path / 'todays_picks.csv').exists() and not (tmp_path / 'data').exists()


def test_main_waits_after_the_sync_then_runs(harness):
    harness.clock.t = pd.Timestamp('2026-11-11 11:15')
    assert rm.main([]) == 0
    assert harness.synced and harness.runs == [pd.Timestamp('2026-11-11 12:05')]


def test_main_no_wait_flag_runs_at_once_with_a_warning(harness, caplog):
    caplog.set_level(logging.INFO)
    harness.clock.t = pd.Timestamp('2026-11-11 11:15')
    assert rm.main(['--no-wait-for-injury-cutoff']) == 0
    assert harness.runs == [pd.Timestamp('2026-11-11 11:15')] and harness.clock.slept == []
    assert any(r.levelno == logging.WARNING and 'DISABLED' in r.getMessage() for r in caplog.records)


def test_main_max_wait_minutes_flag(harness):
    harness.clock.t = pd.Timestamp('2026-11-11 08:00')
    assert rm.main(['--max-wait-minutes', '300']) == 0 and harness.runs == [pd.Timestamp('2026-11-11 12:05')]


@pytest.mark.parametrize('argv, engine_args', [(['--date', '2026-11-10'], ('2026-11-10', False)), (['--only-unpriced'], (None, True))])
def test_backfills_and_reruns_are_never_delayed(harness, argv, engine_args):
    def no_guard(**kw):
        raise AssertionError('the guard must not run for --date or --only-unpriced')
    rm.injury_cutoff_guard, saved = no_guard, rm.injury_cutoff_guard
    try:
        assert rm.main(argv) == 0                                      # 08:00 ET: would be too early for a daily run
    finally:
        rm.injury_cutoff_guard = saved
    assert harness.engines == [engine_args] and harness.clock.slept == []


# ------------------------------------------------------------------ the report actually used

def _reports(*stamps):
    rows = [{'source_file': f"Injury-Report_{pd.Timestamp(s):%Y-%m-%d_%I_%M%p}.pdf", 'report_ts': pd.Timestamp(s), 'game_date': pd.Timestamp('2026-11-11')}
            for s in stamps]
    return pd.DataFrame(rows)


def test_snapshot_report_problem_threshold():
    nominal = pd.Timestamp('2026-11-11 12:00')
    assert rm.snapshot_report_problem(pd.Timestamp('2026-11-11 12:00'), nominal) is None
    assert rm.snapshot_report_problem(pd.Timestamp('2026-11-11 11:30'), nominal) is None      # old-era hourly files: header 30 min after the slot
    assert rm.snapshot_report_problem(pd.Timestamp('2026-11-11 11:00'), nominal) == 60.0      # the post-DST failure; one grid step would have missed it
    assert rm.snapshot_report_problem(None, nominal) is None                                  # no report at all is check_report_freshness's case


def test_expected_grid_slot():
    assert rm.expected_grid_slot(pd.Timestamp('2026-11-11 12:00')) == pd.Timestamp('2026-11-11 12:00')
    assert rm.expected_grid_slot(pd.Timestamp('2026-11-11 11:40')) == pd.Timestamp('2026-11-11 11:00')
    assert rm.expected_grid_slot(pd.Timestamp('2026-11-11 10:30')) is None


def _engine_for_injuries(monkeypatch, tmp_path, now, reports, fetched):
    import core.scrape_injury_reports as sir
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sir, 'now_et', lambda: pd.Timestamp(now).to_pydatetime())
    monkeypatch.setattr(sir, 'live', lambda as_of=None: fetched)
    monkeypatch.setattr(sir, 'time', SimpleNamespace(sleep=lambda s: None))          # the download retry's backoff must not really sleep
    monkeypatch.setattr(rm.irf, 'parse_all_reports', lambda *a, **k: reports)
    monkeypatch.setattr(rm.irf, 'build_injury_features',
                        lambda *a, **k: pd.DataFrame({'gameId': ['1', '1'], 'teamId': [10, 20], 'inj_lost': 0.0, 'inj_lost_new': 0.0,
                                                      'inj_n_out': 0.0, 'inj_missing': 0}))
    import core.injury_downgrades as dg, core.injury_source_compare as sc
    monkeypatch.setattr(dg, 'build_downgrade_features', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('stub')))
    monkeypatch.setattr(sc, 'log_side_by_side', lambda *a, **k: None)
    eng = object.__new__(rm.InferenceEngine)
    eng.db_path, eng.injury_cutoff, eng.injury_report_path = 'x.db', None, None
    games = pd.DataFrame({'gameId': ['1'], 'gameDate': ['2026-11-11'], 'teamId_home': [10], 'teamId_away': [20], 'gameTimeUTC': EVENING})
    return eng, games


def test_run_at_11_19_is_reported_as_using_the_wrong_report(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    reports = _reports('2026-11-11 10:30', '2026-11-11 11:00', '2026-11-11 11:30')
    path = 'data/raw/injury_reports/2026/Injury-Report_2026-11-11_11_00AM.pdf'
    eng, games = _engine_for_injuries(monkeypatch, tmp_path, '2026-11-11 11:19', reports, path)
    eng.get_injury_features(games)
    errs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    # the newest report at 11:19 is the 11:00 one (11:30 is later than the cutoff): 60 min behind the noon snapshot
    assert any('INJURY SNAPSHOT MISMATCH' in m and '11_00AM' in m and '12:00' in m for m in errs), errs
    assert eng.injury_cutoff == pd.Timestamp('2026-11-11 11:19') and eng.injury_report_path.endswith('11_00AM.pdf')


def test_run_after_the_wait_reads_the_noon_report_without_an_error(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    reports = _reports('2026-11-11 11:00', '2026-11-11 11:30', '2026-11-11 12:00')
    path = 'data/raw/injury_reports/2026/Injury-Report_2026-11-11_12_00PM.pdf'
    eng, games = _engine_for_injuries(monkeypatch, tmp_path, '2026-11-11 12:06', reports, path)
    eng.get_injury_features(games)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert eng.injury_cutoff == pd.Timestamp('2026-11-11 12:00') and eng.injury_report_path == path


def test_when_the_download_failed_the_report_really_used_is_recorded(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    reports = _reports('2026-11-11 11:00', '2026-11-11 11:30')
    eng, games = _engine_for_injuries(monkeypatch, tmp_path, '2026-11-11 12:06', reports, None)
    eng.get_injury_features(games)
    assert eng.injury_report_path == 'Injury-Report_2026-11-11_11_30AM.pdf'                    # not None: the file the features came from
    assert any('Injury report download FAILED' in r.getMessage() for r in caplog.records)
    assert not any('MISMATCH' in r.getMessage() for r in caplog.records)                      # 30 min behind: within the tolerance
