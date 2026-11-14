"""Early prop snapshots (props/collect_early.py, scripts/snapshot_props_early.py): no network, a fake vendor, temporary stores."""
import importlib.util
from pathlib import Path

import pytest

from core.odds_vendor.base import BudgetExceeded, CreditBudget, VendorResponse
from core.odds_vendor.theoddsapi import TheOddsApi
from props import collect, collect_early as ce, odds_store

TIP = '2026-11-15T00:10:00Z'                    # 7:10 PM ET on 2026-11-14
EVENT = {'id': 'ev1', 'commence_time': TIP, 'home_team': 'Boston Celtics', 'away_team': 'New York Knicks', 'bookmakers': [
    {'key': 'draftkings', 'markets': [{'key': 'player_rebounds', 'last_update': '2026-11-14T16:00:00Z', 'outcomes': [
        {'name': 'Over', 'description': 'Jayson Tatum', 'price': -115, 'point': 8.5}, {'name': 'Under', 'description': 'Jayson Tatum', 'price': -105, 'point': 8.5}]}]}]}


class FakeVendor:
    def __init__(self, events=None, budget=None):
        self.calls, self._events, self.budget = [], events or [{'id': 'ev1', 'commence_time': TIP}], budget

    def events(self, commence_from=None, commence_to=None):
        self.calls.append(('events', commence_from, commence_to))
        return VendorResponse('/events', self._events, commence_from, commence_from, cost=0)

    def event_odds(self, event_id, markets, regions='us', bookmakers=None):
        if self.budget is not None:
            self.budget.check(len(markets))
        self.calls.append(('odds', event_id, tuple(markets)))
        tip = next(e['commence_time'] for e in self._events if e['id'] == event_id)
        ts = f'2026-11-14T16:10:{len(self.calls):02d}Z'                          # each fetch has its own time, as on the real feed (the log is keyed by time and event)
        resp = VendorResponse('/odds', dict(EVENT, id=event_id, commence_time=tip), ts, ts, cost=11)
        if self.budget is not None:
            self.budget.record('fake', '/odds', {}, len(markets), 11, None, 200)
        return resp


def test_the_caps_are_the_approved_bound():
    assert ce.TOTAL_CAP == 4620 and ce.DAILY_CAP == 330 and ce.MAX_GAME_DAYS == 14 and ce.OFFSETS_MIN == (480, 360)
    assert len(collect.ALL_MARKET_KEYS) == ce.CREDITS_PER_SNAPSHOT


def test_an_offset_is_due_once_inside_its_window_only():
    ev = [{'id': 'ev1', 'commence_time': TIP}]
    plan = lambda now, done=(): [(t.offset, t.mode) for t in ce.plan_early(ev, now, set(done), set())]  # noqa: E731
    assert plan('2026-11-14T16:05:00Z') == []                                   # 485 minutes before tip: too early
    assert plan('2026-11-14T16:10:00Z') == [(480, 'early_T-480')]              # exactly T-480
    assert plan('2026-11-14T16:30:00Z') == [(480, 'early_T-480')]              # 460: still inside the 25-minute grace
    assert plan('2026-11-14T16:36:00Z') == []                                   # 454: the tick came too late, the offset is skipped
    assert plan('2026-11-14T16:20:00Z', [('ev1', 'early_T-480')]) == []         # already taken
    assert plan('2026-11-14T18:12:00Z', [('ev1', 'early_T-480')]) == [(360, 'early_T-360')]
    assert plan('2026-11-14T20:10:00Z') == []                                   # T-240 belongs to the production collector


def test_no_game_on_a_fifteenth_date_is_snapshotted():
    ev = [{'id': 'ev1', 'commence_time': TIP}]
    full = {f'2026-10-{d:02d}' for d in range(1, 15)}
    assert ce.plan_early(ev, '2026-11-14T16:10:00Z', set(), full) == []
    assert len(ce.plan_early(ev, '2026-11-14T16:10:00Z', set(), full | {'2026-11-14'} - {'2026-10-01'})) == 1      # a date already counted keeps its games
    two = ev + [{'id': 'ev2', 'commence_time': '2026-11-16T00:10:00Z'}]                                             # a second, new date in the same tick
    got = ce.plan_early(two, '2026-11-14T16:10:00Z', set(), set(), max_game_days=1)
    assert [t.event_id for t in got] == ['ev1']


def test_one_tick_stores_into_the_early_store_and_is_not_repeated(tmp_path):
    conn = odds_store.connect(tmp_path / 'early.db')
    v = FakeVendor()
    r1 = ce.run_early(v, conn, '2026-11-14T16:10:00Z')
    assert [(x['mode'], x['cost'], x['prop_rows']) for x in r1] == [('early_T-480', 11, 1)]
    assert v.calls[1][2] == tuple(collect.ALL_MARKET_KEYS)
    assert ce.run_early(v, conn, '2026-11-14T16:20:00Z') == []                   # next tick, same offset: nothing
    r3 = ce.run_early(v, conn, '2026-11-14T18:10:00Z')
    assert [x['mode'] for x in r3] == ['early_T-360']
    assert conn.execute("SELECT mode, COUNT(*) FROM snapshot_log GROUP BY 1 ORDER BY 1").fetchall() == [('early_T-360', 1), ('early_T-480', 1)]
    done, days = ce.done_and_days(conn)
    assert done == {('ev1', 'early_T-480'), ('ev1', 'early_T-360')} and days == {'2026-11-14'}


def test_dry_run_spends_nothing_and_writes_no_snapshot(tmp_path):
    conn = odds_store.connect(tmp_path / 'early.db')
    v = FakeVendor()
    r = ce.run_early(v, conn, '2026-11-14T16:10:00Z', dry_run=True)
    assert r[0]['dry_run'] and r[0]['mode'] == 'early_T-480' and [c[0] for c in v.calls] == ['events']
    assert conn.execute('SELECT COUNT(*) FROM snapshot_log').fetchone()[0] == 0


def test_the_total_cap_stops_the_run_before_the_request(tmp_path):
    path = tmp_path / 'early.db'
    conn = odds_store.connect(path)
    budget = CreditBudget(path, per_run=ce.DAILY_CAP, per_day=None, total=15)   # room for one 11-credit snapshot only
    events = [{'id': 'ev1', 'commence_time': TIP}, {'id': 'ev2', 'commence_time': TIP}]
    v = FakeVendor(events, budget)
    with pytest.raises(BudgetExceeded):
        ce.run_early(v, conn, '2026-11-14T16:10:00Z')
    assert [c[1] for c in v.calls if c[0] == 'odds'] == ['ev1']                 # the second request was never sent
    assert conn.execute('SELECT COUNT(*) FROM snapshot_log').fetchone()[0] == 1    # the one paid for is logged


def test_the_script_uses_its_own_store_and_fixed_caps(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('snapshot_props_early', Path(__file__).resolve().parents[2] / 'scripts' / 'snapshot_props_early.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.STORE_PATH == Path('data/props_odds_early.db') and mod.RAW_DIR == Path('data/raw/props_api_early')
    assert mod.STORE_PATH.name != odds_store.DEFAULT_PATH.name if hasattr(odds_store.DEFAULT_PATH, 'name') else True
    s = mod.status(tmp_path / 'missing.db')
    assert s == {'store': str(tmp_path / 'missing.db'), 'exists': False, 'credits_used': 0.0, 'total_cap': 4620, 'game_days': 0, 'max_game_days': 14, 'snapshots': {}}
    assert not (tmp_path / 'missing.db').exists()                                # --status never creates the store
    conn = odds_store.connect(tmp_path / 'early.db')
    budget = CreditBudget(tmp_path / 'early.db', total=ce.TOTAL_CAP)
    ce.run_early(FakeVendor(budget=budget), conn, '2026-11-14T16:10:00Z')
    conn.close()
    s = mod.status(tmp_path / 'early.db')
    assert (s['credits_used'], s['game_days'], s['snapshots']) == (11.0, 1, {'early_T-480': 1})
    with pytest.raises(SystemExit):                                              # the caps are not command-line options
        mod.main(['--total-cap', '99999'])


def test_the_real_vendor_class_accepts_the_separate_raw_folder(tmp_path):
    api = TheOddsApi(CreditBudget(tmp_path / 'early.db', total=ce.TOTAL_CAP), api_key='k', raw_dir=tmp_path / 'raw_early', session=object())
    assert api.raw_dir == tmp_path / 'raw_early'
