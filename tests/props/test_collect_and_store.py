import gzip
import json

import pandas as pd
import pytest

from core.odds_vendor.base import BudgetExceeded, CreditBudget, VendorResponse
from core.odds_vendor.theoddsapi import OddsApiError, TheOddsApi
from props import collect, odds_store
from props.parse import parse_event_odds, prop_market

EVENT = {
    'id': 'ev1', 'commence_time': '2026-10-21T23:30:00Z', 'home_team': 'Boston Celtics', 'away_team': 'New York Knicks',
    'bookmakers': [
        {'key': 'draftkings', 'markets': [
            {'key': 'player_points', 'last_update': '2026-10-21T20:01:00Z', 'outcomes': [
                {'name': 'Over', 'description': 'Jayson Tatum', 'price': -115, 'point': 27.5},
                {'name': 'Under', 'description': 'Jayson Tatum', 'price': -105, 'point': 27.5}]},
            {'key': 'player_points_alternate', 'last_update': '2026-10-21T20:01:00Z', 'outcomes': [
                {'name': 'Over', 'description': 'Jayson Tatum', 'price': 250, 'point': 35.5}]},
            {'key': 'player_double_double', 'outcomes': [{'name': 'Yes', 'description': 'Jayson Tatum', 'price': 300}]},
            {'key': 'h2h', 'last_update': '2026-10-21T20:00:00Z', 'outcomes': [
                {'name': 'Boston Celtics', 'price': -200}, {'name': 'New York Knicks', 'price': 170}]},
            {'key': 'spreads', 'last_update': '2026-10-21T20:02:00Z', 'outcomes': [
                {'name': 'Boston Celtics', 'price': -110, 'point': -5.5},
                {'name': 'New York Knicks', 'price': -110, 'point': 5.5}]},
            {'key': 'totals', 'last_update': '2026-10-21T20:00:30Z', 'outcomes': [
                {'name': 'Over', 'price': -108, 'point': 221.5}, {'name': 'Under', 'price': -112, 'point': 221.5}]}]},
        {'key': 'fanduel', 'markets': [
            {'key': 'player_points', 'last_update': '2026-10-21T19:00:00Z', 'outcomes': [
                {'name': 'Over', 'description': 'Jayson Tatum', 'price': -120, 'point': 28.5}]}]}]}


def test_parse_pairs_over_under_and_keeps_one_sided_and_alt_rows():
    props, teams, stats = parse_event_odds(EVENT, '2026-10-21T20:05:00Z')
    assert len(props) == 3
    dk = props[(props.book == 'draftkings') & (props.is_alt_line == 0)].iloc[0]
    assert (dk.market, dk.line, dk.over_price, dk.under_price) == ('points', 27.5, -115, -105)
    alt = props[props.is_alt_line == 1].iloc[0]
    assert alt.market == 'points' and alt.line == 35.5 and pd.isna(alt.under_price)
    fd = props[props.book == 'fanduel'].iloc[0]
    assert fd.over_price == -120 and pd.isna(fd.under_price)           # one-sided stays a row
    assert stats['unknown_markets'] == {'player_double_double': 1}
    t = teams.iloc[0]
    assert (t.home_ml, t.away_ml, t.home_spread, t.total, t.over_price) == (-200, 170, -5.5, 221.5, -108)
    assert t.book_last_update_utc == '2026-10-21T20:02:00Z'
    assert (props.snapshot_ts_utc == '2026-10-21T20:05:00Z').all()


def test_prop_market_mapping():
    assert prop_market('player_points_rebounds_assists') == ('pra', False)
    assert prop_market('player_threes_alternate') == ('threes', True)
    assert prop_market('h2h') == (None, None)


def test_store_is_idempotent_and_logs_the_snapshot(tmp_path):
    conn = odds_store.connect(tmp_path / 'p.db')
    resp = VendorResponse('/x', EVENT, '2026-10-21T20:05:00Z', '2026-10-21T20:05:01Z', cost=11, raw_path='r.gz')
    out1 = collect.store_event_odds(conn, resp, 'slow')
    out2 = collect.store_event_odds(conn, resp, 'slow')
    assert out1['new_prop_rows'] == 3 and out2['new_prop_rows'] == 0
    assert conn.execute('select count(*) from prop_snapshots').fetchone()[0] == 3
    assert conn.execute('select count(*) from team_lines').fetchone()[0] == 1
    log = conn.execute('select n_prop_rows, books, credit_cost, minutes_to_tip from snapshot_log').fetchone()
    assert log[0] == 3 and log[1] == 'draftkings,fanduel' and log[2] == 11 and log[3] == pytest.approx(205)
    assert conn.execute('select et_game_date from events').fetchone()[0] == '2026-10-21'
    assert odds_store.last_snapshot_ts(conn, 'ev1') == '2026-10-21T20:05:00Z'


def test_plan_live_cadence_and_window():
    ev = [{'id': 'a', 'commence_time': '2026-10-21T23:30:00Z'},     # 150 min away at 21:00Z -> slow
          {'id': 'b', 'commence_time': '2026-10-21T22:30:00Z'},     # 90 min -> fast
          {'id': 'c', 'commence_time': '2026-10-21T20:30:00Z'},     # already started
          {'id': 'd', 'commence_time': '2026-10-22T05:00:00Z'}]     # 8h away, outside the 6h window
    now = '2026-10-21T21:00:00Z'
    plan = collect.plan_live(ev, now, {})
    assert [(t.event_id, t.mode) for t in plan] == [('b', 'fast'), ('a', 'slow')]
    # 8 minutes after a snapshot: fast is due again (10-min cadence, 3-min jitter), slow is not
    last = {'a': '2026-10-21T20:52:00Z', 'b': '2026-10-21T20:52:00Z'}
    assert [t.event_id for t in collect.plan_live(ev, now, last)] == ['b']
    last = {'a': '2026-10-21T20:40:00Z', 'b': '2026-10-21T20:40:00Z'}
    assert [t.event_id for t in collect.plan_live(ev, now, last)] == ['b']      # 20 min: slow still not due
    last = {'a': '2026-10-21T20:29:00Z'}
    assert 'a' in [t.event_id for t in collect.plan_live(ev, now, last)]


def test_plan_historical_offsets():
    assert collect.plan_historical('2026-10-21T23:30:00Z', [180, 90, 5]) == [
        (180, '2026-10-21T20:30:00Z'), (90, '2026-10-21T22:00:00Z'), (5, '2026-10-21T23:25:00Z')]


class FakeVendor:
    def __init__(self):
        self.calls = []

    def events(self, commence_from=None, commence_to=None):
        self.calls.append('events')
        return VendorResponse('/events', [{'id': 'ev1', 'commence_time': '2026-10-21T22:30:00Z'}],
                              '2026-10-21T21:00:00Z', '2026-10-21T21:00:00Z', cost=0)

    def event_odds(self, event_id, markets, regions='us', bookmakers=None):
        self.calls.append(('odds', event_id, tuple(markets)))
        ev = dict(EVENT, id=event_id, commence_time='2026-10-21T22:30:00Z')
        return VendorResponse('/odds', ev, '2026-10-21T21:00:05Z', '2026-10-21T21:00:05Z', cost=11)


def test_run_live_snapshots_due_events_then_waits(tmp_path):
    conn = odds_store.connect(tmp_path / 'p.db')
    v = FakeVendor()
    r1 = collect.run_live(v, conn, '2026-10-21T21:00:10Z')
    assert [x['event_id'] for x in r1] == ['ev1'] and r1[0]['prop_rows'] == 3
    assert v.calls[1][2] == tuple(collect.ALL_MARKET_KEYS)
    r2 = collect.run_live(v, conn, '2026-10-21T21:03:00Z')               # 3 minutes later: not due
    assert r2 == []
    dry = collect.run_live(FakeVendor(), odds_store.connect(tmp_path / 'q.db'), '2026-10-21T21:00:10Z', dry_run=True)
    assert dry[0]['dry_run'] and dry[0]['mode'] == 'fast'


# ---- vendor adapter: no network, a fake session --------------------------------------------------------------
class FakeResp:
    def __init__(self, body, status=200, headers=None):
        self._body, self.status_code = body, status
        self.headers = headers or {'x-requests-last': '11', 'x-requests-remaining': '19989', 'x-requests-used': '11'}
        self.text = json.dumps(body)
        self.ok = status < 400

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        return self.resp


def test_adapter_records_cost_saves_raw_without_the_key(tmp_path):
    budget = CreditBudget(tmp_path / 'p.db', per_run=100)
    sess = FakeSession(FakeResp(EVENT))
    api = TheOddsApi(budget, api_key='SECRETKEY', raw_dir=tmp_path / 'raw', session=sess)
    resp = api.event_odds('ev1', collect.ALL_MARKET_KEYS)
    assert resp.cost == 11 and resp.remaining == 19989 and budget.run_used == 11
    url, params = sess.calls[0]
    assert url.endswith('/sports/basketball_nba/events/ev1/odds') and params['apiKey'] == 'SECRETKEY'
    assert params['oddsFormat'] == 'american' and params['regions'] == 'us'
    raw = gzip.open(resp.raw_path, 'rt').read()
    assert 'SECRETKEY' not in raw and 'Jayson Tatum' in raw
    ledger = budget._used()
    assert ledger == 11


def test_budget_refuses_before_sending(tmp_path):
    budget = CreditBudget(tmp_path / 'p.db', per_run=100)
    sess = FakeSession(FakeResp({'timestamp': '2025-01-15T20:00:00Z', 'data': EVENT}))
    api = TheOddsApi(budget, api_key='k', raw_dir=tmp_path / 'raw', session=sess)
    # historical = 10 x 11 markets x 1 region = 110 credits > per-run cap of 100
    with pytest.raises(BudgetExceeded):
        api.historical_event_odds('ev1', '2025-01-15T20:00:00Z', collect.ALL_MARKET_KEYS)
    assert sess.calls == []
    small = api.historical_event_odds('ev1', '2025-01-15T20:00:00Z', ['player_points'] * 1)   # 10 credits
    assert small.snapshot_ts_utc == '2025-01-15T20:00:00Z' and small.payload['id'] == 'ev1'


def test_total_and_daily_caps_use_the_persisted_ledger(tmp_path):
    p = tmp_path / 'p.db'
    CreditBudget(p).record('v', '/x', {}, 5, 5, 100, 200)
    with pytest.raises(BudgetExceeded):
        CreditBudget(p, total=10).check(6)                          # a new process still sees the earlier spend
    CreditBudget(p, total=10).check(5)
    with pytest.raises(BudgetExceeded):
        CreditBudget(p, per_day=10).check(6)


def test_adapter_errors_do_not_leak_the_key(tmp_path):
    api = TheOddsApi(CreditBudget(tmp_path / 'p.db'), api_key='SECRETKEY', raw_dir=tmp_path / 'raw',
                     session=FakeSession(FakeResp({'message': 'bad key SECRETKEY'}, status=401, headers={})))
    with pytest.raises(OddsApiError) as e:
        api.event_odds('ev1', ['player_points'])
    assert 'SECRETKEY' not in str(e.value)


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setattr('core.odds_vendor.theoddsapi.load_env', lambda: [])      # never read the real .env in tests
    with pytest.raises(RuntimeError, match='ODDS_API_KEY'):
        TheOddsApi(CreditBudget(tmp_path / 'p.db'))


def test_cadence_is_configurable_and_fast_window_can_be_disabled():
    ev = [{'id': 'b', 'commence_time': '2026-10-21T22:30:00Z'}]            # 90 min to tip
    last = {'b': '2026-10-21T20:52:00Z'}                                   # 8 min ago
    now = '2026-10-21T21:00:00Z'
    assert [t.event_id for t in collect.plan_live(ev, now, last)] == ['b']                     # default: fast, due
    assert collect.plan_live(ev, now, last, fast_window_min=0) == []                           # slow only: not due


def test_lean_cadence_with_a_closing_snapshot():
    """30-minute snapshots, no fast window, plus one snapshot in the last 10 minutes (scheduler ticks every 10)."""
    ev = [{'id': 'g', 'commence_time': '2026-10-21T23:00:00Z'}]
    kw = dict(window_hours=4, fast_window_min=0, slow_cadence_min=30, close_snapshot_min=10)
    last = {}
    taken = []
    for minute in range(0, 4 * 60 + 40, 10):                  # ticks from T-4h20 to after tip, every 10 minutes
        now = f'2026-10-21T{(18 + (40 + minute) // 60) % 24:02d}:{(40 + minute) % 60:02d}:00Z'
        for t in collect.plan_live(ev, now, last, **kw):
            last['g'] = now
            taken.append((round(t.minutes_to_tip), t.mode))
    assert taken[0] == (240, 'slow') and taken[-1][1] == 'close' and taken[-1][0] <= 10
    assert [m for m, mode in taken].count(taken[-1][0]) == 1
    assert sum(mode == 'close' for _, mode in taken) == 1
    assert len(taken) == 9                                    # 8 regular (T-240 ... T-30) + 1 close
    assert not any(mode == 'fast' for _, mode in taken)


def test_free_calls_are_not_blocked_by_exhausted_caps(tmp_path):
    p = tmp_path / 'p.db'
    CreditBudget(p).record('v', '/x', {}, 50, 50, 10, 200)
    b = CreditBudget(p, per_run=0, per_day=10, total=10)          # every cap already exceeded
    b.check(0.0)                                                   # a free call passes
    with pytest.raises(BudgetExceeded):
        b.check(1.0)
