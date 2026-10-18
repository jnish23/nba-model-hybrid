import sqlite3

import numpy as np
import pandas as pd
import pytest

from core.odds_vendor.base import BudgetExceeded, CreditBudget, VendorResponse
from props import analysis, backfill, collect, odds_store, quality, schedule, tiers

TIP = '2026-03-18T02:00:00Z'                 # NBA scheduled tip; reference tip = 02:10Z
TEAMS = {'San Antonio Spurs': 1, 'Sacramento Kings': 2, 'Boston Celtics': 3, 'New York Knicks': 4}


def team_ids(name):
    return TEAMS.get(name)


def sched(rows=None):
    rows = rows or [('0022500998', TIP, '2026-03-17', 2, 1, 'Sacramento Kings', 'San Antonio Spurs', '')]
    return pd.DataFrame(rows, columns=schedule.COLS)


def quote(book, player, line, over, under, ts='2026-03-17T21:00:00Z', market='points', eid='e1'):
    return {'snapshot_ts_utc': ts, 'event_id_vendor': eid, 'book': book, 'player_name_raw': player, 'market': market,
            'line': line, 'over_price': over, 'under_price': under, 'is_alt_line': 0,
            'book_last_update_utc': ts, 'source': 't'}


# ---- main line and consensus ---------------------------------------------------------------------------------------
def test_main_line_is_the_balanced_rung_of_a_ladder():
    ladder = [(13.5, -290, 210), (14.5, -215, 160), (15.5, -165, 125), (16.5, -125, -105), (17.5, 100, -130), (18.5, 125, -165)]
    rows = [quote('bovada', 'Ty Jerome', *r) for r in ladder] + [quote('dk', 'Ty Jerome', 17.5, -106, -112)]
    f = analysis.flag_main_lines(analysis.add_devig(pd.DataFrame(rows)))
    main = f[f.is_main_line]
    assert sorted(zip(main.book, main.line)) == [('bovada', 16.5), ('dk', 17.5)]
    assert f[f.book == 'bovada'].is_ladder.all() and not f[f.book == 'dk'].is_ladder.any()
    assert f[f.book == 'bovada'].n_lines.iloc[0] == 6


def test_main_line_with_one_sided_and_alt_rows():
    rows = [quote('a', 'P', 20.5, -110, None), quote('b', 'Q', 10.5, -110, -110), quote('b', 'Q', 11.5, 130, -160)]
    alt = quote('b', 'Q', 15.5, 400, None)
    alt['is_alt_line'] = 1
    f = analysis.flag_main_lines(analysis.add_devig(pd.DataFrame(rows + [alt])))
    assert f.loc[f.book == 'a', 'is_main_line'].all()                      # only quote there is: kept, flagged one-sided
    assert f.loc[f.book == 'a', 'one_sided'].all()
    assert list(f[(f.book == 'b') & f.is_main_line].line) == [10.5]
    assert not f[f.is_alt_line == 1].is_main_line.any()


def test_consensus_requires_min_books_at_the_modal_line():
    rows = [quote(b, 'P', 20.5, -110, -110) for b in ('a', 'b', 'c')] + [quote('d', 'P', 21.5, -110, -110)]
    rows += [quote(b, 'Q', 5.5, -120, 100) for b in ('a', 'b')]
    f = analysis.flag_main_lines(analysis.add_devig(pd.DataFrame(rows)))
    c = analysis.consensus(f, min_books=3).set_index('player_name_raw')
    assert c.loc['P', 'consensus_line'] == 20.5 and c.loc['P', 'n_at_line'] == 3 and c.loc['P', 'n_books'] == 4
    assert c.loc['P', 'book_spread'] == 1.0 and c.loc['P', 'consensus_fair_over'] == pytest.approx(0.5)
    assert np.isnan(c.loc['Q', 'consensus_fair_over']) and c.loc['Q', 'n_at_line'] == 2


def test_stale_and_crossed_quotes_are_excluded_from_consensus():
    rows = [quote(b, 'P', 20.5, -110, -110) for b in ('a', 'b', 'c')]
    rows.append(dict(quote('d', 'P', 20.5, -150, +160), book_last_update_utc='2026-03-17T10:00:00Z'))    # stale and crossed
    f = analysis.flag_main_lines(analysis.add_devig(pd.DataFrame(rows)))
    assert f[f.book == 'd'].stale.all() and f[f.book == 'd'].no_margin.all()
    assert analysis.consensus(f, min_books=2).n_books.iloc[0] == 3


# ---- tiers ---------------------------------------------------------------------------------------------------------
def test_tiers_use_only_games_before_the_date(tmp_path):
    path = tmp_path / 'nba.db'
    g = []
    for i in range(10):
        d = f'2026-01-{i + 1:02d}T00:00:00'
        g += [(1, d, 35.0, 28.0), (2, d, 30.0, 12.0), (3, d, 12.0, 4.0)]
    g.append((3, '2026-01-20T00:00:00', 40.0, 60.0))                         # a huge game ON the date must not count
    with sqlite3.connect(path) as c:
        pd.DataFrame(g, columns=['personId', 'gameDate', 'minutes', 'points']).to_sql('player_basic_stats', c, index=False)
    df = pd.DataFrame({'player_id': [1, 2, 3, 99], 'et_game_date': ['2026-01-20'] * 4})
    out = tiers.assign_tiers(df, path)
    assert list(out.tier) == ['star', 'starter', 'rotation', 'rotation']
    assert out.loc[2, 'mpg'] == pytest.approx(12.0)


# ---- reference clock -----------------------------------------------------------------------------------------------
def test_match_games_ignores_a_wrong_vendor_tip_time():
    ev = pd.DataFrame({'event_id_vendor': ['e1'], 'home_team': ['Sacramento Kings'], 'away_team': ['San Antonio Spurs'],
                       'commence_time_utc': ['2026-03-18T03:10:00Z']})              # vendor says 70 minutes late
    m = schedule.match_games(ev, sched(), team_ids).iloc[0]
    assert m.game_id == '0022500998' and m.tip_ref_utc == '2026-03-18T02:10:00Z'
    far = ev.assign(commence_time_utc='2026-03-25T03:10:00Z')                         # a week away: not the same game
    assert pd.isna(schedule.match_games(far, sched(), team_ids).iloc[0].game_id)


def test_annotate_flags_in_game_snapshots_and_tip_revisions(tmp_path):
    conn = odds_store.connect(tmp_path / 'p.db')
    for ts, tip in (('2026-03-18T01:00:00Z', '2026-03-18T03:10:00Z'), ('2026-03-18T02:20:00Z', '2026-03-18T02:11:00Z')):
        ev = {'id': 'e1', 'commence_time': tip, 'home_team': 'Sacramento Kings', 'away_team': 'San Antonio Spurs',
              'bookmakers': [{'key': 'dk', 'markets': [{'key': 'player_points', 'outcomes': [
                  {'name': 'Over', 'description': 'P', 'price': -110, 'point': 20.5},
                  {'name': 'Under', 'description': 'P', 'price': -110, 'point': 20.5}]}]}]}
        collect.store_event_odds(conn, VendorResponse('/x', ev, ts, ts, cost=11), 'hist_T-60')
    out = schedule.annotate(conn, sched(), team_ids)
    assert out == {'events': 1, 'matched': 1, 'in_game_snapshots': 1, 'revised_events': 1}
    r = dict(conn.execute('select snapshot_ts_utc, in_game from snapshot_ref').fetchall())
    assert r == {'2026-03-18T01:00:00Z': 0, '2026-03-18T02:20:00Z': 1}
    assert conn.execute('select minutes_to_ref from snapshot_ref where in_game = 0').fetchone()[0] == pytest.approx(70.0)


# ---- backfill ------------------------------------------------------------------------------------------------------
class HistVendor:
    def __init__(self, vendor_tip='2026-03-18T03:10:00Z', cost=110):
        self.vendor_tip, self.cost, self.requests = vendor_tip, cost, []

    def historical_events(self, ts):
        ev = [{'id': 'e1', 'home_team': 'Sacramento Kings', 'away_team': 'San Antonio Spurs', 'commence_time': self.vendor_tip}]
        return VendorResponse('/h', ev, ts, ts, cost=1)

    def historical_event_odds(self, eid, ts, markets, regions='us', bookmakers=None):
        self.requests.append(ts)
        ev = {'id': eid, 'commence_time': self.vendor_tip, 'home_team': 'Sacramento Kings',
              'away_team': 'San Antonio Spurs', 'bookmakers': [{'key': 'dk', 'markets': [{'key': 'player_points', 'outcomes': [
                  {'name': 'Over', 'description': 'P', 'price': -110, 'point': 20.5},
                  {'name': 'Under', 'description': 'P', 'price': -110, 'point': 20.5}]}]}]}
        return VendorResponse('/o', ev, ts, ts, cost=self.cost)


def test_plan_prices_and_resumes(tmp_path):
    p = backfill.plan(sched(), '2026-03-01', '2026-03-31', (240, 120, 5), n_markets=11)
    assert p.snapshot_count == 3 and p.credits == 10 * 11 * 3 + 1
    p2 = backfill.plan(sched(), '2026-03-01', '2026-03-31', (240, 120, 5), 11, done={('0022500998', 240), ('0022500998', 5)})
    assert p2.todo == {'0022500998': (120,)} and p2.credits == 111
    p3 = backfill.plan(sched(), '2026-03-01', '2026-03-31', (240,), 11, done={('0022500998', 240)})
    assert p3.todo == {} and p3.credits == 0
    assert backfill.plan(sched(), '2026-04-01', '2026-04-30', (240,), 11).snapshot_count == 0


def test_run_uses_the_nba_clock_never_requests_in_game_and_resumes(tmp_path):
    conn = odds_store.connect(tmp_path / 'p.db')
    conn.executescript(schedule.REF_SCHEMA)
    v = HistVendor()                                                        # its tip time is 70 minutes wrong
    p = backfill.plan(sched(), '2026-03-17', '2026-03-18', (240, 60, 5), 11)
    stats = backfill.run(v, conn, p, team_ids, sleep_s=0)
    assert stats['snapshots'] == 3 and stats['credits'] == 330
    assert v.requests == ['2026-03-17T22:10:00Z', '2026-03-18T01:10:00Z', '2026-03-18T02:05:00Z']      # relative to 02:10Z
    assert all(pd.Timestamp(t) < pd.Timestamp('2026-03-18T02:10:00Z') for t in v.requests)
    schedule.annotate(conn, sched(), team_ids)
    assert backfill.done_snapshots(conn) == {('0022500998', 240), ('0022500998', 60), ('0022500998', 5)}
    again = backfill.plan(sched(), '2026-03-17', '2026-03-18', (240, 60, 5, 30), 11, done=backfill.done_snapshots(conn))
    assert again.todo == {'0022500998': (30,)}


def test_run_stops_cleanly_at_the_credit_cap(tmp_path):
    from core.odds_vendor.theoddsapi import TheOddsApi
    conn = odds_store.connect(tmp_path / 'p.db')
    conn.executescript(schedule.REF_SCHEMA)
    sent = []

    class Resp:
        ok, status_code, text = True, 200, ''
        headers = {'x-requests-last': '1', 'x-requests-remaining': '99'}

        def json(self):
            return {'timestamp': '2026-03-18T01:00:00Z', 'data': [
                {'id': 'e1', 'home_team': 'Sacramento Kings', 'away_team': 'San Antonio Spurs',
                 'commence_time': '2026-03-18T03:10:00Z'}]}

    class Sess:
        def get(self, url, params=None, timeout=None):
            sent.append(url)
            assert url.endswith('/events'), 'a snapshot request was sent past the cap'
            return Resp()

    api = TheOddsApi(CreditBudget(tmp_path / 'p.db', per_run=50), api_key='k', raw_dir=tmp_path / 'raw', session=Sess())
    p = backfill.plan(sched(), '2026-03-17', '2026-03-18', (5,), 11)
    with pytest.raises(BudgetExceeded):                      # the 1-credit listing goes out; the 110-credit snapshot exceeds 50
        backfill.run(api, conn, p, team_ids, sleep_s=0)
    assert len(sent) == 1


# ---- report --------------------------------------------------------------------------------------------------------
def test_quality_report_runs_and_reports_clock_integrity(tmp_path):
    path = tmp_path / 'p.db'
    conn = odds_store.connect(path)
    v = HistVendor()
    collect.store_event_odds(conn, v.historical_event_odds('e1', '2026-03-18T01:10:00Z', ['player_points']), 'hist_T-60')
    collect.store_event_odds(conn, v.historical_event_odds('e1', '2026-03-18T02:20:00Z', ['player_points']), 'hist_T-5')
    schedule.annotate(conn, sched(), team_ids)
    conn.commit()
    conn.close()
    text = quality.build_report(str(path), None)
    for section in ('## 1. Overview', '## 2. Coverage', '## 3b. Clock integrity', '## 4. Missing closes',
                    '## 5. Quote quality', '## 7. Player-ID matching'):
        assert section in text
    assert 'in-game), excluded everywhere below: **1** of 2' in text
    assert 'first vendor tip time differs from the reference by more than 15 minutes: **1**' in text


def test_reference_offset_follows_the_vendor_when_plausible():
    n = '2026-03-18T02:00:00Z'
    assert schedule.ref_offset_min('2026-03-18T02:10:00Z', n) == 10          # the usual listing
    assert schedule.ref_offset_min('2026-03-18T02:00:00Z', n) == 0           # a game that starts on the dot
    assert schedule.ref_offset_min('2026-03-18T02:05:00Z', n) == 5
    assert schedule.ref_offset_min('2026-03-18T02:15:00Z', n) == 10          # clipped to the cap
    assert schedule.ref_offset_min('2026-03-18T03:10:00Z', n) == 10          # 70 minutes off: not trusted
    assert schedule.ref_offset_min('2026-03-18T01:50:00Z', n) == 0           # vendor earlier than the schedule: clipped at 0
    assert schedule.ref_offset_min(None, n) == 10


def test_match_games_uses_the_vendor_offset_for_on_time_games():
    ev = pd.DataFrame({'event_id_vendor': ['e1'], 'home_team': ['Sacramento Kings'], 'away_team': ['San Antonio Spurs'],
                       'commence_time_utc': [TIP]})
    assert schedule.match_games(ev, sched(), team_ids).iloc[0].tip_ref_utc == TIP


def test_annotate_uses_the_earliest_vendor_tip_not_the_last(tmp_path):
    conn = odds_store.connect(tmp_path / 'p.db')
    for ts, tip in (('2026-03-18T01:00:00Z', TIP), ('2026-03-18T02:03:00Z', '2026-03-18T02:03:00Z')):   # on-time game, tip later restated
        ev = {'id': 'e1', 'commence_time': tip, 'home_team': 'Sacramento Kings', 'away_team': 'San Antonio Spurs',
              'bookmakers': [{'key': 'dk', 'markets': [{'key': 'player_points', 'outcomes': [
                  {'name': 'Over', 'description': 'P', 'price': -110, 'point': 20.5},
                  {'name': 'Under', 'description': 'P', 'price': -110, 'point': 20.5}]}]}]}
        collect.store_event_odds(conn, VendorResponse('/x', ev, ts, ts, cost=11), 'hist_T-60')
    schedule.annotate(conn, sched(), team_ids)
    assert conn.execute('select tip_ref_utc from event_ref').fetchone()[0] == TIP


def test_run_aborts_after_repeated_api_errors(tmp_path):
    from core.odds_vendor.theoddsapi import OddsApiError
    conn = odds_store.connect(tmp_path / 'p.db')
    conn.executescript(schedule.REF_SCHEMA)

    class Failing(HistVendor):
        def historical_event_odds(self, *a, **k):
            raise OddsApiError(401, 'OUT_OF_USAGE_CREDITS')

    p = backfill.plan(sched(), '2026-03-17', '2026-03-18', (240, 120, 60, 30, 5), 11)
    with pytest.raises(RuntimeError, match='API errors in a row'):
        backfill.run(Failing(), conn, p, team_ids, sleep_s=0, max_consecutive_errors=3)


def test_annotate_keeps_existing_annotation_for_events_it_cannot_match(tmp_path):
    conn = odds_store.connect(tmp_path / 'p.db')
    ev = {'id': 'e1', 'commence_time': '2026-03-18T02:10:00Z', 'home_team': 'Sacramento Kings',
          'away_team': 'San Antonio Spurs', 'bookmakers': [{'key': 'dk', 'markets': [{'key': 'player_points', 'outcomes': [
              {'name': 'Over', 'description': 'P', 'price': -110, 'point': 20.5},
              {'name': 'Under', 'description': 'P', 'price': -110, 'point': 20.5}]}]}]}
    collect.store_event_odds(conn, VendorResponse('/x', ev, '2026-03-18T01:00:00Z', '2026-03-18T01:00:00Z', cost=11), 'hist_T-60')
    schedule.annotate(conn, sched(), team_ids)
    assert conn.execute('select game_id from event_ref').fetchone()[0] == '0022500998'
    other_season = sched([('0022400001', '2025-03-18T02:00:00Z', '2025-03-17', 2, 1, 'Sacramento Kings', 'San Antonio Spurs', '')])
    schedule.annotate(conn, other_season, team_ids)                                          # this call knows a different season
    assert conn.execute('select game_id, tip_ref_utc from event_ref').fetchone() == ('0022500998', '2026-03-18T02:10:00Z')
