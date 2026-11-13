"""Regression tests for the signal job: a slot where nothing qualifies must not raise, and one bad game must not stop the others of the run (synthetic inputs, no real data)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

from props.live import signal, store

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('props_signal_job_under_test', ROOT / 'scripts' / 'props_signal_job.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)

RULE = {'stats': ['points'], 'k_by_stat': {'points': 1.0}, 'primary': {'e': 0.03, 'min_books': 3}}
PROV = {'model_sha256': 'm', 'features_db_sha256': 'f', 'rule_sha256': 'r', 'roster_rule': 'v2', 'code_commit': 'c'}


def _game(gid):
    return pd.Series({'game_id': gid, 'season': '2026-27', 'game_date': '2026-11-04', 'tip_ref_utc': '2026-11-05T02:30:00Z', 'home_team': 'HOM', 'away_team': 'AWY',
                      'home_team_id': 1, 'away_team_id': 2, 'et_game_date': '2026-11-04'})


def _poisson_pmf(lam, n=80):
    p = np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(n)])
    return p / p.sum()


def _item(gid, over_dec, under_dec, slot=60):
    books = ['b1', 'b2', 'b3']
    cons = pd.DataFrame({'game_id': gid, 'player_id': [7]})
    bets = pd.DataFrame({'game_id': gid, 'player_id': 7, 'minutes_before': slot, 'stat': 'points', 'line': 24.5, 'book': books, 'over_dec': over_dec, 'under_dec': under_dec, 'n_at_line': 3,
                         'consensus_fair_over': 0.5, 'quote_age_min': 1.0})
    return signal.SlotItem(game=_game(gid), slot=slot, snapshot_ts_utc='2026-11-05T01:30:00Z', snapshot_minutes=60.0, event_id='e', cons=cons, bets=bets, names={7: 'Player Seven'})


def _ctx():
    return signal.Context(hist=None, pos={}, reports=None, tips=None, tables={}, bundle={}, rule=RULE, nba_db='', n_sims=10, model_version='v', provenance=PROV)


def _frame(gid, slot=60):
    return pd.DataFrame({'game_id': [gid], 'player_id': [7], 'minutes_before': [slot], 'season': ['2026-27'], 'game_date': ['2026-11-04']})


def _patch_model(monkeypatch):
    monkeypatch.setattr(signal, 'predict_rows', lambda bundle, rows, n_sims, seed: (np.full(len(rows), 30.0), {'points': np.tile(_poisson_pmf(24.0), (len(rows), 1))}))


def test_evaluate_item_with_no_qualifying_market_returns_signals_and_no_bets(monkeypatch):
    """Root cause of the 2026-11-04 / 11-08 KeyError('game_id'): rules.select returned an empty frame and `sel[[]]` (an empty LIST) selected no columns."""
    _patch_model(monkeypatch)
    it = _item('0022600184', over_dec=1.5, under_dec=1.5)             # both sides far below break-even: nothing qualifies
    sig, new, detail = signal.evaluate_item(_ctx(), it, _frame('0022600184'), existing=set())
    assert len(sig) == 1 and int(sig['qualified'].iloc[0]) == 0
    assert new.empty
    assert detail['quoted_players'] == 1


def test_evaluate_item_still_records_a_qualifying_bet_once(monkeypatch):
    _patch_model(monkeypatch)
    it = _item('0022600184', over_dec=2.6, under_dec=2.6)
    sig, new, _ = signal.evaluate_item(_ctx(), it, _frame('0022600184'), existing=set())
    assert len(new) == 1 and new['player_id'].iloc[0] == 7 and int(sig['qualified'].iloc[0]) == 1
    sig2, new2, _ = signal.evaluate_item(_ctx(), it, _frame('0022600184'), existing={('0022600184', 7, 'points')})
    assert new2.empty and int(sig2['already_bet'].iloc[0]) == 1


def _signals(gid):
    return pd.DataFrame({'signal_id': [f'2026-27|{gid}|7|points|60'], 'season': '2026-27', 'game_id': gid, 'slot_minutes': 60, 'player_id': 7, 'stat': 'points', 'qualified': 0, 'already_bet': 0})


def _slots(conn):
    return {(g, s): st for g, s, st in conn.execute('SELECT game_id, slot_minutes, status FROM slot_log')}


def test_one_failing_slot_does_not_stop_the_other_games(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / 'live.db')
    bad, good = _item('BAD', 2.0, 2.0), _item('GOOD', 2.0, 2.0)

    def evaluate(ctx, it, frame, existing):
        if it.game['game_id'] == 'BAD':
            raise KeyError('game_id')
        return _signals('GOOD'), pd.DataFrame(), {'quoted_players': 1}
    monkeypatch.setattr(signal, 'build_frames', lambda ctx, date, group: (pd.DataFrame(), []))
    monkeypatch.setattr(signal, 'evaluate_item', evaluate)
    written, failed = {'signals': 0, 'bets': 0, 'slots': 0}, []
    job.work_date(None, '2026-11-04', [bad, good], conn, 'run1', False, written, failed)          # the bad game comes FIRST
    assert _slots(conn) == {('GOOD', 60): 'done'}                                                   # no slot-log row for BAD: the next run retries it
    assert written == {'signals': 1, 'bets': 0, 'slots': 1}
    assert [(f['game_id'], f['slot'], f['stage']) for f in failed] == [('BAD', 60, 'evaluate')] and 'KeyError' in failed[0]['error']
    assert conn.execute('SELECT count(*) FROM prop_signals WHERE game_id = ?', ('GOOD',)).fetchone()[0] == 1


def test_a_failing_shared_frame_build_is_retried_per_slot(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / 'live.db')
    bad, good = _item('BAD', 2.0, 2.0), _item('GOOD', 2.0, 2.0)

    def build(ctx, date, group):
        if any(i.game['game_id'] == 'BAD' for i in group):
            raise KeyError('game_id')
        return pd.DataFrame(), []
    monkeypatch.setattr(signal, 'build_frames', build)
    monkeypatch.setattr(signal, 'evaluate_item', lambda ctx, it, frame, existing: (_signals(it.game['game_id']), pd.DataFrame(), {}))
    written, failed = {'signals': 0, 'bets': 0, 'slots': 0}, []
    job.work_date(None, '2026-11-04', [bad, good], conn, 'run1', False, written, failed)
    assert _slots(conn) == {('GOOD', 60): 'done'}
    assert [(f['game_id'], f['stage']) for f in failed] == [('BAD', 'frame')]


def test_dry_run_writes_nothing_even_when_a_slot_fails(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / 'live.db')
    monkeypatch.setattr(signal, 'build_frames', lambda ctx, date, group: (pd.DataFrame(), []))
    monkeypatch.setattr(signal, 'evaluate_item', lambda ctx, it, frame, existing: (_signals(it.game['game_id']), pd.DataFrame(), {}))
    written, failed = {'signals': 0, 'bets': 0, 'slots': 0}, []
    job.work_date(None, '2026-11-04', [_item('GOOD', 2.0, 2.0)], conn, 'run1', True, written, failed)
    assert _slots(conn) == {} and written['slots'] == 0 and failed == []


def test_due_items_isolates_a_game_whose_quotes_fail(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / 'live.db')
    tip = '2026-11-05T02:30:00Z'
    now = pd.Timestamp('2026-11-05T01:31:00Z')                                                      # 59 minutes before the tip: the T-60 slot is due, T-240 is missed, T-5 waits
    sched = pd.DataFrame({'game_id': ['BAD', 'GOOD'], 'et_game_date': ['2026-11-04'] * 2, 'home_team_id': [1, 3], 'away_team_id': [2, 4]})
    matched = pd.DataFrame({'game_id': ['BAD', 'GOOD'], 'event_id_vendor': ['eb', 'eg'], 'tip_ref_utc': [tip, tip]})
    monkeypatch.setattr(job.quotes, 'event_snapshots', lambda db, ev, since_utc=None: pd.DataFrame({'snapshot_ts_utc': ['2026-11-05T01:30:00Z']}))

    def slot_quotes(db, ev, gid, ts, T):
        if gid == 'BAD':
            raise ValueError('boom')
        it = _item(gid, 2.0, 2.0, T)
        return it.cons, it.bets, pd.DataFrame({'player_id': [7], 'player_name_raw': ['Player Seven']})
    monkeypatch.setattr(job.quotes, 'slot_quotes', slot_quotes)
    log, failed = [], []
    items = job.due_items(conn, 'odds.db', sched, matched, now, '2026-27', log, failed)
    assert [(i.game['game_id'], i.slot) for i in items] == [('GOOD', 60)]
    assert [(g, T, st) for g, T, st, *_ in log] == [('GOOD', 240, 'missed')]                        # nothing is logged for BAD: its rows were rolled back with the failure
    assert [(f['game_id'], f['stage']) for f in failed] == [('BAD', 'prepare')] and 'boom' in failed[0]['error']
