"""Tests for step S1/S2 of the a11 experiment plan (7a): consensus options, the quote cache, and the de-vig calibration experiment. Synthetic data only; no real database is opened."""
import contextlib
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import a11_cache_quotes as cq
import a11_devig_calibration as dc
from props import analysis, odds_store, schedule
from props.betting import prices

ROOT = Path(__file__).resolve().parents[2]
TS = '2023-11-05T22:00:00Z'


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. analysis.consensus: defaults unchanged, median, bettable-only
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _old_consensus(flagged, min_books=3, exclude_stale=True, exclude_flagged=True):
    """Frozen copy of props.analysis.consensus as of HEAD before the agg/books options were added."""
    GROUP = analysis.GROUP
    m = flagged[flagged['is_main_line']].copy()
    ok = ~m['one_sided']
    if exclude_flagged:
        ok &= ~m['no_margin'] & ~m['wide']
    if exclude_stale:
        ok &= ~m['stale']
    m = m[ok]
    if m.empty:
        return pd.DataFrame(columns=GROUP + ['consensus_line', 'n_books', 'n_at_line', 'book_spread', 'consensus_fair_over'])
    med = m.groupby(GROUP)['line'].transform('median')
    cnt = m.groupby(GROUP + ['line'])['book'].transform('nunique')
    ranked = m.assign(_cnt=cnt, _dist=(m['line'] - med).abs()).sort_values(
        GROUP + ['_cnt', '_dist', 'line'], ascending=[True] * len(GROUP) + [False, True, True], kind='mergesort')
    pick = ranked.drop_duplicates(subset=GROUP)[GROUP + ['line']].rename(columns={'line': 'consensus_line'})
    m = m.merge(pick, on=GROUP, validate='many_to_one')
    at = m[m['line'] == m['consensus_line']]
    g = m.groupby(GROUP)
    res = pick.set_index(GROUP)
    res['n_books'] = g['book'].nunique()
    res['book_spread'] = g['line'].max() - g['line'].min()
    ag = at.groupby(GROUP)
    res['n_at_line'] = ag['book'].nunique()
    res['consensus_fair_over'] = ag['fair_over'].mean()
    res.loc[res['n_at_line'] < min_books, 'consensus_fair_over'] = np.nan
    return res.reset_index()


BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'betrivers', 'unibet_us', 'fanatics', 'bovada', 'betonline', 'mybookie']


def random_quotes(seed, n_events=12):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        ts = f'2023-11-{5 + e % 10:02d}T22:00:00Z'
        for p in range(6):
            for market in ('rebounds', 'assists'):
                base = float(rng.integers(3, 12)) + 0.5
                for b in BOOKS:
                    if rng.random() < 0.15:
                        continue
                    n_lines = 3 if (b == 'bovada' and rng.random() < 0.5) else 1                      # ladders
                    first = base + float(rng.integers(-1, 2))
                    for k in range(n_lines):
                        over = float(-110 + rng.integers(-60, 60))
                        under = float(-110 + rng.integers(-60, 60))
                        if rng.random() < 0.05:
                            under = np.nan
                        lag = int(rng.integers(0, 120))
                        rows.append({'snapshot_ts_utc': ts, 'event_id_vendor': f'e{e}', 'book': b, 'player_name_raw': f'Player {p}', 'market': market, 'line': first + k,
                                     'over_price': over, 'under_price': under, 'is_alt_line': int(rng.random() < 0.05),
                                     'book_last_update_utc': (pd.Timestamp(ts) - pd.Timedelta(minutes=lag)).strftime('%Y-%m-%dT%H:%M:%SZ')})
    return pd.DataFrame(rows)


def test_consensus_defaults_are_unchanged():
    for seed in (1, 2, 3):
        flagged = analysis.flag_main_lines(analysis.add_devig(random_quotes(seed)))
        for kw in ({}, {'min_books': 1}, {'min_books': 4}, {'exclude_stale': False}, {'exclude_flagged': False}, {'exclude_stale': False, 'exclude_flagged': False, 'min_books': 2}):
            old = _old_consensus(flagged, **kw)
            assert len(old) > 50
            assert_frame_equal(analysis.consensus(flagged, **kw), old)
            assert_frame_equal(analysis.consensus(flagged, agg='mean', books=None, **kw), old)


def test_consensus_defaults_unchanged_on_empty_and_bad_agg():
    flagged = analysis.flag_main_lines(analysis.add_devig(random_quotes(1)))
    empty = flagged.iloc[0:0]
    assert_frame_equal(analysis.consensus(empty), _old_consensus(empty))
    with pytest.raises(ValueError):
        analysis.consensus(flagged, agg='mode')


def flagged_from(rows, market='rebounds', player='P'):
    df = pd.DataFrame([{'snapshot_ts_utc': TS, 'event_id_vendor': 'e1', 'book': b, 'player_name_raw': player, 'market': market, 'line': l, 'over_price': o, 'under_price': u,
                        'is_alt_line': 0, 'book_last_update_utc': TS} for b, l, o, u in rows])
    return analysis.flag_main_lines(analysis.add_devig(df))


# fair over probabilities of three price pairs, by hand (multiplicative): q = 120/220 and 1/2 -> (6/11)/(23/22) = 12/23; -110/-110 -> 1/2; q = 130/230 and 1/2 -> (13/23)/(49/46) = 26/49
F_A, F_B, F_C = 12 / 23, 0.5, 26 / 49
P_A, P_B, P_C = (-120, -100), (-110, -110), (-130, 100)


def test_median_vs_mean_on_a_hand_computed_case():
    f = flagged_from([('draftkings', 20.5, *P_A), ('fanduel', 20.5, *P_B), ('betmgm', 20.5, *P_C), ('bovada', 21.5, *P_B)])
    mean = analysis.consensus(f)
    med = analysis.consensus(f, agg='median')
    assert mean['consensus_line'].iloc[0] == 20.5 and mean['n_at_line'].iloc[0] == 3 and mean['n_books'].iloc[0] == 4
    assert mean['consensus_fair_over'].iloc[0] == pytest.approx((F_A + F_B + F_C) / 3)
    assert med['consensus_fair_over'].iloc[0] == pytest.approx(F_A)                  # the middle of 0.5, 0.5217, 0.5306
    assert med['n_at_line'].iloc[0] == 3 and med['consensus_line'].iloc[0] == 20.5


def test_median_of_two_books_is_their_mean_and_min_books_applies():
    f = flagged_from([('draftkings', 20.5, *P_A), ('fanduel', 20.5, *P_C)])
    for agg in ('mean', 'median'):
        r = analysis.consensus(f, agg=agg, min_books=2)
        assert r['consensus_fair_over'].iloc[0] == pytest.approx((F_A + F_C) / 2)
        assert np.isnan(analysis.consensus(f, agg=agg, min_books=3)['consensus_fair_over'].iloc[0])


def test_bettable_only_recomputes_line_counts_and_min_books():
    rows = [('draftkings', 20.5, *P_B), ('fanduel', 21.5, *P_A), ('betmgm', 21.5, *P_C), ('williamhill_us', 21.5, *P_B),
            ('bovada', 20.5, *P_B), ('betonline', 20.5, *P_B), ('mybookie', 20.5, *P_B)]
    f = flagged_from(rows)
    allb = analysis.consensus(f)
    assert allb['consensus_line'].iloc[0] == 20.5 and allb['n_at_line'].iloc[0] == 4 and allb['n_books'].iloc[0] == 7
    bet = analysis.consensus(f, books=prices.BETTABLE)
    # four bettable books: 20.5 has one, 21.5 has three -> the consensus line moves to 21.5, three books agree, the mean is over fanduel, betmgm, williamhill_us
    assert bet['consensus_line'].iloc[0] == 21.5 and bet['n_at_line'].iloc[0] == 3 and bet['n_books'].iloc[0] == 4
    assert bet['book_spread'].iloc[0] == 1.0
    assert bet['consensus_fair_over'].iloc[0] == pytest.approx((F_A + F_C + F_B) / 3)
    assert analysis.consensus(f, books=prices.BETTABLE, agg='median')['consensus_fair_over'].iloc[0] == pytest.approx(F_A)
    # the min-books rule counts only bettable books: with dk, fd, betmgm only, 21.5 has two books -> no consensus value, although all books give a value
    f2 = flagged_from(rows[:3] + rows[4:])
    assert analysis.consensus(f2)['consensus_fair_over'].iloc[0] == pytest.approx(0.5)            # 20.5: dk, bovada, betonline, mybookie
    r = analysis.consensus(f2, books=prices.BETTABLE)
    assert r['consensus_line'].iloc[0] == 21.5 and r['n_at_line'].iloc[0] == 2 and np.isnan(r['consensus_fair_over'].iloc[0])
    # a book list that matches nothing gives an empty frame with the usual columns
    e = analysis.consensus(f, books=['nobody'])
    assert e.empty and list(e.columns) == analysis.GROUP + ['consensus_line', 'n_books', 'n_at_line', 'book_spread', 'consensus_fair_over']


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. The quote cache equals prices.load_season
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

STORE_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'betrivers', 'bovada', 'betonline']


def make_store(path, seed=5):
    """A small SQLite store with the real schema: 8 events in three months, three hist slots each, ladders, alt rows, one-sided and stale quotes, an in-game snapshot,
    a snapshot after its slot, an unresolved player, a team-total ladder (player_id -1), a name variant, an event of the other season and an event whose ET date differs from its tip date."""
    rng = np.random.default_rng(seed)
    conn = odds_store.connect(path)
    conn.executescript(schedule.REF_SCHEMA)
    events = [('v0', '2023-11-06T00:00:00Z', '2023-11-05'), ('v1', '2023-11-21T00:30:00Z', '2023-11-20'), ('v2', '2023-12-04T00:00:00Z', '2023-12-03'), ('v3', '2023-12-21T01:00:00Z', '2023-12-20'),
              ('v4', '2023-12-01T01:00:00Z', '2023-11-30'), ('v5', '2024-01-10T01:00:00Z', '2024-01-09'), ('v6', '2023-11-12T00:00:00Z', '2023-11-11'),
              ('vX', '2024-11-05T00:00:00Z', '2024-11-04')]
    for i, (eid, tip, et) in enumerate(events):
        conn.execute('INSERT INTO events (event_id_vendor, commence_time_utc, home_team, away_team, et_game_date, game_id) VALUES (?,?,?,?,?,?)', (eid, tip, 'H', 'A', et, f'G{i}'))
    prop_rows, log_rows, ref_rows = [], [], []
    for i, (eid, tip, et) in enumerate(events):
        tip_ts = pd.Timestamp(tip)
        for slot in (240, 60, 5):
            snap = (tip_ts - pd.Timedelta(minutes=slot + 3)).strftime('%Y-%m-%dT%H:%M:%SZ')
            log_rows.append((snap, eid, f'hist_T-{slot}', slot + 3.0, 0, 0, '', '', 0.0, None))
            ref_rows.append((snap, eid, slot + 3.0, 0))
            for p in range(5):
                pid, name = 1000 * i + p, f'Player {p}'
                for market in ('rebounds', 'assists', 'points'):
                    base = float(rng.integers(3, 12)) + 0.5
                    for b in STORE_BOOKS:
                        if rng.random() < 0.1:
                            continue
                        nm = 'P. Variant' if (p == 4 and b == 'betonline') else name
                        lines = [base + (float(rng.integers(-1, 2)) if rng.random() < 0.2 else 0.0)]
                        if b == 'bovada':
                            lines = [lines[0] + k for k in (-2, -1, 0, 1, 2)]
                        for l in lines:
                            over, under = float(-110 + rng.integers(-60, 60)), float(-110 + rng.integers(-60, 60))
                            if rng.random() < 0.04:
                                under = None
                            lag = int(rng.integers(0, 75))
                            upd = (pd.Timestamp(snap) - pd.Timedelta(minutes=lag)).strftime('%Y-%m-%dT%H:%M:%SZ')
                            prop_rows.append((snap, f'G{i}', eid, b, pid, nm, market, l, over, under, 0, 'hist', tip, 'H', 'A', upd))
                        if rng.random() < 0.1:                                                          # an alt row (excluded)
                            prop_rows.append((snap, f'G{i}', eid, b, pid, name, market, base + 3, -105.0, -115.0, 1, 'hist', tip, 'H', 'A', snap))
            prop_rows.append((snap, f'G{i}', eid, 'draftkings', None, 'Unresolved Guy', 'rebounds', 5.5, -110.0, -110.0, 0, 'hist', tip, 'H', 'A', snap))
            prop_rows.append((snap, f'G{i}', eid, 'fanduel', -1, 'Team total', 'points', 110.5, -110.0, -110.0, 0, 'hist', tip, 'H', 'A', snap))
        # a snapshot already in game, one stamped after its slot (minutes_to_ref below the slot) and one in a mode the cache must ignore
        late = (tip_ts + pd.Timedelta(minutes=20)).strftime('%Y-%m-%dT%H:%M:%SZ')
        log_rows.append((late, eid, 'hist_T-5', -20.0, 0, 0, '', '', 0.0, None)); ref_rows.append((late, eid, -20.0, 1))
        prop_rows.append((late, f'G{i}', eid, 'draftkings', 1000 * i, 'Player 0', 'rebounds', 7.5, -110.0, -110.0, 0, 'hist', tip, 'H', 'A', late))
        mid = (tip_ts - pd.Timedelta(minutes=50)).strftime('%Y-%m-%dT%H:%M:%SZ')
        log_rows.append((mid, eid, 'hist_T-60', 50.0, 0, 0, '', '', 0.0, None)); ref_rows.append((mid, eid, 50.0, 0))
        prop_rows.append((mid, f'G{i}', eid, 'draftkings', 1000 * i, 'Player 0', 'rebounds', 8.5, -110.0, -110.0, 0, 'hist', tip, 'H', 'A', mid))
        slow = (tip_ts - pd.Timedelta(minutes=500)).strftime('%Y-%m-%dT%H:%M:%SZ')
        log_rows.append((slow, eid, 'slow', 500.0, 0, 0, '', '', 0.0, None)); ref_rows.append((slow, eid, 500.0, 0))
        prop_rows.append((slow, f'G{i}', eid, 'draftkings', 1000 * i, 'Player 0', 'rebounds', 9.5, -110.0, -110.0, 0, 'hist', tip, 'H', 'A', slow))
    conn.executemany('INSERT OR IGNORE INTO prop_snapshots (snapshot_ts_utc, game_id, event_id_vendor, book, player_id, player_name_raw, market, line, over_price, under_price, is_alt_line,'
                     ' source, commence_time_utc, home_team, away_team, book_last_update_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', prop_rows)
    conn.executemany('INSERT INTO snapshot_log VALUES (?,?,?,?,?,?,?,?,?,?)', log_rows)
    conn.executemany('INSERT INTO snapshot_ref VALUES (?,?,?,?)', ref_rows)
    conn.commit()
    conn.close()
    return str(Path(path).as_posix())


@pytest.fixture(scope='module')
def store(tmp_path_factory):
    return make_store(tmp_path_factory.mktemp('odds') / 'props_odds.db')


@pytest.fixture(scope='module')
def season_result(store):
    return cq.collect_season(store, '2023-10-01', '2024-06-30')


def test_cache_prices_equal_load_season(store, season_result):
    cons_ref, bets_ref = prices.load_season(store, '2023-10-01', '2024-06-30')
    cons_ref, bets_ref = cq.sort_prices(cons_ref, bets_ref)
    assert len(cons_ref) > 100 and len(bets_ref) > 200, (len(cons_ref), len(bets_ref))
    assert set(bets_ref['book']) <= set(prices.BETTABLE)
    assert_frame_equal(season_result['cons'], cons_ref)
    assert_frame_equal(season_result['bets'], bets_ref)
    assert list(season_result['cons'].columns) == list(cons_ref.columns)
    assert len(season_result['months']) == 3                                                          # 2023-11, 2023-12 and 2024-01 (by ET date)


def test_other_season_and_bad_rows_are_excluded(season_result):
    c = season_result['cons']
    assert not c['game_id'].isin(['G7']).any()                                                       # the 2024-25 event
    assert not (c['stat'] == 'points').all() and 'points' in set(c['stat'])
    assert set(c['minutes_before']) == {240, 60, 5}
    assert (season_result['bets']['player_id'] >= 0).all()


def test_generic_current_variant_matches_the_price_file(season_result):
    cur = season_result['variants'][cq.CURRENT]
    cons = season_result['cons']
    assert len(cur) == len(cons)
    m = cur.merge(cons, on=cq.KEY, suffixes=('', '_p'), validate='one_to_one')
    assert len(m) == len(cons)
    for c in ('consensus_line', 'consensus_fair_over', 'n_at_line', 'book_spread', 'snapshot_ts_utc'):
        assert (m[c] == m[f'{c}_p']).all()
    assert cur['game_date'].notna().all() and cur['game_date'].min() >= '2023-11-01'


def test_all_nine_variants_are_built_and_differ_where_expected(season_result):
    v = season_result['variants']
    assert len(v) == 9 and list(v)[0] == cq.CURRENT
    assert all(len(t) > 0 for t in v.values())
    bett = v[('multiplicative', 'bettable_mean')]
    cur = v[cq.CURRENT]
    assert len(bett) <= len(cur) and bett['n_at_line'].min() >= 3
    med = v[('multiplicative', 'all_median')]
    assert len(med) == len(cur)                                                                       # same rows and lines, only the aggregate differs
    j = cur.merge(med, on=cq.KEY, suffixes=('_a', '_b'))
    assert (j['consensus_line_a'] == j['consensus_line_b']).all() and (j['consensus_fair_over_a'] != j['consensus_fair_over_b']).any()
    j = cur.merge(v[('shin', 'all_mean')], on=cq.KEY, suffixes=('_a', '_b'))
    assert (j['consensus_fair_over_a'] != j['consensus_fair_over_b']).any()


def test_month_chunking_equals_one_pass(store, season_result):
    one = cq.collect_season(store, '2023-10-01', '2024-06-30', chunk='all')
    assert len(one['months']) == 1 and len(season_result['months']) == 3
    assert_frame_equal(one['cons'], season_result['cons'])
    assert_frame_equal(one['bets'], season_result['bets'])
    for v in cq.VARIANTS:
        assert_frame_equal(one['variants'][v], season_result['variants'][v])


def test_event_id_index_is_used(store):
    with contextlib.closing(cq.connect_ro(store)) as conn:
        ev = cq.season_events(conn, '2023-10-01', '2024-06-30')
        plan = cq.explain_plan(conn, ev['event_id_vendor'].head(2).tolist(), '2023-10-01', '2024-06-30')
    assert any('ix_props_event_ts' in line and 'event_id_vendor=?' in line for line in plan)
    assert not any(line.startswith('SCAN p') or line.startswith('SCAN prop_snapshots') for line in plan)


def test_run_season_writes_only_under_the_given_folders(store, tmp_path):
    out, pout = tmp_path / 'quotes', tmp_path / 'prices'
    before = set(Path(store).parent.iterdir())
    res, man = cq.run_season(store, '2023-24', out, pout)
    assert set(Path(store).parent.iterdir()) == before                                               # nothing next to the store
    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()}
    expected = {f'quotes/2023-24/cons_{cq.variant_name(v)}.pkl' for v in cq.VARIANTS} | {'quotes/2023-24/manifest.json', 'quotes/2023-24/raw_2023-24.parquet', 'prices/prices_2023-24.pkl'}
    assert files == expected
    cons, bets = pd.read_pickle(pout / 'prices_2023-24.pkl')
    assert_frame_equal(cons, res['cons'])
    assert_frame_equal(bets, res['bets'])
    raw = pd.read_parquet(out / '2023-24' / 'raw_2023-24.parquet')
    assert len(raw) == man['rows_total'] == man['raw_cache_rows'] and raw['minutes_before'].isin([240, 60, 5]).all()
    # the raw cache regenerates the price file: running prices.quotes_from_snapshots on it gives the same consensus
    again_cons, again_bets = cq.sort_prices(*prices.quotes_from_snapshots(raw))
    assert_frame_equal(again_cons, res['cons'])
    assert_frame_equal(again_bets, res['bets'])
    with pytest.raises(FileExistsError):
        cq.run_season(store, '2023-24', out, pout)
    cq.run_season(store, '2023-24', out, pout, overwrite=True, skip_raw=True)


def test_limit_months_marks_the_outputs_partial(store, tmp_path):
    res, man = cq.run_season(store, '2023-24', tmp_path / 'q', tmp_path / 'p', limit_months=1)
    assert man['partial'] and len(man['months']) == 1
    assert (tmp_path / 'p' / 'prices_2023-24.partial.pkl').exists() and not (tmp_path / 'p' / 'prices_2023-24.pkl').exists()
    assert (tmp_path / 'q' / '2023-24.partial' / 'manifest.json').exists()


def test_dry_run_prints_counts_and_plan(store, capsys):
    cq.dry_run(store, ['2023-24'])
    out = capsys.readouterr().out
    assert '2023-11' in out and 'EXPLAIN QUERY PLAN' in out and 'ix_props_event_ts' in out


def test_store_is_opened_read_only(store):
    with contextlib.closing(cq.connect_ro(store)) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO unmatched_names (player_name_raw, team_context) VALUES ('x', 'y')")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Refusals
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('season', ['2025-26', '2026-27', '2022-23', 'junk'])
def test_cache_refuses_other_seasons(store, tmp_path, season):
    with pytest.raises(cq.SeasonRefused):
        cq.run_season(store, season, tmp_path / 'q', tmp_path / 'p')
    assert not (tmp_path / 'q').exists() and not (tmp_path / 'p').exists()                          # refused before anything is created


def test_cache_refuses_a_range_after_2025_06_30(store):
    with pytest.raises(cq.SeasonRefused):
        cq.collect_season(store, '2025-10-01', '2026-06-30')


def test_cli_refuses_2025_26(store, tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'a11_cache_quotes.py'), '--props-db', store, '--seasons', '2025-26', '--out', str(tmp_path / 'q'),
                        '--prices-out', str(tmp_path / 'p')], capture_output=True, text=True)
    assert r.returncode != 0 and 'REFUSED' in r.stderr and '2025-26' in r.stderr
    assert not (tmp_path / 'q').exists()
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'a11_devig_calibration.py'), '--quotes', str(tmp_path), '--features-db', store, '--out', str(tmp_path / 'o'),
                        '--report', str(tmp_path / 'r.md'), '--seasons', '2024-25', '2025-26'], capture_output=True, text=True)
    assert r.returncode != 0 and 'REFUSED' in r.stderr
    assert not (tmp_path / 'o').exists() and not (tmp_path / 'r.md').exists()


def test_calibration_refuses_2025_26_and_late_dates():
    with pytest.raises(cq.SeasonRefused):
        dc.evaluate({}, pd.DataFrame(), ['2025-26'])
    with pytest.raises(cq.SeasonRefused):
        dc.load_tables('nowhere', ['2025-26'])
    late = pd.DataFrame({'game_date': ['2025-10-22']})
    with pytest.raises(cq.SeasonRefused):
        dc.check_dates({'2024-25': {'x': late}})


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Calibration on simulated quotes with a known true probability
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def simulate(seed=7, n=24000, n_dates=150, bias_current=0.05, better='shin_bettable_mean', better_one_season='power_all_median', thin='shin_all_median', identical=False):
    """Tables for two seasons x 9 variants and labels. Truth p_true; the current method is p_true + bias + shared noise. `better` removes the bias in both seasons,
    `better_one_season` only in 2023-24, `thin` removes it but has 50% coverage. All other variants copy the current method (or all do when identical)."""
    rng = np.random.default_rng(seed)
    tables, lab_rows = {}, []
    for si, season in enumerate(cq.SEASONS):
        i = np.arange(n)
        p_true = rng.uniform(0.3, 0.7, n)
        hit = (rng.random(n) < p_true).astype(float)
        noise = rng.normal(0, 0.03, n)
        p_cur = np.clip(p_true + (0 if identical else bias_current) + noise, 0.02, 0.98)
        dates = (pd.Timestamp(f'{2023 + si}-11-01') + pd.to_timedelta(i % n_dates, unit='D')).strftime('%Y-%m-%d')
        gid = np.array([f'{season}-{k // 10}' for k in i])
        stat = np.where(i % 2 == 0, 'rebounds', 'assists')
        base = pd.DataFrame({'game_id': gid, 'player_id': i + 1000 * si, 'minutes_before': 60, 'stat': stat, 'game_date': dates, 'snapshot_ts_utc': 's', 'event_id_vendor': 'e',
                             'consensus_line': 8.5, 'n_at_line': 4, 'n_books': 5, 'book_spread': 0.0})
        lab_rows.append(pd.DataFrame({'game_id': gid, 'player_id': i + 1000 * si, 'lbl_played': 1, 'lbl_minutes': 30.0, 'y_points': 20.0, 'y_rebounds': np.where(hit == 1, 9.0, 7.0),
                                      'y_assists': np.where(hit == 1, 9.0, 7.0), 'y_threes': 2.0, 'y_pra': 40.0}))
        per = {}
        for v in cq.VARIANTS:
            name = cq.variant_name(v)
            p = p_cur
            if not identical and name == better:
                p = np.clip(p_true + noise, 0.02, 0.98)
            if not identical and name == better_one_season and si == 0:
                p = np.clip(p_true + noise, 0.02, 0.98)
            t = base.assign(consensus_fair_over=p)
            if name == thin:
                t = t.iloc[::2].copy() if not identical else t
                if not identical:
                    t['consensus_fair_over'] = np.clip(p_true + noise, 0.02, 0.98)[::2]
            per[name] = t.reset_index(drop=True)
        tables[season] = per
    return tables, pd.concat(lab_rows, ignore_index=True)


@pytest.fixture(scope='module')
def sim_result():
    tables, labels = simulate()
    return dc.evaluate(tables, labels, list(cq.SEASONS), n_boot=300, seed=1)


def test_metrics_on_known_probabilities():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.3, 0.7, 200000)
    h = (rng.random(len(p)) < p).astype(float)
    m = dc.seg_metrics(p, h)
    assert abs(m['bias']) < 0.004 and abs(m['recal_intercept']) < 0.03 and abs(m['recal_slope'] - 1) < 0.05
    assert m['logloss'] == pytest.approx(float(np.mean(-(h * np.log(p) + (1 - h) * np.log(1 - p)))))
    assert m['brier'] == pytest.approx(float(np.mean((p - h) ** 2)))
    shifted = dc.seg_metrics(np.clip(p + 0.05, 0, 0.99), h)
    assert shifted['bias'] == pytest.approx(-0.05, abs=0.004) and shifted['logloss'] > m['logloss']
    assert dc.ll(np.array([0.0, 1.0]), np.array([1.0, 0.0])).max() == pytest.approx(-np.log(dc.EPS))      # clipped, finite


def test_the_better_variant_is_chosen(sim_result):
    ch = sim_result['choice']
    assert ch['passed'] is True and ch['variant'] == 'shin_bettable_mean'
    assert (ch['devig_method'], ch['consensus_set']) == ('shin', 'bettable_mean')
    kv = sim_result['keep_variant'].set_index('variant')
    assert kv.loc['shin_bettable_mean', 'passed']
    ks = sim_result['keep_season']
    row = ks[(ks['variant'] == 'shin_bettable_mean') & (ks['season'] == '2023-24')].iloc[0]
    assert row['hi95'] < 0 and row['lo_bonf'] <= row['lo95'] and row['hi_bonf'] >= row['hi95']        # the Bonferroni interval is wider
    assert abs(row['bias']) < abs(row['bias_current'])


def test_variants_failing_a_condition_are_not_chosen(sim_result):
    kv = sim_result['keep_variant'].set_index('variant')
    assert not kv.loc['power_all_median', 'passed'] and '2024-25' in kv.loc['power_all_median', 'why_not']          # better in one season only
    assert not kv.loc['shin_all_median', 'passed'] and 'coverage' in kv.loc['shin_all_median', 'why_not']          # better calibrated but half the coverage
    for name in ('power_all_mean', 'multiplicative_bettable_mean', 'multiplicative_all_median'):                  # copies of the current method: no gain
        assert not kv.loc[name, 'passed']
    assert not kv.loc['power_all_mean', 'passed'] and 'upper bound' in kv.loc['power_all_mean', 'why_not']


def test_paired_difference_and_bias_removal(sim_result):
    p = sim_result['paired']
    r = p[(p['variant'] == 'shin_bettable_mean') & (p['metric'] == 'logloss') & (p['season'] == 'both')].iloc[0]
    assert r['diff'] < 0 and r['hi95'] < 0
    r0 = p[(p['variant'] == 'power_all_mean') & (p['metric'] == 'logloss') & (p['season'] == '2024-25')].iloc[0]
    assert r0['diff'] == 0 and r0['lo95'] == 0 and r0['hi95'] == 0                                                 # an identical method has no difference
    b = sim_result['bias_removed']
    r = b[(b['variant'] == 'shin_bettable_mean') & (b['season'] == '2024-25')].iloc[0]
    assert r['removed_share'] > 0.7 and 3.5 < r['bias_current_pts'] * -1 < 6.5


def test_identical_methods_adopt_nothing():
    tables, labels = simulate(identical=True)
    res = dc.evaluate(tables, labels, list(cq.SEASONS), n_boot=200, seed=1)
    assert res['choice']['passed'] is False and (res['choice']['devig_method'], res['choice']['consensus_set']) == ('multiplicative', 'all_mean')
    assert 'current method kept' in res['choice']['reason'] and not res['keep_variant']['passed'].any()


def test_one_season_alone_cannot_adopt():
    tables, labels = simulate()
    res = dc.evaluate(tables, labels, ['2023-24'], n_boot=200, seed=1)
    assert res['choice']['passed'] is False and 'required' in res['choice']['reason']


def test_the_tie_break_is_the_listed_order():
    tables, labels = simulate(better='shin_bettable_mean', better_one_season='none', thin='none')
    for season in tables:                                                                                          # a second, identical copy of the better variant, earlier in the order
        tables[season]['power_all_mean'] = tables[season]['shin_bettable_mean'].copy()
    res = dc.evaluate(tables, labels, list(cq.SEASONS), n_boot=200, seed=1)
    assert res['choice']['variant'] == 'power_all_mean'


def test_common_sample_requires_one_line_for_all_variants():
    tables, labels = simulate(n=2000, thin='none')
    t = tables['2023-24']
    t['shin_all_mean'] = t['shin_all_mean'].assign(consensus_line=np.where(np.arange(len(t['shin_all_mean'])) < 100, 9.5, 8.5))
    common = dc.common_sample(t)
    assert len(common) == 1900 and (common['line'] == 8.5).all()


def test_outcomes_drop_pushes_dnp_and_unlabelled_rows():
    common = pd.DataFrame({'game_id': ['g1'] * 5, 'player_id': [1, 2, 3, 4, 5], 'minutes_before': 60, 'stat': ['rebounds', 'ra', 'rebounds', 'rebounds', 'rebounds'],
                           'game_date': '2023-11-01', 'line': [8.5, 12.5, 8.5, 8.5, 8.5], **{f'p::{n}': 0.5 for n in dc.VNAMES}})
    lab = pd.DataFrame({'game_id': ['g1'] * 4, 'player_id': [1, 2, 3, 4], 'lbl_played': [1, 1, 0, 1], 'lbl_minutes': [30.0, 25.0, 0.0, 20.0], 'y_points': 0.0, 'y_rebounds': [9.0, 8.0, 0.0, 8.5],
                        'y_assists': [0.0, 4.0, 0.0, 0.0], 'y_threes': 0.0, 'y_pra': 0.0})
    d, c = dc.attach_outcomes(common, lab)
    assert c == {'common_rows': 5, 'no_label': 1, 'did_not_play': 1, 'missing_stat': 0, 'push': 1, 'used': 2}  # player 1 over; player 2: ra = 8 + 4 = 12 < 12.5 (under); 3 did not play; 4 pushes (8.5); 5 has no label
    assert dict(zip(d['player_id'], d['hit'])) == {1: 1.0, 2: 0.0}


def test_labels_are_read_read_only_and_results_written_only_under_out(tmp_path):
    tables, labels = simulate(n=3000)
    quotes = tmp_path / 'quotes'
    for season, per in tables.items():
        (quotes / season).mkdir(parents=True)
        for v in cq.VARIANTS:
            per[cq.variant_name(v)].to_pickle(quotes / season / f'cons_{cq.variant_name(v)}.pkl')
    fdb = tmp_path / 'features.db'
    with contextlib.closing(sqlite3.connect(fdb)) as conn:
        labels.to_sql('player_game_labels', conn, index=False)
    loaded = dc.load_labels(fdb)
    assert len(loaded) == len(labels) and loaded['game_id'].dtype == object
    import json
    out, report = tmp_path / 'results', tmp_path / 'rep' / 'report.md'
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'a11_devig_calibration.py'), '--quotes', str(quotes), '--features-db', str(fdb), '--out', str(out), '--report', str(report),
                        '--n-boot', '100'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    choice = json.loads((out / 'choice.json').read_text())
    assert set(['devig_method', 'consensus_set', 'passed', 'reason']) <= set(choice)
    text = report.read_text(encoding='utf-8')
    assert 'Bonferroni' in text and '0.0022' in text and 'keep rule passes' in text and 'Non-default bootstrap' in text
    for v in dc.VNAMES[1:]:
        assert v in text
    assert {p.name for p in tmp_path.iterdir()} == {'quotes', 'features.db', 'results', 'rep'}
    assert (out / 'keep_variant.csv').exists() and (out / 'paired.csv').exists()
