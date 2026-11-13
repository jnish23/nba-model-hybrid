"""
The live signal path under the A2 candidate rule format (reports/props/audit_2026-11/A2_live_plan.md): rule-file driven p* form (C3), p_market variant (Shin, mean over all books) and the causal
per-player-game cap. Synthetic inputs only (a throw-away SQLite file for the odds store), no data/ files, seconds.
"""
import importlib.util
import json
import math
import sqlite3
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import a11_cache_quotes as cq
import a11_rule_eval as re_
from core import props_board_logic as logic
from props import analysis
from props.betting import a11_rules as ar, evaluate as be, export, prices, rules
from props.live import current, quotes, signal
import test_a11_k_rules as kr  # noqa: E402  (its simulated registry / price files / labels DB, for the script tests below)

ROOT = Path(__file__).resolve().parents[2]
spec_ = importlib.util.spec_from_file_location('props_signal_job_a2_rule', ROOT / 'scripts' / 'props_signal_job.py')
job = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(job)

FROZEN = json.loads((ROOT / 'config' / 'props_frozen_rule.json').read_text(encoding='utf-8'))
STATS = ('rebounds', 'assists', 'threes', 'ra')
PROV = {'model_sha256': 'm', 'features_db_sha256': 'f', 'rule_sha256': 'r', 'roster_rule': 'pregame_all', 'code_commit': 'c'}
# the candidate_rule_a11.json structure (candidate C3, Shin de-vig with the all-books mean), parameters as in that file
CAND = {'stats': list(STATS), 'k_by_stat': {'rebounds': 0.1304, 'assists': 0.2434, 'threes': 0.2911, 'ra': 0.1688}, 'primary': {'e': 0.03, 'time': 'earliest', 'min_books': 3},
        'candidate': 'C3', 'pmarket': {'devig_method': 'shin', 'consensus_set': 'all_mean'},
        'intercept_by_stat': {'rebounds': {'a_pos': -0.0038, 'a_neg': -0.0256}, 'assists': {'a_pos': 0.0029, 'a_neg': -0.0109}, 'threes': {'a_pos': -0.0093, 'a_neg': -0.0099},
                              'ra': {'a_pos': 0.0006, 'a_neg': -0.0201}}, 'p_star_clip': [0.01, 0.99]}
TS, EV, GID = '2026-11-05T01:30:00Z', 'ev1', 'G1'


def _game(gid=GID):
    return pd.Series({'game_id': gid, 'season': '2026-27', 'game_date': '2026-11-04', 'tip_ref_utc': '2026-11-05T02:30:00Z', 'home_team': 'HOM', 'away_team': 'AWY',
                      'home_team_id': 1, 'away_team_id': 2, 'et_game_date': '2026-11-04'})


def _poisson_pmf(lam, n=80):
    p = np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(n)])
    return p / p.sum()


def _ctx(rule):
    return signal.Context(hist=None, pos={}, reports=None, tips=None, tables={}, bundle={}, rule=rule, nba_db='', n_sims=10, model_version='v', provenance=PROV)


def _patch_model(monkeypatch, mean_of=None):
    """predict_rows replaced by Poisson pmfs whose mean depends on (stat, player_id)."""
    mean_of = mean_of or (lambda stat, pid: 3.0 + (int(pid) % 5) + {'rebounds': 0.0, 'assists': 0.5, 'threes': -1.0, 'ra': 3.0}.get(stat, 0.0))
    monkeypatch.setattr(signal, 'predict_rows', lambda bundle, rows, n_sims, seed: (np.full(len(rows), 30.0), {s: np.array([_poisson_pmf(mean_of(s, p)) for p in rows['player_id']]) for s in STATS}))


def _frame(pids, gid=GID, slot=60):
    return pd.DataFrame({'game_id': gid, 'player_id': list(pids), 'minutes_before': slot, 'season': '2026-27', 'game_date': '2026-11-04'})


def _quotes(markets, gid=GID, slot=60, books=('b1', 'b2', 'b3'), n_at_line=3):
    """markets: (player_id, stat, line, over_dec, under_dec, consensus_fair_over). One row per book at the same prices."""
    rows = [{'game_id': gid, 'player_id': p, 'minutes_before': slot, 'stat': s, 'line': ln, 'book': b, 'over_dec': od, 'under_dec': ud, 'n_at_line': n_at_line, 'consensus_fair_over': pm,
             'quote_age_min': 1.0} for p, s, ln, od, ud, pm in markets for b in books]
    bets = pd.DataFrame(rows)
    cons = bets[['game_id', 'player_id']].drop_duplicates().reset_index(drop=True)
    return cons, bets


def _item(cons, bets, gid=GID, slot=60, pmarket=None):
    return signal.SlotItem(game=_game(gid), slot=slot, snapshot_ts_utc=TS, snapshot_minutes=float(slot), event_id='e', cons=cons, bets=bets,
                           names={int(p): f'Player {p}' for p in cons['player_id'].unique()}, pmarket=pmarket)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. a rule file without the new keys gives the signals and bets of the code before this change
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _old_evaluate_item(ctx, it, frame, existing):
    """Verbatim copy of props.live.signal.evaluate_item as it was before the rule-file keys (candidate, pmarket, cap) existed."""
    rule = ctx.rule
    gid = it.game['game_id']
    rows = frame[(frame['game_id'] == gid) & (frame['minutes_before'] == it.slot)]
    quoted = it.cons[['game_id', 'player_id']].drop_duplicates()
    rows = rows.merge(quoted, on=['game_id', 'player_id'], how='inner').sort_values(['player_id']).reset_index(drop=True)
    missing = sorted(set(quoted['player_id']) - set(rows['player_id']))
    detail = {'quoted_players': int(quoted['player_id'].nunique()), 'players_without_features': [int(p) for p in missing]}
    if 'skipped_players' in frame.attrs:
        detail['skip_reasons'] = {str(p): r for p, r in frame.attrs['skipped_players'].get((gid, it.slot), {}).items()}
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), detail
    mm, pm = signal.predict_rows(ctx.bundle, rows, ctx.n_sims, signal.slot_seed(gid, it.slot))
    idx = rows[['game_id', 'player_id', 'minutes_before', 'season', 'game_date']].assign(minutes_mean=mm).reset_index(drop=True)
    q = be.quote_table(it.bets, idx, pm)
    q = q[q['stat'].isin(rule['stats'])].reset_index(drop=True)
    if q.empty:
        return pd.DataFrame(), pd.DataFrame(), detail
    q = rules.add_ev(q, rule['k_by_stat'])
    prim = rule['primary']
    sel = rules.select(q, prim['e'], 'earliest', prim['min_books'])
    m = export.market_level(q)
    key = ['game_id', 'player_id', 'stat', 'minutes_before', 'line']
    best = q.sort_values('best_ev', ascending=False).drop_duplicates(['game_id', 'player_id', 'stat']).merge(m, on=key, how='left', validate='many_to_one')
    over = (best['best_side'] == 'over').to_numpy()
    best['dec_median'] = np.where(over, best['over_med'], best['under_med'])
    eo, eu = rules.ev_over_under(best['p_star'], best['push'], best['over_med'], best['under_med'])
    best['ev_median'] = np.where(over, eo, eu)
    eo, eu = rules.ev_over_under(best['p_star'], best['push'], logic.FLAT_110, logic.FLAT_110)
    best['ev_flat'] = np.where(over, eo, eu)
    qual_keys = set(map(tuple, sel[['game_id', 'player_id', 'stat']].itertuples(index=False, name=None)))
    k3 = list(map(tuple, best[['game_id', 'player_id', 'stat']].itertuples(index=False, name=None)))
    best['qualified'] = [int(k in qual_keys) for k in k3]
    best['already_bet'] = [int(k in existing) for k in k3]
    created = ctx.provenance
    sig = pd.DataFrame({
        'signal_id': [f"{it.game['season']}|{gid}|{p}|{s}|{it.slot}" for p, s in zip(best['player_id'], best['stat'])], 'season': it.game['season'], 'game_id': gid,
        'game_date': it.game['game_date'], 'tip_ref_utc': it.game['tip_ref_utc'], 'slot_minutes': it.slot, 'snapshot_ts_utc': it.snapshot_ts_utc, 'snapshot_minutes': it.snapshot_minutes,
        'player_id': best['player_id'].astype(int), 'player_name': best['player_id'].map(it.names), 'stat': best['stat'], 'line': best['line'], 'n_at_line': best['n_at_line'],
        'p_market': best['consensus_fair_over'], 'p_model': best['p_mod'], 'p_star': best['p_star'], 'push': best['push'], 'side': best['best_side'], 'book': best['book'],
        'dec_best': best['dec_price'], 'dec_median': best['dec_median'], 'ev_best': best['best_ev'], 'ev_median': best['ev_median'], 'ev_flat': best['ev_flat'],
        'expected_minutes': best['expected_minutes'], 'qualified': best['qualified'], 'already_bet': best['already_bet'], 'model_version': ctx.model_version})
    is_new = np.array([tuple(r) not in existing for r in sel[['game_id', 'player_id', 'stat']].itertuples(index=False, name=None)], dtype=bool)
    new = sel[is_new]
    new = new.merge(m, on=key, how='left', validate='many_to_one')
    if new.empty:
        return sig, pd.DataFrame(), detail
    over = (new['best_side'] == 'over').to_numpy()
    dec_med = np.where(over, new['over_med'], new['under_med'])
    eo, eu = rules.ev_over_under(new['p_star'], new['push'], new['over_med'], new['under_med'])
    ev_med = np.where(over, eo, eu)
    eo, eu = rules.ev_over_under(new['p_star'], new['push'], logic.FLAT_110, logic.FLAT_110)
    ev_flat = np.where(over, eo, eu)
    books = {}
    for (g, p, s), grp in q.groupby(['game_id', 'player_id', 'stat']):
        books[(g, p, s)] = grp
    out = []
    tip = pd.to_datetime(it.game['tip_ref_utc'], utc=True)
    for j, r in enumerate(new.itertuples(index=False)):
        grp = books[(r.game_id, r.player_id, r.stat)]
        grp = grp[grp['line'] == r.line]
        px = grp['over_dec'] if r.best_side == 'over' else grp['under_dec']
        out.append({
            'bet_id': f"{it.game['season']}|{gid}|{int(r.player_id)}|{r.stat}", 'season': it.game['season'], 'game_id': gid, 'game_date': it.game['game_date'],
            'tip_ref_utc': it.game['tip_ref_utc'], 'placed_at_utc': (tip - pd.Timedelta(minutes=it.slot)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'minutes_before': it.slot,
            'matchup': f"{it.game['away_team']} @ {it.game['home_team']}", 'home_team': it.game['home_team'], 'away_team': it.game['away_team'], 'player_id': int(r.player_id),
            'player_name': it.names.get(int(r.player_id)), 'stat': r.stat, 'side': r.best_side, 'line': float(r.line), 'book': r.book, 'dec_best': float(r.dec_price),
            'dec_median': float(dec_med[j]), 'dec_flat': float(logic.FLAT_110), 'ev_best': float(r.best_ev), 'ev_median': float(ev_med[j]), 'ev_flat': float(ev_flat[j]),
            'p_over_model': float(r.p_mod), 'p_over_star': float(r.p_star), 'p_over_market': float(r.consensus_fair_over), 'push': float(r.push), 'n_at_line': int(r.n_at_line),
            'n_quotes': int(r.n_quotes), 'expected_minutes': float(r.expected_minutes), 'snapshot_ts_utc': it.snapshot_ts_utc, 'snapshot_minutes': it.snapshot_minutes,
            'quote_age_min': float(r.quote_age_min), 'pmf_json': signal._json(pm[r.stat][int(r.i)]),
            'books_json': signal._json_books(grp['book'], px), 'model_version': ctx.model_version, 'model_sha256': created['model_sha256'], 'features_db_sha256': created['features_db_sha256'],
            'rule_sha256': created['rule_sha256'], 'roster_rule': created['roster_rule'], 'code_commit': created['code_commit']})
    return sig, pd.DataFrame(out), detail


def _random_markets(seed, pids=(1, 2, 3, 4)):
    rng = np.random.default_rng(seed)
    out = []
    for p in pids:
        for s in STATS:
            pm = float(rng.uniform(0.38, 0.62))
            out.append((p, s, float(rng.choice([3.5, 4.5, 5.5])), float(1 / (pm * 1.05) + rng.normal(0, 0.12)), float(1 / ((1 - pm) * 1.05) + rng.normal(0, 0.12)), pm))
    return out


@pytest.mark.parametrize('existing', [set(), {(GID, 2, 'assists'), (GID, 3, 'ra')}])
def test_frozen_format_rule_gives_the_same_signals_and_bets_as_the_code_before(monkeypatch, existing):
    _patch_model(monkeypatch)
    cons, bets = _quotes(_random_markets(7))
    it = _item(cons, bets)
    new_sig, new_bets, new_detail = signal.evaluate_item(_ctx(FROZEN), it, _frame(range(1, 5)), set(existing))
    old_sig, old_bets, old_detail = _old_evaluate_item(_ctx(FROZEN), it, _frame(range(1, 5)), set(existing))
    assert len(old_bets) >= 3 and (old_sig['qualified'] == 0).any()                         # the fixture exercises both qualified and unqualified markets
    assert_frame_equal(new_sig, old_sig, check_exact=True)
    assert_frame_equal(new_bets, old_bets, check_exact=True)
    assert new_detail == old_detail and 'capped_out' not in new_detail
    assert rules.rule_spec(FROZEN) == {'candidate': None, 'params': None, 'clip': rules.P_CLIP, 'pmarket': prices.CURRENT_PMARKET, 'cap': None}


def test_frozen_rule_quotes_are_the_unchanged_current_method(tmp_path):
    db, raw = _store(tmp_path)
    c0, b0 = prices.quotes_from_snapshots(raw)
    for pm in (None, prices.CURRENT_PMARKET):
        c1, b1, _ = quotes.slot_quotes(db, EV, GID, TS, 60, pmarket=pm)
        assert_frame_equal(c1, c0)
        assert_frame_equal(b1, b0)
    c2, b2, _ = quotes.slot_quotes(db, EV, GID, TS, 60)
    assert_frame_equal(c2, c0)
    assert_frame_equal(b2, b0)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# synthetic odds store: snapshot rows of one event, with a ladder market on which Shin and multiplicative de-vig pick different main lines
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'betrivers', 'unibet_us', 'fanatics', 'bovada', 'betonline']
LADDER_PID = 120


def _snapshot_rows(seed=3):
    rng = np.random.default_rng(seed)
    rows = []

    def add(book, pid, market, line, over, under, lag=5):
        rows.append({'snapshot_ts_utc': TS, 'event_id_vendor': EV, 'book': book, 'player_id': pid, 'player_name_raw': f'Player {pid}', 'market': market, 'line': line, 'over_price': over,
                     'under_price': under, 'is_alt_line': 0, 'book_last_update_utc': (pd.Timestamp(TS) - pd.Timedelta(minutes=lag)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                     'home_team': 'HOM', 'away_team': 'AWY'})
    def american(q):
        return float(np.round(-100 * q / (1 - q) if q > 0.5 else 100 * (1 - q) / q))

    for pid in range(101, 109):
        for market in STATS:
            base = float(rng.integers(2, 9)) + 0.5
            p0 = float(rng.uniform(0.42, 0.58))
            for b in BOOKS:
                if rng.random() < 0.08:
                    continue
                n_lines = 3 if (b == 'bovada' and rng.random() < 0.5) else 1                                 # ladders
                first = base + float(rng.integers(-1, 2)) * (rng.random() < 0.2)
                for k in range(n_lines):
                    fo, vig = float(np.clip(p0 + rng.normal(0, 0.02) - 0.08 * k, 0.05, 0.95)), float(rng.uniform(1.03, 1.08))
                    add(b, pid, market, first + k, american(fo * vig), american((1 - fo) * vig), lag=int(rng.integers(0, 70)))
    # the ladder market: multiplicative de-vig makes bovada's main line 5.5, Shin makes it 4.5; consensus line 5.5 (multiplicative) against 4.5 (Shin)
    for b in ('draftkings', 'fanduel'):
        add(b, LADDER_PID, 'rebounds', 4.5, -110.0, -110.0)
    for b in ('betmgm', 'williamhill_us'):
        add(b, LADDER_PID, 'rebounds', 5.5, -110.0, -110.0)
    add('bovada', LADDER_PID, 'rebounds', 4.5, -115.0, 100.0)
    add('bovada', LADDER_PID, 'rebounds', 5.5, -125.0, -145.0)
    return pd.DataFrame(rows)


def _store(tmp_path, seed=3):
    """Writes the rows to a SQLite file with the prop_snapshots columns slot_quotes reads; returns (path, the rows slot_quotes reads back)."""
    db = tmp_path / 'props_odds.db'
    rows = _snapshot_rows(seed)
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE prop_snapshots (snapshot_ts_utc TEXT, event_id_vendor TEXT, book TEXT, player_id INTEGER, player_name_raw TEXT, market TEXT, line REAL, over_price REAL, '
                     'under_price REAL, is_alt_line INTEGER, book_last_update_utc TEXT, home_team TEXT, away_team TEXT)')
        conn.executemany(f"INSERT INTO prop_snapshots VALUES ({','.join('?' * len(rows.columns))})", rows[list(rows.columns)].itertuples(index=False, name=None))
    _, _, raw = quotes.slot_quotes(db, EV, GID, TS, 60)
    return db, raw


def _research_quotes(raw, variant=('shin', 'all_mean')):
    """The research path of scripts/a11_cache_quotes.py + a11_k_estimation.load_season: current-method quotes, the variant table, apply_pmarket."""
    cons, bets = prices.quotes_from_snapshots(raw)
    tables = cq.variant_tables(raw, {EV: '2026-11-04'})
    return ar.apply_pmarket(cons, bets, tables[variant])


def test_the_ladder_market_makes_the_two_methods_pick_different_consensus_lines(tmp_path):
    _, raw = _store(tmp_path)
    cur_cons, _ = prices.quotes_from_snapshots(raw)
    var = cq.variant_tables(raw, {EV: '2026-11-04'})[('shin', 'all_mean')]
    a = cur_cons[(cur_cons['player_id'] == LADDER_PID) & (cur_cons['stat'] == 'rebounds')].iloc[0]
    b = var[(var['player_id'] == LADDER_PID) & (var['stat'] == 'rebounds')].iloc[0]
    assert (a['consensus_line'], b['consensus_line']) == (5.5, 4.5) and a['n_at_line'] == 3 and b['n_at_line'] == 3


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. quotes not on the variant consensus line are dropped as research does; the variant is built by the research functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_live_variant_quotes_equal_the_research_substitution_and_drop_off_line_quotes(tmp_path):
    db, raw = _store(tmp_path)
    cons_live, bets_live, _ = quotes.slot_quotes(db, EV, GID, TS, 60, pmarket=('shin', 'all_mean'))
    cons_r, bets_r, dropped = _research_quotes(raw)
    assert_frame_equal(cons_live.drop(columns=['player_name_raw']), cons_r)                  # the research cache's slim variant tables do not carry player_name_raw; nothing downstream reads it
    assert_frame_equal(bets_live, bets_r)
    assert dropped >= 2                                                                     # the betmgm / williamhill_us quotes at 5.5 of the ladder market
    cur_cons, cur_bets = prices.quotes_from_snapshots(raw)
    assert len(bets_live) == len(cur_bets) - dropped
    lad_cur = cur_bets[(cur_bets['player_id'] == LADDER_PID) & (cur_bets['stat'] == 'rebounds')]
    assert set(lad_cur['book']) == {'betmgm', 'williamhill_us'} and set(lad_cur['line']) == {5.5}
    assert bets_live[(bets_live['player_id'] == LADDER_PID) & (bets_live['stat'] == 'rebounds')].empty        # dropped: they are not on Shin's consensus line (4.5)
    assert (cons_live[(cons_live['player_id'] == LADDER_PID) & (cons_live['stat'] == 'rebounds')]['consensus_line'] == 4.5).all()
    on_line = bets_live.merge(cons_live[['game_id', 'player_id', 'minutes_before', 'stat', 'consensus_line']], on=['game_id', 'player_id', 'minutes_before', 'stat'], validate='many_to_one')
    assert len(on_line) == len(bets_live) and (on_line['line'] == on_line['consensus_line']).all()
    same = bets_live.merge(cur_bets, on=['game_id', 'player_id', 'stat', 'book', 'line'], suffixes=('', '_cur'))
    assert len(same) > 20 and not np.allclose(same['consensus_fair_over'], same['consensus_fair_over_cur'], atol=1e-6)      # Shin differs from multiplicative; n_at_line / p_market are the variant's


def test_the_variant_table_is_unchanged_by_the_move_of_the_consensus_code(tmp_path):
    """scripts/a11_cache_quotes.variant_tables now calls props.betting.prices.variant_consensus; its 9 tables equal the code it had before."""
    _, raw = _store(tmp_path)

    def old_finish(cons_raw, d):
        cons = cons_raw.dropna(subset=['consensus_fair_over'])
        keys = d[['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw', 'game_id', 'player_id', 'minutes_before']].dropna(subset=['player_id']).drop_duplicates(
            ['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw'])
        cons = cons.merge(keys, on=['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw'], how='inner', validate='many_to_one')
        cons['player_id'] = cons['player_id'].astype(int)
        key = ['game_id', 'player_id', 'minutes_before', 'market']
        cons = cons.sort_values(key + ['n_at_line', 'snapshot_ts_utc'], ascending=[True] * len(key) + [False, True]).drop_duplicates(key)
        return cons.rename(columns={'market': 'stat'})[['game_id', 'player_id', 'minutes_before', 'stat', 'snapshot_ts_utc', 'event_id_vendor', 'player_name_raw',
                                                         'consensus_line', 'consensus_fair_over', 'n_at_line', 'n_books', 'book_spread']]
    sets = {'all_mean': {'agg': 'mean', 'books': None}, 'bettable_mean': {'agg': 'mean', 'books': prices.BETTABLE}, 'all_median': {'agg': 'median', 'books': None}}
    new = cq.variant_tables(raw, {EV: '2026-11-04'})
    assert list(new) == cq.VARIANTS and len(new) == 9
    for method in ('multiplicative', 'power', 'shin'):
        flagged = analysis.flag_main_lines(analysis.add_devig(raw, method=method, stale_after_minutes=60))
        for sname, s in sets.items():
            old = old_finish(analysis.consensus(flagged, min_books=3, agg=s['agg'], books=s['books']), raw)
            old['game_date'] = old['event_id_vendor'].map({EV: '2026-11-04'})
            assert_frame_equal(new[(method, sname)], old[cq.SLIM_COLUMNS].reset_index(drop=True))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. the live path under the candidate rule equals the research path, to 1e-12
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _research_params(tmp_path, rule):
    p = tmp_path / 'rule.json'
    p.write_text(json.dumps(rule), encoding='utf-8')
    return re_.load_rule(Namespace(k_file=None, rule_file=str(p)))                                   # the research loader of the rule file


def _research_signals(raw, frame, rule, cand, params, existing=frozenset()):
    """The research chain on the same inputs: variant quotes, quote table, p* and EV by a11_rules, selection by a11_rules.select_candidate."""
    _, bets_r, _ = _research_quotes(raw)
    rows = frame[frame['minutes_before'] == 60]
    mm, pm = signal.predict_rows(None, rows.sort_values('player_id').reset_index(drop=True), 10, 0)
    idx = rows.sort_values('player_id').reset_index(drop=True)[['game_id', 'player_id', 'minutes_before', 'season', 'game_date']].assign(minutes_mean=mm)
    q = be.quote_table(bets_r, idx, pm)
    q = q[q['stat'].isin(rule['stats'])].reset_index(drop=True)
    full = ar.add_ev_pstar(q, ar.p_star(cand, q, params))
    sel = ar.select_candidate(q, cand, params)
    best = full.sort_values('best_ev', ascending=False).drop_duplicates(['game_id', 'player_id', 'stat'])
    return best, sel


def test_live_path_under_the_candidate_rule_equals_the_research_path(tmp_path, monkeypatch):
    _patch_model(monkeypatch)
    db, raw = _store(tmp_path)
    cons, bets, _ = quotes.slot_quotes(db, EV, GID, TS, 60, pmarket=('shin', 'all_mean'))
    pids = sorted(set(cons['player_id']))
    frame = _frame(pids)
    it = signal.SlotItem(game=_game(), slot=60, snapshot_ts_utc=TS, snapshot_minutes=60.0, event_id=EV, cons=cons, bets=bets, names={p: f'Player {p}' for p in pids}, pmarket=('shin', 'all_mean'))
    sig, new, detail = signal.evaluate_item(_ctx(CAND), it, frame, set())
    cand, params, pm, _ = _research_params(tmp_path, CAND)
    assert cand == 'C3' and pm == ('shin', 'all_mean')
    best, sel = _research_signals(raw, frame, CAND, cand, params)
    assert len(sel) >= 5 and len(sel) < len(best)
    key = ['game_id', 'player_id', 'stat']
    s = sig.merge(best, on=key, suffixes=('', '_r'), validate='one_to_one')
    assert len(s) == len(sig) == len(best)

    def col(frame, name):                                           # the research column after the merge (suffixed where the live frame has the same name)
        return frame[name + '_r' if name in frame.columns and name + '_r' in frame.columns else name].to_numpy(float)
    for live, res in (('p_market', 'consensus_fair_over'), ('p_model', 'p_mod'), ('p_star', 'p_star'), ('ev_best', 'best_ev'), ('push', 'push'), ('line', 'line'), ('dec_best', 'dec_price')):
        np.testing.assert_allclose(s[live].to_numpy(float), col(s, res), rtol=0, atol=1e-12, err_msg=live)
    assert (s['side'] == s['best_side']).all() and (s['book'] == s['book_r']).all() and (s['n_at_line'] == s['n_at_line_r']).all()
    assert set(map(tuple, new[key].itertuples(index=False, name=None))) == set(map(tuple, sel[key].itertuples(index=False, name=None)))
    n = new.merge(sel, on=key, suffixes=('', '_r'), validate='one_to_one')
    for live, res in (('p_over_market', 'consensus_fair_over'), ('p_over_star', 'p_star'), ('ev_best', 'best_ev'), ('p_over_model', 'p_mod'), ('line', 'line'), ('dec_best', 'dec_price')):
        np.testing.assert_allclose(n[live].to_numpy(float), col(n, res), rtol=0, atol=1e-12, err_msg=live)
    assert (n['side'] == n['best_side']).all() and (n['book'] == n['book_r']).all()
    assert (sig['qualified'] == 1).sum() == len(new) and 'capped_out' not in detail
    # p* is the C3 form: spot-check every signal by hand, including the sign-dependent intercept
    g = sig['p_model'] - sig['p_market']
    a = np.array([CAND['intercept_by_stat'][s_]['a_pos' if gi >= 0 else 'a_neg'] for s_, gi in zip(sig['stat'], g)])
    kk = sig['stat'].map(CAND['k_by_stat']).to_numpy(float)
    np.testing.assert_allclose(sig['p_star'].to_numpy(float), np.clip(sig['p_market'] + a + kk * g, 0.01, 0.99), rtol=0, atol=1e-12)
    assert (g > 0).any() and (g < 0).any()
    # the variant is in force: the stored p_market is Shin's, not the multiplicative consensus
    cur_cons, _ = prices.quotes_from_snapshots(raw)
    m = sig.merge(cur_cons, on=key, suffixes=('', '_cur'))
    assert len(m) > 20 and not np.allclose(m['p_market'], m['consensus_fair_over'], atol=1e-6)
    nb = new.merge(sig[key + ['p_market']], on=key)
    np.testing.assert_allclose(nb['p_over_market'].to_numpy(float), nb['p_market'].to_numpy(float), rtol=0, atol=1e-12)


def test_g_equal_zero_takes_the_positive_intercept_as_research_p_star_does(tmp_path):
    q = pd.DataFrame({'stat': ['assists'] * 3, 'consensus_fair_over': [0.5, 0.5, 0.5], 'p_mod': [0.5, 0.6, 0.4], 'minutes_before': 60})
    params = {'b': {'assists': 0.3}, 'a_pos': {'assists': -0.01}, 'a_neg': {'assists': -0.04}}
    got = rules.p_star('C3', q, params)
    np.testing.assert_allclose(got, [0.5 - 0.01, 0.5 - 0.01 + 0.03, 0.5 - 0.04 - 0.03], atol=1e-15)
    np.testing.assert_array_equal(got, ar.p_star('C3', q, params))


def test_quotes_built_under_another_p_market_than_the_rules_are_refused(monkeypatch):
    _patch_model(monkeypatch)
    cons, bets = _quotes([(1, 'assists', 4.5, 2.0, 2.0, 0.5)])
    with pytest.raises(ValueError, match='p_market'):
        signal.evaluate_item(_ctx(CAND), _item(cons, bets), _frame([1]), set())                                   # candidate rule, current-method quotes
    with pytest.raises(ValueError, match='p_market'):
        signal.evaluate_item(_ctx(FROZEN), _item(cons, bets, pmarket=('shin', 'all_mean')), _frame([1]), set())   # frozen rule, variant quotes


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. the causal cap
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

CAP_RULE = {'stats': ['rebounds', 'assists'], 'k_by_stat': {'rebounds': 0.3, 'assists': 0.3}, 'primary': {'e': 0.03, 'min_books': 3}}


def _cap_inputs(markets):
    cons, bets = _quotes(markets)
    return _item(cons, bets), _frame(sorted(set(cons['player_id'])))


def _strong(monkeypatch):
    _patch_model(monkeypatch, mean_of=lambda stat, pid: 9.0)                                       # P(over 4.5) close to 1 for every market: every priced market qualifies


def test_a_player_game_that_already_has_a_bet_gets_no_new_bet(monkeypatch):
    _strong(monkeypatch)
    it, frame = _cap_inputs([(7, 'rebounds', 4.5, 2.0, 1.5, 0.5), (7, 'assists', 4.5, 2.1, 1.5, 0.5), (8, 'rebounds', 4.5, 2.0, 1.5, 0.5)])
    rule = {**CAP_RULE, 'cap_per_player_game': 1}
    sig, new, detail = signal.evaluate_item(_ctx(rule), it, frame, {(GID, 7, 'rebounds')})
    assert list(new['player_id']) == [8]                                                           # player 7 has a bet (rebounds, an earlier slot): his assists qualify but are capped
    by = sig.set_index(['player_id', 'stat'])
    assert by.loc[(7, 'rebounds'), 'already_bet'] == 1 and by.loc[(7, 'rebounds'), 'qualified'] == 1
    assert by.loc[(7, 'assists'), 'qualified'] == 0 and by.loc[(7, 'assists'), 'already_bet'] == 0 and by.loc[(7, 'assists'), 'ev_best'] >= 0.03
    assert by.loc[(8, 'rebounds'), 'qualified'] == 1 and len(sig) == 3                              # every evaluated market is still a signal row
    assert detail['capped_out'] == [{'player_id': 7, 'stat': 'assists', 'reason': 'player_game_has_bet'}]
    sig2, new2, _ = signal.evaluate_item(_ctx(rule), it, frame, {(GID, 7, 'rebounds'), (GID, 8, 'rebounds')})
    assert new2.empty
    sig3, new3, _ = signal.evaluate_item(_ctx(rule), it, frame, {('OTHER_GAME', 7, 'rebounds')})        # a bet on another game does not count
    assert sorted(zip(new3['player_id'], new3['stat'])) == [(7, 'assists'), (8, 'rebounds')]


def test_two_qualifying_stats_in_one_slot_give_the_higher_ev_bet(monkeypatch):
    _strong(monkeypatch)
    it, frame = _cap_inputs([(7, 'rebounds', 4.5, 2.0, 1.5, 0.5), (7, 'assists', 4.5, 2.2, 1.5, 0.5), (8, 'rebounds', 4.5, 2.0, 1.5, 0.5)])
    sig, new, detail = signal.evaluate_item(_ctx({**CAP_RULE, 'cap_per_player_game': 1}), it, frame, set())
    assert sorted(zip(new['player_id'], new['stat'])) == [(7, 'assists'), (8, 'rebounds')]         # the cap is per player-game: player 8 keeps his bet
    by = sig.set_index(['player_id', 'stat'])
    assert by.loc[(7, 'assists'), 'ev_best'] > by.loc[(7, 'rebounds'), 'ev_best'] >= 0.03
    assert by.loc[(7, 'rebounds'), 'qualified'] == 0 and by.loc[(7, 'assists'), 'qualified'] == 1
    assert detail['capped_out'] == [{'player_id': 7, 'stat': 'rebounds', 'reason': 'lower_ev_same_slot'}]
    assert (sig['qualified'] == 1).sum() == len(new)                                                # qualified & not already_bet <=> a bet row, as without the cap


def test_equal_ev_is_broken_by_the_stat_name(monkeypatch):
    _strong(monkeypatch)
    for order in ([(7, 'rebounds', 4.5, 2.0, 1.5, 0.5), (7, 'assists', 4.5, 2.0, 1.5, 0.5)], [(7, 'assists', 4.5, 2.0, 1.5, 0.5), (7, 'rebounds', 4.5, 2.0, 1.5, 0.5)]):
        it, frame = _cap_inputs(order)
        sig, new, _ = signal.evaluate_item(_ctx({**CAP_RULE, 'cap_per_player_game': 1}), it, frame, set())
        assert list(new['stat']) == ['assists']                                                    # identical best-price EV: alphabetical stat first, whatever the row order


def test_cap_two_and_no_cap_key(monkeypatch):
    _strong(monkeypatch)
    it, frame = _cap_inputs([(7, 'rebounds', 4.5, 2.0, 1.5, 0.5), (7, 'assists', 4.5, 2.2, 1.5, 0.5)])
    _, none, d0 = signal.evaluate_item(_ctx(CAP_RULE), it, frame, set())
    assert sorted(none['stat']) == ['assists', 'rebounds'] and 'capped_out' not in d0               # no cap_per_player_game key: no cap
    _, none2, _ = signal.evaluate_item(_ctx(CAP_RULE), it, frame, {(GID, 7, 'rebounds')})
    assert list(none2['stat']) == ['assists']                                                      # and a bet on another stat of the player-game does not block (unchanged behaviour)
    _, two, _ = signal.evaluate_item(_ctx({**CAP_RULE, 'cap_per_player_game': 2}), it, frame, set())
    assert sorted(two['stat']) == ['assists', 'rebounds']
    _, two2, _ = signal.evaluate_item(_ctx({**CAP_RULE, 'cap_per_player_game': 2}), it, frame, {(GID, 7, 'rebounds')})
    assert list(two2['stat']) == ['assists']                                                       # one bet taken, room for one more
    _, two3, _ = signal.evaluate_item(_ctx({**CAP_RULE, 'cap_per_player_game': 2}), it, frame, {(GID, 7, 'rebounds'), (GID, 7, 'threes')})
    assert two3.empty


def test_causal_cap_is_the_earliest_slot_not_the_best_bet_overall():
    sel = pd.DataFrame({'game_id': 'g', 'player_id': [1, 1, 1, 2, 2], 'stat': ['assists', 'rebounds', 'threes', 'assists', 'ra'], 'minutes_before': [240, 60, 60, 60, 60],
                        'best_ev': [0.04, 0.10, 0.12, 0.05, 0.05]})
    assert sorted(zip(rules.cap_per_player_game(sel, 1)['player_id'], rules.cap_per_player_game(sel, 1)['stat'])) == [(1, 'threes'), (2, 'assists')]        # non-causal: the best bet of the player-game
    causal = rules.cap_per_player_game_causal(sel, 1)
    assert sorted(zip(causal['player_id'], causal['stat'])) == [(1, 'assists'), (2, 'assists')]                    # causal: player 1's T-240 bet stands; player 2's tie goes to the first stat name
    c2 = rules.cap_per_player_game_causal(sel, 2)
    assert sorted(zip(c2['player_id'], c2['stat'])) == [(1, 'assists'), (1, 'threes'), (2, 'assists'), (2, 'ra')]
    assert len(rules.cap_per_player_game_causal(sel.iloc[0:0], 1)) == 0


def test_the_slot_by_slot_live_calls_equal_the_research_causal_cap(monkeypatch):
    """Replaying the slots in time order through evaluate_item (each slot's new bets joined to `existing`) gives the bets of rules.cap_per_player_game_causal on the uncapped selection."""
    _strong(monkeypatch)
    rng = np.random.default_rng(5)
    rule_nocap = {**CAP_RULE, 'stats': ['rebounds', 'assists']}
    existing_cap, existing_free, got, free_rows = set(), set(), [], []
    for slot in (240, 60, 5):
        mk = [(p, s, 4.5, float(rng.uniform(1.75, 2.4)), 1.5, 0.5) for p in range(1, 6) for s in ('rebounds', 'assists') if rng.random() < 0.7]
        cons, bets = _quotes(mk, slot=slot)
        it, frame = _item(cons, bets, slot=slot), _frame(sorted(set(cons['player_id'])), slot=slot)
        _, n_cap, _ = signal.evaluate_item(_ctx({**rule_nocap, 'cap_per_player_game': 1}), it, frame, existing_cap)
        existing_cap |= set(zip(n_cap['game_id'], n_cap['player_id'], n_cap['stat'])) if len(n_cap) else set()
        got.append(n_cap)
        _, n_free, _ = signal.evaluate_item(_ctx(rule_nocap), it, frame, existing_free)
        existing_free |= set(zip(n_free['game_id'], n_free['player_id'], n_free['stat'])) if len(n_free) else set()
        free_rows.append(n_free)
    free = pd.concat(free_rows, ignore_index=True).assign(best_ev=lambda d: d['ev_best'])             # the uncapped selection (rules.select output has best_ev)
    live = pd.concat(got, ignore_index=True)
    ref = rules.cap_per_player_game_causal(free, 1)
    key = ['player_id', 'stat', 'minutes_before']
    assert len(live) == len(ref) > 3 and len(free) > len(ref)
    assert sorted(map(tuple, live[key].itertuples(index=False, name=None))) == sorted(map(tuple, ref[key].itertuples(index=False, name=None)))
    assert live.groupby(['game_id', 'player_id']).size().max() == 1


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 5. a rule file that is not understood fails loudly at load
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _bad(**changes):
    r = json.loads(json.dumps(CAND))
    for k, v in changes.items():
        if v is KeyError:
            r.pop(k, None)
        else:
            r[k] = v
    return r


def test_the_candidate_file_and_the_frozen_file_are_accepted():
    s = rules.rule_spec(CAND)
    assert s['candidate'] == 'C3' and s['pmarket'] == ('shin', 'all_mean') and s['cap'] is None and s['clip'] == (0.01, 0.99)
    assert s['params']['a_neg']['rebounds'] == CAND['intercept_by_stat']['rebounds']['a_neg']
    assert rules.rule_spec({**CAND, 'cap_per_player_game': 1})['cap'] == 1
    path = ROOT / 'reports' / 'props' / 'audit_2026-11' / 'candidate_rule_a11.json'
    if path.exists():
        real = json.loads(path.read_text(encoding='utf-8'))
        rs = rules.rule_spec(real)
        assert rs['candidate'] == 'C3' and rs['pmarket'] == ('shin', 'all_mean') and set(rs['params']['a_pos']) == set(real['stats'])
        cand, params, pm, _ = re_.load_rule(Namespace(k_file=None, rule_file=str(path)))
        assert (cand, pm) == ('C3', ('shin', 'all_mean')) and params == {k: v for k, v in rs['params'].items()}


@pytest.mark.parametrize('rule, why', [
    (_bad(candidate='C9'), 'unknown candidate'),
    (_bad(candidate='P0'), 'unknown candidate'),
    (_bad(intercept_by_stat={s: v for s, v in CAND['intercept_by_stat'].items() if s != 'ra'}), 'intercept_by_stat has no entry'),
    (_bad(intercept_by_stat=KeyError), 'needs intercept_by_stat'),
    (_bad(intercept_by_stat={**CAND['intercept_by_stat'], 'ra': {'a_pos': 0.0}}), 'a_pos, a_neg'),
    (_bad(intercept_by_stat={**CAND['intercept_by_stat'], 'ra': {'a_pos': 0.0, 'a_neg': float('nan')}}), 'finite'),
    (_bad(p_star_clip=KeyError), 'p_star_clip'),
    (_bad(p_star_clip=[0.5, 0.2]), 'p_star_clip'),
    (_bad(k_by_stat={s: v for s, v in CAND['k_by_stat'].items() if s != 'threes'}), 'k_by_stat has no entry'),
    (_bad(pmarket={'devig_method': 'logit', 'consensus_set': 'all_mean'}), 'devig_method'),
    (_bad(pmarket={'devig_method': 'shin', 'consensus_set': 'trimmed_mean'}), 'consensus_set'),
    (_bad(pmarket={'devig_method': 'shin'}), 'pmarket must be'),
    (_bad(candidate=KeyError), 'without a candidate'),                                              # intercepts present but no form named: never fall back to k-only silently
    (_bad(cap_per_player_game=0), 'cap_per_player_game'),
    (_bad(cap_per_player_game=1.5), 'cap_per_player_game'),
    (_bad(cap_per_player_game=True), 'cap_per_player_game'),
    (_bad(cap_per_player_game='1'), 'cap_per_player_game')])
def test_unknown_or_incomplete_rule_files_raise_at_load(rule, why):
    with pytest.raises(rules.RuleFileError, match=why):
        rules.rule_spec(rule)
    with pytest.raises(rules.RuleFileError):                                                       # and evaluate_item / add_ev_rule refuse them too, never a fallback to the old form
        rules.add_ev_rule(pd.DataFrame({'stat': ['assists'], 'consensus_fair_over': [0.5], 'p_mod': [0.6], 'push': [0.0], 'over_dec': [2.0], 'under_dec': [1.9]}), rule)


def test_evaluate_item_raises_on_an_unreadable_rule(monkeypatch):
    _patch_model(monkeypatch)
    cons, bets = _quotes([(1, 'assists', 4.5, 2.0, 2.0, 0.5)])
    with pytest.raises(rules.RuleFileError):
        signal.evaluate_item(_ctx(_bad(candidate='C9')), _item(cons, bets, pmarket=('shin', 'all_mean')), _frame([1]), set())


def test_the_job_passes_the_rules_p_market_to_the_quote_builder(tmp_path, monkeypatch):
    from props.live import store
    conn = store.connect(tmp_path / 'live.db')
    tip = '2026-11-05T02:30:00Z'
    now = pd.Timestamp('2026-11-05T01:31:00Z')
    sched = pd.DataFrame({'game_id': ['G1'], 'et_game_date': ['2026-11-04'], 'home_team_id': [1], 'away_team_id': [2]})
    matched = pd.DataFrame({'game_id': ['G1'], 'event_id_vendor': ['e1'], 'tip_ref_utc': [tip]})
    monkeypatch.setattr(job.quotes, 'event_snapshots', lambda db, ev, since_utc=None: pd.DataFrame({'snapshot_ts_utc': ['2026-11-05T01:30:00Z']}))
    seen = []

    def slot_quotes(db, ev, gid, ts, T, **kw):
        seen.append(kw)
        cons, bets = _quotes([(7, 'assists', 4.5, 2.0, 2.0, 0.5)], slot=T)
        return cons, bets, pd.DataFrame({'player_id': [7], 'player_name_raw': ['Player Seven']})
    monkeypatch.setattr(job.quotes, 'slot_quotes', slot_quotes)
    for pmarket, kw, tag in ((None, {}, None), (prices.CURRENT_PMARKET, {}, None), (('shin', 'all_mean'), {'pmarket': ('shin', 'all_mean')}, ('shin', 'all_mean'))):
        seen.clear()
        items = job.due_items(conn, 'odds.db', sched, matched, now, '2026-27', [], [], pmarket=pmarket)
        assert seen == [kw] and [i.pmarket for i in items] == [tag]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the "odds now" view follows the rule form
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_bet_now_uses_the_rules_p_star_form():
    pmf = _poisson_pmf(7.0)
    mkt = pd.DataFrame({'book': ['fanduel', 'draftkings', 'betmgm'], 'over_dec': [1.95, 2.0, 1.9], 'under_dec': [1.9, 1.85, 1.95], 'consensus_fair_over': 0.45, 'n_at_line': 5, 'line': 6.5})
    bet = {'side': 'over', 'line': 6.5, 'pmf': pmf}
    p_mod = float(pmf[7:].sum())
    old = current.bet_now(bet, mkt, {'primary': {'e': 0.03, 'min_books': 3}}, 0.3)
    assert np.isclose(old['ev_best_now'], (0.45 + 0.3 * (p_mod - 0.45)) * 1.0 - (1 - (0.45 + 0.3 * (p_mod - 0.45))))
    new = current.bet_now(bet, mkt, {**CAND, 'primary': {'e': 0.03, 'min_books': 3}}, 0.2434, stat='assists')
    p = float(np.clip(0.45 + CAND['intercept_by_stat']['assists']['a_pos'] + 0.2434 * (p_mod - 0.45), 0.01, 0.99))
    assert np.isclose(new['ev_best_now'], p * 1.0 - (1 - p)) and not np.isclose(new['ev_best_now'], old['ev_best_now'])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# research scripts: --causal-cap of a11_rule_eval.py, --cap-per-player-game of a11_k_estimation.py --mode final
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp('a2world')
    return (root,) + kr._write_world(root, n_dates=12)


def _common(world):
    _, reg, prices_dir, db = world
    return ['--p5-version', 'a11_test_p5', '--registry-root', str(reg), '--prices-dir', str(prices_dir), '--features-db', str(db), '--n-boot', '40']


def test_rule_eval_causal_cap_option_adds_the_causal_variants_and_leaves_the_default_unchanged(world):
    root = world[0]
    rule = {**CAND, 'pmarket': {'devig_method': 'multiplicative', 'consensus_set': 'all_mean'}}
    rp = root / 'cand_for_eval.json'
    rp.write_text(json.dumps(rule), encoding='utf-8')
    base = _common(world) + ['--rule-file', str(rp), '--experiments', '7c']
    default = re_.main(base + ['--out', str(root / 'eval_default')])
    assert set(default['7c']['metrics']['variant']) == {'flat', 'cap1', 'cap2', 'kelly'} and set(default['7c']['decisions']) == {'cap1', 'cap2', 'kelly'}
    assert not any('causal' in c for c in default['7c']['paired']['comparison'])
    both = re_.main(base + ['--causal-cap', '--out', str(root / 'eval_causal')])
    met = both['7c']['metrics']
    assert set(met['variant']) == {'flat', 'cap1', 'cap2', 'kelly', 'cap1_causal', 'cap2_causal'}
    assert set(both['7c']['decisions']) == {'cap1', 'cap2', 'kelly', 'cap1_causal', 'cap2_causal'}
    m = met.set_index(['variant', 'season'])
    for season in ('2023-24', '2024-25'):
        assert 0 < m.loc[('cap1_causal', season), 'bets'] <= m.loc[('flat', season), 'bets']
        assert m.loc[('cap1_causal', season), 'bets'] == m.loc[('cap1', season), 'bets']                    # one bet per player-game with a bet at all, under either cap (the bet chosen can differ)
    comps = set(both['7c']['paired']['comparison'])
    assert {'cap1_causal - flat (median ROI)', 'cap1_causal - cap1 (median ROI)', 'cap2_causal - cap2 (median ROI)'} <= comps
    for name in ('flat', 'cap1', 'cap2', 'kelly'):                                                   # the shared variants are the same numbers with and without the option
        a = default['7c']['metrics'].set_index(['variant', 'season']).loc[name]
        b = m.loc[name]
        assert_frame_equal(a, b)


def test_k_estimation_final_writes_the_cap_key_only_when_asked_and_the_live_loader_accepts_the_file(world):
    root = world[0]
    args = _common(world) + ['--mode', 'final', '--train', '2023-24', '2024-25', '--candidate', 'C3']
    plain = kr.ke.main(args + ['--write-rule', str(root / 'rule_plain.json')])
    capped = kr.ke.main(args + ['--cap-per-player-game', '1', '--write-rule', str(root / 'rule_cap1.json')])
    assert 'cap_per_player_game' not in plain and capped['cap_per_player_game'] == 1
    on_disk = json.loads((root / 'rule_cap1.json').read_text(encoding='utf-8'))
    assert on_disk['cap_per_player_game'] == 1 and (root / 'rule_cap1.json.sha256').exists()
    assert {k: v for k, v in on_disk.items() if k != 'cap_per_player_game'}.keys() == json.loads((root / 'rule_plain.json').read_text(encoding='utf-8')).keys()
    sp = rules.rule_spec(on_disk)
    assert sp['candidate'] == 'C3' and sp['cap'] == 1 and sp['pmarket'] == prices.CURRENT_PMARKET and set(sp['params']['a_pos']) == set(ar.BET_STATS)
    assert rules.rule_spec(json.loads((root / 'rule_plain.json').read_text(encoding='utf-8')))['cap'] is None
    with pytest.raises(SystemExit):
        kr.ke.main(args + ['--cap-per-player-game', '0', '--write-rule', str(root / 'rule_bad.json')])
    with pytest.raises(SystemExit):
        kr.ke.main(_common(world) + ['--mode', 'reproduce', '--out', str(root / 'x.json'), '--cap-per-player-game', '1'])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. the research guards are intact
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_research_guards_still_refuse_2025_26():
    for s in ('2025-26', '2026-27'):
        with pytest.raises(SystemExit):
            ar.check_season(s)
    assert ar.check_season('2024-25') == '2024-25'
    with pytest.raises(SystemExit):
        ar.check_dates(['2025-06-30', '2025-07-01'])
    with pytest.raises(cq.SeasonRefused):
        cq.check_season('2025-26')
    with pytest.raises(SystemExit, match='2025-26'):
        re_.main(['--p5-version', 'x', '--registry-root', 'nowhere', '--prices-dir', 'nowhere', '--k-file', 'k.json', '--seasons', '2024-25', '2025-26', '--out', 'o'])
    # the pure math the live path calls is shared, not copied
    assert ar.p_star is rules.p_star and ar.add_ev_pstar is rules.add_ev_pstar and ar.apply_pmarket is prices.apply_pmarket and cq.finish_consensus is prices.finish_consensus
