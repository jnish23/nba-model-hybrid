"""The dashboard's betting record (dashboard/bet_record.py, dashboard/app.py, scripts/export_dashboard_data.py, scripts/maintenance/backfill_bet_recommendations.py):
the record shown is the record of the bets the live run recommended, graded at the logged line and the side's own price, a push is a push, and anything
that is not a recommended bet is labelled as such. No network, no production data: every database here is a temporary one."""
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import prediction_logger as pl
from dashboard import bet_record as br
from team.betting import DEFAULT_PROB_EDGE, DEFAULT_SPREAD_STD, implied_prob_edge, points_threshold

ROOT = Path(__file__).resolve().parent.parent.parent
SIGMAS = {'spread_std': 14.287, 'total_std': 18.59, 'approximate': []}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log_row(game_id, run_ts, **kw):
    row = {c: None for c in pl.BET_LOG_NAMES}
    row.update(game_id=game_id, game_date='2026-11-02', run_ts_utc=run_ts, **kw)
    return row


def _games(**over):
    """predictions_log_v2-like frame with the three markets priced; one completed game."""
    base = dict(game_id='0022600001', game_date=pd.Timestamp('2026-11-02').date(), home_team='A', away_team='B', matchup='B @ A', ml_prob=0.55,
                spread_pred=0.0, total_pred=220.0, odds_home_ml=-110.0, odds_away_ml=-110.0, odds_home_spread=-3.5, odds_total=221.5,
                home_score=100.0, away_score=96.0, status='COMPLETED')
    base.update(over)
    return pd.DataFrame([base])


# --------------------------------------------------------------------------------------------------------------- grading

@pytest.mark.parametrize('market, side, line, hs, as_, want', [
    ('spread', 'home', -3.0, 103, 100, 'P'), ('spread', 'away', -3.0, 103, 100, 'P'), ('spread', 'home', -3.5, 103, 100, 'L'), ('spread', 'away', -3.5, 103, 100, 'W'),
    ('total', 'over', 200.0, 100, 100, 'P'), ('total', 'under', 200.0, 100, 100, 'P'), ('total', 'over', 200.5, 100, 101, 'W'), ('total', 'under', 200.5, 100, 101, 'L'),
    ('ml', 'home', None, 101, 100, 'W'), ('ml', 'away', None, 101, 100, 'L')])
def test_grade_is_push_aware(market, side, line, hs, as_, want):
    assert br.grade(market, side, line, hs, as_) == want


def test_grade_without_a_bet_or_a_score_is_none():
    assert br.grade('spread', None, -3.5, 100, 90) is None and br.grade('spread', 'home', -3.5, np.nan, 90) is None and br.grade('total', 'over', np.nan, 100, 90) is None


def test_a_push_returns_the_stake_and_is_neither_win_nor_loss():
    assert br.bet_profit('P', -110) == 0.0
    # a logged whole-number spread that lands on the line: shown as W-L-P, not as a loss
    log = pd.DataFrame([_log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.0, spread_pred=5.0, spread_std=14.287, spread_bet='home', spread_edge=0.07)])
    df = br.attach_recommendations(_games(odds_home_spread=-3.0, spread_pred=5.0, home_score=103.0, away_score=100.0), log, SIGMAS)
    bets = br.settled_bets(df, 'spread')
    rec = br.summarize(bets)
    assert list(bets['result']) == ['P'] and (rec['wins'], rec['losses'], rec['pushes'], rec['units'], rec['roi']) == (0, 0, 1, 0.0, 0.0)
    assert br.record_text(rec) == '0-0-1'
    mixed = br.summarize(pd.DataFrame({'result': ['W', 'L', 'P'], 'profit': [90.91, -100.0, 0.0], 'clv': [np.nan] * 3, 'src': ['run'] * 3}))
    assert br.record_text(mixed) == '1-1-1' and mixed['roi'] == pytest.approx(-9.09 / 200 * 100, abs=0.01)     # the push is not in the ROI base


# ------------------------------------------------------------------------------------------------- the side's own price

def test_a_logged_bet_is_graded_at_its_own_price_and_logged_line():
    log = pd.DataFrame([_log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.5, curr_price_home_spread_odds=-125.0, curr_price_away_spread_odds=105.0,
                                 spread_pred=5.0, spread_std=14.287, spread_bet='home', spread_edge=0.06)])
    # the page's stored line is -2.5 (a later move); the bet was judged at -3.5 and -125
    df = br.attach_recommendations(_games(odds_home_spread=-2.5, spread_pred=5.0, home_score=104.0, away_score=100.0), log, SIGMAS)
    dec = br.decide(df, 'spread')
    assert (dec['side'].iloc[0], dec['line'].iloc[0], dec['price'].iloc[0], bool(dec['price_assumed'].iloc[0]), dec['src'].iloc[0]) == ('home', -3.5, -125.0, False, 'run')
    b = br.settled_bets(df, 'spread')
    assert list(b['result']) == ['W'] and b['profit'].iloc[0] == pytest.approx(80.0)          # 100 * 100/125, not the -110 payout 90.91
    lost = br.attach_recommendations(_games(home_score=100.0, away_score=104.0), log, SIGMAS)
    assert br.settled_bets(lost, 'spread')['profit'].iloc[0] == -100.0


def test_a_missing_spread_price_is_settled_at_minus_110_and_marked():
    log = pd.DataFrame([_log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.5, spread_pred=5.0, spread_std=14.287, spread_bet='home', spread_edge=0.06)])
    df = br.attach_recommendations(_games(spread_pred=5.0), log, SIGMAS)
    dec = br.decide(df, 'spread')
    assert dec['price'].iloc[0] == -110 and bool(dec['price_assumed'].iloc[0])
    assert br.bet_label('spread', 'home', -3.5, -110, 0.06, 'A', 'B', True) == 'BET A -3.5 @ -110* (edge 6.0%)'


def test_moneyline_uses_each_sides_own_logged_price():
    log = pd.DataFrame([_log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_ml=-150.0, curr_odds_away_ml=130.0, ml_prob=0.45, ml_bet='away', ml_edge=0.06)])
    df = br.attach_recommendations(_games(home_score=95.0, away_score=100.0), log, SIGMAS)
    b = br.settled_bets(df, 'ml')
    assert list(b['result']) == ['W'] and b['price'].iloc[0] == 130.0 and b['profit'].iloc[0] == pytest.approx(130.0)


# ------------------------------------------------------------------------------------------------------ first-run rule

def test_the_bet_of_a_game_is_the_first_run_that_had_a_line_and_a_later_run_never_replaces_it():
    runs = [
        # run 3 (latest): says bet AWAY at another line; run 1: no spread line yet, the moneyline had one; run 2: first spread line, bet HOME
        _log_row('0022600001', '2026-11-02T20:00:00.000000Z', curr_odds_home_ml=-120.0, curr_odds_away_ml=100.0, ml_prob=0.7, ml_bet='home', ml_edge=0.1,
                 curr_odds_home_spread=-1.5, spread_pred=-9.0, spread_std=14.287, spread_bet='away', spread_edge=0.1, curr_odds_total=230.0, total_pred=240.0,
                 total_std=18.5, total_bet='over', total_edge=0.1),
        _log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_ml=-110.0, curr_odds_away_ml=-110.0, ml_prob=0.55, ml_bet=None, ml_edge=0.02),
        _log_row('0022600001', '2026-11-02T18:00:00.000000Z', curr_odds_home_ml=-115.0, curr_odds_away_ml=-105.0, ml_prob=0.6, ml_bet='home', ml_edge=0.07,
                 curr_odds_home_spread=-3.5, spread_pred=5.0, spread_std=14.287, spread_bet='home', spread_edge=0.06)]
    first = br.first_line_runs(pd.DataFrame(runs))
    assert first['spread'].loc['0022600001', 'run_ts_utc'] == '2026-11-02T18:00:00.000000Z' and first['spread'].loc['0022600001', 'spread_bet'] == 'home'
    assert first['ml'].loc['0022600001', 'run_ts_utc'] == '2026-11-02T16:00:00.000000Z'          # the earliest run with both moneyline prices
    assert first['total'].loc['0022600001', 'run_ts_utc'] == '2026-11-02T20:00:00.000000Z'       # the only run with a total line
    df = br.attach_recommendations(_games(), pd.DataFrame(runs), SIGMAS)
    assert br.decide(df, 'spread')['side'].iloc[0] == 'home' and br.decide(df, 'spread')['line'].iloc[0] == -3.5
    assert br.decide(df, 'ml')['side'].iloc[0] is None           # the first moneyline run said no bet: a later run that bets does not change the recommendation
    assert br.decide(df, 'total')['side'].iloc[0] == 'over'


def test_a_game_without_log_rows_is_recomputed_by_the_page_and_marked():
    other = pd.DataFrame([_log_row('0022600999', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.5, spread_pred=5.0, spread_std=14.287, spread_bet='home')])
    df = br.attach_recommendations(_games(spread_pred=7.0), other, SIGMAS)
    dec = br.decide(df, 'spread')
    assert dec['src'].iloc[0] == 'page' and dec['side'].iloc[0] == 'home'


def test_provenance_of_backfilled_rows_reaches_the_source_labels():
    log = pd.DataFrame([_log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.5, spread_pred=5.0, spread_std=14.287, spread_bet='home', spread_edge=0.06,
                                 prov_ml='recomputed', prov_spread='printed', prov_total='recomputed')])
    df = br.attach_recommendations(_games(), log, SIGMAS)
    assert br.decide(df, 'spread')['src'].iloc[0] == 'printed'
    by_src = br.summarize(br.settled_bets(df, 'spread'))['by_src']
    assert br.logged_share(by_src) == (1, 0) and br.source_sentence(by_src) == '1 printed in the daily log'


# ----------------------------------------------------------------------------------------- sigma source and the fallback

def test_the_fallback_uses_the_model_sets_sigma_not_the_betting_py_default():
    # a 2.55-point spread edge: a bet at the default std 13.4 (5% rule = 2.49 pts), not at the model set's 14.287 (2.66 pts)
    df_model = br.attach_recommendations(_games(spread_pred=2.55, odds_home_spread=0.0), None, br.sigmas_from_meta({'spread_std': 14.287, 'total_std': 18.59}))
    df_default = br.attach_recommendations(_games(spread_pred=2.55, odds_home_spread=0.0), None, br.sigmas_from_meta(None))
    assert br.sigmas_from_meta(None)['approximate'] == ['spread_std', 'total_std'] and br.sigmas_from_meta(None)['spread_std'] == DEFAULT_SPREAD_STD
    assert br.decide(df_default, 'spread')['side'].iloc[0] == 'home' and br.decide(df_model, 'spread')['side'].iloc[0] is None


def test_deployed_sigmas_are_read_from_the_checked_in_meta_file(tmp_path):
    (tmp_path / 'team').mkdir()
    (tmp_path / 'models' / 'saved').mkdir(parents=True)
    (tmp_path / 'team' / 'model_registry.py').write_text("DEPLOYED_SUFFIX = '_x_20260101'\n")
    assert br.deployed_meta_path(tmp_path) is None                                   # no meta file yet
    (tmp_path / 'models' / 'saved' / 'calibration_meta_x_20260101.json').write_text(json.dumps({'spread_std': 14.0, 'total_std': 18.0}))
    found = br.deployed_meta_path(tmp_path)
    assert found.name == 'calibration_meta_x_20260101.json'
    assert br.sigmas_from_meta(json.loads(found.read_text())) == {'spread_std': 14.0, 'total_std': 18.0, 'approximate': []}
    real = br.deployed_meta_path(ROOT)                                               # and the real repo's deployed set resolves
    assert real is not None and real.name.startswith('calibration_meta')


# -------------------------------------------------------------------------------------------------------- slider points

def test_slider_conversion_knows_a_whole_number_line_can_push():
    sigma, pts = 14.287, 2.6
    half, whole = br.points_to_prob_threshold(pts, sigma), br.points_to_prob_threshold(pts, sigma, line=-3.0)
    assert half == pytest.approx(implied_prob_edge(pts, sigma)) and whole != pytest.approx(half, abs=1e-4)
    # model number 2.55 points past a WHOLE line: the half-point conversion (the old behaviour) lets it through, the per-game conversion does not
    old_edge_whole = implied_prob_edge(2.55, sigma, line=-3.0)
    assert old_edge_whole > half
    whole_game = br.attach_recommendations(_games(spread_pred=2.55 + 3.0, odds_home_spread=-3.0), None, SIGMAS)
    half_game = br.attach_recommendations(_games(spread_pred=2.65 + 3.5, odds_home_spread=-3.5), None, SIGMAS)
    assert br.decide(whole_game, 'spread', points_edge=pts, recorded=False)['side'].iloc[0] is None
    assert br.decide(half_game, 'spread', points_edge=pts, recorded=False)['side'].iloc[0] == 'home'
    assert points_threshold(sigma, DEFAULT_PROB_EDGE) == pytest.approx(2.66, abs=0.01)


def test_the_what_if_rule_is_the_recorded_rule_at_the_default_edge():
    log = pd.DataFrame([_log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.5, spread_pred=8.0, spread_std=14.287, spread_bet='home', spread_edge=0.06)])
    df = br.attach_recommendations(_games(spread_pred=8.0), log, SIGMAS)
    rec, whatif = br.decide(df, 'spread'), br.decide(df, 'spread', DEFAULT_PROB_EDGE, recorded=False)
    assert rec['side'].iloc[0] == whatif['side'].iloc[0] == 'home'
    assert br.decide(df, 'spread', 0.2, recorded=False)['side'].iloc[0] is None and br.decide(df, 'spread')['side'].iloc[0] == 'home'     # the sliders never touch the record


# ------------------------------------------------------------------------------------------------------------------- CLV

def test_clv_is_positive_when_the_line_moved_toward_the_bet():
    assert br.clv('spread', 'home', -3.5, -110, -110, -4.5) == 1.0
    assert br.clv('spread', 'away', -3.5, -110, -110, -4.5) == -1.0               # bet the away side at +3.5, line moved to +4.5: the home side got the better number, not this bet
    assert br.clv('total', 'under', 221.5, -110, -110, 220.5) == 1.0 and br.clv('total', 'over', 221.5, -110, -110, 220.5) == -1.0
    pp = br.clv('ml', 'home', None, -120.0, 100.0, 0.55)
    fair = (120 / 220) / ((120 / 220) + 0.5)
    assert pp == pytest.approx(100 * (0.55 - fair)) and np.isnan(br.clv('ml', 'home', None, -120.0, 100.0, np.nan))


# --------------------------------------------------------------------------------------------------- export and backfill

def _db(tmp_path, with_log=True):
    path = tmp_path / 'nba.db'
    games = pd.DataFrame({
        'game_id': ['0022600001', '0022600002', '0022600003'], 'game_date': ['2026-10-20', '2026-11-02', '2026-11-02'],
        'prediction_date': ['2026-10-20 09:17:59', '2026-11-02 08:18:00', '2026-11-02 08:18:00'], 'home_team': ['A', 'C', 'E'], 'away_team': ['B', 'D', 'F'],
        'matchup': ['B @ A', 'D @ C', 'F @ E'], 'ml_prob': [0.7, 0.4, 0.5], 'spread_pred': [6.0, -2.0, 0.0], 'total_pred': [225.0, 215.0, 220.0],
        'odds_home_ml': [-150.0, 120.0, -110.0], 'odds_away_ml': [130.0, -140.0, -110.0], 'odds_home_spread': [-3.5, 2.5, -1.5], 'odds_total': [221.5, 219.5, 220.5],
        'home_score': [110, 100, 99], 'away_score': [100, 104, 101], 'status': ['COMPLETED'] * 3, 'model_set_hash': ['h'] * 3})
    expl = pd.DataFrame([dict(game_id='0022600001', prediction_date='2026-10-20 09:17:59', market=m, kind='piece', feature='movement', label='x', group='g', value=v,
                              contribution=0.0, pctile=None, severity=None) for m, v in (('spread', 4.5), ('total', 223.5), ('moneyline', 0.55))])
    with sqlite3.connect(path) as conn:
        conn.execute(pl.CREATE_TABLE_SQL.format(table=pl.TABLE))
        games.to_sql(pl.TABLE, conn, if_exists='append', index=False)
        expl.to_sql('prediction_explanations', conn, index=False)
        conn.execute("CREATE TABLE odds_snapshots (fetched_at_et TEXT, game_date TEXT, game_start TEXT, game_status TEXT, in_game INTEGER, home_team TEXT, away_team TEXT, "
                     "home_team_abbr TEXT, away_team_abbr TEXT, line_type TEXT, book TEXT, home_spread REAL, home_spread_odds REAL, away_spread REAL, away_spread_odds REAL, "
                     "home_ml REAL, away_ml REAL, total REAL, over_odds REAL, under_odds REAL)")
        snaps = []
        for fetched, sp, tt in (('2026-11-02 17:00:00', -1.0, 220.0), ('2026-11-02 18:30:00', -2.0, 219.0)):      # the later fetch is the close
            for book in ('b1', 'b2', 'b3'):
                snaps.append((fetched, '2026-11-02', '', '', 0, 'E', 'F', '', '', 'current', book, sp, -110, -sp, -110, -110, -110, tt, -110, -110))
        snaps.append(('2026-11-02 19:30:00', '2026-11-02', '', '', 1, 'E', 'F', '', '', 'current', 'b1', -9.0, -110, 9.0, -110, -110, -110, 250.0, -110, -110))    # in-game: ignored
        conn.executemany("INSERT INTO odds_snapshots VALUES (" + ", ".join("?" * 20) + ")", snaps)
        if with_log:
            conn.execute(pl.BET_LOG_CREATE_SQL)
    return path


def _bets_csv(tmp_path):
    p = tmp_path / 'bets.csv'
    pd.DataFrame([dict(game_id='0022600001', market='Spread', side='home', edge=0.077, price=-115.0, price_source='snapshot', source='recomputed'),
                  dict(game_id='0022600002', market='Total', side='under', edge=0.061, price=np.nan, price_source='-110 assumed', source='log'),
                  dict(game_id='0022600001', market='ML', side='home', edge=0.09, price=-150.0, price_source='logged moneyline', source='recomputed')]).to_csv(p, index=False)
    return p


def test_export_writes_log_with_closing_lines_and_tolerates_a_missing_table(tmp_path):
    exp = _load('export_dashboard_data_b1', 'scripts/export_dashboard_data.py')
    db = _db(tmp_path, with_log=False)
    assert exp.load_bet_recommendations(db).empty                                   # no table yet: nothing to export
    out = tmp_path / 'out'
    exp.write_files(out, exp.load_predictions(db), None, recommendations=exp.load_bet_recommendations(db))
    assert (out / 'predictions_log.csv').exists() and not (out / exp.RECOMMENDATIONS_FILE_NAME).exists() and not (out / exp.RECORD_META_FILE_NAME).exists()

    (tmp_path / 'with').mkdir()
    db = _db(tmp_path / 'with', with_log=True)
    with sqlite3.connect(db) as conn:
        pl_rows = [_log_row('0022600003', '2026-11-02T16:19:00.000000Z', curr_odds_home_spread=-1.5, spread_bet='home'),
                   _log_row('0022600003', '2026-11-02T17:30:00.000000Z', curr_odds_home_spread=-2.0, spread_bet=None)]
        conn.executemany(f"INSERT INTO {pl.BET_LOG_TABLE} ({', '.join(pl.BET_LOG_NAMES)}) VALUES ({', '.join('?' * len(pl.BET_LOG_NAMES))})",
                         [[r[c] for c in pl.BET_LOG_NAMES] for r in pl_rows])
    rec = exp.load_bet_recommendations(db)
    assert len(rec) == 2                                                             # every run is exported: the dashboard picks the first one
    assert rec['close_home_spread'].eq(-2.0).all() and rec['close_total'].eq(219.0).all()     # last pre-game fetch, in-game rows ignored, median of the books
    assert rec['close_p_home'].round(3).eq(0.5).all()
    exp.write_files(out, exp.load_predictions(db), None, recommendations=rec, record_meta={'spread_std': 14.0, 'total_std': 18.0})
    assert (out / exp.RECOMMENDATIONS_FILE_NAME).exists() and json.loads((out / exp.RECORD_META_FILE_NAME).read_text())['spread_std'] == 14.0
    exp.write_files(out, exp.load_predictions(db), None)                              # the source went away: the stale copies go too
    assert not (out / exp.RECOMMENDATIONS_FILE_NAME).exists() and not (out / exp.RECORD_META_FILE_NAME).exists()


def test_backfill_dry_run_writes_nothing_applies_once_and_never_touches_a_real_run(tmp_path, capsys):
    bf = _load('backfill_bet_recommendations_b1', 'scripts/maintenance/backfill_bet_recommendations.py')
    db, bets = _db(tmp_path, with_log=True), _bets_csv(tmp_path)
    args = ['--db', str(db), '--bets', str(bets), '--recon', str(tmp_path / 'none.json'), '--date-from', '2026-10-20', '--date-to', '2026-11-11']

    def count(table):
        with sqlite3.connect(db) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # a game that a REAL run already logged (a log row with no provenance row) is skipped, never overwritten
    with sqlite3.connect(db) as conn:
        real = _log_row('0022600003', '2026-11-02T16:19:00.000000Z', curr_odds_home_spread=-1.5, spread_bet='away')
        conn.execute(f"INSERT INTO {pl.BET_LOG_TABLE} ({', '.join(pl.BET_LOG_NAMES)}) VALUES ({', '.join('?' * len(pl.BET_LOG_NAMES))})", [real[c] for c in pl.BET_LOG_NAMES])
    assert bf.main(args) == 0 and count(pl.BET_LOG_TABLE) == 1                       # default is a dry run
    assert 'DRY RUN' in capsys.readouterr().out
    assert bf.main(args + ['--apply']) == 0
    assert count(pl.BET_LOG_TABLE) == 3 and count(br.PROVENANCE_TABLE) == 2
    assert bf.main(args + ['--apply']) == 0 and count(pl.BET_LOG_TABLE) == 3          # idempotent
    with sqlite3.connect(db) as conn:
        kept = conn.execute(f"SELECT spread_bet FROM {pl.BET_LOG_TABLE} WHERE game_id='0022600003'").fetchall()
        assert kept == [('away',)]
        log = br.read_log_from_db(conn)
    first = log[log['game_id'] == '0022600001'].iloc[0]
    assert (first['spread_bet'], first['curr_price_home_spread_odds'], first['ml_bet'], first['prov_spread'], first['open_source']) == ('home', -115.0, 'home', 'recomputed', 'explanations')
    assert first['odds_home_spread'] == -4.5 and first['odds_total'] == 223.5            # the opening line the model read, from the explanation pieces
    assert first['spread_own_view'] == pytest.approx(6.0 - 4.5) and first['spread_line_move'] == pytest.approx(-4.5 - (-3.5))
    nov = log[log['game_id'] == '0022600002'].iloc[0]
    assert (nov['total_bet'], nov['prov_total'], nov['run_ts_utc']) == ('under', 'printed', '2026-11-02T16:18:00.000000Z')
    # the dashboard side: the backfilled bets are the recommendations, with their provenance
    games = pd.DataFrame({'game_id': ['0022600001', '0022600002'], 'game_date': [pd.Timestamp('2026-10-20').date(), pd.Timestamp('2026-11-02').date()], 'home_team': ['A', 'C'], 'away_team': ['B', 'D'],
                          'ml_prob': [0.7, 0.4], 'spread_pred': [6.0, -2.0], 'total_pred': [225.0, 215.0], 'odds_home_ml': [-150.0, 120.0], 'odds_away_ml': [130.0, -140.0],
                          'odds_home_spread': [-3.5, 2.5], 'odds_total': [221.5, 219.5], 'home_score': [110.0, 100.0], 'away_score': [100.0, 104.0], 'status': ['COMPLETED'] * 2})
    df = br.attach_recommendations(games, log, SIGMAS)
    tot = br.settled_bets(df, 'total')
    assert list(tot['side']) == ['under'] and list(tot['src']) == ['printed'] and list(tot['result']) == ['W']       # 204 points against an under 219.5: it won


def test_a_missing_recommendations_file_leaves_the_page_recomputing(tmp_path):
    pytest.importorskip('streamlit')
    pytest.importorskip('plotly')
    app = _load('dashboard_app_b1_missing', 'dashboard/app.py')
    app.DB_PATH = tmp_path / 'missing.db'
    app.RECS_URL = (tmp_path / 'not_published.csv').as_uri()
    app.RECS_SNAPSHOT_PATH = tmp_path / 'no_snapshot.csv'
    log, label = app._read_recommendations()
    assert log is None and label == 'none logged yet'
    app.RECORD_META_URL = (tmp_path / 'not_published.json').as_uri()
    app.RECORD_META_SNAPSHOT_PATH = tmp_path / 'no_meta_snapshot.json'
    sig, label = app._read_sigmas()                                                  # no published or bundled file: the checked-in model-set meta of this repo
    assert sig['spread_std'] == pytest.approx(14.287, abs=1e-3) and 'calibration_meta' in label and not sig['approximate']
    app.root_path = tmp_path                                                          # and with nothing at all: betting.py's defaults, flagged
    sig, label = app._read_sigmas()
    assert sig['approximate'] == ['spread_std', 'total_std'] and 'approximate' in label
    (tmp_path / 'meta.json').write_text(json.dumps({'spread_std': 14.1, 'total_std': 18.2}))
    app.RECORD_META_SNAPSHOT_PATH = tmp_path / 'meta.json'                           # the bundled snapshot wins over the repo file when the database is not local
    assert app._read_sigmas()[0]['spread_std'] == 14.1


def test_simulation_at_its_default_settings_is_the_recorded_record_and_the_what_if_is_not():
    pytest.importorskip('streamlit')
    pytest.importorskip('plotly')
    app = _load('dashboard_app_b1_sim', 'dashboard/app.py')
    log = pd.DataFrame([
        _log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.0, curr_price_home_spread_odds=-120.0, spread_pred=5.0, spread_std=14.287, spread_bet='home', spread_edge=0.07),
        _log_row('0022600002', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.5, spread_pred=5.0, spread_std=14.287, spread_bet='home', spread_edge=0.07)])
    games = pd.concat([_games(game_id='0022600001', odds_home_spread=-3.0, spread_pred=5.0, home_score=103.0, away_score=100.0),                    # a push at -3
                       _games(game_id='0022600002', odds_home_spread=-3.5, spread_pred=5.0, home_score=110.0, away_score=100.0)], ignore_index=True)     # a win at -110
    df = br.attach_recommendations(games, log, SIGMAS)
    sim = app.simulate_strategies(df, 0.05, 'Spread')
    assert sorted(sim['bet_outcome']) == ['Push', 'Win'] and sim['bet_result'].sum() == pytest.approx(100 * 100 / 110)
    rec = app.record_of(sim)
    assert (rec['wins'], rec['losses'], rec['pushes'], app.bet_record.record_text(rec)) == (1, 0, 1, '1-0-1')
    wide = app.simulate_strategies(df, 0.05, 'Spread', recorded=False, points_edge=9.0)          # a what-if with a huge minimum edge: no bets, the record is untouched
    assert (wide['bet_outcome'] == 'No Bet').all() and len(app.simulate_strategies(df, 0.05, 'Spread')[lambda x: x['bet_outcome'] != 'No Bet']) == 2
    assert set(app.recommended_bets_table({'Spread': sim})['Result']) == {'Push', 'Win'}


def test_the_page_runs_with_and_without_the_log_shows_only_recommended_bets_and_labels_what_ifs(tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    pytest.importorskip('plotly')
    from streamlit.testing.v1 import AppTest
    script = tmp_path / 'run_app.py'
    script.write_text("import sys, runpy\nsys.path.insert(0, r'%s')\nrunpy.run_path(r'%s', run_name='__main__')\n" % (ROOT, ROOT / 'dashboard' / 'app.py'))
    from core import live_feed
    monkeypatch.setattr(live_feed, 'scoreboard', lambda *a, **k: pd.DataFrame())          # no network; undone when the test ends
    import streamlit as st
    try:
        _run_page_with_and_without_the_log(tmp_path, monkeypatch, script, AppTest)
    finally:
        st.cache_data.clear()                  # the loaders are cached per process: leave nothing of this test's databases for the next test's AppTest


def _elements_mentioning_lean(at):
    """Every element of the rendered page (main area and sidebar, every tab and expander) whose text, label, help, options or table cells contain 'lean' (any case)."""
    found = []

    def walk(block):
        for el in block:
            if hasattr(el, 'children'):
                walk(el.children.values())
            pieces = [str(el.proto)]                              # the element's own text, label, help and options as streamlit sent them
            value = getattr(el, 'value', None)
            if isinstance(value, pd.DataFrame):                    # tables: column names and cells
                pieces += [str(c) for c in value.columns] + [str(v) for v in value.to_numpy().ravel()]
            if any('lean' in p.lower() for p in pieces):
                found.append((el.type, getattr(el, 'label', None)))

    walk([at.main, at.sidebar])
    return found


def _run_page_with_and_without_the_log(tmp_path, monkeypatch, script, AppTest):
    for with_log in (False, True):
        sub = tmp_path / f'db{with_log}'
        sub.mkdir()
        db = _db(sub, with_log=True)
        if not with_log:
            with sqlite3.connect(db) as conn:
                conn.execute(f"DROP TABLE {pl.BET_LOG_TABLE}")
        else:
            bf = _load('backfill_for_page', 'scripts/maintenance/backfill_bet_recommendations.py')
            bf.main(['--db', str(db), '--bets', str(_bets_csv(sub)), '--recon', str(sub / 'none.json'), '--apply'])
        monkeypatch.setenv('NBA_DASHBOARD_DB', str(db))
        import streamlit as st
        st.cache_data.clear()                                                         # the loaders are cached per process, not per database path
        at = AppTest.from_file(str(script), default_timeout=120).run()
        assert not at.exception, [e.value for e in at.exception]
        text = ' '.join(c.value for c in at.caption) + ' ' + ' '.join(m.value for m in at.markdown)
        assert 'Record of the recommended bets' in text
        assert not _elements_mentioning_lean(at), _elements_mentioning_lean(at)          # recommended bets only: no model-lean display anywhere on the page
        assert ('printed in the daily log' in text) if with_log else ('recomputed by this page' in text)
        # moving the Spread what-if slider switches to the what-if wording; the default says it is the recorded bets
        [b for b in at.sidebar.selectbox if b.label == 'Bet Type'][0].select('Spread').run()
        assert any('RECOMMENDED bets' in c.value for c in at.caption)
        at.sidebar.slider[0].set_value(4.0).run()
        assert not at.exception and any('WHAT-IF' in c.value for c in at.caption)
        assert not _elements_mentioning_lean(at), _elements_mentioning_lean(at)
        # the same page on a day that HAS games (the test database holds none for the real today): Today's Games shows the recommended bets only, no lean column
        from core import season_utils
        real_today = season_utils.nba_today
        monkeypatch.setattr(season_utils, 'nba_today', lambda: pd.Timestamp('2026-11-02').date())
        at = AppTest.from_file(str(script), default_timeout=120).run()
        assert not at.exception, [e.value for e in at.exception]
        tables = [list(el.value.columns) for el in at.get('dataframe') if isinstance(el.value, pd.DataFrame)]
        assert ['Why?', 'Away Team', 'Home Team', 'Home win % (model / market)', 'Moneyline (away / home)', 'ML Bet', 'Spread Bet', 'Total Bet'] in tables
        assert not _elements_mentioning_lean(at), _elements_mentioning_lean(at)
        monkeypatch.setattr(season_utils, 'nba_today', real_today)

def test_live_score_picks_are_the_recommended_bets_not_a_lean_on_every_game():
    from dashboard import live_scores
    log = pd.DataFrame([_log_row('0022600001', '2026-11-02T16:00:00.000000Z', curr_odds_home_spread=-3.0, spread_pred=8.0, spread_std=14.287, spread_bet='home', spread_edge=0.07)])
    games = br.attach_recommendations(_games(spread_pred=8.0, odds_home_spread=-2.5, odds_total=221.5), log, SIGMAS)       # the page's current line moved to -2.5 since the bet
    sides = br.override_pick_sides(live_scores.pick_sides(games))
    assert live_scores.pick_sides(games)['total_side'].iloc[0] == 'under'                          # the lean exists on every game ...
    assert sides['total_side'].iloc[0] is None and sides['ml_side'].iloc[0] is None and sides['spread_side'].iloc[0] == 'home'      # ... but only the bet is a pick
    assert sides['odds_home_spread'].iloc[0] == -3.0                                              # shown against the line the bet was judged at
