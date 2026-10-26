import json

import numpy as np
import pandas as pd
import pytest

from dashboard_props import logic as L


def make_bets(n=60, seed=0):
    rng = np.random.default_rng(seed)
    games = [f'002250{i:04d}' for i in range(12)]
    rows = []
    for i in range(n):
        g = games[i % len(games)]
        side = 'over' if i % 3 == 0 else 'under'
        stat = ['rebounds', 'assists', 'threes', 'ra'][i % 4]
        win = rng.random() < 0.53
        st_ = 'win' if win else 'loss'
        rows.append({'bet_id': f'2025-26|{g}|{i}|{stat}', 'season': '2025-26' if i % 2 else '2024-25', 'data_kind': 'test', 'game_id': g, 'game_date': f'2026-01-{1 + (i % len(games)):02d}',
                     'tip_ref_utc': f'2026-01-{1 + (i % len(games)):02d}T00:10:00Z', 'placed_at_utc': '2026-01-01T20:10:00Z', 'minutes_before': [240, 60, 5][i % 3],
                     'matchup': 'AAA @ BBB', 'home_team': 'BBB', 'away_team': 'AAA', 'player_id': i, 'player_name': f'Player {i}', 'stat': stat, 'side': side, 'line': 5.5, 'book': 'draftkings',
                     'dec_best': 2.05, 'dec_median': 1.91, 'dec_flat': L.FLAT_110, 'ev_best': 0.03 + 0.001 * i, 'ev_median': -0.01 + 0.001 * i, 'ev_flat': -0.005 + 0.001 * i,
                     'p_over_model': 0.4, 'p_over_star': 0.45, 'p_over_market': 0.5, 'push': 0.0, 'n_at_line': 5, 'n_quotes': 6, 'expected_minutes': 30.0, 'minutes_actual': 31.0, 'y': 6.0,
                     'played': True, 'status_stands': st_, 'status_flagged': st_, 'status_all': st_, 'clv_prob': 0.004, 'line_move_favour': 0.0, 'close_line': 5.5,
                     'line_240': 5.5, 'line_60': 5.5, 'line_5': 5.5, 'mkt_240': 0.5, 'mkt_60': 0.5, 'mkt_5': 0.5})
    return pd.DataFrame(rows)


def test_american_conversion():
    assert list(L.american([L.FLAT_110, 2.5, 1.5, 2.0])) == ['-110', '+150', '-200', '+100']


def test_profit_and_treatments():
    d = make_bets(6)
    d['status_stands'] = ['win', 'loss', 'push', 'void', 'win', 'loss']
    d['status_flagged'] = ['win', 'loss', 'push', 'void', 'void', 'loss']
    w = L.with_profit(d, 'best', 'stands')
    assert np.allclose(w['profit'], [1.05, -1, 0, 0, 1.05, -1]) and list(w['stake']) == [1, 1, 0, 0, 1, 1]
    w2 = L.with_profit(d, 'flat', 'flagged')
    assert np.allclose(w2['profit'], [L.FLAT_110 - 1, -1, 0, 0, 0, -1])


def test_board_ranks_by_ev_at_the_chosen_price_and_counts_the_hidden():
    d = make_bets(30)
    shown, hidden = L.board(d, 'median', min_ev=0.0, top_n=5)
    assert len(shown) == 5 and (shown['ev_median'] >= 0).all() and shown['ev_median'].is_monotonic_decreasing
    assert hidden == int((d['ev_median'] >= 0).sum()) - 5
    best_shown, _ = L.board(d, 'best', min_ev=0.0, top_n=5)
    assert best_shown['ev_best'].is_monotonic_decreasing and len(best_shown) == 5
    only, _ = L.board(d, 'median', stats=['threes'], sides=['over'], top_n=50)
    assert set(only['stat']) == {'threes'} and set(only['side']) == {'over'}


def test_roi_summary_and_interval():
    d = make_bets(400, seed=1)
    s = L.roi_summary(d, 'best', 'stands', n_boot=300)
    assert s['bets'] == 400 and s['lo'] < s['roi'] < s['hi']
    expected = (d['status_stands'].eq('win') * (d['dec_best'] - 1) - d['status_stands'].eq('loss')).sum() / 400
    assert s['roi'] == pytest.approx(expected)
    d2 = d.assign(status_stands='void')
    assert L.roi_summary(d2, 'best', 'stands')['bets'] == 0


def test_cumulative_and_breakdown_and_noise_note():
    d = make_bets(120, seed=2)
    c = L.cumulative(d, 'median', 'stands')
    assert c['n'].iloc[-1] == 120 and (c['hi'] >= c['cum_units']).all() and (c['lo'] <= c['cum_units']).all()
    b = L.breakdown(d, 'stat', 'median', 'stands')
    assert b['bets'].sum() == 120 and set(b['stat']) == {'rebounds', 'assists', 'threes', 'ra'}
    assert '1.0 points' in L.noise_note(10000) and 'No settled bets' in L.noise_note(0)


def test_streamlit_app_boots_on_synthetic_data(tmp_path, monkeypatch):
    at_mod = pytest.importorskip('streamlit.testing.v1')
    pytest.importorskip('plotly')
    d = make_bets(60)
    d.to_csv(tmp_path / 'bets.csv', index=False)
    rows = [{'season': s, 'kind': 'test', 'bets': 30, 'player_games': 20, 'roi_best_stands': 0.04, 'roi_median_stands': 0.0, 'roi_flat_stands': -0.01, 'clv_real_pts': 0.4,
             'clv_placebo_mean_pts': 0.2, 'clv_excess_pts': 0.2, 'clv_excess_lower_pts': 0.1, 'placebo_roi': -0.03, 'slope4': 0.2, 'slope4_lo': 0.1, 'slope4_hi': 0.3} for s in ('2024-25', '2025-26')]
    pd.DataFrame(rows).to_csv(tmp_path / 'season_summary.csv', index=False)
    pd.DataFrame({'season': '2025-26', 'mean_p_model': [0.3, 0.5, 0.7], 'mean_p_market': [0.3, 0.5, 0.7], 'hit_rate': [0.32, 0.5, 0.66], 'n': 10}).to_csv(tmp_path / 'reliability.csv', index=False)
    pd.DataFrame({'season': '2025-26', 'stat': 'rebounds', 'month': ['2025-11', '2025-12'], 'n': 10, 'observed': 10, 'predicted': 10, 'ratio': [1.0, 0.98]}).to_csv(tmp_path / 'bias_monthly.csv', index=False)
    meta = {'mode': 'backtest', 'built_at_utc': 'now', 'rule_sha256': 'a' * 64, 'features_db_sha256': 'b' * 64, 'code_commit': 'c' * 40,
            'rule': {'stats': ['rebounds', 'assists', 'threes', 'ra'], 'primary': {'e': 0.03, 'min_books': 3}, 'k_by_stat': {'rebounds': 0.26, 'assists': 0.29, 'threes': 0.31, 'ra': 0.23}}}
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    (tmp_path / 'pmf.json').write_text(json.dumps({b: [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02] for b in d['bet_id']}))
    (tmp_path / 'books.json').write_text(json.dumps({b: {'draftkings': 2.05, 'fanduel': 1.91, 'betmgm': 1.87} for b in d['bet_id']}))
    monkeypatch.setenv('PROPS_DASH_DATA', str(tmp_path))
    monkeypatch.setenv('PROPS_DASH_LIVE', str(tmp_path / 'no_live_here'))
    from pathlib import Path
    app = at_mod.AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'dashboard_props' / 'app.py'), default_timeout=60)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any('BACKTEST' in w.value for w in app.warning)
    app.sidebar.radio[1].set_value('best').run()                                   # switching the pricing mode must not break any page
    assert not app.exception, [e.value for e in app.exception]
    [c for c in app.checkbox if 'Reveal' in c.label][0].set_value(True).run()      # showing results must not break the bet detail
    assert not app.exception, [e.value for e in app.exception]


def test_streamlit_app_boots_on_the_live_export(tmp_path, monkeypatch):
    at_mod = pytest.importorskip('streamlit.testing.v1')
    pytest.importorskip('pdfplumber')                      # the exporter lives in the full environment; the app itself only needs streamlit
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('live_export_helpers', Path(__file__).with_name('test_live_export.py'))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    CONFIG, _bet = helpers.CONFIG, helpers._bet
    from props.live import export, store
    conn = store.connect(tmp_path / 'live.db')
    store.write_slot(conn, 'g1', 60, 'done', '2026-10-23T00:42:00Z', 'r1', bets=pd.concat([_bet(1), _bet(2)]))
    export.export_live(tmp_path / 'live.db', tmp_path / 'live_out', CONFIG, n_boot=20)
    monkeypatch.setenv('PROPS_DASH_LIVE', str(tmp_path / 'live_out'))
    monkeypatch.setenv('PROPS_DASH_DATA', str(tmp_path / 'no_backtest_here'))
    app = at_mod.AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'dashboard_props' / 'app.py'), default_timeout=60)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any('LIVE PAPER RECORD' in w.value for w in app.warning)
    app.sidebar.radio[1].set_value('best').run()
    assert not app.exception, [e.value for e in app.exception]


def test_board_sort_options_and_upcoming_filter():
    d = make_bets(24)
    now = pd.Timestamp('2026-01-06T12:00:00Z')
    up, _ = L.board(d, 'best', min_ev=-1, top_n=100, upcoming_only=True, now=now)
    assert (pd.to_datetime(up['tip_ref_utc'], utc=True) > now).all() and 0 < len(up) < len(d)
    by_tip, _ = L.board(d, 'best', min_ev=-1, top_n=100, sort='tip')
    tips = pd.to_datetime(by_tip['tip_ref_utc'], utc=True)
    assert tips.is_monotonic_increasing and by_tip.groupby('tip_ref_utc')['ev_best'].apply(lambda s: s.is_monotonic_decreasing).all()
    cur = pd.DataFrame({'bet_id': d['bet_id'].iloc[:3], 'ev_best_now': [0.01, 0.09, 0.05]})
    by_now, _ = L.board(d, 'best', min_ev=-1, top_n=100, sort='ev_now', cur=cur)
    assert list(by_now['bet_id'].iloc[:3]) == [d['bet_id'].iloc[1], d['bet_id'].iloc[2], d['bet_id'].iloc[0]] and len(by_now) == len(d)       # no quote: last


def test_results_for_display_prefers_the_official_result_and_marks_provisional_ones():
    d = make_bets(3)
    d['status_stands'] = ['win', np.nan, np.nan]
    d['y'] = [6.0, np.nan, np.nan]
    prov = pd.DataFrame({'bet_id': d['bet_id'], 'provisional': ['loss', 'loss', 'void'], 'actual': [1.0, 2.0, np.nan]})
    r = L.results_for_display(d, prov, 'best', 'stands')
    assert list(r['Result']) == ['win', 'loss (prov.)', 'void (prov.)'] and r['Actual'].iloc[1] == 2.0 and r['Actual'].iloc[0] == 6.0         # official wins over provisional
    assert np.allclose(r['Units'].iloc[:2], [1.05, -1.0]) and r['Units'].iloc[2] == 0
    none = L.results_for_display(d, None, 'best', 'stands')
    assert list(none['Result']) == ['win', '', ''] and np.isnan(none['Units'].iloc[1])
    assert [L.result_kind(x) for x in ['win', 'loss (prov.)', 'push', 'void (prov.)', '']] == ['win', 'loss', 'other', 'other', '']


def test_ev_band_labels():
    assert list(L.ev_band([-0.02, 0.0, 0.029, 0.03, 0.08, 0.1, 0.4])) == ['below 0%', '0 to 3%', '0 to 3%', '3 to 6%', '6 to 10%', '10% and up', '10% and up']
