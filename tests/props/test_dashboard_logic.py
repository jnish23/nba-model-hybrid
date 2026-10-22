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
    from pathlib import Path
    app = at_mod.AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'dashboard_props' / 'app.py'), default_timeout=60)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any('BACKTEST' in w.value for w in app.warning)
    app.sidebar.radio[0].set_value('best').run()                                   # switching the pricing mode must not break any page
    assert not app.exception, [e.value for e in app.exception]
