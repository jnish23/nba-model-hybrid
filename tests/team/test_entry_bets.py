"""Bets priced at the entry line (first line at/after the feature cutoff) rather than the opener."""
import numpy as np
import pandas as pd

from experiments import metrics as M


def frame():
    return pd.DataFrame({
        'gameId': ['0022600001', '0022600002'], 'gameDate': ['2026-10-20', '2026-10-21'], 'seasonYear': ['2026-27'] * 2,
        'ml_prob': [0.5, 0.5], 'spread_pred': [-8.0, -8.0], 'total_pred': [235.0, 235.0], 'target_home_win': [1, 1],
        'target_point_diff': [10.0, 10.0], 'target_total_pts': [225.0, 225.0],
        'odds_home_ml': [-110.0, -110.0], 'odds_away_ml': [-110.0, -110.0], 'odds_home_spread': [-4.0, -4.0], 'odds_total': [224.0, 224.0],
        'close_home_spread': [-6.0, -6.0], 'close_total': [226.0, 226.0],
    })


def entry():
    return pd.DataFrame({'gameId': ['0022600001'], 'entry_home_spread': [-6.5], 'entry_total': [227.0],
                         'entry_home_ml': [-110.0], 'entry_away_ml': [-110.0]})


def test_default_is_labelled_open_basis():
    b = M.make_bets(frame(), markets=('spread', 'total'))
    assert set(b['entry_basis']) == {'open'} and b.attrs['entry_basis'] == 'open'
    assert set(b['line']) == {-4.0, 224.0}


def test_entry_prices_replace_the_opener_and_drop_games_without_one():
    b = M.make_bets(frame(), markets=('spread', 'total'), entry=entry())
    assert set(b['entry_basis']) == {'entry'}
    assert set(b['gameId']) == {'0022600001'}                  # game 2 has no entry price: skipped, not priced at the open
    sp = b[b['market'] == 'spread'].iloc[0]
    assert sp['line'] == -6.5 and sp['side'] == 'away'         # model -8 vs the -6.5 entry line: edge is measured against the entry line
    tt = b[b['market'] == 'total'].iloc[0]
    assert tt['line'] == 227.0 and tt['side'] == 'over'


def test_edge_uses_the_entry_line_not_the_open():
    f = frame()
    f['spread_pred'] = [-10.0, -10.0]                          # point edge is pred + line: 14 vs the -4 open, 16.5 vs the -6.5 entry line
    open_edge = M.make_bets(f, markets=('spread',)).iloc[0]['edge']
    entry_edge = M.make_bets(f, markets=('spread',), entry=entry()).iloc[0]['edge']
    assert entry_edge > open_edge


def test_clv_is_measured_from_the_entry_line():
    b = M.make_bets(frame(), markets=('spread',), entry=entry())
    # away +6.5 at the entry, the close is home -6.0 (away +6.0): the line moved against an away bettor by 0.5
    assert np.isclose(M.clv_points(b).iloc[0], -((-6.5) - (-6.0)))


def test_report_marks_open_basis_results():
    from experiments import report
    t = frame()
    t['gameDate'] = pd.to_datetime(t['gameDate'])
    res = report.evaluate('x', test=t, save=False)
    assert res['entry_basis'] == 'open' and 'OPENING' in res['entry_basis_caveat']
    res2 = report.evaluate('x', test=t, save=False, entry=entry())
    assert res2['entry_basis'] == 'entry' and 'entry_basis_caveat' not in res2
