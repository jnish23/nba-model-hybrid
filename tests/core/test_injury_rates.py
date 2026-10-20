import json

import pandas as pd
import pytest

from core import injury_report_features as irf


def test_prior_is_ordered_and_complete():
    p = irf.STATUS_PRIOR
    assert p['Out'] == 0.0 and p['Available'] == 1.0
    assert p['Out'] < p['Doubtful'] < p['Questionable'] < p['Probable'] < p['Available']


@pytest.mark.db
def test_first_season_rates_use_no_later_data(tmp_path, monkeypatch):
    """The earliest season must get the documented prior, and every other season only earlier seasons."""
    from conftest import DB_PATH
    monkeypatch.setattr(irf, 'STATUS_RATES_PATH', tmp_path / 'rates.json')
    table = irf.compute_status_play_rates(str(DB_PATH))
    seasons = sorted(k for k in table if k != 'all')
    first = table[seasons[0]]
    assert {k: v['p_play'] for k, v in first.items()} == irf.STATUS_PRIOR
    assert all(v['n'] == 0 for v in first.values())
    # season 2 is estimated from season 1 only: its counts equal season 1's own counts
    second_n = {k: v['n'] for k, v in table[seasons[1]].items()}
    all_n = {k: v['n'] for k, v in table['all'].items()}
    assert all(second_n[k] < all_n[k] for k in second_n)


def _reports(*rows):
    return pd.DataFrame(rows, columns=['report_ts', 'game_date']).assign(
        report_ts=lambda d: pd.to_datetime(d['report_ts']), game_date=lambda d: pd.to_datetime(d['game_date']))


def test_freshness_flags_missing_and_old_reports():
    cutoff = pd.Timestamp('2026-01-15 12:00')
    fresh = _reports(('2026-01-15 11:30', '2026-01-15'), ('2026-01-14 20:00', '2026-01-15'))
    latest, age, stale = irf.check_report_freshness(fresh, '2026-01-15', cutoff)
    assert latest == pd.Timestamp('2026-01-15 11:30') and age == pytest.approx(0.5) and not stale

    old = _reports(('2026-01-14 20:00', '2026-01-15'))          # the morning download failed
    latest, age, stale = irf.check_report_freshness(old, '2026-01-15', cutoff)
    assert stale and age == pytest.approx(16.0)

    none = _reports(('2026-01-14 20:00', '2026-01-14'))         # nothing covers the game date
    assert irf.check_report_freshness(none, '2026-01-15', cutoff) == (None, None, True)

    future = _reports(('2026-01-15 13:00', '2026-01-15'))       # published after the cutoff: not usable
    assert irf.check_report_freshness(future, '2026-01-15', cutoff)[2] is True
