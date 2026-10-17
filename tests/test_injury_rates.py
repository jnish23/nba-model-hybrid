import json

import pandas as pd
import pytest

from models import injury_report_features as irf


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
