import re

import pytest

from conftest import ROOT
from props.features import build, dictionary

DATASET = ROOT / 'data' / 'props_features.db'


def test_describe_covers_each_naming_pattern():
    for col in ('m_min_h15', 'm_games_eff_h40', 'm_min_sd_h15', 'm_min_blowout_h40', 'm_blowout_ratio_h40', 'm_bucket', 'role_p_played_h5',
                'role_start_share_h15', 'rest_days_since_played', 'r_points_h15', 'r_reb_chances_off_h40', 'r_usage_pct_h5', 'c_fg3_pct_h15',
                'tm_off_rtg', 'opp_pts_allowed_5g', 'opp_zone_ra_share_allowed', 'tm_zone_c3_pct_allowed', 'ctx_rest_advantage', 'inj_p_play',
                'av_miss_minutes', 'opp_av_n_out', 'av_miss_starter_minutes', 'env_blowout_prob', 'pos_group', 'tm_is_b2b', 'opp_dist_miles'):
        assert dictionary.describe(col) is not None, col
    assert dictionary.describe('r_points_h15')[0] == 'rates' and dictionary.describe('opp_pts')[0] == 'opponent'
    assert dictionary.describe('points') is None and dictionary.describe('y_points') is None          # labels have no feature entry
    assert dictionary.is_key_or_label('y_points') and dictionary.is_key_or_label('lbl_minutes') and dictionary.is_key_or_label('game_id')


@pytest.mark.db
@pytest.mark.skipif(not DATASET.exists(), reason='dataset not built (scripts/build_props_features.py)')
def test_dataset_tables_are_documented_clean_and_consistent():
    t = build.read_tables(DATASET)
    base, labels, at_t = t['player_game_base'], t['player_game_labels'], t['player_game_at_T']
    assert not dictionary.missing_entries(t)
    # keys
    assert not base.duplicated(['game_id', 'player_id']).any() and not labels.duplicated(['game_id', 'player_id']).any()
    assert not at_t.duplicated(['game_id', 'player_id', 'minutes_before']).any()
    assert len(at_t) == 3 * len(base) and set(at_t['minutes_before']) == {240, 60, 5}
    # no outcome leaks into a feature table, by name
    outcome_names = set(labels.columns) - {'game_id', 'player_id'}
    for df in (base, at_t):
        assert not outcome_names & set(df.columns)
        assert not [c for c in df.columns if c.startswith(('y_', 'lbl_'))]
    # time order: no report is newer than T; probabilities are probabilities
    assert (at_t['inj_report_age_min'].dropna() >= 0).all()
    assert at_t['inj_p_play'].between(0, 1).all()
    # provenance of the market environment: snapshots only exist from 2023-24
    m = at_t.merge(base[['game_id', 'player_id', 'season']], on=['game_id', 'player_id'])
    assert not ((m['env_source'] == 1) & (m['season'] < '2023-24')).any()
    assert not ((m['env_source'].isin([2, 3])) & (m['season'] >= '2023-24')).any()
