"""Train/serve parity: live features for a historical date must equal the training-set features for that date.

Needs data/nba_stats.db (skipped in CI). The training set is rebuilt from the current code, and the
live path is run as of the morning of each date (history strictly before it, the injury report as of noon).
"""
import numpy as np
import pandas as pd
import pytest

from conftest import DB_PATH
from models.game_features_v2 import GameFeatureBuilder
from models.live_features import build_live_game_frame

pytestmark = pytest.mark.db

DATES = ['2026-01-15', '2025-03-12', '2022-11-23', '2020-01-17']   # in-season, mid-season, 2 earlier seasons
SKIP_PREFIXES = ('odds_', 'target_', 'wl_', 'points_', 'is_home_', 'matchup_home', 'matchup_away')


@pytest.fixture(scope='module')
def training_set():
    ts = GameFeatureBuilder(str(DB_PATH)).build_training_set()
    ts['gameDate'] = pd.to_datetime(ts['gameDate'])
    return ts


def _live_frame(ts, date):
    from models import injury_report_features as irf
    day = ts[ts['gameDate'] == date]
    assert len(day), f'no games on {date}'
    games = pd.DataFrame({'gameId': day['gameId'].astype(str).str.zfill(10).values, 'gameDate': date,
                          'teamId_home': day['teamId_home'].astype(int).values,
                          'teamId_away': day['teamId_away'].astype(int).values})
    inj = None
    season = day['seasonYear'].iloc[0]
    if season >= GameFeatureBuilder.INJURY_FIRST_SEASON:
        up = pd.concat([pd.DataFrame({'gameId': games['gameId'], 'teamId': games[f'teamId_{s}']})
                        for s in ('home', 'away')], ignore_index=True)
        up['gameDate'], up['seasonYear'] = date, season
        inj = irf.build_injury_features(str(DB_PATH), [season], snapshot_name=GameFeatureBuilder.INJURY_SNAPSHOT,
                                        as_of=pd.Timestamp(date) + pd.Timedelta(hours=12), upcoming=up,
                                        reports=irf.parse_all_reports())
        inj = inj[inj['gameId'].astype(str).str.zfill(10).isin(games['gameId'])]
    return day, build_live_game_frame(str(DB_PATH), games, injury_features=inj)


@pytest.mark.parametrize('date', DATES)
def test_live_features_equal_training_features(training_set, date):
    day, live = _live_frame(training_set, date)
    day = day.assign(_gid=day['gameId'].astype(str).str.zfill(10)).set_index('_gid')
    live = live.assign(_gid=live['gameId'].astype(str).str.zfill(10)).set_index('_gid').loc[day.index]
    cols = [c for c in day.columns if not c.startswith(SKIP_PREFIXES) and c in live.columns
            and pd.api.types.is_numeric_dtype(day[c]) and c not in ('gameId', 'seasonYear')]
    only_training = [c for c in day.columns if c not in live.columns and not c.startswith(SKIP_PREFIXES)
                     and c not in ('gameDate', 'gameId', 'seasonYear')]
    assert not only_training, f'columns the training set has but live does not build: {only_training}'
    bad = {}
    for c in cols:
        a, b = live[c].astype(float).to_numpy(), day[c].astype(float).to_numpy()
        ok = np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
        if not ok.all():
            bad[c] = list(zip(a[~ok][:3], b[~ok][:3]))
    assert not bad, f'{len(bad)} columns differ on {date}: {dict(list(bad.items())[:8])}'
    assert len(cols) > 300     # the comparison really covered the feature set


def test_live_frame_has_every_model_input(training_set):
    """All features the deployed/OOS model configs ask for (except odds) exist in the live frame."""
    import json
    from models.model_registry import MODEL_DIR
    _, live = _live_frame(training_set, DATES[0])
    for cfg in MODEL_DIR.glob('feature_config_oos_*.json'):
        feats = json.loads(cfg.read_text())
        wanted = {f for fs in feats.values() for f in fs if not f.startswith('odds_')}
        missing = wanted - set(live.columns)
        # older sets used the buggy roster_bpm_ewma_x/_y names; those cannot exist any more
        missing -= {'roster_bpm_ewma_x', 'roster_bpm_ewma_y'}
        assert not missing, f'{cfg.name}: {sorted(missing)}'
