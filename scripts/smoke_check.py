"""
Lightweight checks that need no database or network (run by .github/workflows/checks.yml).
Production inference runs locally (Windows Task Scheduler: scripts/run_daily_inference.bat),
because it needs data/nba_stats.db and the injury-report archive, which are not in the repo.
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]


def main():
    # Imports used on the production path
    import core.odds_clean  # noqa: F401
    import core.injury_report_parser  # noqa: F401
    import core.availability_features  # noqa: F401
    import team.game_features_v2  # noqa: F401
    import run_inference_v2  # noqa: F401
    from core import injury_report_features as irf
    from core.season_utils import season_for_date
    from core import snapshot_odds, scrape_injury_reports

    # Season boundaries
    assert season_for_date('2026-10-20') == '2026-27'
    assert season_for_date('2027-06-15') == '2026-27'
    assert season_for_date('2026-09-30') == '2025-26'

    # Injury-report filename slots and snapshot cutoffs (capped at tip - 30 min)
    assert irf.report_slot('Injury-Report_2024-01-15_05PM.pdf') == (17, 0)
    assert irf.report_slot('Injury-Report_2026-01-15_05_30PM.pdf') == (17, 30)
    assert {(h % 12 + (12 if ap == 'PM' else 0), 0) for h, ap in scrape_injury_reports.SLOTS} == irf.GRID_SLOTS
    g = pd.DataFrame({'game_date': pd.to_datetime(['2026-01-15', '2026-01-15']),
                      'tip_ts': pd.to_datetime(['2026-01-15 19:30', '2026-01-15 12:00'])})
    noon = irf.SNAPSHOT_CUTOFFS['game_day_noon'](g)
    assert noon.iloc[0] == pd.Timestamp('2026-01-15 12:00') and noon.iloc[1] == pd.Timestamp('2026-01-15 11:30')

    # In-game detection for odds snapshots
    fetched = pd.Timestamp('2026-10-21 19:00')
    assert snapshot_odds.in_game_flag('7:30 PM ET', '2026-10-21T23:30:00Z', fetched) == 0
    assert snapshot_odds.in_game_flag('7:30 PM ET', '2026-10-21T22:30:00Z', fetched) == 1
    assert snapshot_odds.in_game_flag('Final', None, fetched) == 1
    assert snapshot_odds._num('PK') == 0.0

    print("smoke checks passed")


if __name__ == "__main__":
    main()
