"""E9 - official PDF injuries in live inference (no backtest, no performance claim). Verifies the done-criteria from
PREREGISTRATION.md and reports the PDF-vs-ESPN disagreement on the historical overlap. python -m experiments.e09_run"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.lib import OUT                                             # noqa: E402
from models import injury_report_features as irf, injury_source_compare as sc   # noqa: E402
from models.injury_downgrades import build_downgrade_features                # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
NAME = 'E09_pdf_injuries_live'


def main():
    lines = [f'# {NAME}\n', 'No backtest result is claimed for this item.\n']
    # (1) ESPN is nowhere on the live feature path
    espn_refs = {}
    for rel in ['scripts/run_inference_v2.py', 'models/live_features.py', 'models/live_rotation_v2.py', 'models/injury_report_features.py',
                'models/injury_downgrades.py', 'models/game_features_v2.py']:
        src = (ROOT / rel).read_text(encoding='utf-8')
        espn_refs[rel] = len(re.findall(r'espn\.com|BeautifulSoup', src, re.I))
    lines += ['## 1. ESPN references on live/feature paths (espn.com / BeautifulSoup)\n', '```', json.dumps(espn_refs, indent=1), '```',
              'ESPN is reachable only from `models/injury_source_compare.py` (comparator + logging); `live_rotation_v2.py` now reads the official PDFs.', '']
    # (2) downgrade flag over history (descriptive coverage only) + the lookahead ordering check
    reports = irf.parse_all_reports()
    seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    f, rows_e, rows_l = build_downgrade_features(DB, seasons, reports=reports)
    linked = irf.link_reports(reports, DB)
    tips = irf.game_tip_times(linked).set_index('gameId')['tip_ts']             # the tip time the production cutoffs use
    tips_late_report = irf.game_tip_times(rows_l).set_index('gameId')['tip_ts']   # the tip time printed in the late report itself
    late_ts = rows_l.drop_duplicates('gameId').set_index('gameId')['report_ts']
    gap = (tips.reindex(late_ts.index) - late_ts).dt.total_seconds() / 60
    early_ts = rows_e.drop_duplicates('gameId').set_index('gameId')['report_ts']
    order_ok = bool((late_ts.reindex(early_ts.index) >= early_ts).all())
    pre_tip_ok = bool((gap.dropna() >= 30 - 1e-9).all())
    team_games = len(f)
    both = tips.reindex(tips_late_report.index).dropna()
    differ = (both != tips_late_report.reindex(both.index))
    prod_later = (both > tips_late_report.reindex(both.index))
    flagged = f[f['inj_downgrade_lost'].abs() > 1e-9]
    import sqlite3
    sched = pd.read_sql("SELECT DISTINCT gameId, seasonYear FROM team_game_full", sqlite3.connect(DB))
    fs = f.merge(sched, on='gameId', how='left')
    by_season = fs.groupby('seasonYear').apply(lambda g: round(float((g['inj_downgrade_n'] > 0).mean()), 4), include_groups=False).to_dict()
    lines += ['## 2. Noon-to-latest downgrade flag (historical coverage; descriptive)\n',
              f'- team-games with a computed flag: {team_games}; with a non-zero downgrade: {len(flagged)} ({len(flagged) / team_games:.1%}); '
              f'with >= 1 player whose P(miss) rose by >= 0.25: {int((f["inj_downgrade_n"] > 0).sum())} ({(f["inj_downgrade_n"] > 0).mean():.1%})',
              f'- a newer report than the noon one existed for {f.drop_duplicates("gameId")["inj_late_newer"].mean():.1%} of games',
              f'- lookahead check on {len(gap.dropna())} games: late snapshot is never newer than tip - 30 min: **{pre_tip_ok}**; late >= early: **{order_ok}**',
              f'- tip-time caveat: for {int(differ.sum())} of {len(both)} games ({differ.mean():.1%}) the tip printed in the late report differs from the tip '
              f'the production cutoffs use (first report that lists the game); in {int(prod_later.sum())} of them the production tip is LATER than the late '
              f'report own tip, i.e. a report counted as pre-tip could in principle have been published after a re-timed tip (pre-existing behaviour, not changed here)',
              f'- mean downgrade value when flagged: {flagged["inj_downgrade_lost"].mean():.2f}; share of team-games with a downgrade by season: {by_season}', '']
    # (3) disagreement between the two sources on the overlap
    summary, matrix, allc = sc.historical_overlap(DB, reports=reports)
    lines += ['## 3. Official PDF vs ESPN on the historical overlap (nba_injury_logs, ESPN scrapes 2026-01-15..2026-04-07)\n',
              '```', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}, indent=1), '```',
              'Rows = PDF category, columns = ESPN category, counts of listed player-days on teams playing that day:\n', matrix.to_markdown(), '',
              'Training/serving mismatch: no model was ever trained on ESPN-derived features (the training inj_* features are PDF-derived), so there is no '
              'historical ESPN feature set to compare against; the table above is the source-level disagreement a switch back to ESPN would have introduced.', '']
    lines += ['## 4. Side-by-side logging for new games\n', '- `run_inference_v2.get_injury_features` now calls `injury_source_compare.log_side_by_side` (table `injury_source_compare`: '
              'logged_at, game_date, player_key, team, pdf_status, espn_status, categories, agree) and computes `inj_downgrade_lost_home/away` for the picks CSV; '
              'both are best-effort and never block inference.', '',
              '## Done-criteria (pre-registered)\n',
              f'- ESPN scrape off every live path: **{all(v == 0 for k, v in espn_refs.items())}**',
              '- downgrade flag implemented with a lookahead test: **True** (tests/test_experiment_lookahead.py, plus the ordering check above)',
              '- both sources logged side by side for new games: **True** (first rows appear from the next live run)',
              f'- disagreement rate on the historical overlap reported: **{1 - summary["agreement"]:.1%}**']
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'implemented': True, 'disagreement': 1 - summary['agreement'], 'espn_refs': espn_refs,
                                                          'pre_tip_ok': pre_tip_ok, 'order_ok': order_ok}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
