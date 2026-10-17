# E09_pdf_injuries_live

No backtest result is claimed for this item.

## 1. ESPN references on live/feature paths (espn.com / BeautifulSoup)

```
{
 "scripts/run_inference_v2.py": 0,
 "models/live_features.py": 0,
 "models/live_rotation_v2.py": 0,
 "models/injury_report_features.py": 0,
 "models/injury_downgrades.py": 0,
 "models/game_features_v2.py": 0
}
```
ESPN is reachable only from `models/injury_source_compare.py` (comparator + logging); `live_rotation_v2.py` now reads the official PDFs.

## 2. Noon-to-latest downgrade flag (historical coverage; descriptive)

- team-games with a computed flag: 17622; with a non-zero downgrade: 7928 (45.0%); with >= 1 player whose P(miss) rose by >= 0.25: 6645 (37.7%)
- a newer report than the noon one existed for 97.1% of games
- lookahead check on 8824 games: late snapshot is never newer than tip - 30 min: **True**; late >= early: **True**
- tip-time caveat: for 16 of 8824 games (0.2%) the tip printed in the late report differs from the tip the production cutoffs use (first report that lists the game); in 9 of them the production tip is LATER than the late report own tip, i.e. a report counted as pre-tip could in principle have been published after a re-timed tip (pre-existing behaviour, not changed here)
- mean downgrade value when flagged: 2.95; share of team-games with a downgrade by season: {'2019-20': 0.3298, '2020-21': 0.3378, '2021-22': 0.401, '2022-23': 0.3847, '2023-24': 0.3868, '2024-25': 0.4054, '2025-26': 0.3832}

## 3. Official PDF vs ESPN on the historical overlap (nba_injury_logs, ESPN scrapes 2026-01-15..2026-04-07)

```
{
 "scrapes compared": 125,
 "listed player-days": 9761,
 "agreement": 0.5958,
 "ESPN OUT but PDF not OUT": 0.1057,
 "PDF OUT but ESPN not OUT": 0.3532
}
```
Rows = PDF category, columns = ESPN category, counts of listed player-days on teams playing that day:

| pdf_category   |   OUT |   PLAYING |   UNCERTAIN |
|:---------------|------:|----------:|------------:|
| OUT            |  4427 |      2106 |         311 |
| PLAYING        |   299 |       657 |         597 |
| UNCERTAIN      |   224 |       408 |         732 |

Training/serving mismatch: no model was ever trained on ESPN-derived features (the training inj_* features are PDF-derived), so there is no historical ESPN feature set to compare against; the table above is the source-level disagreement a switch back to ESPN would have introduced.

## 4. Side-by-side logging for new games

- `run_inference_v2.get_injury_features` now calls `injury_source_compare.log_side_by_side` (table `injury_source_compare`: logged_at, game_date, player_key, team, pdf_status, espn_status, categories, agree) and computes `inj_downgrade_lost_home/away` for the picks CSV; both are best-effort and never block inference.

## Done-criteria (pre-registered)

- ESPN scrape off every live path: **True**
- downgrade flag implemented with a lookahead test: **True** (tests/test_experiment_lookahead.py, plus the ordering check above)
- both sources logged side by side for new games: **True** (first rows appear from the next live run)
- disagreement rate on the historical overlap reported: **40.4%**