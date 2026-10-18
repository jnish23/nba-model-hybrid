# Phase 0: audit and `core/` refactor (behavior-preserving)

Branch `props-phase0-core-refactor` (worktree `C:\Users\jorda\nba-model-hybrid-props`), cut from `78b87f4`
(content-identical to `origin/main`). Nothing has been merged; the main checkout, its uncommitted files and the
two Windows scheduled tasks are untouched.

## 1. Gate result

| Check | Result |
|---|---|
| pytest, before (main checkout, real DB) | 98 passed |
| pytest, after (worktree, frozen copy of the DB, DB-marked parity tests included) | 100 passed (98 + 2 new package-boundary tests) |
| `compileall core team props scripts`, `scripts/smoke_check.py` (the CI steps) | pass |
| Every `core`/`team`/`props`/`scripts` import in the repo resolves; no code imports `models.*` any more | pass |
| `--help` of `snapshot_odds`, `scrape_injury_reports`, `run_inference_v2`, `generate_training_set`, `generate_season_predictions`, `evaluate_vs_market`, `team/train_baseline.py` | pass |
| **Training set** from the real entry point `scripts/generate_training_set.py` (16,795 rows x 430 cols) | **byte-identical** (sha256 3c1a0c358bdf...) |
| **Inference feature path + deployed model scoring** (`build_live_game_frame`, then the deployed `_with_odds_v5` set: raw ML prob, calibrated prob, spread, total) on 5 historical dates (52 games, 2020 to 2026-04-12) | **byte-identical** (live frames and predictions) |

How the comparison was made: both code trees (old commit, refactored tree) ran with `data/` pointing at one frozen
copy of `nba_stats.db`, the odds archive and the injury-report archive, with the parsed-report cache restored to the
same starting state. `scripts/phase0_equivalence_check.py --out DIR` produces the dumps and
`--compare A B` hashes them (it works on both layouts, so it can be reused for later refactors).

**What was not compared:** a live `run_inference_v2.py` run. It calls the NBA API and SBR, syncs and writes to the
production DB and appends to the prediction log, so two runs are not comparable and running one now would change
production state. The replay above exercises the same feature builders, model loading and scoring code; the
network, odds-mapping and logging code in that script moved only in its import lines. There is no game today
(season opens 2026-10-20); the first scheduled inference run is the real end-to-end test, and I suggest watching it.

## 2. Audit: what the repo is

**Code (before):** `models/` (21 modules, library code), `scripts/` (~30 entry points, but also three library modules
that others imported: `update_database`, `scrape_injury_reports`, `snapshot_odds`), `dashboard/app.py`, `tests/`,
`notebooks/`, `data/` (gitignored), `reports/`. `pipeline/` and `backtesting/` are empty (only `__pycache__`).

**Data flow**
```
nba_api box scores --update_database--> nba_stats.db (player_*, team_*, team_game_full)
SBR (opening/closing team lines) -----> nba_odds, nba_odds_opening            (no timestamps)
SBR current lines ---snapshot_odds----> odds_snapshots                        (timestamped; see finding 1)
NBA injury report PDFs --scrape_injury_reports--> data/raw/injury_reports/**.pdf
      --injury_report_parser--> data/processed/injury_reports_parsed.csv (cache) --injury_report_features--> inj_* + play rates
box scores --team_features_v2 / bpm_builder / player_features / travel_builder--> game_features_v2.build_training_set
      --> data/processed/training_set_v1_reduced_<date>.csv --train_baseline--> models/saved/*_with_odds_v5.* (+ calibrator)
live: run_inference_v2 --> live_features.build_live_game_frame (same builders) --> model_registry.load_bundle
      --> prediction_logger (predictions_log_v2 + predictions/*.csv) --export_dashboard_data--> dashboard-data branch --> Streamlit
```

**DB (`nba_stats.db`, 1.09 GB, 46 tables).** Groups: player box scores in several endpoint families (`player_basic_stats`
356k rows with minutes; `player_advanced/usage/scoring/track_stats_v3` 433k rows each; `player_hustle_stats`;
`player_shot_location_boxscores`; `player_miscellaneous_stats[_v3]`), the same families per team, `team_game_full`
(169 cols, the feature source), `player_info`, `nba_arenas`, `nba_odds` / `nba_odds_opening`, `nba_injury_logs`,
`player_bpm` (cache), prediction logs, and clutter (`*_backup*`, `predictions_log_v2_backup_*`, `predictions_*_2024_25`).

**Scheduled jobs (Windows Task Scheduler, working dir = main checkout)**

| Task | Runs | Trigger | State |
|---|---|---|---|
| NBA Odds Snapshots | `scripts\run_snapshot_odds.bat` -> `scripts\snapshot_odds.py` | from **2026-10-19 06:00 PT**, every 30 min for 18 h | Ready; last ran 2026-10-10 |
| NBA Daily Inference | `scripts\run_daily_inference.bat` -> `run_inference_v2.py` then `export_dashboard_data.py --publish` | **2026-10-20 09:15 PT**, daily | Ready; never run |

### Findings that matter for the props work
1. **`odds_snapshots` does not exist in the DB.** The only run (2026-10-10) got no SBR data, so the table was never
   created. There is no timestamped line history at all yet. Anything H1/H2 needs starts accumulating from the first
   successful run.
2. **Player box-score coverage is good:** basic, advanced, usage, tracking and scoring tables cover 2013-14 to 2025-26
   (min 23k to 28k player rows per season). Hustle is sparse in early seasons (min 825 rows/season), `*_misc_v3` has only 3
   seasons, `player_bpm` only 2023-24 onward (it is a cache). Phase 2 should build on basic/advanced/usage/track/scoring.
3. **Injury-report archive** parses to 2019-20 onward; `nba_injury_logs` holds only 13k rows (2026-01-15 to 2026-04-07).
4. `team/betting.py` mixes team-market bet selection with generic price math (implied probability, payout, EV). Props
   needs the generic part. Not split in this phase (move-only); proposed for Phase 1 together with the de-vig code.

## 3. New layout

| Package | Contents |
|---|---|
| `core/` | `update_database`, `scrape_injury_reports`, `snapshot_odds`, `sbr` (ingestion and collectors); `injury_report_parser`, `injury_report_features`, `availability_features` (injury parsing, status play probabilities); `bpm_builder`, `player_features`, `travel_builder` (shared feature utilities, incl. travel/rest/altitude); `odds_clean`; `prediction_logger`; `season_utils` |
| `team/` | `team_features_v2`, `game_features_v2`, `live_features`, `live_odds`, `live_rotation_v2`, `model_registry`, `train_baseline`, `backtest`, `betting`, `betting_simulation`, `data_paths` |
| `props/` | empty package (new code goes here) |
| `scripts/` | unchanged paths; `update_database.py`, `scrape_injury_reports.py`, `snapshot_odds.py` are now 8-line wrappers that call `core.<name>.main()` |
| `tests/` | `core/`, `team/`, `props/` per package; `conftest.py`, `test_requirements.py` and the new `test_package_boundaries.py` stay shared |

`test_package_boundaries.py` enforces `core` imports neither `team` nor `props`, `team` never imports `props`, `props`
never imports `team`, and nothing imports the old `models` package.

Code changes beyond import rewrites (all 3 are path or entry-point mechanics):
- `team/model_registry.py`: `MODEL_DIR` = `<repo>/models/saved` (it used to be relative to the module file).
- `team/betting_simulation.py`: same fix in its `__main__` block.
- `core/update_database.py`: its `__main__` block became `main()` so the wrapper can call it.

## 4. Decisions I made (say if you want any reversed)
1. **`models/saved/` stays where it is** and now holds artifacts only. Moving it means rewriting every `*.json`/report
   path and touching 52 tracked artifacts for no behavior gain. A later `git mv` is easy if you prefer `team/saved/`.
   Saved pickles reference no moved module (3 old `*_oos_2025-26` pickles reference `models.market_models`, a module
   removed long ago; they were unloadable before and still are).
2. **`scripts/` stays flat** and the two `.bat` files stay at the same paths. Your Task Scheduler entries point at them,
   and changing scheduler config is your call. New props entry points will go in `scripts/props/`.
3. **`prediction_logger` went to `core/`**, because `update_database` calls it after every sync, and `core` must not
   depend on `team`.
4. **`injury_report_features` moved intact to `core/`.** It holds both the generic status-to-play-probability code (shared)
   and the team-level aggregation. I will split the team-level part off only when Phase 2 needs to.
5. No compatibility shims for `models.*`: nothing in the repo or the notebooks imports it. External code that does
   (none known) would break.

## 5. Risks and what I need from you
- **Merge timing.** Both jobs run from the main checkout. If this is merged while a run is mid-flight, that run could fail.
  The first snapshot fires 2026-10-19 06:00 PT. Suggested: merge tonight (nothing fires before then), then
  `git pull`/merge in the main checkout with your uncommitted notebooks and `predictions_2025-26.csv` left alone
  (none of them are touched by this branch). Tell me whether to merge, and by which route (PR or local merge).
- **CI** was not run on GitHub; I reproduced its three steps locally on Windows (`compileall`, `smoke_check`, `pytest`).
- `README.md` layout table and `team/train_baseline.py` command paths are updated; the dashboard imports `team.betting`
  (still free of scipy, covered by an existing test).

## 6. Proposed next step (Phase 1, not started)
Props odds store (`data/props_odds.db`), vendor-agnostic loader, collector extension of `snapshot_odds`, player-ID
mapping, de-vig, quality report. Decisions I will need from you first, since they cost money or touch production:
which odds vendor and plan for live collection (Odds API credit cost per props snapshot grows with books x markets x
games), the cadence/credit budget, and the historical-backfill density (I will send a cost table before running anything).
