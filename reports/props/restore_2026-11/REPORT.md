# Props model restore, 2026-11 (running log)

Restores the pregame player-props model after the 2026-10-31 `data\` loss.
Restored model version: `v3_live_2026-27r1`. Pre-loss record: `v3_live_2026-27` (kept as is).

## Step 0. Preflight (2026-11-03 08:50-08:56 PT)

- `git branch --show-current` -> `main`.
- Scheduled tasks: NBA Props Publish PM `Disabled`, NBA Props Settle `Disabled`, NBA Props Signals `Disabled`, NBA Props Snapshots `Ready` (untouched).
- Missing as expected: `data\props_live.db`, `data\props_features_v3.db`, `data\props_models\`, `data\raw\injury_reports_hourly\`.
- Present: `data\nba_stats.db` (1,092,714,496 B), `data\props_odds.db` (3,945,582,592 B), `data\props_dashboard\`, `data\props_dashboard_live\`, `data\raw\`.
- Backup: `D:\nba_data_rescue\props_dashboard_pre_restore_20261103-0853\` (Copy-Item -Recurse).
  - `props_dashboard_live`: src 12 files / 748,430 B; dst 12 files / 748,430 B.
  - `props_dashboard`: src 7 files / 39,047,624 B; dst 7 files / 39,047,624 B.
- Baseline:
  - `data\props_dashboard_live\bets.csv`: 814 rows (814 unique `bet_id`), sha256 `8ac8e547a6c6d05ae15697da2bb652d940308a6570207bbce7410c35ab239ae4`.
  - `season_summary.csv`: bets 733, player_games 461, roi_best_stands 0.019716, roi_median_stands -0.020376, roi_flat_stands -0.015503 (matches expected).
  - Old pinned config saved as `live_config_v3_live_2026-27.old.json` (from `git show HEAD:reports/props/live_config.json`; working copy has no diff vs HEAD). Bundle sha256 `c416a8a5...746a`, features 684,130,304 B sha256 `2d221a8d...fab9`, rule sha256 `b597763a...b6fb`, `n_sims` 20000, effective_from 2026-10-22.
- Tests: `pytest tests/props -q -m "not db"` -> 179 passed, 9 deselected, 1 warning (pandas FutureWarning), 55 s. The live-export bets.csv hash was unchanged afterwards (the dashboard boot test uses tmp_path).

## Step 1. Hourly injury-report archive (started 08:55 PT)

- Dry run (`--start 2021-10-01 --end 2026-06-30 --dry-run`): 1,056 game days x 8 new slots = 8,448 requests (~701 MB if all exist, estimate 73 min).
- Real run in the background, log `data/raw/download_injury_hourly_restore.log`. Results below when it finishes.

## Step 2. `v2_base` rate-family stub (done while step 1 downloaded; independent of it)

- Wrote `data/props_models/v2_base/manifest.json` with `families` read programmatically from `reports/props/frozen_config.json` -> `models` -> `rate_families_per_component (selected on dev folds in v2_base)`, plus `restored_from` and `note`.
- Check (asserted in code): reloaded `families` == frozen dict; keys exactly {fg2a, fg3a, fta, reb, ast, p2, p3, pft}. Values: fg2a hier, fg3a lgb, fta hier, reb hier, ast lgb, p2 hier, p3 hier, pft hier.

## Step 5. Live record import (done while step 1 downloaded; it does not depend on steps 1-4)

Script: `import_live_record.py` (this folder). Checks: `check_import.py`, output `check_import_output.json`.
First run against a scratch copy (outside the repo) to test, then the real one at 2026-11-03T16:58:17Z.

**Count difference from the brief.** The brief expected 733 results and 81 unsettled bets. The export actually holds:
- 742 settled bets (`y` or `status_stands` present): 378 win, 355 loss, **9 void** (player did not play; `y` empty, `status_stands` = void). 733 = stakes only (win + loss), which is what `season_summary.bets` counts.
- **72 unsettled**, all on 2026-10-31 (not 81; 81 = 814 - 733 counted the voids as unsettled).
- I followed the brief's definition (`y` or `status_stands` present), so `prop_results` has **742** rows. Leaving the voids out would only make the settle job settle them again as void. Flagged for the owner.

Import result (`data/props_live.db`, 1,093,632 B, sha256 `fe5ef19c...b2d5` at import time):
- `prop_bets` 814, `prop_results` 742, `prop_signals` / `slot_log` / `pipeline_health` 0.
- `game_id`: all 814 are 10-character strings with leading zeros, asserted equal to the `game_id` part of `bet_id`; `player_id` stored as SQLite integer (814/814).
- `pmf_json` / `books_json` from pmf.json / books.json (keys = the 814 bet_ids exactly). Old hashes from the old config (bundle `c416a8a5...`, features `2d221a8d...`, rule `b597763a...`), `roster_rule` v2, `run_id` `restored_from_dashboard_export_2026-11`, `snapshot_ts_utc`/`quote_age_min`/`code_commit` NULL, `model_version` v3_live_2026-27 (asserted).
- Profits from `props.betting.settle.settle(..., 'stands')` with dec_best / dec_median / dec_flat; the recomputed stands status equals the exported `status_stands` on all 742 (asserted).

Checks:
1. Counts 814 / 742 (742, not 733: see above).
2. Re-export into `export_check/` (old config, n_boot 1000):
   - same 814 bet_ids; **no difference on any column for the 742 settled bets** (tolerance 1e-9).
   - differences only on the 72 unsettled bets' line history: `line_240` 55, `mkt_240` 55, `line_60` 72, `mkt_60` 72 cells. The old export filled these from `prop_signals`, which was lost. The values are kept in `unsettled_line_history_from_export.csv` (this folder) and in the D: backup.
   - season_summary: bets 733, player_games 461, roi_best 0.0197160333552556 (equal), roi_median -0.0203756330623458 (equal), roi_flat -0.0155029145479352 vs -0.0155029145479349 (3e-16, float summation order).
   - pmf.json equal, books.json equal.
3. Triggers: UPDATE and DELETE inside a rolled-back transaction both raise `prop_bets is immutable`; 814 rows afterwards.

**CLV.** `clv_prob`, `line_move_favour` and `line_5`/`mkt_5` were empty for **all** 814 bets in the export, settled ones included; the pre-loss record never had CLV filled. The 72 bets that settle late will also have no CLV and no line history (`line_*`/`mkt_*`), because the settle job reads them from the lost `prop_signals`.

### Step 1 result (finished 10:11:50 PT, about 76 min)

- 8,448 slots requested, **8,442 PDFs downloaded** (488 MB), **6 missing**: all on 2023-10-24 (opening night 2023-24: 08AM, 09AM, 10AM, 02PM, 03PM, 04PM). The manifest records 403 and 404 alike as `missing`, so I cannot tell them apart; either way they are 0.07% of the slots, far under the 10% stop line. Zero transient errors in the log.
- By year (ok): 2021 576, 2022 1,704, 2023 1,618, 2024 1,680, 2025 1,728, 2026 1,136.
- Dry run again: **0 requests left**. `data\raw\injury_reports_hourly\manifest.csv` has 8,448 rows.

## Step 3. Features database (started 10:13 PT)

`build_props_features.py --out-db data/props_features_v3.db --max-date 2026-06-30 --dictionary-dir reports/props/restore_2026-11`, in the background, log `data/raw/build_props_features_restore.log`. Checks: `step3_parity.py` -> `step3_parity.json`, `step3_dictionary_diff.csv`.

### Step 3 result (finished 10:53:16 PT, 39 min, exit 0)

- The build seeded `props_injury_main_parsed.csv` from the team cache (113.5 MB), parsed the hourly PDFs into `props_injury_hourly_parsed.csv` (100.0 MB) and linked them (`props_injury_linked.pkl`, 158.1 MB). It fetched the NBA schedules for 2019-20..2022-23 (free endpoint; log lines "fetched N scheduled games"). No errors or warnings in the log.
- `data/props_features_v3.db`: **684,220,416 B** (old 684,130,304; +90,112 B), sha256 `7350a26f027280cf63f0d193fdc93cba91ebd8d1e4caf692a270a625c624390c`.
- Tables `player_game_base` 232,782, `player_game_labels` 232,782, `player_game_at_T` 698,346 (232,782 at each of T-240 / T-60 / T-5). Same player-game count as the Phase 2 gate (232,782).
- Rows by season (base = labels; at_T = 3x): 2019-20 28,577; 2020-21 31,676; 2021-22 33,848; 2022-23 34,024; 2023-24 34,815; 2024-25 35,087; 2025-26 34,755.
- Max `game_date` 2026-06-13 (on or before 2026-06-30).
- Feature dictionary vs tracked `reports/props/phase2_feature_dictionary.csv`: **(table, feature) sets identical (223 = 223)**; group, source, dtype and description unchanged for all.
  - `first_season_with_data`: no change.
  - `non_null_share`: 10 features changed, all `market_env` (at-T), each by +0.0023..+0.0024 (e.g. `env_total` 0.7958 -> 0.7982, `env_spread_dispersion` 0.4468 -> 0.4491). None moved more than 0.01. Likely cause: the restored paid backfill in `props_odds.db` (3,958 events) gives slightly more market-era games a team-line snapshot than the snapshots the original build saw. Not fixed.
  - Injury features (6) and teammate features (15): unchanged non-null shares, so the rebuilt hourly archive gives the same coverage as the lost one (to 4 dp).
  - `non_null_by_season`: no change for any base feature; the at-T features have no by-season string in either file.
  - Full list: `step3_dictionary_diff.csv`; all checks: `step3_parity.json`.
- Side note: `data/props_models/config_log.jsonl` (an append-only log of model configs tried) appeared at 08:55:00, written by the step 0 test run (model tests append to the real log). Harmless, left as is.

## Step 4. Training `v3_live_2026-27r1` (started ~10:57 PT)

`train_props_live.py --version v3_live_2026-27r1 --next-season 2026-27` (defaults: `--families-from v2_base --sims 4000 --seed 0`), log `data/raw/train_props_live_restore.log`.

### Step 4 result (finished 11:27:32 PT, 1,906 s total; fit 1,666 s)

- `data/props_models/v3_live_2026-27r1/manifest.json` and `.../2026-27/bundle.pkl` (29,358,975 B) exist.
- **bundle sha256 `7045a4ba34c4feb231ec3c7da1d6cfdd4cf6376a9503935667829525480ea2d1`**
- `trained_through` 2026-06-13 (the last 2025-26 game date); `calibration_starts` 2025-03-27.
- `fit_rows` 562,062; `calib_rows` 123,396; `rows` 698,346; `sims` 4000; `seed` 0; families as the v2_base stub; `code_commit` 8dd0aa3.
- `features_db_sha256` 7350a26f027280cf63f0d193fdc93cba91ebd8d1e4caf692a270a625c624390c (= the step 3 file).
- `dispersion_c` (stat:regime): points 1.1112 / 1.1234 / 1.0; rebounds 1.0552 / 1.0248 / 1.0; assists 1.0602 / 1.0769 / 1.0; threes 1.0329 / 1.0888 / 1.0384; pra 1.1613 / 1.1662 / 1.0361; pr 1.1486 / 1.155 / 1.037; pa 1.1296 / 1.1357 / 1.0; ra 1.1023 / 1.0872 / 1.0407.
- The old `v3_live_2026-27` manifest was lost with `data\` and no tracked report records its row counts or dispersion values, so there is nothing to compare these with.

## Step 6. Candidate config and dry run

- `live_config_candidate.json` (this folder): same keys in the same order as the old config / `pin()`; `model_version` v3_live_2026-27r1, bundle `7045a4ba...a2d1`, features 684,220,416 B / `7350a26f...390c`, rule file and rule sha256 unchanged (`b597763a...b6fb`), `n_sims` 20000, roster v2, waivers and note unchanged, `effective_from` "TBD". Built by hand (not `pin()`); the real `reports/props/live_config.json` is untouched (no git diff).
- `config.verify(candidate, check_features_db=True)`: **passed** (no exception).
- Dry run (`step6_dry_run.py` runs `scripts/props_signal_job.py` unchanged and keeps a copy of what `signal.evaluate_item` returns): `--config reports/props/restore_2026-11/live_config_candidate.json --db reports/props/restore_2026-11/scratch_live.db --now 2026-11-03T00:12:00Z --dry-run`. Log `data/raw/props_signal_dryrun_restore.log`; outputs `step6_signals.csv`, `step6_bets.csv`.
  - `--now` = T-60 of the three 2026-11-02 ET games with reference tip 2026-11-03T01:10:00Z (0022600164, 0022600165, 0022600166). Snapshots used: 00:10:06-00:10:08Z (59.9 min before tip).
  - **Exit 0, no `refused`**, history check passed, context ready in 26 s, 1 min 18 s in all.
  - **116 signals** (41 players: 12 / 16 / 13 per game; rebounds 40, threes 28, assists 27, ra 21), **16 qualified would-be bets** (6 / 9 / 1). Informational only; nothing changed because of it.
  - `p_model` 0.308-0.725 (all in (0,1), median 0.471); `p_star` 0.355-0.627; `p_market` 0.345-0.639; `expected_minutes` 15.6-36.0 (all in [0,48], median 27.3).
  - Report linking: team 100%, game 100%, player 88.4% of player rows.
  - Side effects (pipeline files, expected): `data/processed/live_play_tables_2026-27.pkl` created (from the new features DB); `props_injury_main_parsed.csv` gained the 2 newest main-archive PDFs. The schedule for 2026-27 was fetched (free).
  - No writes: the scratch DB still hashes `fe5ef19c...` with 0 signals / slot_log / health rows; `data/props_live.db` and the live export hashes are unchanged.
- **Optional old-bet comparison: skipped.** Rebuilding an October decision frame needs the slot's full quoted-player set (`SlotItem.cons`). It decides which players get rows and, through roster rule v2, the teammate-availability features of every row. Those quotes (2026-10-20..31) are lost. Using only the players we bet on would change the teammate inputs, so the frames cannot be built honestly.

## STOP 1 (2026-11-03 ~11:35 PT): waiting for the owner

Asked to approve: (a) pinning `v3_live_2026-27r1`; (b) `effective_from`; (c) how to commit the config (default: worktree `../nba-props-restore`, branch `props-restore-2026-11`, PR on a yes). Steps 7-12 have not started.

## Owner decisions at STOP 1 (2026-11-03, written)

- 742 results (voids included) kept; the brief's 733 counted wins and losses only.
- Line history of the 72 late bets: write the 55 T-240 and 72 T-60 values (line and mkt) from `unsettled_line_history_from_export.csv` back into their result rows after step 7, labelled as restored.
- CLV: empty for all 814 before the loss; nothing lost.
- Old-bet comparison: skipping it confirmed.
- (a) pin `v3_live_2026-27r1`: yes. (b) `effective_from` 2026-11-04: yes (no games on 11-03; gap = the 11-01 and 11-02 slates). The code never reads `effective_from`: the record really starts when the Signals task is enabled; enable it after the config is committed and before 12:00 PM PT on 11-04 (T-240 of the first game). (c) commit via worktree + branch, push and open the PR: yes; merge is not needed before the restart (the job reads the main checkout's working tree).
- Follow-ups after the restart: per-model-version split in the live export (the 2026-27 summary combines both segments); tests must stop appending to the real config log.

## config_log.jsonl (owner request)

The file counts model configurations tried, for the multiple-testing corrections; the original was lost on 10-31. The new file (142 lines, 13,451 B, sha256 `e19d02a5...51fb`) held two things, split at byte 11,609 (a line boundary) into:
- `data/props_models/config_log.TEST_RUN_2026-11-03_not_a_real_count.jsonl`: 122 lines from the step 0 pytest run (08:55).
- `data/props_models/config_log.train_v3_live_2026-27r1_2026-11-03.jsonl`: 20 lines appended by the step 4 training (fit on 562,062 rows, then refit on 685,458).
The two files concatenate to the original byte for byte. No `config_log.jsonl` remains, so the next writer starts a new file that is clearly not the historical count.

## Step 7. Settle: CHECK FAILED (2026-11-03 11:47 PT), stopped

`props_settle_job.py --dry-run` exited 1: `KeyError: "['clv_prob'] not in index"` at `scripts/props_settle_job.py:45`.
- Cause: `props/live/settle.py:settle_bets` adds `clv_prob`, `line_move_favour`, `close_line` and `line_*`/`mkt_*` only when the batch has signal rows. The 72 restored bets have none (`prop_signals` was lost), so the result frame lacks `clv_prob` and the job's print fails. It fails before the insert (and the dry run would not insert anyway): `prop_results` is still 742.
- The failed run wrote one `pipeline_health` row to `data/props_live.db`: run `20261103T194718Z`, job settle, status `error`. Left as the record of that run.
- Same failure would hit the Settle task on any morning when the only unsettled bets are the restored ones (e.g. 11-04). Once a batch includes a bet with signals, the columns exist and the 72 get NaN CLV.
- Waiting on the owner's choice of fix.

Owner's choice (written): fix the gap in code on the restore branch and carry on with steps 8, 9 and the Signals task meanwhile; step 7 runs once the fix is in the main checkout.

## Step 8. Pinned config (2026-11-03 ~11:55 PT)

- Main checkout (branch `main`, config clean vs HEAD before): `config.pin('v3_live_2026-27r1', '2026-27', '2026-11-04')`, then `supersedes` added with a JSON load / add / dump (`indent=1`).
- `config.verify(config.load(), check_features_db=True)`: passed.
- Differences from `live_config_candidate.json`: only `effective_from` (TBD -> 2026-11-04) and the new `supersedes` key.
- `reports/props/live_config.json` sha256 `5f464fc8c0d4b912d81e46a24b8a4b5ec99576a94a3786bbe7dde7edf5cb7399`; copy in `live_config.json.new` (main checkout only).
- `effective_from` is not read by any code: the new segment really starts when the Signals task is enabled (owner: after this commit, before 12:00 PM PT on 11-04).
- Worktree `../nba-props-restore`, branch `props-restore-2026-11` from `main` at 8dd0aa3 (= origin/main); it has no `data` entry.
