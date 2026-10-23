# D1 plan: the live pipeline behind the props dashboard

Date 2026-10-22. Status: plan for approval; nothing built. D0 (the dashboard on backtest data, `dashboard_props/`) is the front end; D1 is what makes it live and starts the paper-trading record. Companion to [dashboard_plan.md](dashboard_plan.md) and [phase5_final_report.md](phase5_final_report.md).

## 1. Goal and what "live" must preserve

At each decision time (T-4h, T-1h, T-5m) for every game, produce the same rule decisions the backtest would have produced, record them immutably, settle them after the games, and expose them to the dashboard with the same table schema as the backtest export. Three non-negotiables carried over from the project rules:

1. **Point in time.** Everything used at T is stamped at or before T (injury reports with the 15-minute publish lag, quotes from snapshots at or before T).
2. **Training/live parity.** The live feature code is the training feature code, proven by a parity test, not by inspection.
3. **Nothing about the rule changes.** The frozen rule file (`phase5_revised_rule.frozen_rule.json`: rebounds, assists, threes, ra; pooled k; e = 3%; earliest time with positive EV; at least 3 books) is loaded by hash and the job refuses to run if the hash differs.

## 2. Components

| # | Component | What it does | Size |
|---|---|---|---|
| 1 | **Feeds** | Already running: prop snapshots (the collector, every 10 minutes, with the T-grid). Needed: daily box-score sync (`update_database`), injury report PDFs at least hourly (reuse the existing downloader), the NBA schedule for upcoming games (`props/schedule.py`). Add a freshness check for each. | S |
| 2 | **Live feature builder** | For an upcoming game and decision time T, builds the base-spec feature row for each quoted player in exactly the training column set. The history, context, travel/rest and injury-at-T functions already exist as batch code (`props/features/*`); they need an "as of this date" entry point and a roster rule (below). | **L** |
| 3 | **Parity harness** | Re-creates 30+ historical dates as if live (database truncated to before the date, roster from the rule below, injury reports and quotes at or before T) and compares features and predictions with the stored training rows. | **L** |
| 4 | **Model for 2026-27** | Same frozen pipeline (v3, rate families from v2_base, refit on fit + calibration, no recalibration), trained once on everything through the end of 2025-26 as a new fold `v3_live_2026-27`, with manifest hashes. About 35 minutes of compute. Retrain policy in section 5. | S |
| 5 | **Signal job** | Scheduled every 10 minutes; for each game whose next decision time has arrived and which has no signal yet: build features, 20,000-draw predictions for the quoted players (seconds per game), apply the frozen rule, write `prop_signals` and, on first qualification only, an immutable row in `prop_bets`. Idempotent and safe to re-run. | M |
| 6 | **Settlement job** | Nightly after the box scores sync: actuals, minutes, the three settlement treatments, profit under the three price views, CLV to T-5, written to `prop_results`. | M |
| 7 | **Export** | `export_props_dashboard_data.py --live` writes the same CSV/JSON files as the backtest export, plus open bets with status `pending`; the dashboard needs no schema change. | S |
| 8 | **Health and alerts** | A `pipeline_health` table (last run per job, rows written, snapshot lag, errors) shown on the Model-health page; a log line and an optional desktop notification when a job fails or a game was missed. | S |

## 3. The hard part: the roster rule and live/training parity

Training rows came from box-score rosters (every listed player, played or not), which do not exist before the game. The live roster must be built from information available at T:

- **Traded set** = players with a consensus quote at T (that is the universe for betting, and it is what the backtest bets used).
- **Teammate-availability features** (the model's most valuable group, +0.8% to +1.0% CRPS) need the team's whole expected rotation: players who appeared for the team in its last 15 games plus anyone on the injury report, with the same weights as training.
- **Differences from training** (players listed in the training roster but not in the live one, and the reverse) are exactly what the parity harness measures. Acceptance criteria, fixed before the build: for rows in the traded set, at least 95% of rows have an absolute change in the model's mean of under 0.5% and the mean absolute change in P(over) at the book line is under 0.5 points; the live pipeline's backtest ROI on the replayed dates is within the bootstrap interval of the training-feature ROI. If it fails, the roster rule changes, not the model.

## 4. Tables (SQLite `data/props_live.db`; CSV export for the dashboard)

- `prop_signals`: every evaluated market at each decision time (model P, market P, EV at best/median price, the books' quotes summary); also the future "market explorer".
- `prop_bets`: **immutable**; one row per player-game-stat the first time it qualifies, with price, book, pricing views, and provenance: code commit, model version and hash, dataset hash, rule file hash, snapshot timestamp. A database constraint and a test forbid updates.
- `prop_results`: settlement and CLV, one row per bet.
- `pipeline_health`: job runs.

## 5. Policies to settle before building

1. **Retrain policy.** Proposal: train once before the season on all data through 2025-26 (the same configuration, a new fold, not a new model); refit on a fixed calendar (the first day of each month), never in response to results; every refit is a new immutable version and the paper record carries the version per bet. The team audit showed what happens when nothing retrains; this keeps retraining mechanical.
2. **What counts as the paper record.** All rule-qualified bets at first qualification, flat 1 unit, one per player-game-stat, reported at the median bettable price (default), the best price and flat -110. No hand-picking. An optional personal "I took this" flag is stored locally and never feeds the record.
3. **Start date.** The record starts the day the signal job first runs end to end; earlier games are not back-filled with paper bets (it would be look-ahead). Games the collector missed are logged as missed.
4. **Where it runs.** Your PC under Task Scheduler (like the collector), so it must be on during game windows; one machine, local database, dashboard served locally.

## 6. Build order and what each step delivers

| Step | Delivers | Depends on |
|---|---|---|
| D1.1 | Live feature builder + parity harness on historical dates (go/no-go for everything after) | none |
| D1.2 | `v3_live_2026-27` trained and hashed | none, can run in parallel with D1.1 |
| D1.3 | Signal job writing `prop_signals` and `prop_bets` for the current slate; dashboard Board live | D1.1, D1.2 |
| D1.4 | Settlement job and live Tracker | D1.3 |
| D1.5 | Health page, alerts, scheduled tasks | D1.3 |

Every step ships with tests of the same kind as Phases 2 and 3 (no look-ahead, determinism, immutability, hash refusal, parity), and the full suite stays green.

## 7. Risks

Roster mismatch (section 3) is the main one; injury reports that arrive late or not at all (the model then uses the previous report, as in training with the 15-minute lag); quotes that vanish between flag and placement (not modelled, recorded as price at flag time); thin history for rookies and new arrivals (the shrunk-rate features handle it, and the first weeks are flagged as low-information); the 2026-27 season has already started, so the record starts late by however long D1.3 takes.

## 8. Decisions for you

1. Approve D1.1 and D1.2 now, in parallel (the parity harness is the gating risk).
2. The retrain policy in section 5.1.
3. Confirm the paper-record definition in 5.2 and that the record starts when the job first runs (no back-fill).

## 9. Status 2026-10-22

D1.2 done (`v3_live_2026-27`). D1.1 built and tested; the parity test is reported in [d1_parity_report.md](d1_parity_report.md): the pre-registered criteria are not met (teammate-availability inputs cannot be reproduced before the game because the training roster is the post-game box-score list), A2 and A3 pass, rule v2 coverage is ~100% from November. D1.3-D1.5 wait on the owner's decision in section 5 of that report.

## 10. D1.3-D1.5 as built (2026-10-23)

**Decision (owner, 2026-10-22):** start the live pipeline with the frozen model, the frozen rule and roster rule v2, and waive A1/A4/A5/A6 as preconditions; the gap is disclosed in every report ([d1_parity_report.md](d1_parity_report.md); the dashboard banner and `meta.json` carry the text).

| Piece | Where | Notes |
|---|---|---|
| Pinned configuration | `reports/props/live_config.json`, `props/live/config.py` | rule file hash, model bundle hash, features database size; the job exits 2 and writes a `refused` health row on any mismatch. A monthly refit is a new version and a reviewed edit of this file. |
| Signal job | `scripts/props_signal_job.py`, `props/live/{quotes,signal,store,reports,freshness}.py` | every 10 minutes; one transaction per slot (signals, new bets, slot log); idempotent; `--dry-run`, `--now` for testing. |
| Store | `data/props_live.db` | `prop_bets` is immutable (triggers abort UPDATE and DELETE; unique per game-player-stat); `prop_signals`, `prop_results`, `slot_log`, `pipeline_health`. |
| Settlement | `scripts/props_settle_job.py`, `props/live/settle.py` | box scores -> outcome, three early-exit treatments, profit at best / median / flat -110, CLV to T-5 from the recorded signals. |
| Export and dashboard | `scripts/export_props_dashboard_data.py --live` -> `data/props_dashboard_live/`; sidebar source switch in `dashboard_props/app.py` | same schema as the backtest export; pending bets have blank results. |
| Health | `props/live/health.py`, dashboard Model-health tab | collector age, injury-report age, signal-job age, missed slots, unsettled bets. |

**Slot rule (live differs from the backtest here).** The collector snapshots every 30 minutes plus a closing snapshot, not at exactly T-240/T-60/T-5. A slot with nominal time T uses the snapshot taken up to 10 minutes after T if one exists (the job waits for it), else the freshest snapshot before T if it is at most 40 minutes old, else the slot is missed; a slot is never worked more than 25 minutes after its nominal time; the T-5 slot does not wait. Every other input is limited to what was stamped at or before the snapshot used. The age of the snapshot (`snapshot_minutes`) is stored with every signal and bet.

**Known live limits, all disclosed:** (1) the box-score sync (task "NBA Daily Inference", 09:15 local) runs after the earliest slots of afternoon games, so on days with afternoon tips the T-240 slot of those games is refused (history stale) and missed; moving that task earlier would fix it (not done: it is the team pipeline's task). (2) A quoted player with no listing this season and no report entry is left without a row, so October coverage is low. (3) The record starts when the job first writes a bet; nothing earlier is back-filled. (4) The job runs from this worktree (`C:\Users\jorda\nba-props-integrate`, branch `props-phase4`, not merged to main); the scheduled tasks break if the worktree moves.

**Verification before start.** 9 + 7 + 4 + 3 + 2 unit tests for the slot rule, store, guard, settlement, export, health and freshness; replay of four historical slates through the job's evaluation code reproduced the parity harness's bets (Jaccard 0.978, EV correlation 0.997); one dry run on the live feeds.
