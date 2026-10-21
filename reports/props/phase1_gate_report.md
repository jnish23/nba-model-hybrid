# Phase 1 gate report: prop odds data

Status: ready for your review. Date: 2026-10-18. Branch `props-phase0-core-refactor` (worktree `nba-model-hybrid-props`), Phase 1 not merged to `main`.
Detailed numbers: [`data_quality_backfill_2026-10-18.md`](data_quality_backfill_2026-10-18.md) (three seasons), [`data_quality_pilot_2026-10-18.md`](data_quality_pilot_2026-10-18.md)
(pilot), [`phase1_plan_and_costs.md`](phase1_plan_and_costs.md) (vendor choice, costs, density pilot).

## 1. Gate criteria

| Criterion (from the spec) | Status |
|---|---|
| Separate props store | **Done.** `data/props_odds.db` (SQLite, 6.5 GB, 14.2M quotes); `nba_stats.db` untouched |
| Live collector, 30-minute cadence, team lines on the same timestamps | **Built and scheduled** (Task Scheduler "NBA Props Snapshots", every 10 min; per game: a snapshot every 30 min over the last 4 h plus a closing one). **Not yet seen on a real game day.** First check is 2026-10-20/21 |
| Historical backfill, vendor-agnostic loader | **Done for 2023-24 to 2025-26**: 19,933 snapshots over 3,958 games at T-240, 120, 60, 30, 5 minutes, 11 markets (8 prop markets + game spread, total, moneyline), 1 region (8 to 11 books). Loader is resumable and plans cost before spending |
| Player-ID mapping, 99%+ | **99.995%** (703 unresolved of 14.2M quotes, 0.005%; list in section 4) |
| De-vig (multiplicative, power, Shin), per book and consensus, flags | **Done** (`core/devig.py`, `props/analysis.py`), with one-sided, crossed, wide and stale flags and a same-line consensus requiring N books |
| Data-quality report | **Done** for the backfill; live sections fill in after Oct 20 |

## 2. What is in the store

- **Coverage:** 3,956 of the 3,976 scheduled games in the window have props in at least one snapshot. The other 20: 17 All-Star and Rising Stars exhibition games
  (no player props exist), one real game with no vendor event (Bulls vs Pistons, 2025-02-11), and two with an event but no props posted. Two NBA Cup games
  (Nov 29, 2024) lost their T-30 snapshot to an API "event not found" error. (The planner now skips the exhibition game types.)
- **Books differ by season.** Constant core: DraftKings, FanDuel, BetMGM, Caesars (`williamhill_us`), BetRivers, Bovada, BetOnline (each on 95% to 100% of
  games). 2023-24 also has PointsBet, Unibet, MyBookie, Barstool (5% of games), WynnBet (8%), all gone in later seasons. Fanatics appears in 2024-25
  (39% of games) and covers 97% of 2025-26. **No Pinnacle or other sharp reference.** CLV will be measured against a consensus of the core books.
- **Players priced:** star (trailing 20-game mean points of 20 or more) 99.9%, starter 98.6%, rotation 73.6% of box-score players with at least 15 minutes.
  A quarter of rotation players never get a line, so every backtest must treat "no line" as missing, not as zero.
- **Quote quality:** one-sided quotes are rare except MyBookie (70% of its rows). Bovada, BetRivers and Unibet post ladders of lines under the main market key
  (65%, 78% and 96% of their rows), so analysis uses a main line per snapshot, book, player and market (the rung closest to 50%) and keeps the ladder.
  No quote had a book last-update time more than 60 minutes before its snapshot (the stale flag fired on none).

## 3. Problems found and fixed during Phase 1 (these changed results)

1. **The vendor's historical tip time can be wrong.** One pilot game was listed 70 minutes late, so its "T-45 to T-5" snapshots were in-game; 340 of 3,958 games
   (8.6%) have a tip time that changes between snapshots, and 13 start more than 15 minutes from the NBA schedule. All timing now uses the NBA's own
   schedule plus 0 to 10 minutes (the vendor's live convention). In-game snapshots are flagged and excluded: 7 of 19,933. The first density-pilot numbers
   were wrong because of this and were corrected in the report.
2. **Snapshot staleness.** 99.8% of historical snapshots are within 5 minutes of the requested instant. 41 (14 games) are 10 to 1,444 minutes older: vendor data
   gaps; 5 of them are empty.
3. **Name matching.** The first loose rules turned "Trayce Jackson" into three other Jacksons and "J. Green" into AJ Green; those were found by reviewing the
   audit table and fixed (last-name-only rule removed, a token-subset rule added, fuzzy matching limited to long names, team-specific overrides). Team-total
   ladders posted under a player market (Bovada "Alternate Total", 7 rows) are tagged and excluded.
4. **Concurrency.** The live collector and the backfill both write to one SQLite file; a lock killed the first backfill attempt after 4 minutes. Fixed with a
   longer wait, retries, partial indexes and a pending check; the second run completed with no data lost.

## 4. Known gaps (not blocking Phase 2)

- 703 quotes unresolved: "J. Green" on Houston (605 rows; Jalen and Jeff Green were both on that roster at different times, so a team-only override would be
  wrong), "D. Robinson", "C. Martin", "D. Wade", "T. Young" and seven two-way players with one row each. They stay `player_id = NULL` and out of any player-level analysis.
- Alternate lines are not collected (main market keys only). One region (`us`) only; the `us2` books are not covered.
- Per-book rules for DNP and voids are not yet documented (a Phase 3 requirement).
- Oct 20 is the first time the live collector meets real games; unmatched rookies are expected on the first day.

## 5. What the data can and cannot test (hypotheses)

| | Testable now? | Notes |
|---|---|---|
| H1 news lag | Partly | Same-day status changes between snapshots (T-240 to T-5) can be studied; the official injury report's earlier updates fall outside the grid, and 5 snapshots is a coarse event-study clock. The live collector (30-minute spacing) gives finer timing from 2026-27 |
| H2 team-line lag | Yes | Spread, total and moneyline sit on the same timestamps as props, 11 books |
| H3 distribution shape / calibration at specific lines | Yes | Ladders from Bovada and BetRivers give many lines per player; outcomes are in the box scores |
| H4 opponent effects | Yes | Needs only outcomes and market lines |

Early signal, 19 pilot games and the full data (descriptive, not a test): consensus line movement rises toward tip. In the three-season data the average absolute
move per hour in the points market goes 0.08 (T-240 to T-120), 0.20 (T-120 to T-60), 0.28 (T-60 to T-30), 0.37 (T-30 to T-5); threes and assists move much less.

## 6. Spend

Plan: 5M credits this cycle (the upgrade reset the quota; the 5,135 credits of earlier pilots were on the old 20K plan). Used so far: 2,163,058, all of it the
backfill (the one-week stage, the first attempt that stopped on the lock, and the completed run), and it matches the vendor's own count to the credit. The live
collector has spent nothing yet because no game has been within four hours. Remaining: **2,836,942**. Live collection will cost about 100 credits per game
(about 20,000 a month in season).

## 7. Recommendation

**Pass the gate with one condition.** The historical data is complete, matched and quality-checked, so Phase 2 (player-game dataset and features) can start now and
does not depend on live collection. The condition: the live collector must be confirmed on Oct 20/21 before we rely on it for forward validation; if it fails, I fix it
before anything else. Not yet decided, and needing you: (a) whether to spend about 8 credits to check if the `eu` region carries NBA props from a sharp book (better CLV
reference), (b) merging Phase 1 into `main` (no rush; the live task runs from the worktree).
