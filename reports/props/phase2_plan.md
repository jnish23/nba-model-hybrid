# Phase 2 plan: player-game dataset and features

Status: plan for your decisions (section 5). Nothing built yet. Phase 1 gate is approved on condition that the live collector is confirmed on Oct 20/21;
Phase 2 is offline and adds new modules only, so it cannot disturb the collector.

## 1. Output

One row per (player, game, decision time T), every feature as of T and nothing later, built by one function that the live pipeline will later call unchanged
(`props/features/`). Deliverables at the gate: the dataset (parquet), a feature dictionary (name, group, formula, source table, availability by season), leakage
tests (including "shuffling future rows changes no past feature"), parity tests, and a short report.

## 2. What the data offers (checked 2026-10-18)

| Need | Source | Coverage | Caveat |
|---|---|---|---|
| Player box score, minutes | `player_basic_stats` | 2013-14 to 2025-26 | ~28K rows with minutes per season; 190K for 2019-20 to 2025-26 (245K with DNP rows) |
| Usage, rebound chances, touches, passes, potential assists, contested shots | `player_usage_stats_v3`, `player_track_stats_v3`, `player_advanced_stats_v3`, `player_scoring_stats_v3` | 2013-14 on, all four complete | Hustle is sparse in early seasons; I will not depend on it |
| Shot-location mix | `player_shot_location_boxscores` | per game | feeds 3PA share and rim share |
| DNP reasons | `comment` in the v3 tables ("DNP - Coach's Decision", "DND - Injury/Illness", rest, suspension, not with team) | full | separates coach decisions from injuries; also the void-rule labels for Phase 3 |
| **Starting lineup** | `position` in the v3 tables (F/G/C = starter, blank = bench) | full | **post-game truth, not knowable at T**; see decision D2 |
| Injury status at T | parsed official injury reports, 750K rows | 2019-10 on, hundreds of reports per half-season, publish time `report_ts` (ET) | only listed players have a status; "not listed" means no report entry |
| Status to play probability | `models/saved/injury_status_play_rates.json` | per season, built from earlier seasons only | measured at tip minus 30 min; the rate at T-240 differs, so I will estimate it per T (decision-free default below) |
| Team form, pace, ratings, shot profile allowed, rebound chances allowed | `team_game_full`, `team_shot_location_boxscores`, `team_track_stats_v3` | full | team level only; no player-vs-team splits |
| Travel, rest, altitude | `core/travel_builder.py` | full | reused as is |
| Game spread, total, moneyline at T | `team_lines` (our snapshots) | 2023-24 on, 11 books, T-240/120/60/30/5 | **nothing at T for 2019-20 to 2022-23**; see decision D4 |
| Player prop lines at T | `prop_snapshots` | 2023-24 on | **never a feature of the base model** (below) |

## 3. Dataset design

- **Rows:** every player with a box-score row, played or DNP, from 2019-20 (the injury-report era; also Phase 4's start). DNP rows are kept with `played = 0` and the DNP reason,
  because the minutes model must predict who plays and the void rules depend on it. About 245K player-games, three decision times, so about 735K rows. Small enough for parquet files by season.
- **Two tables, not one:** `player_game_base` (features that do not depend on T: rate and minutes history, rest, opponent, schedule) and `player_game_at_T`
  (features that do: injury status and probability, teammate availability, market game environment, minutes to tip). Joined by `(player_id, game_id)`.
- **Prop lines are not features.** Phase 5's information test asks whether the model adds anything beyond the line, so the base model never sees the line. A separate
  "market-aware" variant can be built later for an ablation.
- **Rate history reaches back to 2013-14** for cold starts; only rows from 2019-20 on enter the dataset.

## 4. Feature groups

1. **Game environment** (T-dependent): spread, total, implied team totals, blowout probability from the spread (normal approximation, fixed sd, documented), consensus across the core books at T, plus the books' spread of opinion. Team-model predictions only as a separate ablation set.
2. **Minutes inputs:** exponentially weighted minutes at three half-lives (5, 15, 40 games); share of recent games started (the role proxy, using games before the date); minutes in blowouts versus close games (blowout sensitivity from the player's own history); rest days, back-to-back, games since returning from an absence (from the `comment` and gaps); foul rate; the player's own status probability at T.
3. **Teammate availability** (T-dependent, the core of H1): for each player-game, from the latest injury report at or before T, each teammate's play probability times that teammate's trailing minutes, usage, rebound share, assist share and BPM. Detail in D3.
4. **Rate features** per minute or per possession: FGA, 3PA, FTA, usage, rebound chances, contested-rebound share, potential assists, touches, passes, shot-location mix. Exponentially weighted at the same half-lives, shrunk toward a position and minutes-bucket prior (pseudo-count of 10 games, to be tuned on 2019-22 only), carried across seasons with a fixed offseason decay.
5. **Opponent (team only):** pace, defensive rating, shot profile and FG% by zone allowed, offensive and defensive rebound rates, turnover and assist rates allowed, all exponentially weighted and strictly before the game. No player-versus-team features anywhere.
6. **Schedule:** the existing travel, rest and altitude features, unchanged.

Point-in-time rule for every group: only games with a date strictly before the game date, injury reports published at or before T, market snapshots at or before T. Anything from the game itself (minutes, starters, box score) is a label, never a feature.

## 5. Decisions I need from you

**D1. Decision-time grid.** The spec says open, T-90 and close; our grid is T-240, 120, 60, 30, 5 and the true open is not observable.
 (a) **T-240 ("early"), T-60 ("mid"), T-5 ("close")** (recommended): T-60 is the nearest grid point to the spec's T-90 that still sees most injury and lineup news; the code takes T as a parameter, so other points can be added cheaply.
 (b) T-240, T-120, T-5: closer to the spec's T-90 but before most news.
 (c) all five.

**D2. Starting lineups.** The only lineup data is the post-game starter flag. (a) **Never use actual starters; model the role from past starts and injuries** (recommended: knowable at every T and the same in production). (b) Allow the actual starters at T-5 only: lineups are public about 30 minutes before tip, but we have no live lineup feed, so production could not reproduce the feature, and it would mostly restate what the close already prices.

**D3. Teammate on/off.** (a) **Weighted-missing-share features** (recommended first): teammates' play probability times their trailing usage, minutes, rebound and assist shares, no pairwise modelling. (b) Add explicit with/without shifts per player pair, estimated with shrinkage (the spec asks for this). It is heavier and noisier, so my suggestion is to build (a) now, run the Phase 4 ablation, and build (b) only if (a) shows the group matters. Say if you want (b) built now.

**D4. Market game environment before 2023-24.** We have spread and total at T only from 2023-24. (a) **Base features for all seasons; market-environment features from 2023-24, with a flagged proxy for 2019-22** (recommended): SBR opening line for T-240 and SBR closing line for T-5, a column `env_source` saying which, and nothing for T-60. More training years for the ablation, clearly labelled. (b) Market features 2023-24 on only: cleaner, but the market-aware variant then has two training seasons.

Defaults I will apply unless you object: rows from 2019-20 including DNPs; status-to-play-probability estimated per T from earlier seasons only, shrunk toward the spec's rates (0, 2%, 57%, 93%); half-lives 5, 15, 40 games; position from `player_info` plus a minutes bucket for the priors; tiers as defined in `props/tiers.py`; no prop lines as features.

## 6. Tests

- **Truncation test:** rebuild features for sampled player-games from a database truncated to rows before T and compare with the full build; they must match exactly.
- **Shuffle test:** permute every row dated after T (box scores, reports, snapshots) and confirm no feature at T changes.
- **Forbidden-source test:** features may not reference label columns (`minutes`, `points`, `rebounds`, `assists`, `position`, `comment` of the target game).
- **Parity test:** the training builder and the function a future live pipeline calls are the same code; the test calls both on historical dates and compares.
- **Coverage tests:** every feature has a dictionary entry, availability by season is recorded, and nulls are the documented ones.

## 7. Build order

1. Skeleton and labels (rows, `played`, DNP reason, targets).  2. Minutes and rate histories with shrinkage.  3. Opponent and schedule.  4. Injury status at T and teammate availability.
5. Market environment.  6. Dictionary, tests, report.  Each step ends with its tests passing before the next.

## 8. Not doing in this phase

Models (Phase 3), any evaluation against outcomes (Phase 4) or against the market (Phase 5), player-versus-team splits, live lineups, same-game correlations.
