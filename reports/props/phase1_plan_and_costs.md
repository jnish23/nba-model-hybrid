# Phase 1 plan: prop odds data (planning only, nothing built or spent)

Status: draft for approval. No API key exists in the repo, no account was created, no request was made.
All vendor facts below are from the vendors' public docs, fetched 2026-10-18. Items marked **unverified** come from
third-party listings and must be checked on the vendor's own pricing page before any money is spent.

## 1. What exists today (relevant facts)

- `core/snapshot_odds.py` scrapes **SBR** (team lines only: spread, moneyline, total for 6 books) into `odds_snapshots`
  in `nba_stats.db`. It has no player props, and **the table does not exist yet**: the only run (2026-10-10, preseason)
  got no data. There is no timestamped line history at all.
- SBR cannot supply props. Live props need a paid or free-tier API key from some vendor, and the key must live in an
  environment variable (`ODDS_API_KEY`), never in the repo or in anything I see.
- Games to cover (from `team_game_full`): 2023-24: 1,312, 2024-25: 1,314, 2025-26: 1,315 (regular season + playoffs),
  3,941 in total; 3,690 regular season only. About 205 regular-season games per month.

## 2. Vendor comparison

| | The Odds API | DunkAPI | OddsPapi |
|---|---|---|---|
| Historical props start | 2023-05-03 (documented) | May 2023 (**unverified**, third-party listing) | claimed, NBA depth **unverified** |
| Snapshot spacing | 5 min since Sep 2022 | unknown | unknown |
| Price | Free 500 credits; $30 / 20K; **$59 / 100K**; **$119 / 5M**; $249 / 15M per month | Starter $19 (10K credits), Pro $49 (30K), Scale $149 (125K), **unverified** | unverified; says historical is free-tier |
| Credit rule | Live: 1 per market per region per event. Historical: **10x** that. Responses with no data are free | unknown | unknown |
| US books | `us`: DraftKings, FanDuel, BetMGM, Caesars*, BetRivers, Fanatics*, Bovada, BetOnline, LowVig, BetUS, MyBookie; `us2`: theScore Bet, Hard Rock, Bally, betPARX, others (*paid plans only) | n/a | claims 117+ books for NBA |
| Sharp reference (Pinnacle) | not in the `us` region; would need the `eu` region (extra credits); whether it lists NBA props is **unverified** | n/a | claims Pinnacle |

Recommendation: **The Odds API.** It is the only one where I can state the credit formula and the history start date from
its own documentation, and it covers both the live collector and the backfill with one key. DunkAPI and OddsPapi are
worth a free-tier look only if you want a cross-check; I cannot create accounts for you.

## 3. Credit model (The Odds API)

- Live props: `/events/{id}/odds`, cost = markets returned x regions, per event per snapshot.
- Historical: `/historical/sports/basketball_nba/events/{id}/odds`, cost = **10 x** markets returned x regions.
  The per-day historical events lookup costs 1 credit (about 600 calls in total, negligible).
- Markets: 4 core (`player_points`, `player_rebounds`, `player_assists`, `player_threes`) + 4 combos (`player_points_rebounds_assists`,
  `player_points_rebounds`, `player_points_assists`, `player_rebounds_assists`) + 3 team lines (`h2h`, `spreads`, `totals`,
  needed on the same timestamps for H2) = 11. Alternate-line markets are excluded (extra credits, and the main-line spec).
- All numbers below are **upper bounds**: only markets that actually return data are charged. A small pilot (section 6) measures
  the real cost per call before the big run.

### 3a. Backfill, 2023-24 to 2025-26, 3,941 games, 1 region (`us`)

Credits = games x snapshots x 10 x markets. Snapshot sets: close only; open+close; open, T-90m, close; open, T-3h, T-90m, close.

| Markets | 1 snapshot | 2 | 3 | 4 |
|---|---|---|---|---|
| 2 (points, rebounds) | 78,820 | 157,640 | 236,460 | 315,280 |
| 4 core | 157,640 | 315,280 | 472,920 | 630,560 |
| 8 (core + combos) | 315,280 | 630,560 | 945,840 | 1,261,120 |
| 11 (8 + team lines) | 433,510 | 867,020 | 1,300,530 | 1,734,040 |

Second region (`us2`, 5 more books): double every cell. Only 2023-24 as a pilot season (1,312 games), 8 markets: 105K / 210K / 315K / 420K.

Dollar cost: the monthly plans are the only cost lever. 100K credits ($59) only covers the smallest cells; **anything above
~100K needs the 5M plan ($119 for one month)**, which covers every cell in the 1-region table with room to spare. Cheapest
sensible full run = one month of the 5M plan = **$119**, if it can be cancelled or downgraded after one month
(**unverified**, check the account terms). The 15M plan ($249) is not needed even for 2 regions x 11 markets x 4 snapshots (3.47M fits in 5M).

### 3b. Live collection, per month of regular season (~205 games), no 10x

Snapshots per game: 30-min cadence over the window, 10-min cadence in the last 2 hours where marked.

| Window | Snaps/game | 4 mkts 1 reg | 8 mkts 1 reg | 8 mkts 2 reg | 11 mkts 1 reg | 11 mkts 2 reg |
|---|---|---|---|---|---|---|
| last 3h @30m | 6 | 4,920 | 9,840 | 19,680 | 13,530 | 27,060 |
| last 6h @30m | 12 | 9,840 | 19,680 | 39,360 | 27,060 | 54,120 |
| last 6h @30m, last 2h @10m | 20 | 16,400 | 32,800 | 65,600 | 45,100 | 90,200 |
| last 10h @30m, last 2h @10m | 28 | 22,960 | 45,920 | 91,840 | 63,140 | 126,280 |

The **100K plan ($59)** covers every cell except 11 markets x 2 regions in the two 20+-snapshot rows (90K and 126K are the edge: 126K overflows).

## 4. Proposed build (Phase 1 tasks, none started)

1. **Storage:** `data/props_odds.db` (SQLite, separate from `nba_stats.db`), table `prop_snapshots` with the columns in your spec
   plus `ingest_ts_utc`, `commence_time_utc`, `home_team`, `away_team`, `raw_market_key`, and a `team_lines` table keyed on the
   same `snapshot_ts_utc` and `event_id_vendor`. All times stored UTC; an ET conversion helper lives in `core/`.
2. **Vendor-agnostic loader:** `core/odds_vendor/` with a small `Vendor` protocol (`list_events`, `event_odds`, `historical_*`) and a
   `TheOddsApi` adapter that records the raw JSON response to disk (gzip) before parsing, so a parser fix never costs credits again.
3. **Collector:** `scripts/props/snapshot_props.py`, a separate entry point (see question 4), 30-min cadence, 10-min in the last 2 h,
   a hard per-day credit cap, and reading the remaining-credits response headers to stop itself.
4. **Player-ID mapping:** deterministic matcher on normalised name + team + date window against `player_info` / box scores, a manual
   override CSV, an `unmatched_names` log. Target 99%+ of prop rows; report failures.
5. **De-vig:** multiplicative and power (Shin as an option), per book and consensus; flags for one-sided, stale, and crossed lines.
6. **Quality report:** coverage by season, book, market, player tier; snapshot timing relative to tip; missing closes; line movement.
7. Tests: schema, ID matcher, de-vig properties (probabilities sum to 1, monotone), parser against saved fixtures, UTC/ET handling.

Everything in items 1, 2 (parser), 4, 5 and 7 can be built and tested offline against fixtures without spending a credit.
The collector needs a key for its first live run, and the backfill needs your decisions below.

## 5. Open risks

- **Collection start.** The season opens 2026-10-20. Missed live snapshots can be recovered later with the historical endpoint, but
  at 10x the credit cost, and the 5-minute snapshot grid means the recovery is approximate rather than exact.
- **"Open" is not observable** from a fixed grid. Books post props on game morning at different times. I propose defining open as the
  first snapshot at or after 09:00 ET game day in which the line exists, and recording how late it first appears.
- **Pulled and missing lines are information.** The loader will store "present at T-90m, absent at close" explicitly instead of dropping rows.
- Pinnacle as the sharp reference may not be available for NBA props at all; the plan falls back to a consensus of `us` books
  (CLV against the consensus close) unless the pilot shows `eu` coverage.
- Merge of Phase 0 is on hold at your request; Phase 1 work happens on the same worktree branch.

## 6. Decisions needed (see the questions sent with this plan)

Vendor and plan; live cadence, markets and regions; backfill scope and snapshot density; separate scheduled task vs extending the
existing job; who creates the account and sets `ODDS_API_KEY`. Suggested first paid step is a **pilot of about 50 calls** on the
chosen plan (a few thousand credits at most) to measure real per-call cost and book/market coverage before the full backfill.

## 7. Update 2026-10-18: prototype on the $30 / 20,000-credit plan

Decision (yours): start on the 20K plan to prove the collector and see what the data contains, then upgrade for the historical backfill.
Built so far on this branch (all tested offline, 121 tests pass): `core/odds_vendor` (Odds API adapter, persisted credit ledger with
per-run / per-day / total caps checked before every request, raw responses saved gzip without the key), `core/devig.py`
(multiplicative, power, Shin, quality flags, consensus), `core/timeutil.py`, `props/parse.py`, `props/odds_store.py`
(`data/props_odds.db`), `props/player_ids.py` (matcher, override CSV, unmatched log, game-id resolver), `props/collect.py`
(cadence planner) and `scripts/snapshot_props.py`. Nothing is scheduled and no request has been made.

Prototype budget (20K credits; caps default to 600 per run, 2,000 per day, 18,000 total, leaving a 2,000 reserve):
- Live, full spec (11 markets, 1 region, 30 min over 6 h, 10 min in the last 2 h): about 1,760 credits per 8-game day.
  Four game days = about 7K.
- Historical pilot: 3 game dates x 2 games x 3 snapshots (T-180, T-90, T-5) x 11 markets x 10 = about 2K.
- That leaves about 10K for retries and a second look. Do not leave the live task running past about 8 game days on this plan.

## 8. Pilot result (2026-10-18): Memphis @ Philadelphia, 2026-03-10, snapshots T-180 / T-90 / T-5

Cost: 331 credits (1 for the events listing + 110 per snapshot), 19,669 left. 110 = 10 x 11 markets x 1 region, so the
upper-bound formula in section 3 is exact when all 11 markets are requested and returned.

What came back (1,709 prop rows, 33 team-line rows): 8 books with props (BetMGM, BetOnline, BetRivers, Bovada, DraftKings,
Fanatics, FanDuel, Caesars) and 11 with team lines (adds BetUS, LowVig, MyBookie). All 1,709 rows matched to a player id
(100%), the event resolved to NBA game 0022500936, no one-sided or crossed quotes. Each book's own last-update time was within
~1 minute of the snapshot (Bovada up to 4 min). The requested "T-5" snapshot came back 9.2 minutes before tip: history is on a
5-minute grid and returns the latest snapshot at or before the requested instant, so "close" means about 5 to 10 minutes pre-tip.

Findings that change the design:
1. **Ladders under the main market key.** Bovada posts a ladder of lines for the same player and market (92% of its player-markets
   have 2 to 8 lines, e.g. points 13.5 to 20.5 with prices from -290 to +190); BetRivers does it for 38%. They are not marked
   alternate by the vendor. The store keeps every row, but main-line analysis needs a rule. Proposed: per snapshot, book, player
   and market, the main line is the one whose de-vigged over probability is closest to 50%, flagged `is_main_line`, with ladders
   kept for H3 (calibration at specific lines).
2. **Props exist for only about 15 players per game, and the set grows toward tip** (11, then 14, then 15 players priced across the
   three snapshots). Four players who played 22 to 27 minutes had no line at all. Lines appearing, moving and being absent are
   the H1 data; the snapshot log and the raw files keep that, nothing is imputed.
3. **This game was unusually thin:** neither team's stars played (the box score has only the depleted rotations), so the 15
   priced players are those rotations, not a typical slate. Read the coverage numbers as a floor.
4. **Book coverage is uneven:** Fanatics prices one player, BetOnline and Caesars have no PRA-style combos for most players,
   BetMGM had no rebounds at T-5. A consensus must require a minimum number of books per player-market.
5. Team lines moved from -3.0 to about -4.5 (home spread) between T-180 and T-5 with a flat total, and the featured player's points
   line moved 16.5 to 17.5 at most books: the aligned data H2 needs is present.

## 9. Density pilot (2026-10-18): where do lines and coverage change relative to tip?

Design: 20 games (5 spread evenly across the slate on each of 2026-03-01, 03-17, 03-21, 03-25), points and rebounds only,
12 snapshots per game at T-360, 300, 240, 180, 150, 120, 90, 60, 45, 30, 15, 5 minutes (denser near tip). Cost 4,804 credits
(240 calls x 20, plus 4 events listings); with the first pilot, 5,135 of 20,000 credits used, 14,865 left. Reproduce with
`scripts/analyze_snapshot_density.py`. Main line = the line whose de-vigged over probability is closest to 50% (needed because of the
ladders); consensus = median of the books' main lines (at least 2 books; about 6.4 books per player-market on average).
**Caveats:** 20 games, two markets, about 250 to 290 player-markets per interval, correlated within games; no confidence intervals
were computed, so read the numbers as shape, not precision. Line changes undercount news that only moves the juice, so section 2b below
also tracks price moves at an unchanged line.

**CORRECTION (same day).** The first version of this section measured minutes to tip with the vendor's tip time. For one game (Spurs @ Kings) the
vendor's historical listing gave a tip 70 minutes late, so its "T-45 to T-5" snapshots were really in-game (the last one had 6 quotes left).
Everything below is recomputed on the reference clock (NBA scheduled tip + 0 to 10 minutes, `props/schedule.py`) with in-game snapshots dropped
(4 of 243; 19 games remain with the full 12-snapshot grid). Three claims in the first version were wrong and are replaced: the movement peak at
T-45 to T-30 (0.76 points/hour), the 0.6 lines per game pulled between T-15 and T-5, and the books' spread widening to 0.57.

Results (points; rebounds show the same shape at roughly 40 to 50% of the size):
- **Coverage:** 13.7 players have a points line at T-6h, rising to 15.5 by T-5. New lines arrive mostly between T-120 and T-45 (0.68, 0.63 and 0.37
  new players per game in the three intervals, vs 0.05 to 0.26 in every earlier one). Once in-game snapshots are removed, no lines disappear
  between T-15 and T-5.
- **Movement intensity** (mean |change in consensus line| per hour): 0.13 to 0.15 points/hour from T-6h to T-2.5h, 0.42 at T-120 to T-90,
  then 0.25, 0.33, 0.40, 0.49 and 0.44 over the five intervals from T-90 to T-5. It rises toward tip and plateaus at roughly 0.4 to 0.5 per hour from
  T-45 to T-5; there is no sharp peak. About 56% of the total line movement from T-6h to T-5 happens in the last two hours and 30% in the
  last hour. Price moves at an unchanged line (about 0.3 to 0.5 percentage points even early) rise the same way and are largest from T-30 to T-5.
- **Residual movement still to come before the close** (mean |close - line| in points, share of players whose consensus line still moves):
  T-240: 0.42 (36%); T-120: 0.34 (29%); T-90: 0.22 (22%); T-60: 0.19 (19%); T-30: 0.15 (16%); T-15: 0.07 (9%). A T-60 snapshot is
  therefore not close to the close, and a T-15 snapshot still misses 0.07 points.
- **Direction:** the average consensus line change is slightly negative after T-120 (about -0.02 points per interval), so lines drift down
  toward tip on average. 19 games is not evidence of a systematic shade; it is a thing to test with the full backfill.
- **Books diverge slightly at the very end:** the spread between the highest and lowest book line is 0.42 to 0.47 points from T-6h to T-30, and
  0.50 at T-15 and 0.53 at T-5, consistent with books updating at different speeds on late news.

Implication for the backfill grid (not yet run): the information is concentrated in roughly T-120 to T-5 with a steady, high rate over the last
45 minutes, so T-60 alone is too early and T-15 alone too late. Credits do not constrain this choice: every grid I priced fits in one month of the 5M plan, at 433,510 credits per snapshot for 3,941 games and 11 markets (4 snapshots 1.73M, 5 snapshots 2.17M, 6 snapshots 2.60M).
Suggested: **T-240, T-120, T-60, T-30, T-5** (5 snapshots, 2.17M credits, 43% of the 5M plan).
Suggested live setting while on the 20K plan: hourly from T-5h to T-2h, then every 10 minutes (about 16 snapshots and 176 credits per
game, about 1,400 per 8-game day, roughly 10 game days).

Also in this step: four vendor nicknames (Nicolas Claxton, Herb Jones, Moe Wagner, Ron Holland) were missing; they are now in
`props/player_overrides.csv` with ids looked up in the NBA database. Match rate before the overrides was 99.2% of 69,052 rows, 100% after.

## 10. Built after the density pilot (2026-10-18)

- `props/analysis.py`: de-vig (multiplicative, power, Shin), quality flags (one-sided, crossed, wide, stale), **main-line selection** per
  snapshot, book, player and market (ladders kept, flagged), and a same-line consensus probability that needs at least N books.
- `props/tiers.py`: star / starter / rotation from the 20 games before each game date (thresholds fixed in the file).
- `props/schedule.py` and `scripts/annotate_props.py`: **reference clock** = NBA scheduled tip + 0 to 10 minutes, the offset taken from the vendor's earliest tip estimate when it is within 15 minutes of the NBA time, else 10 (cached schedule per season),
  `event_ref` and `snapshot_ref` tables, in-game snapshots flagged and excluded, vendor tip revisions counted. This came out of the
  Spurs @ Kings finding (vendor listing 70 minutes late): the pilot report, the density analysis and the backfill all use it now.
- `props/quality.py` and `scripts/report_props_quality.py`: the data-quality report (coverage by season, book, market and tier, timing,
  clock integrity, missing closes, quote quality, line movement, ID matching), chunked so it scales to the full backfill. First run on
  the pilot data: `reports/props/data_quality_pilot_2026-10-18.md`.
- `props/backfill.py` and `scripts/backfill_props.py`: plan first (spends nothing), run only with `--confirm-credits <exact planned figure>`,
  resumable, never requests an instant at or after the reference tip. Plan for 2023-24 through 2025-26 at T-240/120/60/30/5, 11 markets,
  1 region: **19,779 snapshots, 2,176,327 credits** (3,976 games incl. play-in and Cup; 101 snapshots already stored are skipped).

Live collector: unchanged and still running from the same files; the vendor's live tip time matched the NBA schedule to within 10 minutes for all 46
upcoming games on 2026-10-18 (35 at exactly +10 minutes after the NBA time, 11 at +0, none outside that range), so the live clock is sound.
