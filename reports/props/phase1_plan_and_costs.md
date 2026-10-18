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
