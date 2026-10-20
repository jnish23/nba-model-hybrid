# Props odds data quality

## 1. Overview

| events | with_game_id | snapshots | prop_rows | player_id_matched | first_game_date | last_game_date |
|---|---|---|---|---|---|---|
| 21 | 21 | 243 | 69052 | 1 | 2026-03-01 | 2026-03-25 |

| kind | snapshots | events | snapshot_credits |
|---|---|---|---|
| historical | 243 | 21 | 5130 |

## 2. Coverage

### By season and book (share_of_events = events with at least one quote / events in the store)

| season | book | events | share_of_events | player_markets | rows |
|---|---|---|---|---|---|
| 2025-26 | betmgm | 21 | 1 | 646 | 6030 |
| 2025-26 | betonlineag | 21 | 1 | 667 | 6466 |
| 2025-26 | betrivers | 21 | 1 | 539 | 9500 |
| 2025-26 | bovada | 21 | 1 | 621 | 27568 |
| 2025-26 | draftkings | 21 | 1 | 677 | 6392 |
| 2025-26 | fanatics | 21 | 1 | 80 | 487 |
| 2025-26 | fanduel | 21 | 1 | 660 | 6095 |
| 2025-26 | williamhill_us | 21 | 1 | 600 | 5918 |

### By season and market

| season | market | rows | player_markets |
|---|---|---|---|
| 2025-26 | assists | 161 | 13 |
| 2025-26 | pa | 125 | 13 |
| 2025-26 | points | 42956 | 336 |
| 2025-26 | pr | 162 | 15 |
| 2025-26 | pra | 203 | 14 |
| 2025-26 | ra | 110 | 13 |
| 2025-26 | rebounds | 24528 | 325 |
| 2025-26 | threes | 211 | 11 |

### Selection: box-score players with >= 15 minutes, by trailing-form tier, and the share that had a line in any snapshot (games with a resolved game id only)

| tier | players | priced | share_priced |
|---|---|---|---|
| rotation | 153 | 107 | 0.699 |
| star | 38 | 38 | 1 |
| starter | 161 | 157 | 0.975 |

_Tiers: star = trailing 20-game mean points >= 20; starter = mean minutes >= 24; rotation = the rest (games strictly before the date). Unpriced players are the missing-line selection the backtests must respect._

## 3. Snapshot timing

### Historical: requested minutes before tip vs the snapshot actually returned (history is on a 5-minute grid and returns the latest snapshot at or before the requested instant)

| requested | snapshots | median_actual_min | median_early_by | max_early_by | empty |
|---|---|---|---|---|---|
| 5 | 20 | 9.367 | 4.367 | 4.383 | 0 |
| 15 | 19 | 19.383 | 4.383 | 4.383 | 0 |
| 30 | 19 | 34.383 | 4.383 | 4.383 | 0 |
| 45 | 19 | 49.383 | 4.383 | 4.383 | 0 |
| 60 | 20 | 64.383 | 4.383 | 4.383 | 0 |
| 90 | 21 | 94.383 | 4.383 | 4.383 | 0 |
| 120 | 20 | 124.367 | 4.367 | 4.383 | 0 |
| 150 | 20 | 154.383 | 4.383 | 4.383 | 0 |
| 180 | 21 | 184.383 | 4.383 | 4.383 | 0 |
| 240 | 20 | 244.383 | 4.383 | 4.383 | 0 |
| 300 | 20 | 304.383 | 4.383 | 4.383 | 0 |
| 360 | 20 | 364.383 | 4.383 | 4.383 | 0 |

## 3b. Clock integrity (reference tip = NBA scheduled tip + 0 to 10 minutes)

- snapshots at or after the reference tip (in-game), excluded everywhere below: **4** of 243
- games whose vendor tip time changed between snapshots: **3** of 21
- games whose first vendor tip time differs from the reference by more than 15 minutes: **1**
- games not matched to the NBA schedule: **0**

## 4. Missing closes

- live games without a snapshot within 12 minutes of tip: **0** of 0
- historical games without a closing-side (<= T-10) snapshot: **1** of 21
- historical games whose closing-side snapshot came back empty: **0**

## 5. Quote quality by book (shares of rows)

| book | rows | one_sided_share | no_margin_share | wide_share | stale_share | ladder_rows_share | main_rows |
|---|---|---|---|---|---|---|---|
| betmgm | 6030 | 0 | 0 | 0 | 0 | 0 | 6030 |
| betonlineag | 6466 | 0 | 0 | 0 | 0 | 0.001 | 6464 |
| betrivers | 9500 | 0 | 0 | 0 | 0 | 0.7 | 5544 |
| bovada | 27568 | 0 | 0 | 0 | 0 | 0.998 | 5817 |
| draftkings | 6392 | 0 | 0 | 0 | 0 | 0 | 6392 |
| fanatics | 487 | 0 | 0 | 0 | 0 | 0 | 487 |
| fanduel | 6095 | 0 | 0 | 0 | 0 | 0 | 6095 |
| williamhill_us | 5918 | 0 | 0 | 0 | 0 | 0 | 5918 |

_Ladder rows = quotes from a book that posted more than one line for that player and market in the same snapshot; analysis uses the main line (closest to 50%) and keeps the rest._

## 6. Consensus line movement (>= 2 books; consecutive snapshots of the same game, binned by the interval midpoint in minutes before tip; abs_move_per_hour = total |change| / total hours)

| market | window | n | share_moved | mean_abs_move | abs_move_per_hour |
|---|---|---|---|---|---|
| assists | T-120..180 | 10 | 0.1 | 0.1 | 0.067 |
| assists | T-30..60 | 12 | 0 | 0 | 0 |
| pa | T-120..180 | 10 | 0.4 | 0.4 | 0.267 |
| pa | T-30..60 | 12 | 0.333 | 0.333 | 0.235 |
| points | T-240+ | 530 | 0.157 | 0.164 | 0.163 |
| points | T-180..240 | 276 | 0.13 | 0.134 | 0.134 |
| points | T-120..180 | 555 | 0.094 | 0.095 | 0.176 |
| points | T-90..120 | 274 | 0.212 | 0.248 | 0.491 |
| points | T-60..90 | 284 | 0.144 | 0.158 | 0.31 |
| points | T-30..60 | 595 | 0.096 | 0.096 | 0.339 |
| points | T-15..30 | 300 | 0.137 | 0.14 | 0.539 |
| points | T-5..15 | 292 | 0.065 | 0.065 | 0.389 |
| pr | T-120..180 | 10 | 0.4 | 0.4 | 0.267 |
| pr | T-90..120 | 1 | 1 | 1 | 0.343 |
| pr | T-30..60 | 12 | 0.417 | 0.417 | 0.294 |
| pra | T-120..180 | 9 | 0.444 | 0.444 | 0.296 |
| pra | T-90..120 | 1 | 1 | 1 | 0.343 |
| pra | T-30..60 | 11 | 0.364 | 0.364 | 0.256 |
| ra | T-120..180 | 9 | 0.222 | 0.222 | 0.148 |
| ra | T-90..120 | 1 | 0 | 0 | 0 |
| ra | T-30..60 | 11 | 0.273 | 0.273 | 0.192 |
| rebounds | T-240+ | 520 | 0.06 | 0.06 | 0.059 |
| rebounds | T-180..240 | 270 | 0.033 | 0.033 | 0.033 |
| rebounds | T-120..180 | 543 | 0.033 | 0.033 | 0.061 |
| rebounds | T-90..120 | 268 | 0.112 | 0.112 | 0.221 |
| rebounds | T-60..90 | 275 | 0.073 | 0.073 | 0.142 |
| rebounds | T-30..60 | 572 | 0.045 | 0.045 | 0.161 |
| rebounds | T-15..30 | 287 | 0.07 | 0.07 | 0.268 |
| rebounds | T-5..15 | 280 | 0.036 | 0.036 | 0.214 |
| threes | T-120..180 | 9 | 0.222 | 0.222 | 0.148 |
| threes | T-30..60 | 11 | 0.091 | 0.091 | 0.064 |

## 7. Player-ID matching

- matched rows: 69,052 of 69,052 (100.00%); unmatched names: 0

_none_
