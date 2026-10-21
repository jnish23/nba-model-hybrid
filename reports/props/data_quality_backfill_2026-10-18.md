# Props odds data quality

## 1. Overview

| events | with_game_id | snapshots | player_prop_rows | non_player_rows_excluded | player_id_matched | first_game_date | last_game_date |
|---|---|---|---|---|---|---|---|
| 3958 | 3956 | 19933 | 14221921 | 7 | 1 | 2023-10-24 | 2026-06-13 |

| kind | snapshots | events | snapshot_credits |
|---|---|---|---|
| historical | 19933 | 3958 | 2,167,220 |

## 2. Coverage

### By season and book (share_of_events = events with at least one quote / events in the store)

| season | book | events | share_of_events | player_markets | rows |
|---|---|---|---|---|---|
| 2023-24 | barstool | 59 | 0.045 | 7095 | 28595 |
| 2023-24 | betmgm | 1254 | 0.951 | 128684 | 493602 |
| 2023-24 | betonlineag | 1252 | 0.95 | 84295 | 339720 |
| 2023-24 | betrivers | 1311 | 0.995 | 46751 | 865243 |
| 2023-24 | bovada | 1317 | 0.999 | 124271 | 495328 |
| 2023-24 | draftkings | 1318 | 1 | 126039 | 519646 |
| 2023-24 | fanduel | 1315 | 0.998 | 96324 | 418927 |
| 2023-24 | mybookieag | 1246 | 0.945 | 61062 | 812095 |
| 2023-24 | pointsbetus | 1280 | 0.971 | 110840 | 459708 |
| 2023-24 | unibet_us | 1306 | 0.991 | 46532 | 860984 |
| 2023-24 | williamhill_us | 1318 | 1 | 136320 | 560183 |
| 2023-24 | wynnbet | 110 | 0.083 | 2575 | 9986 |
| 2024-25 | betmgm | 1299 | 0.985 | 152484 | 633599 |
| 2024-25 | betonlineag | 1308 | 0.992 | 84044 | 363027 |
| 2024-25 | betrivers | 1311 | 0.994 | 69524 | 581059 |
| 2024-25 | bovada | 1317 | 0.998 | 93800 | 412894 |
| 2024-25 | draftkings | 1305 | 0.989 | 112788 | 505906 |
| 2024-25 | fanatics | 520 | 0.394 | 14589 | 63633 |
| 2024-25 | fanduel | 1316 | 0.998 | 95755 | 417576 |
| 2024-25 | williamhill_us | 1289 | 0.977 | 135366 | 544827 |
| 2025-26 | betmgm | 1302 | 0.986 | 130865 | 593283 |
| 2025-26 | betonlineag | 1306 | 0.989 | 83996 | 381617 |
| 2025-26 | betrivers | 1284 | 0.972 | 61471 | 511341 |
| 2025-26 | bovada | 1285 | 0.973 | 106918 | 1768201 |
| 2025-26 | draftkings | 1298 | 0.983 | 123297 | 568065 |
| 2025-26 | fanatics | 1283 | 0.971 | 50769 | 224034 |
| 2025-26 | fanduel | 1299 | 0.983 | 108224 | 494363 |
| 2025-26 | williamhill_us | 1312 | 0.993 | 64997 | 291774 |

### By season and market

| season | market | rows | player_markets |
|---|---|---|---|
| 2023-24 | assists | 636115 | 22101 |
| 2023-24 | pa | 354554 | 22475 |
| 2023-24 | points | 1524276 | 24014 |
| 2023-24 | pr | 413655 | 23159 |
| 2023-24 | pra | 920422 | 23050 |
| 2023-24 | ra | 334243 | 22549 |
| 2023-24 | rebounds | 952323 | 24158 |
| 2023-24 | threes | 728429 | 21843 |
| 2024-25 | assists | 374085 | 20643 |
| 2024-25 | pa | 297203 | 19619 |
| 2024-25 | points | 726974 | 21358 |
| 2024-25 | pr | 385915 | 20176 |
| 2024-25 | pra | 520789 | 21157 |
| 2024-25 | ra | 280723 | 19468 |
| 2024-25 | rebounds | 496589 | 21046 |
| 2024-25 | threes | 440243 | 19246 |
| 2025-26 | assists | 465171 | 18761 |
| 2025-26 | pa | 372857 | 17496 |
| 2025-26 | points | 1207882 | 21190 |
| 2025-26 | pr | 481862 | 19117 |
| 2025-26 | pra | 778951 | 20654 |
| 2025-26 | ra | 342043 | 17086 |
| 2025-26 | rebounds | 680424 | 20693 |
| 2025-26 | threes | 503488 | 18126 |

### Selection: box-score players with >= 15 minutes, by trailing-form tier, and the share that had a line in any snapshot (games with a resolved game id only)

| tier | players | priced | share_priced |
|---|---|---|---|
| rotation | 24197 | 17807 | 0.736 |
| star | 9699 | 9688 | 0.999 |
| starter | 28900 | 28481 | 0.986 |

_Tiers: star = trailing 20-game mean points >= 20; starter = mean minutes >= 24; rotation = the rest (games strictly before the date). Unpriced players are the missing-line selection the backtests must respect._

## 3. Snapshot timing

### Historical: requested minutes before tip vs the snapshot actually returned (history is on a 5-minute grid and returns the latest snapshot at or before the requested instant)

| requested | snapshots | median_actual_min | median_early_by | max_early_by | empty |
|---|---|---|---|---|---|
| 5 | 3955 | 9.367 | 4.367 | 1444.367 | 4 |
| 15 | 19 | 19.383 | 4.383 | 4.383 | 0 |
| 30 | 3956 | 34.367 | 4.367 | 1444.367 | 4 |
| 45 | 19 | 49.383 | 4.383 | 4.383 | 0 |
| 60 | 3958 | 64.367 | 4.367 | 1444.367 | 3 |
| 90 | 21 | 94.383 | 4.383 | 4.383 | 0 |
| 120 | 3959 | 124.367 | 4.367 | 1444.367 | 5 |
| 150 | 20 | 154.383 | 4.383 | 4.383 | 0 |
| 180 | 21 | 184.383 | 4.383 | 4.383 | 0 |
| 240 | 3958 | 244.367 | 4.367 | 1444.367 | 4 |
| 300 | 20 | 304.383 | 4.383 | 4.383 | 0 |
| 360 | 20 | 364.383 | 4.383 | 4.383 | 0 |

### How far before the requested instant the returned snapshot is (vendor data gaps show up as large values)

| bucket | snapshots | share |
|---|---|---|
| within 5 min (the grid) | 19884 | 0.998 |
| 5 to 10 min | 1 | 0 |
| 10 to 60 min | 34 | 0.002 |
| over 60 min | 7 | 0 |

- lagged by more than 10 minutes: **41** snapshots in **14** games, of which **5** are empty (no props posted yet at that time)

## 3b. Clock integrity (reference tip = NBA scheduled tip + 0 to 10 minutes)

- snapshots at or after the reference tip (in-game), excluded everywhere below: **7** of 19933
- games whose vendor tip time changed between snapshots: **340** of 3958
- games whose first vendor tip time differs from the reference by more than 15 minutes: **13**
- games not matched to the NBA schedule: **0**

## 4. Missing closes

- live games without a snapshot within 12 minutes of tip: **0** of 0
- historical games without a closing-side (<= T-10) snapshot: **4** of 3958
- historical games whose closing-side snapshot came back empty: **4**

## 5. Quote quality by book (shares of rows)

| book | rows | one_sided_share | no_margin_share | wide_share | stale_share | ladder_rows_share | main_rows |
|---|---|---|---|---|---|---|---|
| barstool | 28595 | 0 | 0 | 0 | 0 | 0 | 28591 |
| betmgm | 1720484 | 0 | 0 | 0 | 0 | 0 | 1720078 |
| betonlineag | 1084364 | 0.001 | 0 | 0 | 0 | 0.001 | 1083934 |
| betrivers | 1957643 | 0.029 | 0 | 0 | 0 | 0.775 | 824948 |
| bovada | 2676423 | 0.001 | 0 | 0 | 0 | 0.651 | 1377387 |
| draftkings | 1593617 | 0 | 0 | 0 | 0 | 0 | 1593557 |
| fanatics | 287667 | 0 | 0 | 0 | 0 | 0 | 287613 |
| fanduel | 1330866 | 0 | 0 | 0 | 0 | 0 | 1330866 |
| mybookieag | 812095 | 0.703 | 0 | 0 | 0 | 0.809 | 247816 |
| pointsbetus | 459708 | 0 | 0 | 0 | 0 | 0 | 459691 |
| unibet_us | 860984 | 0 | 0 | 0 | 0 | 0.956 | 213330 |
| williamhill_us | 1396784 | 0 | 0 | 0 | 0 | 0 | 1396766 |
| wynnbet | 9986 | 0 | 0 | 0 | 0 | 0.005 | 9960 |

_Ladder rows = quotes from a book that posted more than one line for that player and market in the same snapshot; analysis uses the main line (closest to 50%) and keeps the rest._

## 6. Consensus line movement (>= 2 books; consecutive snapshots of the same game, binned by the interval midpoint in minutes before tip; abs_move_per_hour = total |change| / total hours)

| market | window | n | share_moved | mean_abs_move | abs_move_per_hour |
|---|---|---|---|---|---|
| assists | T-240+ | 16 | 0.125 | 0.125 | 0.021 |
| assists | T-180..240 | 47538 | 0.045 | 0.045 | 0.022 |
| assists | T-120..180 | 235 | 0.123 | 0.128 | 0.044 |
| assists | T-90..120 | 48748 | 0.057 | 0.057 | 0.057 |
| assists | T-60..90 | 411 | 0.202 | 0.221 | 0.144 |
| assists | T-30..60 | 52014 | 0.041 | 0.041 | 0.081 |
| assists | T-15..30 | 53287 | 0.045 | 0.045 | 0.109 |
| pa | T-240+ | 14 | 0.429 | 0.429 | 0.072 |
| pa | T-180..240 | 45790 | 0.162 | 0.172 | 0.086 |
| pa | T-120..180 | 235 | 0.387 | 0.587 | 0.204 |
| pa | T-90..120 | 46961 | 0.208 | 0.245 | 0.245 |
| pa | T-60..90 | 409 | 0.523 | 0.922 | 0.598 |
| pa | T-30..60 | 50141 | 0.154 | 0.178 | 0.353 |
| pa | T-15..30 | 51635 | 0.147 | 0.172 | 0.414 |
| points | T-240+ | 546 | 0.161 | 0.168 | 0.147 |
| points | T-180..240 | 51706 | 0.143 | 0.151 | 0.076 |
| points | T-120..180 | 761 | 0.138 | 0.179 | 0.154 |
| points | T-90..120 | 53514 | 0.175 | 0.2 | 0.201 |
| points | T-60..90 | 618 | 0.322 | 0.44 | 0.42 |
| points | T-30..60 | 57513 | 0.126 | 0.14 | 0.28 |
| points | T-15..30 | 59172 | 0.133 | 0.152 | 0.365 |
| points | T-5..15 | 292 | 0.065 | 0.065 | 0.389 |
| pr | T-240+ | 14 | 0.357 | 0.357 | 0.06 |
| pr | T-180..240 | 48363 | 0.182 | 0.195 | 0.098 |
| pr | T-120..180 | 227 | 0.396 | 0.551 | 0.192 |
| pr | T-90..120 | 49879 | 0.224 | 0.263 | 0.264 |
| pr | T-60..90 | 317 | 0.533 | 0.975 | 0.64 |
| pr | T-30..60 | 53229 | 0.169 | 0.196 | 0.39 |
| pr | T-15..30 | 55048 | 0.17 | 0.203 | 0.486 |
| pra | T-240+ | 16 | 0.438 | 0.438 | 0.074 |
| pra | T-180..240 | 50124 | 0.184 | 0.2 | 0.1 |
| pra | T-120..180 | 227 | 0.357 | 0.665 | 0.234 |
| pra | T-90..120 | 51566 | 0.235 | 0.289 | 0.289 |
| pra | T-60..90 | 360 | 0.522 | 1.053 | 0.684 |
| pra | T-30..60 | 55087 | 0.172 | 0.208 | 0.413 |
| pra | T-15..30 | 56868 | 0.174 | 0.218 | 0.522 |
| ra | T-240+ | 13 | 0.077 | 0.077 | 0.013 |
| ra | T-180..240 | 45316 | 0.088 | 0.089 | 0.044 |
| ra | T-120..180 | 239 | 0.222 | 0.259 | 0.089 |
| ra | T-90..120 | 46420 | 0.111 | 0.114 | 0.115 |
| ra | T-60..90 | 422 | 0.346 | 0.45 | 0.29 |
| ra | T-30..60 | 49512 | 0.086 | 0.089 | 0.176 |
| ra | T-15..30 | 50962 | 0.086 | 0.091 | 0.218 |
| rebounds | T-240+ | 536 | 0.062 | 0.062 | 0.053 |
| rebounds | T-180..240 | 50973 | 0.06 | 0.06 | 0.03 |
| rebounds | T-120..180 | 734 | 0.049 | 0.052 | 0.045 |
| rebounds | T-90..120 | 52608 | 0.073 | 0.075 | 0.075 |
| rebounds | T-60..90 | 585 | 0.174 | 0.191 | 0.184 |
| rebounds | T-30..60 | 56537 | 0.054 | 0.055 | 0.11 |
| rebounds | T-15..30 | 58276 | 0.062 | 0.065 | 0.155 |
| rebounds | T-5..15 | 280 | 0.036 | 0.036 | 0.214 |
| threes | T-240+ | 14 | 0.071 | 0.071 | 0.012 |
| threes | T-180..240 | 45811 | 0.022 | 0.022 | 0.011 |
| threes | T-120..180 | 180 | 0.078 | 0.078 | 0.028 |
| threes | T-90..120 | 47234 | 0.028 | 0.028 | 0.028 |
| threes | T-60..90 | 314 | 0.096 | 0.096 | 0.064 |
| threes | T-30..60 | 50460 | 0.021 | 0.021 | 0.042 |
| threes | T-15..30 | 52152 | 0.023 | 0.023 | 0.056 |

## 7. Player-ID matching

- matched rows: 14,221,218 of 14,221,921 (100.00%); unmatched names: 40

| player_name_raw | team_context | reason | n_rows |
|---|---|---|---|
| J. Green | Houston Rockets @ New Orleans Pelicans | ambiguous | 40 |
| J. Green | Charlotte Hornets @ Houston Rockets | ambiguous | 35 |
| J. Green | Sacramento Kings @ Houston Rockets | ambiguous | 35 |
| J. Green | Houston Rockets @ Los Angeles Clippers | ambiguous | 35 |
| J. Green | Denver Nuggets @ Houston Rockets | ambiguous | 35 |
| J. Green | Houston Rockets @ Chicago Bulls | ambiguous | 35 |
| J. Green | Houston Rockets @ Miami Heat | ambiguous | 35 |
| J. Green | Houston Rockets @ Detroit Pistons | ambiguous | 35 |
| D. Robinson | Washington Wizards @ Miami Heat | ambiguous | 28 |
| J. Green | Memphis Grizzlies @ Houston Rockets | ambiguous | 28 |
| J. Green | Houston Rockets @ Minnesota Timberwolves | ambiguous | 28 |
| J. Green | Los Angeles Lakers @ Houston Rockets | ambiguous | 26 |
| J. Green | Philadelphia 76ers @ Houston Rockets | ambiguous | 21 |
| J. Green | Houston Rockets @ Philadelphia 76ers | ambiguous | 21 |
| J. Green | New York Knicks @ Houston Rockets | ambiguous | 21 |
| J. Green | Houston Rockets @ Milwaukee Bucks | ambiguous | 21 |
| J. Green | Houston Rockets @ Golden State Warriors | ambiguous | 14 |
| J. Green | Houston Rockets @ Denver Nuggets | ambiguous | 14 |
| J. Green | Houston Rockets @ Cleveland Cavaliers | ambiguous | 14 |
| J. Green | Atlanta Hawks @ Houston Rockets | ambiguous | 14 |
| J. Green | Brooklyn Nets @ Houston Rockets | ambiguous | 14 |
| J. Green | Utah Jazz @ Houston Rockets | ambiguous | 14 |
| C. Martin | Charlotte Hornets @ Detroit Pistons | ambiguous | 14 |
| J. Green | Houston Rockets @ Brooklyn Nets | ambiguous | 14 |
| J. Green | Houston Rockets @ Indiana Pacers | ambiguous | 14 |
