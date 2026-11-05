# E28 audit - tracking, hustle and schedule data

Registered in `reports/experiments/PREREGISTRATION.md` ("E28-audit") before any number was computed. Descriptive only. Built by `experiments/e28_audit.py`; training set `training_set_v1_reduced_2026-10-19.csv`.

## Summary (numbers computed in this run; the wording was written after reading them)

- **History is complete for team and player tracking**, 2013-14 to 2025-26: every game in `team_game_full` has both teams' rows (team tracking coverage 1.000 in every season). Player distances sum to the team figure within 1% in 0.9998 of team-games. Team `distance` averages 16.8-18.3 miles a game in 2013-14..2024-25 and player speed averages 4.2-4.4 mph, as expected.
- **Hustle**: full from 2016-17. 2013-14 and 2014-15 are fragments (74 and 39 games); in 2015-16 almost every game has a hustle row for only one team (both teams present in 16.5% of games), so hustle features start in 2016-17 (NaN before).
- **Team `speed` is NULL in every row**; team speed has to be built from players (`player_track_stats_v3.speed`, filled for 99%+ of played rows in 2013-14..2024-25).
- **Zero rows**: 458 team-game rows carry no tracking (distance 0 and touches 0). 350 of them are in 2025-26 (13.2% of that season's team-games, 2025-11-26 .. 2026-04-12); every other season has 0 to 20. They must be read as missing (NaN), never as a very low load, and the 2025-26 fold (the last test season) is thinner for any tracking feature. Player tracking in 2025-26 has speed filled for only 87.1% of played rows. **The plan's "2024-25 has a lower fill share (24,964 of 34,788)" is 2025-26**: those counts are the 2025-26 player rows with distance filled; 2024-25 is 99.9% filled.
- **Live availability (decisive)**: the log covers only 5 runs (2026-10-20 to 2026-10-24) and no team had a back-to-back yet, so the share of back-to-back second nights with the previous night's tracking at 9:15 **could not be measured (0 cases)**. What the window shows: on three of the four runs that had games to fetch (10/21, 10/22, 10/23) every previous-night game's team tracking was in the database after the sync, with a lag of exactly one day. On 10/24 **all 12 games of 10/23 failed** (`'NoneType' object has no attribute 'get'`, the endpoint returned nothing), and a read-only probe at about 13:35 ET the same day (more than 12 hours later) still returned nothing for them, while the 10/21 games return normally. Hustle data arrived for all 28 games on schedule, including the 12 games whose tracking failed. Overall 57% (16 of 28 games) had team tracking by the next morning.
- **Reading of the stop rule**: this is not 'usually missing', but it is not 'reliably there' either, and one bad day fell on the 12-game slate. Per the plan, the lagged-load families (F1, F2, F3) are therefore built with the rule **'most recent game with tracking data'** (a game with an all-zero row is skipped as if absent) from the start, in training and in any production use. In history the previous game nearly always has data, so for the walk-forward tests the rule equals 'previous game' outside the 2025-26 gap; the live gap is a step 6 problem and is flagged there. The audit window is far too short to say how often tracking is late; it should be rechecked once a month of daily runs exists.
- **Opening-line timing on back-to-backs**: not measurable yet. `nba_odds_opening` has no timestamps and `odds_snapshots` starts 2026-10-19 (18 games, none a back-to-back second night). Every game already had a line at the collector's first look, 30-47 hours before tip, but that figure is bounded by when the collector started, so it only shows that lines exist at least that early (a 'current' line, not necessarily the opening-table value). For a second night that is before the previous night's game ends, so the open-to-close or opening line can in principle predate the previous game; whether the archived opening values do is unknown, and the market-residual screen has to be read with that in mind.
- **Opening-line coverage**: `nba_odds_opening` itself starts in 2019-20 (19 empty rows in 2013-14). The training set's `odds_total` and `odds_home_spread` for 2013-14..2018-19 come from `data/nba_odds_archive.csv` (opening spread and total columns; closing moneyline only) and are 99.8% plausible after the `odds_clean` bounds (2013-14: 98.5%), so the market-residual screen can use those six seasons as extra discovery data, **labelled as a different source** (same treatment as E14's supplementary seasons).

## 1. Coverage

Seasons are taken from digits 3-4 of the game id. `games_in_tgf` = distinct games in `team_game_full`.

| season | team_games_in_tgf | games_in_tgf | team_track_rows | team_track_games | team_hustle_rows | team_hustle_games | player_track_rows | player_track_games | player_hustle_rows | player_hustle_games | track_cov_vs_tgf | hustle_cov_vs_tgf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013-14 | 2638 | 1319 | 2638 | 1319 | 148 | 74 | 33681 | 1319 | 1540 | 74 | 1.000 | 0.056 |
| 2014-15 | 2622 | 1311 | 2622 | 1311 | 78 | 39 | 33522 | 1311 | 825 | 39 | 1.000 | 0.030 |
| 2015-16 | 2632 | 1316 | 2632 | 1316 | 1527 | 1311 | 33659 | 1316 | 5756 | 1311 | 1.000 | 0.996 |
| 2016-17 | 2618 | 1309 | 2618 | 1309 | 2618 | 1309 | 33610 | 1309 | 27876 | 1309 | 1.000 | 1.000 |
| 2017-18 | 2624 | 1312 | 2624 | 1312 | 2624 | 1312 | 33090 | 1312 | 27836 | 1312 | 1.000 | 1.000 |
| 2018-19 | 2624 | 1312 | 2624 | 1312 | 2624 | 1312 | 32928 | 1312 | 27862 | 1312 | 1.000 | 1.000 |
| 2019-20 | 2286 | 1143 | 2286 | 1143 | 2286 | 1143 | 28707 | 1143 | 24105 | 1143 | 1.000 | 1.000 |
| 2020-21 | 2342 | 1171 | 2342 | 1171 | 2342 | 1171 | 31546 | 1171 | 25054 | 1171 | 1.000 | 1.000 |
| 2021-22 | 2646 | 1323 | 2646 | 1323 | 2646 | 1323 | 33848 | 1323 | 28050 | 1323 | 1.000 | 1.000 |
| 2022-23 | 2640 | 1320 | 2640 | 1320 | 2800 | 1400 | 34024 | 1320 | 29901 | 1400 | 1.000 | 1.061 |
| 2023-24 | 2636 | 1318 | 2636 | 1318 | 2636 | 1318 | 34815 | 1318 | 28206 | 1318 | 1.000 | 1.000 |
| 2024-25 | 2640 | 1320 | 2640 | 1320 | 2640 | 1320 | 35087 | 1320 | 27586 | 1320 | 1.000 | 1.000 |
| 2025-26 | 2642 | 1321 | 2642 | 1321 | 2637 | 1321 | 34788 | 1321 | 26954 | 1321 | 1.000 | 1.000 |
| 2026-27 | 56 | 28 | 32 | 16 | 56 | 28 | 553 | 16 | 542 | 28 | 0.571 | 1.000 |

Share of games with both teams present:

| season | team_track | team_hustle |
|---|---|---|
| 2013-14 | 1.000 | 1.000 |
| 2014-15 | 1.000 | 1.000 |
| 2015-16 | 1.000 | 0.165 |
| 2016-17 | 1.000 | 1.000 |
| 2017-18 | 1.000 | 1.000 |
| 2018-19 | 1.000 | 1.000 |
| 2019-20 | 1.000 | 1.000 |
| 2020-21 | 1.000 | 1.000 |
| 2021-22 | 1.000 | 1.000 |
| 2022-23 | 1.000 | 1.000 |
| 2023-24 | 1.000 | 1.000 |
| 2024-25 | 1.000 | 1.000 |
| 2025-26 | 1.000 | 0.996 |
| 2026-27 | 1.000 | 1.000 |

Game kinds (`002` regular season, `004` playoffs, `005` play-in, others as found):

| kind | team_track_games | team_hustle_games | in_team_game_full |
|---|---|---|---|
| 002 | 15685 | 13322 | 15697 |
| 004 | 1089 | 942 | 1089 |
| 005 | 37 | 37 | 37 |

All-zero team tracking rows (distance 0 and touches 0; present in the table but carrying no data, to be treated as missing):

| season | team_rows | all_zero_rows | all_zero_share |
|---|---|---|---|
| 2013-14 | 2638 | 8 | 0.003 |
| 2014-15 | 2622 | 6 | 0.002 |
| 2015-16 | 2632 | 4 | 0.002 |
| 2016-17 | 2618 | 8 | 0.003 |
| 2017-18 | 2624 | 12 | 0.005 |
| 2018-19 | 2624 | 10 | 0.004 |
| 2019-20 | 2286 | 12 | 0.005 |
| 2020-21 | 2342 | 10 | 0.004 |
| 2021-22 | 2646 | 10 | 0.004 |
| 2022-23 | 2640 | 8 | 0.003 |
| 2023-24 | 2636 | 20 | 0.008 |
| 2024-25 | 2640 | 0 | 0.000 |
| 2025-26 | 2642 | 350 | 0.132 |
| 2026-27 | 32 | 0 | 0.000 |

2025-26 all-zero rows by month (span 2025-11-26 .. 2026-04-12):

| month | all_zero_team_rows |
|---|---|
| 2025-11 | 4 |
| 2026-01 | 96 |
| 2026-02 | 94 |
| 2026-03 | 108 |
| 2026-04 | 48 |

Hustle game ids not in `team_game_full` (preseason `001`, all-star `003` and malformed ids; ignored by any join):

| season | kind | games |
|---|---|---|
| 2022-23 | 001 | 70 |
| 2022-23 | 003 | 5 |
| 2022-23 | 202 | 2 |
| 2022-23 | 204 | 1 |
| 2022-23 | 205 | 2 |

Games in `team_game_full` that are missing from tracking / hustle:

| season | kind | team_track_missing | team_hustle_missing |
|---|---|---|---|
| 2026-27 | 002 | 12 | 0 |
| 2013-14 | 002 | 0 | 1166 |
| 2013-14 | 004 | 0 | 79 |
| 2014-15 | 002 | 0 | 1204 |
| 2014-15 | 004 | 0 | 68 |
| 2015-16 | 002 | 0 | 5 |

Player tracking fill (a played row = minutes > 0; the plan flagged 2024-25):

| season | rows | dnp_share | played_rows | speed_filled | dist_filled | speed_fill_share_of_played | dist_fill_share_of_played | rows_with_speed | rows_with_speed_share_all |
|---|---|---|---|---|---|---|---|---|---|
| 2013-14 | 33681 | 0.183 | 27516 | 27429 | 27396 | 0.997 | 0.996 | 27429 | 0.814 |
| 2014-15 | 33522 | 0.176 | 27630 | 27559 | 27525 | 0.997 | 0.996 | 27559 | 0.822 |
| 2015-16 | 33659 | 0.169 | 27970 | 27917 | 27896 | 0.998 | 0.997 | 27917 | 0.829 |
| 2016-17 | 33610 | 0.171 | 27872 | 27774 | 27754 | 0.996 | 0.996 | 27774 | 0.826 |
| 2017-18 | 33090 | 0.159 | 27828 | 27669 | 27653 | 0.994 | 0.994 | 27669 | 0.836 |
| 2018-19 | 32928 | 0.154 | 27855 | 27723 | 27714 | 0.995 | 0.995 | 27723 | 0.842 |
| 2019-20 | 28707 | 0.161 | 24097 | 23640 | 23635 | 0.981 | 0.981 | 23642 | 0.824 |
| 2020-21 | 31546 | 0.206 | 25053 | 24882 | 24875 | 0.993 | 0.993 | 24882 | 0.789 |
| 2021-22 | 33848 | 0.171 | 28045 | 27878 | 27875 | 0.994 | 0.994 | 27878 | 0.824 |
| 2022-23 | 34024 | 0.185 | 27737 | 27646 | 27635 | 0.997 | 0.996 | 27646 | 0.813 |
| 2023-24 | 34815 | 0.190 | 28197 | 27930 | 27898 | 0.991 | 0.989 | 27936 | 0.802 |
| 2024-25 | 35087 | 0.195 | 28236 | 28217 | 28204 | 0.999 | 0.999 | 28218 | 0.804 |
| 2025-26 | 34788 | 0.175 | 28695 | 24980 | 24964 | 0.871 | 0.870 | 24981 | 0.718 |
| 2026-27 | 553 | 0.336 | 367 | 366 | 366 | 0.997 | 0.997 | 366 | 0.662 |

Special seasons: {"bubble_games_2019_20": 172, "bubble_games_with_team_track": 172, "bubble_games_with_team_hustle": 172, "games_2020_21": 1171, "games_2020_21_with_team_track": 1171}

## 2. Units and sanity

Team distance (miles, regular season) by season:

| season | count | mean | std | min | max | team_min_mean |
|---|---|---|---|---|---|---|
| 2013-14 | 2452 | 16.884 | 0.756 | 13.070 | 21.780 | 241.958 |
| 2014-15 | 2454 | 16.827 | 0.694 | 14.670 | 22.730 | 241.956 |
| 2015-16 | 2456 | 16.982 | 0.668 | 15.000 | 23.110 | 241.792 |
| 2016-17 | 2452 | 16.984 | 0.694 | 13.830 | 22.980 | 241.631 |
| 2017-18 | 2448 | 17.243 | 0.924 | 0.680 | 21.520 | 241.409 |
| 2018-19 | 2450 | 18.180 | 0.937 | 0.000 | 25.910 | 241.653 |
| 2019-20 | 2106 | 17.968 | 0.796 | 12.790 | 21.920 | 241.733 |
| 2020-21 | 2150 | 17.994 | 1.000 | 2.210 | 21.880 | 241.442 |
| 2021-22 | 2450 | 17.900 | 0.697 | 15.840 | 23.420 | 241.367 |
| 2022-23 | 2452 | 17.941 | 0.792 | 12.710 | 22.330 | 241.815 |
| 2023-24 | 2440 | 18.200 | 1.160 | 0.010 | 22.580 | 241.352 |
| 2024-25 | 2460 | 18.338 | 1.051 | 0.040 | 23.320 | 241.301 |
| 2025-26 | 2110 | 17.908 | 1.054 | 14.560 | 22.200 | 241.374 |
| 2026-27 | 32 | 17.744 | 0.627 | 15.750 | 18.950 | 240.000 |

Team speed (plan says NULL everywhere):

| rows | speed_non_null | speed_distinct_values |
|---|---|---|
| 33622 | 0 | [] |

Player speed (mph) among regular-season rows with 5+ minutes and speed > 0:

| season | count | mean | std | min | 1% | 50% | 99% | max |
|---|---|---|---|---|---|---|---|---|
| 2013-14 | 23738.000 | 4.228 | 0.272 | 3.300 | 3.620 | 4.230 | 4.880 | 5.370 |
| 2014-15 | 24261.000 | 4.204 | 0.266 | 3.280 | 3.610 | 4.200 | 4.840 | 5.280 |
| 2015-16 | 24334.000 | 4.249 | 0.266 | 3.100 | 3.650 | 4.250 | 4.890 | 5.560 |
| 2016-17 | 24297.000 | 4.255 | 0.264 | 2.980 | 3.640 | 4.250 | 4.890 | 5.490 |
| 2017-18 | 24288.000 | 4.335 | 0.275 | 3.100 | 3.660 | 4.340 | 4.980 | 5.630 |
| 2018-19 | 24260.000 | 4.251 | 0.260 | 3.210 | 3.610 | 4.250 | 4.860 | 5.470 |
| 2019-20 | 20632.000 | 4.212 | 0.253 | 3.180 | 3.590 | 4.210 | 4.810 | 5.420 |
| 2020-21 | 21083.000 | 4.233 | 0.255 | 3.200 | 3.610 | 4.240 | 4.830 | 5.320 |
| 2021-22 | 23841.000 | 4.190 | 0.250 | 3.310 | 3.610 | 4.190 | 4.790 | 5.250 |
| 2022-23 | 23846.000 | 4.185 | 0.258 | 3.000 | 3.560 | 4.190 | 4.780 | 5.230 |
| 2023-24 | 23764.000 | 4.258 | 0.288 | 0.200 | 3.590 | 4.260 | 4.910 | 5.380 |
| 2024-25 | 23980.000 | 4.293 | 0.281 | 2.310 | 3.610 | 4.300 | 4.970 | 5.620 |
| 2025-26 | 21050.000 | 4.356 | 0.291 | 3.190 | 3.650 | 4.360 | 5.050 | 5.690 |
| 2026-27 | 329.000 | 4.501 | 0.286 | 3.630 | 3.843 | 4.510 | 5.296 | 5.390 |

Outlier counts: `{"distance_zero_with_minutes_gt_0": 5708, "distance_zero_with_minutes_gt_5": 4757, "speed_gt_6": 179, "speed_lt_2_with_minutes_ge_5": 4757, "rows_minutes_unparsed_nonblank": 0, "team_distance_zero": 460, "team_distance_gt_30": 0, "team_minutes_lt_235": 0}`

Sum of player distances against team `distance` (share within 1% in column `within_1pct`):

| season | team_games | within_1pct | median_ratio | p05 | p95 |
|---|---|---|---|---|---|
| 2013-14 | 2630 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2014-15 | 2616 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2015-16 | 2628 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2016-17 | 2610 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2017-18 | 2612 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2018-19 | 2612 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2019-20 | 2274 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2020-21 | 2332 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2021-22 | 2636 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2022-23 | 2632 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2023-24 | 2616 | 0.999 | 1.000 | 0.999 | 1.001 |
| 2024-25 | 2640 | 0.999 | 1.000 | 0.999 | 1.001 |
| 2025-26 | 2292 | 1.000 | 1.000 | 0.999 | 1.001 |
| 2026-27 | 32 | 1.000 | 1.000 | 0.999 | 1.002 |

Overall share within 1%: 1.000. Player minutes summed vs team minutes: `{"median_abs_diff": 0.0, "share_within_1min": 1.0}`

## 3. Live availability at the 9:15 ET run

Log `data\raw\daily_inference.log`: 2026-10-20 .. 2026-10-24 (5 runs). Only 2026-27 games are covered, so this is a short window and is stated as such.

Per table, how the 2026-27 games reached the database (`lag1_next_morning` = added by the next morning's 9:15 run; `not_in_db_now` = still absent):

| table | games_played_2026_27 | in_db_now | added_in_logged_runs | lag0_same_day_as_game | lag1_next_morning | lag2plus | errored_at_least_once | not_in_db_now |
|---|---|---|---|---|---|---|---|---|
| team_track_stats_v3 | 28 | 16 | 16.000 | 0.000 | 16.000 | 0.000 | 12.000 | 12 |
| player_track_stats_v3 | 28 | 16 |  |  |  |  |  | 12 |
| team_hustle_stats | 28 | 28 | 28.000 | 0.000 | 28.000 | 0.000 | 0.000 | 0 |
| player_hustle_stats | 28 | 28 |  |  |  |  |  | 0 |

Missing-game counts reported by each run:

| run_date | team_advanced_stats_v3 | team_hustle_stats | team_scoring_stats_v3 | team_track_stats_v3 | team_usage_stats_v3 |
|---|---|---|---|---|---|
| 2026-10-21 | 3 | 3 | 3 | 3 | 3 |
| 2026-10-22 | 11 | 11 | 11 | 11 | 11 |
| 2026-10-23 | 2 | 2 | 2 | 2 | 2 |
| 2026-10-24 | 12 | 12 | 12 | 12 | 12 |

Errors:

| table | error | n | games | first_run | last_run |
|---|---|---|---|---|---|
| team_track_stats_v3 | 'NoneType' object has no attribute 'get' | 12 | 12 | 2026-10-24 | 2026-10-24 |

Previous night's games with team tracking in the database after each run's sync:

| run_date | previous_night_games | with_team_tracking_after_sync | share |
|---|---|---|---|
| 2026-10-21 | 3 | 3 | 1.000 |
| 2026-10-22 | 11 | 11 | 1.000 |
| 2026-10-23 | 2 | 2 | 1.000 |
| 2026-10-24 | 12 | 0 | 0.000 |

Overall: 0.571 of previous-night games had team tracking after the sync.

Back-to-back second nights in the log window: `{"b2b_second_nights_in_window": 0, "prev_game_team_tracking_in_by_that_morning": 0}`

Historical completeness, regular-season back-to-back second nights whose previous game has team tracking in the database at all (not the same as available at 9:15, which history cannot show):

| season | b2b_second_nights | prev_has_team_track |
|---|---|---|
| 2013-14 | 562 | 1.000 |
| 2014-15 | 580 | 1.000 |
| 2015-16 | 534 | 1.000 |
| 2016-17 | 491 | 1.000 |
| 2017-18 | 433 | 1.000 |
| 2018-19 | 398 | 1.000 |
| 2019-20 | 315 | 1.000 |
| 2020-21 | 457 | 1.000 |
| 2021-22 | 423 | 1.000 |
| 2022-23 | 401 | 1.000 |
| 2023-24 | 422 | 1.000 |
| 2024-25 | 450 | 1.000 |
| 2025-26 | 441 | 1.000 |

Per game, team tracking, 2026-27:

| gameId | gameDate | first_added_run | lag_days | ever_error | in_db_now |
|---|---|---|---|---|---|
| 0022600001 | 2026-10-20 00:00:00 | 2026-10-21 | 1.000 | False | True |
| 0022600002 | 2026-10-20 00:00:00 | 2026-10-21 | 1.000 | False | True |
| 0022600003 | 2026-10-20 00:00:00 | 2026-10-21 | 1.000 | False | True |
| 0022600004 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600005 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600006 | 2026-10-22 00:00:00 | 2026-10-23 | 1.000 | False | True |
| 0022600007 | 2026-10-22 00:00:00 | 2026-10-23 | 1.000 | False | True |
| 0022600008 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600009 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600085 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600086 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600087 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600088 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600089 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600090 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600091 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600092 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600093 | 2026-10-21 00:00:00 | 2026-10-22 | 1.000 | False | True |
| 0022600094 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600095 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600096 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600097 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600098 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600099 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600100 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600101 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600102 | 2026-10-23 00:00:00 |  |  | True | False |
| 0022600103 | 2026-10-23 00:00:00 |  |  | True | False |


## 4. Opening-line timing on back-to-backs

`nba_odds_opening` has no timestamps. `odds_snapshots` covers: `{"rows": 6273, "first_fetch": "2026-10-19 09:00:01", "last_fetch": "2026-10-23 23:30:02", "games": 18}`

Hours between the first pre-game snapshot of a game and its tip: `{"count": 18.0, "mean": 42.5, "std": 4.44, "min": 30.0, "25%": 43.0, "50%": 43.5, "75%": 45.25, "max": 46.5}`

Back-to-back second nights inside the snapshot window: `{"n": 0, "first_snapshot_before_prev_game_end": 0, "note": "snapshot history starts 2026-10-19, so only second nights whose previous game is also inside the snapshot window can be checked; previous-game end is estimated as tip + 2.5 h"}`

_(empty)_

Not measurable: when the historical opening lines (`nba_odds_opening`) were posted relative to the previous game ending.

## 5. Opening-line coverage

`nba_odds_opening` by season:

| yr | rows | with_total | with_spread | games_in_team_game_full |
|---|---|---|---|---|
| 2013-14 | 19.000 | 0.000 | 0.000 | 1319 |
| 2014-15 | 0.000 | 0.000 | 0.000 | 1311 |
| 2015-16 | 0.000 | 0.000 | 0.000 | 1316 |
| 2016-17 | 0.000 | 0.000 | 0.000 | 1309 |
| 2017-18 | 0.000 | 0.000 | 0.000 | 1312 |
| 2018-19 | 0.000 | 0.000 | 0.000 | 1312 |
| 2019-20 | 1144.000 | 1142.000 | 1142.000 | 1143 |
| 2020-21 | 1205.000 | 1171.000 | 1171.000 | 1171 |
| 2021-22 | 1334.000 | 1323.000 | 1323.000 | 1323 |
| 2022-23 | 1321.000 | 1320.000 | 1320.000 | 1320 |
| 2023-24 | 1318.000 | 1306.000 | 1306.000 | 1318 |
| 2024-25 | 1325.000 | 1325.000 | 1325.000 | 1320 |
| 2025-26 | 1324.000 | 1324.000 | 1324.000 | 1321 |
| 2026-27 | 28.000 | 28.000 | 28.000 | 28 |

Opening odds carried in the training set (`odds_total`, `odds_home_spread`), with the plausibility bounds of `core/odds_clean.py`:

| seasonYear | games | odds_nonnull | odds_plausible | plausible_share |
|---|---|---|---|---|
| 2013-14 | 1319 | 1299 | 1299 | 0.985 |
| 2014-15 | 1311 | 1310 | 1309 | 0.998 |
| 2015-16 | 1316 | 1315 | 1315 | 0.999 |
| 2016-17 | 1309 | 1308 | 1307 | 0.998 |
| 2017-18 | 1312 | 1311 | 1311 | 0.999 |
| 2018-19 | 1312 | 1311 | 1311 | 0.999 |
| 2019-20 | 1143 | 1140 | 1140 | 0.997 |
| 2020-21 | 1171 | 1171 | 1171 | 1.000 |
| 2021-22 | 1323 | 1323 | 1323 | 1.000 |
| 2022-23 | 1320 | 1320 | 1320 | 1.000 |
| 2023-24 | 1318 | 1304 | 1304 | 0.989 |
| 2024-25 | 1320 | 1318 | 1318 | 0.998 |
| 2025-26 | 1321 | 1321 | 1321 | 1.000 |


## Addendum (manual, not produced by the script): read-only probes of the tracking endpoint on 2026-10-24

`BoxScorePlayerTrackV3` was called directly (no database write) for games 0022600094, 0022600008 and 0022600100 (all played the night of 10-23) at about 13:35 ET and again at about 14:05 ET: the endpoint still returned nothing (`'NoneType' object has no attribute 'get'`), roughly 14-15 hours after tip. A 10-21 game (0022600085) returned normally. So the 10-24 failure was not a late-morning timing miss that a later run would have fixed; it is a gap of at least a day. Whether it clears is not known yet, and one slate is the only evidence of lateness the log window holds.
