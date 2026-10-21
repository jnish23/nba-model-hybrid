# Phase 2 dataset report

## 1. Size

232,782 player-games (every listed player, played or not), 698,346 rows at the three decision times (T-240, T-60, T-5 minutes before the reference tip), tables joined on (player_id, game_id [, minutes_before]).

| season | player_games | played | games | players | played_share |
|---|---|---|---|---|---|
| 2019-20 | 28577 | 24006 | 1138 | 536 | 0.84 |
| 2020-21 | 31676 | 25150 | 1176 | 545 | 0.794 |
| 2021-22 | 33848 | 28046 | 1323 | 620 | 0.829 |
| 2022-23 | 34024 | 27737 | 1320 | 548 | 0.815 |
| 2023-24 | 34815 | 28203 | 1318 | 586 | 0.81 |
| 2024-25 | 35087 | 28237 | 1320 | 577 | 0.805 |
| 2025-26 | 34755 | 28696 | 1321 | 591 | 0.826 |

## 2. Features by group

223 features in total; each documented in `phase2_feature_dictionary.csv` with availability by season.

| table | group | features |
|---|---|---|
| player_game_at_T | injury | 6 |
| player_game_at_T | market_env | 11 |
| player_game_at_T | teammates | 15 |
| player_game_base | conversion | 6 |
| player_game_base | minutes | 11 |
| player_game_base | opponent | 37 |
| player_game_base | rates | 72 |
| player_game_base | rest | 4 |
| player_game_base | role | 5 |
| player_game_base | schedule | 19 |
| player_game_base | team_context | 37 |

## 3. Missing values by season (share null)

Players with no earlier game have no minutes history (NaN, not zero); their per-minute rates fall back to the league prior, so those are never null.

| season | m_min_h15 | r_points_h15 | c_fg3_pct_h15 | tm_off_rtg | opp_zone_ra_share_allowed |
|---|---|---|---|---|---|
| 2019-20 | 0.008 | 0 | 0 | 0 | 0 |
| 2020-21 | 0.007 | 0 | 0 | 0 | 0 |
| 2021-22 | 0.008 | 0 | 0 | 0 | 0 |
| 2022-23 | 0.006 | 0 | 0 | 0 | 0 |
| 2023-24 | 0.006 | 0 | 0 | 0 | 0 |
| 2024-25 | 0.006 | 0 | 0 | 0 | 0 |
| 2025-26 | 0.006 | 0 | 0 | 0 | 0 |

## 4. Injury information

### Staleness of the report used (minutes before T). The archive is hourly from 2021-22 and sparse before; production will see fresher reports than the 2019-21 training rows did.

| season | median age at T-5 (min) | median age at T-60 (min) | median age at T-240 (min) | share with no report covering the game |
|---|---|---|---|---|
| 2019-20 | 125 | 70 | 130 | 0.001 |
| 2020-21 | 95 | 70 | 130 | 0 |
| 2021-22 | 35 | 40 | 40 | 0 |
| 2022-23 | 35 | 40 | 40 | 0 |
| 2023-24 | 35 | 40 | 40 | 0.001 |
| 2024-25 | 35 | 40 | 40 | 0 |
| 2025-26 | 20 | 10 | 10 | 0 |

### P(played) by reported status and decision time (all seasons, realized)

| inj_status | 5 | 60 | 240 |
|---|---|---|---|
| Available | 0.871 | 0.844 | 0.802 |
| Doubtful | 0.073 | 0.069 | 0.12 |
| None | 0.837 | 0.836 | 0.835 |
| Out | 0.006 | 0.015 | 0.052 |
| Probable | 0.922 | 0.933 | 0.93 |
| Questionable | 0.716 | 0.732 | 0.732 |
| no_report | 0.6 | 0.6 | 0.836 |

### Status mix at T-60

| inj_status | share |
|---|---|
| None | 0.907 |
| Available | 0.032 |
| Questionable | 0.025 |
| Out | 0.021 |
| Probable | 0.014 |
| Doubtful | 0.001 |
| no_report | 0 |

### Calibration of the play probability (T-60, 2023-24 onward, by decile)

| decile | predicted | realized | n |
|---|---|---|---|
| (0.0015063, 0.5373] | 0.399 | 0.409 | 11545 |
| (0.5373, 0.54147] | 0.54 | 0.561 | 12902 |
| (0.54147, 0.77845] | 0.77 | 0.783 | 11012 |
| (0.77845, 0.78267] | 0.783 | 0.797 | 8885 |
| (0.78267, 0.78481] | 0.785 | 0.811 | 8897 |
| (0.78481, 0.95414] | 0.932 | 0.94 | 9739 |
| (0.95414, 0.95865] | 0.957 | 0.97 | 15919 |
| (0.95865, 0.99535] | 0.994 | 0.994 | 10566 |
| (0.99535, 0.99544] | 0.995 | 0.998 | 7913 |
| (0.99544, 0.99593] | 0.996 | 0.998 | 7279 |

_The probability for a season is estimated from earlier seasons only (checked by a test that shuffles the labels)._

## 5. Market game environment: where it comes from

0 none, 1 our snapshot at T, 2 SBR opening line (proxy for T-240), 3 SBR closing line (proxy for T-5).

| season | minutes_before | env_source shares |
|---|---|---|
| 2019-20 | 5 | {3: 1.0} |
| 2019-20 | 60 | {0: 1.0} |
| 2019-20 | 240 | {2: 1.0} |
| 2020-21 | 5 | {3: 1.0} |
| 2020-21 | 60 | {0: 1.0} |
| 2020-21 | 240 | {2: 1.0} |
| 2021-22 | 5 | {3: 1.0} |
| 2021-22 | 60 | {0: 1.0} |
| 2021-22 | 240 | {2: 1.0} |
| 2022-23 | 5 | {3: 1.0} |
| 2022-23 | 60 | {0: 1.0} |
| 2022-23 | 240 | {2: 1.0} |
| 2023-24 | 5 | {1: 1.0, 0: 0.0} |
| 2023-24 | 60 | {1: 1.0, 0: 0.0} |
| 2023-24 | 240 | {1: 1.0, 0: 0.0} |
| 2024-25 | 5 | {1: 1.0, 0: 0.0} |
| 2024-25 | 60 | {1: 1.0, 0: 0.0} |
| 2024-25 | 240 | {1: 1.0, 0: 0.0} |
| 2025-26 | 5 | {1: 0.98, 0: 0.02} |
| 2025-26 | 60 | {1: 0.98, 0: 0.02} |
| 2025-26 | 240 | {1: 0.98, 0: 0.02} |

Proxy check on 3,918 games with both: the T-5 snapshot spread differs from the SBR closing spread by 0.20 points on average (total: 0.32); the T-240 snapshot differs from the SBR opening spread by 1.22 points. The closing proxy is close; the opening proxy is a looser stand-in for T-240.

## 6. Do the features carry signal? (Spearman rank correlation with the realized value, players who played)

A plain sanity check that nothing is broken or reversed, **not** a model evaluation (that is Phase 4).

| feature | target | spearman |
|---|---|---|
| m_min_h15 | lbl_minutes | 0.761 |
| role_start_share_h15 | lbl_minutes | 0.673 |
| exp_points = rate x minutes | y_points | 0.694 |
| exp_reb = rate x minutes | y_rebounds | 0.632 |
| exp_ast = rate x minutes | y_assists | 0.661 |
| exp_3pm = rate x minutes | y_threes | 0.596 |
