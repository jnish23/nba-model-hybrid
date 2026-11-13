# Feature database comparison

A: `data/props_features_v3.db`

B: `data/props_experiments/audit_2026-11/features/props_features_a11_ctl.db`

Seasons compared: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25 (2025-26 and later are never read, plan A.1). Tolerance 1e-09 relative; NaN equals NaN. Allowed columns: none.

sha256 A `7350a26f027280cf63f0d193fdc93cba91ebd8d1e4caf692a270a625c624390c`; B `84a175dc2783270d13f456dc7f077f0aeff0ce15572892d9b154c2b7954f61f0`

**Identical (no difference at all): True.**
**Violations outside the allowed columns: 0.**


Share of player_game_base cells that differ beyond tolerance: 0.000000 (plan B.6.3: more than 0.001 re-bases the reference database: False).

## Row counts by table and season

| table | season | rows_a | rows_b | shared | only_a | only_b | duplicate_keys_a | duplicate_keys_b |
|---|---|---|---|---|---|---|---|---|
| player_game_at_T | 2019-20 | 86121 | 86121 | 86121 | 0 | 0 | 0 | 0 |
| player_game_at_T | 2020-21 | 94638 | 94638 | 94638 | 0 | 0 | 0 | 0 |
| player_game_at_T | 2021-22 | 101544 | 101544 | 101544 | 0 | 0 | 0 | 0 |
| player_game_at_T | 2022-23 | 102072 | 102072 | 102072 | 0 | 0 | 0 | 0 |
| player_game_at_T | 2023-24 | 104445 | 104445 | 104445 | 0 | 0 | 0 | 0 |
| player_game_at_T | 2024-25 | 105261 | 105261 | 105261 | 0 | 0 | 0 | 0 |
| player_game_base | 2019-20 | 28707 | 28707 | 28707 | 0 | 0 | 0 | 0 |
| player_game_base | 2020-21 | 31546 | 31546 | 31546 | 0 | 0 | 0 | 0 |
| player_game_base | 2021-22 | 33848 | 33848 | 33848 | 0 | 0 | 0 | 0 |
| player_game_base | 2022-23 | 34024 | 34024 | 34024 | 0 | 0 | 0 | 0 |
| player_game_base | 2023-24 | 34815 | 34815 | 34815 | 0 | 0 | 0 | 0 |
| player_game_base | 2024-25 | 35087 | 35087 | 35087 | 0 | 0 | 0 | 0 |
| player_game_labels | 2019-20 | 28707 | 28707 | 28707 | 0 | 0 | 0 | 0 |
| player_game_labels | 2020-21 | 31546 | 31546 | 31546 | 0 | 0 | 0 | 0 |
| player_game_labels | 2021-22 | 33848 | 33848 | 33848 | 0 | 0 | 0 | 0 |
| player_game_labels | 2022-23 | 34024 | 34024 | 34024 | 0 | 0 | 0 | 0 |
| player_game_labels | 2023-24 | 34815 | 34815 | 34815 | 0 | 0 | 0 | 0 |
| player_game_labels | 2024-25 | 35087 | 35087 | 35087 | 0 | 0 | 0 | 0 |

## By feature group (cells and rows that differ beyond tolerance)

| table | group | columns | columns_differing | cells | cells_differing | share_cells_differing | rows | rows_any_column_differing | share_rows_any_differing |
|---|---|---|---|---|---|---|---|---|---|
| player_game_at_T | injury | 6 | 0 | 3564486 | 0 | 0 | 594081 | 0 | 0 |
| player_game_at_T | teammates | 15 | 0 | 8911215 | 0 | 0 | 594081 | 0 | 0 |
| player_game_at_T | market_env | 11 | 0 | 6534891 | 0 | 0 | 594081 | 0 | 0 |
| player_game_base | key | 6 | 0 | 1188162 | 0 | 0 | 198027 | 0 | 0 |
| player_game_base | minutes | 11 | 0 | 2178297 | 0 | 0 | 198027 | 0 | 0 |
| player_game_base | role | 5 | 0 | 990135 | 0 | 0 | 198027 | 0 | 0 |
| player_game_base | rest | 4 | 0 | 792108 | 0 | 0 | 198027 | 0 | 0 |
| player_game_base | rates | 72 | 0 | 14257944 | 0 | 0 | 198027 | 0 | 0 |
| player_game_base | conversion | 6 | 0 | 1188162 | 0 | 0 | 198027 | 0 | 0 |
| player_game_base | team_context | 37 | 0 | 7326999 | 0 | 0 | 198027 | 0 | 0 |
| player_game_base | schedule | 19 | 0 | 3762513 | 0 | 0 | 198027 | 0 | 0 |
| player_game_base | opponent | 37 | 0 | 7326999 | 0 | 0 | 198027 | 0 | 0 |
| player_game_labels | label | 33 | 0 | 6534891 | 0 | 0 | 198027 | 0 | 0 |

## Rows with any differing column of a group, by season

| table | season | group | rows | rows_any_column_differing | share |
|---|---|---|---|---|---|
| player_game_at_T | 2019-20 | injury | 86121 | 0 | 0 |
| player_game_at_T | 2019-20 | market_env | 86121 | 0 | 0 |
| player_game_at_T | 2019-20 | teammates | 86121 | 0 | 0 |
| player_game_at_T | 2020-21 | injury | 94638 | 0 | 0 |
| player_game_at_T | 2020-21 | market_env | 94638 | 0 | 0 |
| player_game_at_T | 2020-21 | teammates | 94638 | 0 | 0 |
| player_game_at_T | 2021-22 | injury | 101544 | 0 | 0 |
| player_game_at_T | 2021-22 | market_env | 101544 | 0 | 0 |
| player_game_at_T | 2021-22 | teammates | 101544 | 0 | 0 |
| player_game_at_T | 2022-23 | injury | 102072 | 0 | 0 |
| player_game_at_T | 2022-23 | market_env | 102072 | 0 | 0 |
| player_game_at_T | 2022-23 | teammates | 102072 | 0 | 0 |
| player_game_at_T | 2023-24 | injury | 104445 | 0 | 0 |
| player_game_at_T | 2023-24 | market_env | 104445 | 0 | 0 |
| player_game_at_T | 2023-24 | teammates | 104445 | 0 | 0 |
| player_game_at_T | 2024-25 | injury | 105261 | 0 | 0 |
| player_game_at_T | 2024-25 | market_env | 105261 | 0 | 0 |
| player_game_at_T | 2024-25 | teammates | 105261 | 0 | 0 |
| player_game_base | 2019-20 | conversion | 28707 | 0 | 0 |
| player_game_base | 2019-20 | key | 28707 | 0 | 0 |
| player_game_base | 2019-20 | minutes | 28707 | 0 | 0 |
| player_game_base | 2019-20 | opponent | 28707 | 0 | 0 |
| player_game_base | 2019-20 | rates | 28707 | 0 | 0 |
| player_game_base | 2019-20 | rest | 28707 | 0 | 0 |
| player_game_base | 2019-20 | role | 28707 | 0 | 0 |
| player_game_base | 2019-20 | schedule | 28707 | 0 | 0 |
| player_game_base | 2019-20 | team_context | 28707 | 0 | 0 |
| player_game_base | 2020-21 | conversion | 31546 | 0 | 0 |
| player_game_base | 2020-21 | key | 31546 | 0 | 0 |
| player_game_base | 2020-21 | minutes | 31546 | 0 | 0 |
| player_game_base | 2020-21 | opponent | 31546 | 0 | 0 |
| player_game_base | 2020-21 | rates | 31546 | 0 | 0 |
| player_game_base | 2020-21 | rest | 31546 | 0 | 0 |
| player_game_base | 2020-21 | role | 31546 | 0 | 0 |
| player_game_base | 2020-21 | schedule | 31546 | 0 | 0 |
| player_game_base | 2020-21 | team_context | 31546 | 0 | 0 |
| player_game_base | 2021-22 | conversion | 33848 | 0 | 0 |
| player_game_base | 2021-22 | key | 33848 | 0 | 0 |
| player_game_base | 2021-22 | minutes | 33848 | 0 | 0 |
| player_game_base | 2021-22 | opponent | 33848 | 0 | 0 |
| player_game_base | 2021-22 | rates | 33848 | 0 | 0 |
| player_game_base | 2021-22 | rest | 33848 | 0 | 0 |
| player_game_base | 2021-22 | role | 33848 | 0 | 0 |
| player_game_base | 2021-22 | schedule | 33848 | 0 | 0 |
| player_game_base | 2021-22 | team_context | 33848 | 0 | 0 |
| player_game_base | 2022-23 | conversion | 34024 | 0 | 0 |
| player_game_base | 2022-23 | key | 34024 | 0 | 0 |
| player_game_base | 2022-23 | minutes | 34024 | 0 | 0 |
| player_game_base | 2022-23 | opponent | 34024 | 0 | 0 |
| player_game_base | 2022-23 | rates | 34024 | 0 | 0 |
| player_game_base | 2022-23 | rest | 34024 | 0 | 0 |
| player_game_base | 2022-23 | role | 34024 | 0 | 0 |
| player_game_base | 2022-23 | schedule | 34024 | 0 | 0 |
| player_game_base | 2022-23 | team_context | 34024 | 0 | 0 |
| player_game_base | 2023-24 | conversion | 34815 | 0 | 0 |
| player_game_base | 2023-24 | key | 34815 | 0 | 0 |
| player_game_base | 2023-24 | minutes | 34815 | 0 | 0 |
| player_game_base | 2023-24 | opponent | 34815 | 0 | 0 |
| player_game_base | 2023-24 | rates | 34815 | 0 | 0 |
| player_game_base | 2023-24 | rest | 34815 | 0 | 0 |
| player_game_base | 2023-24 | role | 34815 | 0 | 0 |
| player_game_base | 2023-24 | schedule | 34815 | 0 | 0 |
| player_game_base | 2023-24 | team_context | 34815 | 0 | 0 |
| player_game_base | 2024-25 | conversion | 35087 | 0 | 0 |
| player_game_base | 2024-25 | key | 35087 | 0 | 0 |
| player_game_base | 2024-25 | minutes | 35087 | 0 | 0 |
| player_game_base | 2024-25 | opponent | 35087 | 0 | 0 |
| player_game_base | 2024-25 | rates | 35087 | 0 | 0 |
| player_game_base | 2024-25 | rest | 35087 | 0 | 0 |
| player_game_base | 2024-25 | role | 35087 | 0 | 0 |
| player_game_base | 2024-25 | schedule | 35087 | 0 | 0 |
| player_game_base | 2024-25 | team_context | 35087 | 0 | 0 |
| player_game_labels | 2019-20 | label | 28707 | 0 | 0 |
| player_game_labels | 2020-21 | label | 31546 | 0 | 0 |
| player_game_labels | 2021-22 | label | 33848 | 0 | 0 |
| player_game_labels | 2022-23 | label | 34024 | 0 | 0 |
| player_game_labels | 2023-24 | label | 34815 | 0 | 0 |
| player_game_labels | 2024-25 | label | 35087 | 0 | 0 |

## Share of rows with av_n_out > 0

| table | season | side | column | share_gt0 | n |
|---|---|---|---|---|---|
| player_game_at_T | 2019-20 | A | av_n_out | 0.159 | 86121 |
| player_game_at_T | 2019-20 | A | opp_av_n_out | 0.1726 | 86121 |
| player_game_at_T | 2019-20 | B | av_n_out | 0.159 | 86121 |
| player_game_at_T | 2019-20 | B | opp_av_n_out | 0.1726 | 86121 |
| player_game_at_T | 2020-21 | A | av_n_out | 0.1822 | 94638 |
| player_game_at_T | 2020-21 | A | opp_av_n_out | 0.1981 | 94638 |
| player_game_at_T | 2020-21 | B | av_n_out | 0.1822 | 94638 |
| player_game_at_T | 2020-21 | B | opp_av_n_out | 0.1981 | 94638 |
| player_game_at_T | 2021-22 | A | av_n_out | 0.1172 | 101544 |
| player_game_at_T | 2021-22 | A | opp_av_n_out | 0.1296 | 101544 |
| player_game_at_T | 2021-22 | B | av_n_out | 0.1172 | 101544 |
| player_game_at_T | 2021-22 | B | opp_av_n_out | 0.1296 | 101544 |
| player_game_at_T | 2022-23 | A | av_n_out | 0.1025 | 102072 |
| player_game_at_T | 2022-23 | A | opp_av_n_out | 0.1151 | 102072 |
| player_game_at_T | 2022-23 | B | av_n_out | 0.1025 | 102072 |
| player_game_at_T | 2022-23 | B | opp_av_n_out | 0.1151 | 102072 |
| player_game_at_T | 2023-24 | A | av_n_out | 0.1557 | 104445 |
| player_game_at_T | 2023-24 | A | opp_av_n_out | 0.1737 | 104445 |
| player_game_at_T | 2023-24 | B | av_n_out | 0.1557 | 104445 |
| player_game_at_T | 2023-24 | B | opp_av_n_out | 0.1737 | 104445 |
| player_game_at_T | 2024-25 | A | av_n_out | 0.2575 | 105261 |
| player_game_at_T | 2024-25 | A | opp_av_n_out | 0.2703 | 105261 |
| player_game_at_T | 2024-25 | B | av_n_out | 0.2575 | 105261 |
| player_game_at_T | 2024-25 | B | opp_av_n_out | 0.2703 | 105261 |

## player_game_at_T: differing or one-sided columns (0 of 32; every column is in player_game_at_T.csv)

_none_

## player_game_base: differing or one-sided columns (0 of 197; every column is in player_game_base.csv)

_none_

## player_game_labels: differing or one-sided columns (0 of 33; every column is in player_game_labels.csv)

_none_
