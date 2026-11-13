# Feature database comparison

A: `data/props_experiments/audit_2026-11/features/props_features_a11_ctl.db`

B: `data/props_experiments/audit_2026-11/features/props_features_a11_pg1.db`

Seasons compared: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25 (2025-26 and later are never read, plan A.1). Tolerance 1e-09 relative; NaN equals NaN. Allowed columns: ['group:teammates'].

sha256 A `84a175dc2783270d13f456dc7f077f0aeff0ce15572892d9b154c2b7954f61f0`; B `e7c502eaefdf66e97725f8608c70cd821273fbcb96c1cbefaea98573ea02363f`

**Identical (no difference at all): False.**
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
| player_game_at_T | teammates | 15 | 15 | 8911215 | 3848942 | 0.431921 | 594081 | 462272 | 0.77813 |
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
| player_game_at_T | 2019-20 | teammates | 86121 | 63029 | 0.731866 |
| player_game_at_T | 2020-21 | injury | 94638 | 0 | 0 |
| player_game_at_T | 2020-21 | market_env | 94638 | 0 | 0 |
| player_game_at_T | 2020-21 | teammates | 94638 | 74831 | 0.790708 |
| player_game_at_T | 2021-22 | injury | 101544 | 0 | 0 |
| player_game_at_T | 2021-22 | market_env | 101544 | 0 | 0 |
| player_game_at_T | 2021-22 | teammates | 101544 | 76976 | 0.758056 |
| player_game_at_T | 2022-23 | injury | 102072 | 0 | 0 |
| player_game_at_T | 2022-23 | market_env | 102072 | 0 | 0 |
| player_game_at_T | 2022-23 | teammates | 102072 | 74778 | 0.732601 |
| player_game_at_T | 2023-24 | injury | 104445 | 0 | 0 |
| player_game_at_T | 2023-24 | market_env | 104445 | 0 | 0 |
| player_game_at_T | 2023-24 | teammates | 104445 | 84690 | 0.810857 |
| player_game_at_T | 2024-25 | injury | 105261 | 0 | 0 |
| player_game_at_T | 2024-25 | market_env | 105261 | 0 | 0 |
| player_game_at_T | 2024-25 | teammates | 105261 | 87968 | 0.835713 |
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
| player_game_at_T | 2019-20 | B | av_n_out | 0 | 86121 |
| player_game_at_T | 2019-20 | B | opp_av_n_out | 0 | 86121 |
| player_game_at_T | 2020-21 | A | av_n_out | 0.1822 | 94638 |
| player_game_at_T | 2020-21 | A | opp_av_n_out | 0.1981 | 94638 |
| player_game_at_T | 2020-21 | B | av_n_out | 0 | 94638 |
| player_game_at_T | 2020-21 | B | opp_av_n_out | 0 | 94638 |
| player_game_at_T | 2021-22 | A | av_n_out | 0.1172 | 101544 |
| player_game_at_T | 2021-22 | A | opp_av_n_out | 0.1296 | 101544 |
| player_game_at_T | 2021-22 | B | av_n_out | 0 | 101544 |
| player_game_at_T | 2021-22 | B | opp_av_n_out | 0 | 101544 |
| player_game_at_T | 2022-23 | A | av_n_out | 0.1025 | 102072 |
| player_game_at_T | 2022-23 | A | opp_av_n_out | 0.1151 | 102072 |
| player_game_at_T | 2022-23 | B | av_n_out | 0 | 102072 |
| player_game_at_T | 2022-23 | B | opp_av_n_out | 0 | 102072 |
| player_game_at_T | 2023-24 | A | av_n_out | 0.1557 | 104445 |
| player_game_at_T | 2023-24 | A | opp_av_n_out | 0.1737 | 104445 |
| player_game_at_T | 2023-24 | B | av_n_out | 0 | 104445 |
| player_game_at_T | 2023-24 | B | opp_av_n_out | 0 | 104445 |
| player_game_at_T | 2024-25 | A | av_n_out | 0.2575 | 105261 |
| player_game_at_T | 2024-25 | A | opp_av_n_out | 0.2703 | 105261 |
| player_game_at_T | 2024-25 | B | av_n_out | 0 | 105261 |
| player_game_at_T | 2024-25 | B | opp_av_n_out | 0 | 105261 |

## player_game_at_T: differing or one-sided columns (15 of 32; every column is in player_game_at_T.csv)

| column | group | status | n_compared | n_diff_exact | share_diff_exact | n_diff_tol | share_diff_tol | max_abs_diff | allowed | first_differing_keys |
|---|---|---|---|---|---|---|---|---|---|---|
| av_miss_minutes | teammates | both | 594081 | 326829 | 0.550142 | 317533 | 0.534494 | 0.913749 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| av_miss_usage | teammates | both | 594081 | 326246 | 0.549161 | 317576 | 0.534567 | 0.920607 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| av_miss_reb | teammates | both | 594081 | 326189 | 0.549065 | 317534 | 0.534496 | 0.916661 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| av_miss_ast | teammates | both | 594081 | 327069 | 0.550546 | 317534 | 0.534496 | 0.930414 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| av_miss_pts | teammates | both | 594081 | 327219 | 0.550799 | 317576 | 0.534567 | 0.920966 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| av_n_out | teammates | both | 594081 | 96672 | 0.162725 | 96672 | 0.162725 | 6 | True | ('0021900006', np.int64(201599), np.int64(5)): a=1 b=0; ('0021900006', np.int64(201599), np.int64(60)): a=1 b=0; ('0021900006', np.int64(201599), np.int64(240)): a=1 b=0; ('0021900006', np.int64(202066), np.int64(5)): a=1 b=0; ('0021900006', np.int64(202066), np.int64(60)): a=1 b=0 |
| av_n_questionable | teammates | both | 594081 | 92942 | 0.156447 | 92942 | 0.156447 | 6 | True | ('0021900002', np.int64(2544), np.int64(5)): a=0 b=1; ('0021900002', np.int64(2544), np.int64(60)): a=0 b=1; ('0021900002', np.int64(2544), np.int64(240)): a=0 b=1; ('0021900002', np.int64(2730), np.int64(5)): a=0 b=1; ('0021900002', np.int64(2730), np.int64(60)): a=0 b=1 |
| av_miss_starter_minutes | teammates | both | 594081 | 161852 | 0.272441 | 158328 | 0.266509 | 158.031971 | True | ('0021900001', np.int64(200755), np.int64(5)): a=30.6026346566 b=0; ('0021900001', np.int64(200755), np.int64(60)): a=30.6026346566 b=0; ('0021900001', np.int64(200755), np.int64(240)): a=30.6026346566 b=0; ('0021900001', np.int64(200768), np.int64(5)): a=13.8310598594 b=0; ('0021900001', np.int64(200768), np.int64(60)): a=13.8310598594 b=0 |
| opp_av_miss_minutes | teammates | both | 594081 | 331088 | 0.557311 | 327355 | 0.551028 | 0.8973 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| opp_av_miss_usage | teammates | both | 594081 | 330434 | 0.55621 | 327400 | 0.551103 | 0.908087 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| opp_av_miss_reb | teammates | both | 594081 | 330474 | 0.556278 | 327355 | 0.551028 | 0.896885 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| opp_av_miss_ast | teammates | both | 594081 | 331746 | 0.558419 | 327355 | 0.551028 | 0.919817 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| opp_av_miss_pts | teammates | both | 594081 | 331408 | 0.55785 | 327400 | 0.551103 | 0.908018 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0.15 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0.15 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0.15 b=NaN |
| opp_av_n_out | teammates | both | 594081 | 109827 | 0.184869 | 109827 | 0.184869 | 6 | True | ('0021900001', np.int64(200755), np.int64(5)): a=0 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=0 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=0 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=0 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=0 b=NaN |
| opp_av_miss_starter_minutes | teammates | both | 594081 | 166591 | 0.280418 | 166555 | 0.280357 | 158.031971 | True | ('0021900001', np.int64(200755), np.int64(5)): a=19.2877285884 b=NaN; ('0021900001', np.int64(200755), np.int64(60)): a=19.2877285884 b=NaN; ('0021900001', np.int64(200755), np.int64(240)): a=19.2877285884 b=NaN; ('0021900001', np.int64(200768), np.int64(5)): a=35.3282481902 b=NaN; ('0021900001', np.int64(200768), np.int64(60)): a=35.3282481902 b=NaN |

## player_game_base: differing or one-sided columns (0 of 197; every column is in player_game_base.csv)

_none_

## player_game_labels: differing or one-sided columns (0 of 33; every column is in player_game_labels.csv)

_none_
