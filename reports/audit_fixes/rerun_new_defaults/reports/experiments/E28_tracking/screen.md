# E28 - residual screen of the tracking / fatigue families

Registered in `reports/experiments/PREREGISTRATION.md` ("E28") before any residual was computed. Built by `experiments/e28_screen.py`; raw numbers in `screen.json`. Find: (a) 2019-20..2021-22, (b) total and margin 2013-14..2021-22 (the first six from the archive lines), home win 2019-20..2021-22; confirmation 2022-23..2025-26. Regular season only. `gamma` = slope of the confirmation residual on the find-period signal standardised to the find-period SD: points of margin or total (log-odds for the home win) per SD of the signal; full replication of the find fit would give `signal_sd_find_units` (about 0.4-0.5 points for the margin signals, 1.5 for a|F1|total). Day-cluster bootstrap, 1,000 draws.

## (a) residual of the deployed-form out-of-sample prediction

| family | target | n find | n conf | df | find Wald p | gamma | gamma 95% CI | one-sided p | q (BH) | conf Wald p | tier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | total | 3354 | 4778 | 5 | 0.0207 | 0.592 | [0.18, 0.97] | 0.004 | 0.06 | 0.0 | CANDIDATE |
| F1 | margin | 3354 | 4778 | 5 | 0.4854 | 0.348 | [-0.04, 0.72] | 0.037 | 1.0 | 0.0363 | Reject (fails find step) |
| F1 | win | 3354 | 4778 | 5 | 0.7812 | 0.053 | [-0.01, 0.12] | 0.038 | 1.0 | 0.3232 | Reject (fails find step) |
| F2 | total | 3366 | 4786 | 3 | 0.7214 | -0.863 | [-1.32, -0.37] | 1.0 | 1.0 | 0.0026 | Reject (fails find step) |
| F2 | margin | 3366 | 4786 | 3 | 0.9524 | -0.568 | [-0.94, -0.17] | 1.0 | 1.0 | 0.0002 | Reject (fails find step) |
| F2 | win | 3366 | 4786 | 3 | 0.992 | -0.027 | [-0.09, 0.03] | 0.8202 | 1.0 | 0.0097 | Reject (fails find step) |
| F3 | total | 3366 | 4896 | 5 | 0.6625 | 0.341 | [-0.17, 0.82] | 0.0819 | 1.0 | 0.0022 | Reject (fails find step) |
| F3 | margin | 3366 | 4896 | 5 | 0.1248 | 0.335 | [-0.02, 0.76] | 0.036 | 1.0 | 0.0158 | Reject (fails find step) |
| F3 | win | 3366 | 4896 | 5 | 0.0972 | 0.045 | [-0.02, 0.10] | 0.0749 | 0.562 | 0.0386 | Reject (not confirmed, q >= 0.10) |
| F4 | total | 3366 | 4904 | 6 | 0.9545 | -0.161 | [-0.74, 0.45] | 0.7183 | 1.0 | 0.4122 | Reject (fails find step) |
| F4 | margin | 3366 | 4904 | 6 | 0.5286 | 0.212 | [-0.19, 0.61] | 0.1558 | 1.0 | 0.0696 | Reject (fails find step) |
| F4 | win | 3366 | 4904 | 6 | 0.1435 | 0.016 | [-0.04, 0.08] | 0.3177 | 1.0 | 0.668 | Reject (fails find step) |
| F5 | total | 3354 | 4778 | 3 | 0.4277 | 0.46 | [-0.07, 0.94] | 0.0539 | 1.0 | 0.024 | Reject (fails find step) |
| F5 | margin | 3354 | 4778 | 3 | 0.2435 | -0.062 | [-0.44, 0.33] | 0.6374 | 1.0 | 0.8243 | Reject (fails find step) |
| F5 | win | 3354 | 4778 | 3 | 0.2674 | -0.012 | [-0.07, 0.05] | 0.6184 | 1.0 | 0.9558 | Reject (fails find step) |

Candidates: ['a|F1|total']

## (b) residual against the opening line

| family | target | n find | n conf | df | find Wald p | gamma | gamma 95% CI | one-sided p | q (BH) | conf Wald p | tier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | total | 10555 | 4778 | 5 | 0.1376 | 0.28 | [-0.20, 0.74] | 0.1309 | 1.0 | 0.0022 | Reject (fails find step) |
| F1 | margin | 10555 | 4778 | 5 | 0.0383 | 0.485 | [0.12, 0.83] | 0.009 | 0.067 | 0.0289 | CANDIDATE |
| F1 | win | 3342 | 4778 | 5 | 0.8371 | 0.057 | [-0.01, 0.12] | 0.038 | 1.0 | 0.3479 | Reject (fails find step) |
| F2 | total | 10569 | 4786 | 3 | 0.9074 | 0.329 | [-0.22, 0.84] | 0.0979 | 1.0 | 0.007 | Reject (fails find step) |
| F2 | margin | 10569 | 4786 | 3 | 0.9309 | -0.139 | [-0.52, 0.27] | 0.7582 | 1.0 | 0.0003 | Reject (fails find step) |
| F2 | win | 3353 | 4786 | 3 | 0.9809 | -0.007 | [-0.07, 0.05] | 0.6014 | 1.0 | 0.0082 | Reject (fails find step) |
| F3 | total | 7036 | 4896 | 5 | 0.2682 | 0.905 | [0.41, 1.39] | 0.002 | 1.0 | 0.0008 | Reject (fails find step) |
| F3 | margin | 7036 | 4896 | 5 | 0.1161 | 0.538 | [0.15, 0.95] | 0.006 | 1.0 | 0.0152 | Reject (fails find step) |
| F3 | win | 3353 | 4896 | 5 | 0.1139 | 0.053 | [-0.01, 0.11] | 0.043 | 1.0 | 0.0311 | Reject (fails find step) |
| F4 | total | 10725 | 4904 | 6 | 0.0717 | 0.192 | [-0.36, 0.78] | 0.2408 | 0.722 | 0.4549 | Reject (not confirmed, q >= 0.10) |
| F4 | margin | 10725 | 4904 | 6 | 0.0651 | 0.42 | [0.03, 0.80] | 0.022 | 0.11 | 0.017 | Reject (not confirmed, q >= 0.10) |
| F4 | win | 3353 | 4904 | 6 | 0.0739 | 0.025 | [-0.03, 0.09] | 0.2318 | 0.722 | 0.5259 | Reject (not confirmed, q >= 0.10) |
| F5 | total | 10459 | 4778 | 3 | 0.0285 | 0.747 | [0.24, 1.21] | 0.003 | 0.045 | 0.0298 | CANDIDATE |
| F5 | margin | 10459 | 4778 | 3 | 0.9213 | -0.142 | [-0.53, 0.28] | 0.7273 | 1.0 | 0.8344 | Reject (fails find step) |
| F5 | win | 3342 | 4778 | 3 | 0.2746 | -0.011 | [-0.07, 0.05] | 0.6134 | 1.0 | 0.9526 | Reject (fails find step) |

Candidates: ['b|F1|margin', 'b|F5|total']

## (c) mechanism: own team's shooting, pace and efficiency in game t

| family | outcome | n find | df | find Wald p | gamma | gamma 95% CI | one-sided p | replicates (p < 0.0167) |
|---|---|---|---|---|---|---|---|---|
| F1 | 3P% | 21120 | 5 | 0.1886 | 0.306 | [0.11, 0.49] | 0.001 | True |
| F1 | FT% | 21120 | 5 | 0.0 | 0.486 | [0.22, 0.76] | 0.001 | True |
| F1 | rim FG% | 21120 | 5 | 0.0 | 0.823 | [0.60, 1.06] | 0.001 | True |
| F1 | pace | 21120 | 5 | 0.0 | 1.652 | [1.55, 1.76] | 0.001 | True |
| F1 | points/100 poss | 21120 | 5 | 0.0 | 1.16 | [0.92, 1.41] | 0.001 | True |
| F2 | 3P% | 21148 | 3 | 0.9658 | -0.03 | [-0.19, 0.12] | 0.6803 | False |
| F2 | FT% | 21148 | 3 | 0.1141 | 0.231 | [0.03, 0.43] | 0.013 | True |
| F2 | rim FG% | 21148 | 3 | 0.6905 | 0.009 | [-0.19, 0.21] | 0.4805 | False |
| F2 | pace | 21148 | 3 | 0.0 | 0.264 | [0.18, 0.35] | 0.001 | True |
| F2 | points/100 poss | 21148 | 3 | 0.2243 | -0.126 | [-0.36, 0.09] | 0.8382 | False |
| F3 | 3P% | 14080 | 5 | 0.0034 | 0.039 | [-0.15, 0.23] | 0.3377 | False |
| F3 | FT% | 14080 | 5 | 0.2272 | 0.13 | [-0.07, 0.34] | 0.0999 | False |
| F3 | rim FG% | 14080 | 5 | 0.9559 | -0.048 | [-0.26, 0.18] | 0.6653 | False |
| F3 | pace | 14080 | 5 | 0.0 | 0.264 | [0.17, 0.36] | 0.001 | True |
| F3 | points/100 poss | 14080 | 5 | 0.0158 | 0.314 | [0.07, 0.58] | 0.01 | True |
| F4 | 3P% | 21464 | 6 | 0.8686 | 0.054 | [-0.11, 0.24] | 0.2438 | False |
| F4 | FT% | 21464 | 6 | 0.0102 | 0.232 | [0.04, 0.43] | 0.009 | True |
| F4 | rim FG% | 21464 | 6 | 0.0692 | 0.282 | [0.04, 0.50] | 0.009 | True |
| F4 | pace | 21464 | 6 | 0.0 | 0.265 | [0.19, 0.34] | 0.001 | True |
| F4 | points/100 poss | 21464 | 6 | 0.0021 | 0.333 | [0.11, 0.58] | 0.002 | True |
| F5 | 3P% | 20928 | 3 | 0.0336 | 0.003 | [-0.17, 0.19] | 0.4775 | False |
| F5 | FT% | 20928 | 3 | 0.4442 | 0.053 | [-0.12, 0.22] | 0.2697 | False |
| F5 | rim FG% | 20928 | 3 | 0.3133 | -0.005 | [-0.22, 0.23] | 0.5205 | False |
| F5 | pace | 20928 | 2 | 0.0 | 0.262 | [0.16, 0.36] | 0.001 | True |
| F5 | points/100 poss | 20928 | 3 | 0.0075 | 0.149 | [-0.10, 0.40] | 0.1189 | False |

**Fatigue story (F1, F2 move 3P%, FT% or pace, rule registered):** {'F1': True, 'F2': True}

## Coefficients in plain units (find period; points per 1 SD for total and margin, log-odds per 1 SD for home win)

| residual | family | target | feature | per 1 SD | 95% CI |
|---|---|---|---|---|---|
| a | F1 | total | dist_per_min_last | 0.3733 | [-0.573, 1.325] |
| a | F1 | total | dist_per_poss_last | -0.9249 | [-2.239, 0.314] |
| a | F1 | total | xload_last | -0.4355 | [-1.639, 0.892] |
| a | F1 | total | xload_4n | 0.3164 | [-0.760, 1.349] |
| a | F1 | total | rot_xload_last | -1.0998 | [-1.986, -0.264] |
| a | F1 | margin | dist_per_min_last | 0.0121 | [-0.617, 0.676] |
| a | F1 | margin | dist_per_poss_last | 0.3553 | [-0.474, 1.191] |
| a | F1 | margin | xload_last | -0.4991 | [-1.499, 0.588] |
| a | F1 | margin | xload_4n | 0.5408 | [-0.239, 1.284] |
| a | F1 | margin | rot_xload_last | 0.0095 | [-0.660, 0.671] |
| a | F1 | win | dist_per_min_last | -0.0429 | [-0.149, 0.061] |
| a | F1 | win | dist_per_poss_last | 0.0164 | [-0.122, 0.149] |
| a | F1 | win | xload_last | -0.0694 | [-0.225, 0.089] |
| a | F1 | win | xload_4n | 0.0784 | [-0.044, 0.191] |
| a | F1 | win | rot_xload_last | 0.0397 | [-0.059, 0.145] |
| a | F2 | total | rot_speed_z_last | -0.0766 | [-1.440, 1.221] |
| a | F2 | total | rot_speed_trend5 | 0.0745 | [-0.700, 0.907] |
| a | F2 | total | star_speed_z_last | -0.3218 | [-1.481, 0.846] |
| a | F2 | margin | rot_speed_z_last | -0.2659 | [-1.276, 0.736] |
| a | F2 | margin | rot_speed_trend5 | 0.0288 | [-0.551, 0.658] |
| a | F2 | margin | star_speed_z_last | 0.2598 | [-0.650, 1.151] |
| a | F2 | win | rot_speed_z_last | -0.0166 | [-0.180, 0.137] |
| a | F2 | win | rot_speed_trend5 | 0.0053 | [-0.092, 0.098] |
| a | F2 | win | star_speed_z_last | 0.0036 | [-0.144, 0.148] |
| a | F3 | total | prev_ot | -0.2936 | [-1.165, 0.550] |
| a | F3 | total | prev_close | -0.1301 | [-0.714, 0.484] |
| a | F3 | total | prev_n40 | 0.285 | [-0.498, 1.137] |
| a | F3 | total | prev_hustle_z | -0.4961 | [-1.157, 0.180] |
| a | F3 | total | prev_opp_pace | -0.0641 | [-0.659, 0.567] |
| a | F3 | margin | prev_ot | -0.5743 | [-1.113, 0.061] |
| a | F3 | margin | prev_close | -0.1662 | [-0.595, 0.251] |
| a | F3 | margin | prev_n40 | 0.0193 | [-0.574, 0.629] |
| a | F3 | margin | prev_hustle_z | 0.284 | [-0.215, 0.738] |
| a | F3 | margin | prev_opp_pace | -0.2364 | [-0.734, 0.169] |
| a | F3 | win | prev_ot | -0.1206 | [-0.214, -0.022] |
| a | F3 | win | prev_close | 0.0126 | [-0.056, 0.082] |
| a | F3 | win | prev_n40 | 0.015 | [-0.081, 0.114] |
| a | F3 | win | prev_hustle_z | 0.0586 | [-0.021, 0.129] |
| a | F3 | win | prev_opp_pace | -0.0088 | [-0.090, 0.061] |
| a | F4 | total | g3in4 | 0.0443 | [-0.814, 0.889] |
| a | F4 | total | g4in6 | 0.1762 | [-0.813, 1.131] |
| a | F4 | total | games_last7 | -0.4117 | [-1.334, 0.466] |
| a | F4 | total | trip_last_game | -0.1741 | [-0.758, 0.444] |
| a | F4 | total | homecoming | -0.1837 | [-0.876, 0.458] |
| a | F4 | total | prev_altitude | -0.0712 | [-0.709, 0.560] |
| a | F4 | margin | g3in4 | -0.1098 | [-0.696, 0.456] |
| a | F4 | margin | g4in6 | -0.3333 | [-0.984, 0.324] |
| a | F4 | margin | games_last7 | 0.3766 | [-0.186, 0.937] |
| a | F4 | margin | trip_last_game | -0.4223 | [-0.882, 0.060] |
| a | F4 | margin | homecoming | -0.0245 | [-0.439, 0.388] |
| a | F4 | margin | prev_altitude | 0.1065 | [-0.428, 0.634] |
| a | F4 | win | g3in4 | -0.0297 | [-0.114, 0.054] |
| a | F4 | win | g4in6 | 0.0313 | [-0.075, 0.136] |
| a | F4 | win | games_last7 | -0.0254 | [-0.114, 0.067] |
| a | F4 | win | trip_last_game | -0.09 | [-0.168, -0.013] |
| a | F4 | win | homecoming | -0.0039 | [-0.074, 0.069] |
| a | F4 | win | prev_altitude | 0.0704 | [-0.005, 0.151] |
| a | F5 | total | xload_4n_x_b2b | 0.2815 | [-0.354, 0.942] |
| a | F5 | total | xload_last_x_dist | -0.444 | [-1.094, 0.192] |
| a | F5 | total | one_side_tired | -0.2107 | [-0.935, 0.413] |
| a | F5 | margin | xload_4n_x_b2b | -0.2299 | [-0.691, 0.275] |
| a | F5 | margin | xload_last_x_dist | 0.4461 | [-0.090, 0.965] |
| a | F5 | margin | one_side_tired | -0.272 | [-0.748, 0.246] |
| a | F5 | win | xload_4n_x_b2b | 0.0368 | [-0.035, 0.115] |
| a | F5 | win | xload_last_x_dist | 0.048 | [-0.030, 0.125] |
| a | F5 | win | one_side_tired | -0.0703 | [-0.148, 0.004] |

| residual | family | target | feature | per 1 SD | 95% CI |
|---|---|---|---|---|---|
| b | F1 | total | dist_per_min_last | -0.2786 | [-1.162, 0.593] |
| b | F1 | total | dist_per_poss_last | 0.1991 | [-0.438, 0.875] |
| b | F1 | total | xload_last | -0.4165 | [-1.157, 0.248] |
| b | F1 | total | xload_4n | -0.2853 | [-0.813, 0.268] |
| b | F1 | total | rot_xload_last | 0.0452 | [-0.487, 0.602] |
| b | F1 | margin | dist_per_min_last | 0.1782 | [-0.186, 0.556] |
| b | F1 | margin | dist_per_poss_last | 0.4629 | [0.025, 0.870] |
| b | F1 | margin | xload_last | -0.6522 | [-1.152, -0.167] |
| b | F1 | margin | xload_4n | 0.3195 | [-0.057, 0.700] |
| b | F1 | margin | rot_xload_last | -0.0774 | [-0.421, 0.240] |
| b | F1 | win | dist_per_min_last | -0.0192 | [-0.132, 0.098] |
| b | F1 | win | dist_per_poss_last | 0.0183 | [-0.124, 0.161] |
| b | F1 | win | xload_last | -0.0798 | [-0.230, 0.093] |
| b | F1 | win | xload_4n | 0.0832 | [-0.043, 0.199] |
| b | F1 | win | rot_xload_last | 0.0212 | [-0.083, 0.121] |
| b | F2 | total | rot_speed_z_last | -0.1178 | [-0.860, 0.604] |
| b | F2 | total | rot_speed_trend5 | 0.1142 | [-0.318, 0.562] |
| b | F2 | total | star_speed_z_last | 0.1346 | [-0.503, 0.776] |
| b | F2 | margin | rot_speed_z_last | -0.1493 | [-0.666, 0.357] |
| b | F2 | margin | rot_speed_trend5 | 0.0742 | [-0.222, 0.386] |
| b | F2 | margin | star_speed_z_last | 0.1111 | [-0.350, 0.572] |
| b | F2 | win | rot_speed_z_last | -0.0268 | [-0.180, 0.136] |
| b | F2 | win | rot_speed_trend5 | 0.0108 | [-0.088, 0.108] |
| b | F2 | win | star_speed_z_last | 0.0089 | [-0.140, 0.154] |
| b | F3 | total | prev_ot | -0.5162 | [-1.049, 0.002] |
| b | F3 | total | prev_close | -0.2129 | [-0.640, 0.235] |
| b | F3 | total | prev_n40 | 0.4394 | [-0.054, 0.942] |
| b | F3 | total | prev_hustle_z | -0.189 | [-0.652, 0.247] |
| b | F3 | total | prev_opp_pace | 0.0663 | [-0.369, 0.442] |
| b | F3 | margin | prev_ot | -0.317 | [-0.703, 0.097] |
| b | F3 | margin | prev_close | -0.2067 | [-0.530, 0.099] |
| b | F3 | margin | prev_n40 | -0.0337 | [-0.437, 0.354] |
| b | F3 | margin | prev_hustle_z | 0.2947 | [-0.024, 0.642] |
| b | F3 | margin | prev_opp_pace | -0.0676 | [-0.354, 0.225] |
| b | F3 | win | prev_ot | -0.1098 | [-0.207, -0.009] |
| b | F3 | win | prev_close | 0.0179 | [-0.052, 0.090] |
| b | F3 | win | prev_n40 | 0.002 | [-0.097, 0.098] |
| b | F3 | win | prev_hustle_z | 0.0668 | [-0.010, 0.139] |
| b | F3 | win | prev_opp_pace | -0.004 | [-0.082, 0.070] |
| b | F4 | total | g3in4 | -0.2659 | [-0.683, 0.128] |
| b | F4 | total | g4in6 | -0.1001 | [-0.563, 0.366] |
| b | F4 | total | games_last7 | -0.2259 | [-0.671, 0.266] |
| b | F4 | total | trip_last_game | -0.2014 | [-0.537, 0.143] |
| b | F4 | total | homecoming | -0.2029 | [-0.556, 0.166] |
| b | F4 | total | prev_altitude | -0.2843 | [-0.649, 0.080] |
| b | F4 | margin | g3in4 | -0.1775 | [-0.464, 0.156] |
| b | F4 | margin | g4in6 | 0.1056 | [-0.226, 0.396] |
| b | F4 | margin | games_last7 | -0.3693 | [-0.663, -0.081] |
| b | F4 | margin | trip_last_game | -0.0463 | [-0.276, 0.182] |
| b | F4 | margin | homecoming | 0.0923 | [-0.147, 0.319] |
| b | F4 | margin | prev_altitude | 0.0464 | [-0.199, 0.281] |
| b | F4 | win | g3in4 | -0.0275 | [-0.117, 0.057] |
| b | F4 | win | g4in6 | 0.0386 | [-0.067, 0.139] |
| b | F4 | win | games_last7 | -0.0496 | [-0.138, 0.044] |
| b | F4 | win | trip_last_game | -0.0941 | [-0.167, -0.017] |
| b | F4 | win | homecoming | -0.0064 | [-0.079, 0.067] |
| b | F4 | win | prev_altitude | 0.0682 | [-0.005, 0.146] |
| b | F5 | total | xload_4n_x_b2b | -0.089 | [-0.483, 0.257] |
| b | F5 | total | xload_last_x_dist | -0.4639 | [-0.815, -0.117] |
| b | F5 | total | one_side_tired | -0.17 | [-0.504, 0.201] |
| b | F5 | margin | xload_4n_x_b2b | 0.0296 | [-0.226, 0.309] |
| b | F5 | margin | xload_last_x_dist | 0.0776 | [-0.228, 0.383] |
| b | F5 | margin | one_side_tired | -0.0866 | [-0.351, 0.180] |
| b | F5 | win | xload_4n_x_b2b | 0.0425 | [-0.036, 0.121] |
| b | F5 | win | xload_last_x_dist | 0.0432 | [-0.036, 0.123] |
| b | F5 | win | one_side_tired | -0.0719 | [-0.153, 0.011] |

## Mechanism coefficients (find period; percentage points / possessions per 1 SD of the own-team feature)

| family | outcome | feature | per 1 SD | 95% CI |
|---|---|---|---|---|
| F1 | 3P% | dist_per_min_last | -0.3166 | [-0.578, -0.047] |
| F1 | 3P% | dist_per_poss_last | 0.2225 | [-0.011, 0.464] |
| F1 | 3P% | xload_last | -0.0415 | [-0.333, 0.225] |
| F1 | 3P% | xload_4n | -0.0552 | [-0.266, 0.155] |
| F1 | 3P% | rot_xload_last | 0.1817 | [-0.004, 0.365] |
| F1 | FT% | dist_per_min_last | -0.7788 | [-1.084, -0.490] |
| F1 | FT% | dist_per_poss_last | 0.5714 | [0.320, 0.819] |
| F1 | FT% | xload_last | -0.4286 | [-0.704, -0.130] |
| F1 | FT% | xload_4n | 0.0077 | [-0.212, 0.214] |
| F1 | FT% | rot_xload_last | 0.5114 | [0.282, 0.740] |
| F1 | rim FG% | dist_per_min_last | -0.6749 | [-0.981, -0.368] |
| F1 | rim FG% | dist_per_poss_last | -0.6734 | [-0.922, -0.406] |
| F1 | rim FG% | xload_last | 0.4428 | [0.172, 0.725] |
| F1 | rim FG% | xload_4n | -0.0014 | [-0.213, 0.206] |
| F1 | rim FG% | rot_xload_last | 0.3019 | [0.079, 0.514] |
| F1 | pace | dist_per_min_last | 2.2486 | [2.072, 2.434] |
| F1 | pace | dist_per_poss_last | -2.9106 | [-3.023, -2.794] |
| F1 | pace | xload_last | 1.8255 | [1.701, 1.957] |
| F1 | pace | xload_4n | -0.0946 | [-0.190, -0.009] |
| F1 | pace | rot_xload_last | -0.9313 | [-1.069, -0.805] |
| F1 | points/100 poss | dist_per_min_last | -1.6014 | [-1.927, -1.270] |
| F1 | points/100 poss | dist_per_poss_last | -0.6399 | [-0.938, -0.383] |
| F1 | points/100 poss | xload_last | 0.4848 | [0.170, 0.807] |
| F1 | points/100 poss | xload_4n | -0.0879 | [-0.325, 0.154] |
| F1 | points/100 poss | rot_xload_last | 0.7083 | [0.491, 0.928] |
| F2 | 3P% | rot_speed_z_last | 0.0018 | [-0.257, 0.267] |
| F2 | 3P% | rot_speed_trend5 | -0.036 | [-0.190, 0.118] |
| F2 | 3P% | star_speed_z_last | 0.0052 | [-0.236, 0.236] |
| F2 | FT% | rot_speed_z_last | 0.0308 | [-0.270, 0.337] |
| F2 | FT% | rot_speed_trend5 | -0.0726 | [-0.254, 0.112] |
| F2 | FT% | star_speed_z_last | 0.1831 | [-0.088, 0.450] |
| F2 | rim FG% | rot_speed_z_last | -0.0251 | [-0.325, 0.277] |
| F2 | rim FG% | rot_speed_trend5 | 0.0995 | [-0.084, 0.284] |
| F2 | rim FG% | star_speed_z_last | -0.0014 | [-0.284, 0.273] |
| F2 | pace | rot_speed_z_last | 0.3264 | [0.202, 0.454] |
| F2 | pace | rot_speed_trend5 | -0.0741 | [-0.160, 0.011] |
| F2 | pace | star_speed_z_last | -0.0405 | [-0.171, 0.081] |
| F2 | points/100 poss | rot_speed_z_last | -0.3389 | [-0.676, -0.034] |
| F2 | points/100 poss | rot_speed_trend5 | 0.0662 | [-0.120, 0.261] |
| F2 | points/100 poss | star_speed_z_last | 0.2402 | [-0.059, 0.527] |
| F3 | 3P% | prev_ot | -0.3184 | [-0.503, -0.132] |
| F3 | 3P% | prev_close | -0.1142 | [-0.257, 0.023] |
| F3 | 3P% | prev_n40 | 0.2091 | [0.031, 0.401] |
| F3 | 3P% | prev_hustle_z | 0.017 | [-0.127, 0.169] |
| F3 | 3P% | prev_opp_pace | -0.1293 | [-0.274, 0.016] |
| F3 | FT% | prev_ot | -0.0762 | [-0.284, 0.130] |
| F3 | FT% | prev_close | -0.022 | [-0.188, 0.150] |
| F3 | FT% | prev_n40 | 0.1694 | [-0.039, 0.369] |
| F3 | FT% | prev_hustle_z | 0.007 | [-0.166, 0.182] |
| F3 | FT% | prev_opp_pace | 0.1851 | [0.001, 0.359] |
| F3 | rim FG% | prev_ot | 0.031 | [-0.196, 0.262] |
| F3 | rim FG% | prev_close | 0.0622 | [-0.100, 0.228] |
| F3 | rim FG% | prev_n40 | -0.0415 | [-0.245, 0.170] |
| F3 | rim FG% | prev_hustle_z | 0.0188 | [-0.160, 0.210] |
| F3 | rim FG% | prev_opp_pace | -0.0554 | [-0.235, 0.131] |
| F3 | pace | prev_ot | -0.0451 | [-0.134, 0.045] |
| F3 | pace | prev_close | -0.1017 | [-0.188, -0.022] |
| F3 | pace | prev_n40 | 0.0551 | [-0.041, 0.150] |
| F3 | pace | prev_hustle_z | -0.0284 | [-0.106, 0.060] |
| F3 | pace | prev_opp_pace | 0.3043 | [0.219, 0.384] |
| F3 | points/100 poss | prev_ot | -0.3832 | [-0.615, -0.150] |
| F3 | points/100 poss | prev_close | -0.1098 | [-0.285, 0.076] |
| F3 | points/100 poss | prev_n40 | 0.3259 | [0.085, 0.577] |
| F3 | points/100 poss | prev_hustle_z | -0.0642 | [-0.258, 0.149] |
| F3 | points/100 poss | prev_opp_pace | -0.1127 | [-0.305, 0.061] |
| F4 | 3P% | g3in4 | -0.0609 | [-0.212, 0.101] |
| F4 | 3P% | g4in6 | -0.0472 | [-0.217, 0.131] |
| F4 | 3P% | games_last7 | 0.0056 | [-0.178, 0.190] |
| F4 | 3P% | trip_last_game | -0.038 | [-0.192, 0.103] |
| F4 | 3P% | homecoming | -0.0037 | [-0.135, 0.132] |
| F4 | 3P% | prev_altitude | -0.0319 | [-0.150, 0.082] |
| F4 | FT% | g3in4 | -0.0013 | [-0.173, 0.183] |
| F4 | FT% | g4in6 | 0.0421 | [-0.151, 0.229] |
| F4 | FT% | games_last7 | 0.174 | [-0.045, 0.379] |
| F4 | FT% | trip_last_game | -0.0061 | [-0.163, 0.150] |
| F4 | FT% | homecoming | -0.206 | [-0.342, -0.060] |
| F4 | FT% | prev_altitude | -0.1063 | [-0.243, 0.037] |
| F4 | rim FG% | g3in4 | -0.108 | [-0.290, 0.070] |
| F4 | rim FG% | g4in6 | -0.1564 | [-0.334, 0.032] |
| F4 | rim FG% | games_last7 | 0.0455 | [-0.163, 0.236] |
| F4 | rim FG% | trip_last_game | -0.1079 | [-0.270, 0.052] |
| F4 | rim FG% | homecoming | 0.0503 | [-0.103, 0.202] |
| F4 | rim FG% | prev_altitude | -0.0588 | [-0.201, 0.082] |
| F4 | pace | g3in4 | -0.0477 | [-0.116, 0.023] |
| F4 | pace | g4in6 | 0.0519 | [-0.030, 0.131] |
| F4 | pace | games_last7 | -0.3504 | [-0.406, -0.289] |
| F4 | pace | trip_last_game | -0.0623 | [-0.138, 0.008] |
| F4 | pace | homecoming | 0.0222 | [-0.045, 0.090] |
| F4 | pace | prev_altitude | 0.0306 | [-0.042, 0.092] |
| F4 | points/100 poss | g3in4 | -0.2636 | [-0.462, -0.080] |
| F4 | points/100 poss | g4in6 | -0.1093 | [-0.312, 0.084] |
| F4 | points/100 poss | games_last7 | 0.1044 | [-0.107, 0.307] |
| F4 | points/100 poss | trip_last_game | -0.1885 | [-0.351, -0.017] |
| F4 | points/100 poss | homecoming | -0.0071 | [-0.171, 0.161] |
| F4 | points/100 poss | prev_altitude | 0.0462 | [-0.091, 0.209] |
| F5 | 3P% | xload_4n_x_b2b | 0.0518 | [-0.084, 0.187] |
| F5 | 3P% | xload_last_x_dist | 0.117 | [-0.026, 0.242] |
| F5 | 3P% | one_side_tired | -0.1918 | [-0.330, -0.051] |
| F5 | FT% | xload_4n_x_b2b | 0.1136 | [-0.026, 0.243] |
| F5 | FT% | xload_last_x_dist | -0.0282 | [-0.184, 0.112] |
| F5 | FT% | one_side_tired | -0.0204 | [-0.174, 0.135] |
| F5 | rim FG% | xload_4n_x_b2b | -0.0062 | [-0.152, 0.133] |
| F5 | rim FG% | xload_last_x_dist | -0.1252 | [-0.271, 0.013] |
| F5 | rim FG% | one_side_tired | -0.0242 | [-0.172, 0.122] |
| F5 | pace | xload_4n_x_b2b | -0.0823 | [-0.146, -0.019] |
| F5 | pace | xload_last_x_dist | -0.2392 | [-0.301, -0.173] |
| F5 | pace | one_side_tired | 0.0 | [0.000, 0.000] |
| F5 | points/100 poss | xload_4n_x_b2b | -0.0147 | [-0.176, 0.152] |
| F5 | points/100 poss | xload_last_x_dist | 0.0162 | [-0.145, 0.178] |
| F5 | points/100 poss | one_side_tired | -0.2458 | [-0.393, -0.088] |

## Sensitivity (decides nothing)

Wald p on the six archive seasons alone, (b) total and margin:

| test | n | p |
|---|---|---|
| F1|total | 7201 | 0.04128358050478778 |
| F1|margin | 7201 | 0.061629925946184744 |
| F2|total | 7203 | 0.5664723134606968 |
| F2|margin | 7203 | 0.9399108615093607 |
| F3|total | 3670 | 0.10594134347836406 |
| F3|margin | 3670 | 0.6407371558951656 |
| F4|total | 7359 | 0.007628085725309943 |
| F4|margin | 7359 | 0.009594593642235775 |
| F5|total | 7105 | 0.02394304612870754 |
| F5|margin | 7105 | 0.8002989628084523 |

Playoffs and play-in added:

| test | n_find | n_conf | p_find | gamma | p_one |
|---|---|---|---|---|---|
| a|F1|total | 3622 | 5135 | 0.0381 | 0.2928 | 0.0589 |
| a|F1|margin | 3622 | 5135 | 0.5751 | 0.3382 | 0.04 |
| a|F1|win | 3622 | 5135 | 0.8408 | 0.0423 | 0.0869 |
| a|F2|total | 3634 | 5145 | 0.7239 | -0.9401 | 1.0 |
| a|F2|margin | 3634 | 5145 | 0.9432 | -0.6567 | 0.998 |
| a|F2|win | 3634 | 5145 | 0.9022 | -0.0593 | 0.976 |
| a|F3|total | 3634 | 5255 | 0.6127 | 0.0706 | 0.3796 |
| a|F3|margin | 3634 | 5255 | 0.0435 | 0.2257 | 0.1189 |
| a|F3|win | 3634 | 5255 | 0.0378 | 0.0413 | 0.0909 |
| a|F4|total | 3634 | 5263 | 0.9569 | -0.4674 | 0.973 |
| a|F4|margin | 3634 | 5263 | 0.2905 | 0.2175 | 0.1169 |
| a|F4|win | 3634 | 5263 | 0.1109 | 0.0261 | 0.1858 |
| a|F5|total | 3622 | 5135 | 0.3716 | 0.4427 | 0.039 |
| a|F5|margin | 3622 | 5135 | 0.2796 | -0.0434 | 0.5864 |
| a|F5|win | 3622 | 5135 | 0.364 | -0.0074 | 0.6094 |
| b|F1|total | 11316 | 5135 | 0.1999 | 0.1222 | 0.3207 |
| b|F1|margin | 11316 | 5135 | 0.0716 | 0.4999 | 0.002 |
| b|F1|win | 3609 | 5135 | 0.8979 | 0.0513 | 0.048 |
| b|F2|total | 11330 | 5145 | 0.7652 | 0.198 | 0.1898 |
| b|F2|margin | 11330 | 5145 | 0.9467 | -0.4135 | 0.98 |
| b|F2|win | 3620 | 5145 | 0.8923 | -0.0456 | 0.9331 |
| b|F3|total | 7614 | 5255 | 0.5778 | 0.7085 | 0.005 |
| b|F3|margin | 7614 | 5255 | 0.0373 | 0.475 | 0.007 |
| b|F3|win | 3620 | 5255 | 0.0713 | 0.0514 | 0.041 |
| b|F4|total | 11486 | 5263 | 0.1608 | 0.0764 | 0.3966 |
| b|F4|margin | 11486 | 5263 | 0.0844 | 0.483 | 0.008 |
| b|F4|win | 3620 | 5263 | 0.0481 | 0.0356 | 0.1059 |
| b|F5|total | 11220 | 5135 | 0.0844 | 0.6923 | 0.005 |
| b|F5|margin | 11220 | 5135 | 0.9468 | -0.1173 | 0.7203 |
| b|F5|win | 3609 | 5135 | 0.409 | -0.0075 | 0.6134 |
