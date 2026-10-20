# Phase 2 feature dictionary

223 features. Full detail (with availability by season) in phase2_feature_dictionary.csv.

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

| table | feature | group | source | first_season_with_data | non_null_share | description |
|---|---|---|---|---|---|---|
| player_game_base | m_min_h5 | minutes | history | 2019-20 | 0.993 | Decayed mean minutes per game played (half-life 5 games) |
| player_game_base | m_games_eff_h5 | minutes | history | 2019-20 | 1 | Effective number of earlier games, the sum of decay weights (half-life 5 games): how much history backs the estimates |
| player_game_base | m_min_h15 | minutes | history | 2019-20 | 0.993 | Decayed mean minutes per game played (half-life 15 games) |
| player_game_base | m_games_eff_h15 | minutes | history | 2019-20 | 1 | Effective number of earlier games, the sum of decay weights (half-life 15 games): how much history backs the estimates |
| player_game_base | m_min_sd_h15 | minutes | history | 2019-20 | 0.993 | Decayed standard deviation of minutes (half-life 15 games) |
| player_game_base | m_min_h40 | minutes | history | 2019-20 | 0.993 | Decayed mean minutes per game played (half-life 40 games) |
| player_game_base | m_games_eff_h40 | minutes | history | 2019-20 | 1 | Effective number of earlier games, the sum of decay weights (half-life 40 games): how much history backs the estimates |
| player_game_base | role_p_played_h5 | role | history | 2019-20 | 0.997 | Decayed share of his listed games in which he played (half-life 5 games) |
| player_game_base | role_start_share_h5 | role | history | 2019-20 | 0.997 | Decayed share of his listed games that he started (half-life 5 games); uses EARLIER games only |
| player_game_base | role_p_played_h15 | role | history | 2019-20 | 0.997 | Decayed share of his listed games in which he played (half-life 15 games) |
| player_game_base | role_start_share_h15 | role | history | 2019-20 | 0.997 | Decayed share of his listed games that he started (half-life 15 games); uses EARLIER games only |
| player_game_base | m_min_blowout_h40 | minutes | history | 2019-20 | 0.993 | Decayed mean minutes in blowout games (|final margin| >= 15 is a blowout), shrunk toward his overall mean (half-life 40) |
| player_game_base | m_min_close_h40 | minutes | history | 2019-20 | 0.993 | Decayed mean minutes in close games (|final margin| >= 15 is a blowout), shrunk toward his overall mean (half-life 40) |
| player_game_base | m_blowout_ratio_h40 | minutes | history | 2019-20 | 0.993 | Minutes in blowouts divided by minutes in close games: blowout sensitivity |
| player_game_base | rest_days_since_played | rest | history | 2019-20 | 0.993 | Days since he last played |
| player_game_base | rest_returning | rest | history | 2019-20 | 1 | Returning from a gap of 10 or more days within a season |
| player_game_base | rest_games_since_return | rest | history | 2019-20 | 0.929 | Games played since the latest return from such a gap (NaN if none) |
| player_game_base | rest_games_played_season | rest | history | 2019-20 | 1 | Games played earlier this season |
| player_game_base | r_points_h5 | rates | history | 2019-20 | 1 | points per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fga_h5 | rates | history | 2019-20 | 1 | field-goal attempts per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fgm_h5 | rates | history | 2019-20 | 1 | field goals made per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fg3a_h5 | rates | history | 2019-20 | 1 | three-point attempts per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fg3m_h5 | rates | history | 2019-20 | 1 | threes made per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fta_h5 | rates | history | 2019-20 | 1 | free-throw attempts per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_ftm_h5 | rates | history | 2019-20 | 1 | free throws made per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_rebounds_h5 | rates | history | 2019-20 | 1 | rebounds per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_oreb_h5 | rates | history | 2019-20 | 1 | offensive rebounds per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_dreb_h5 | rates | history | 2019-20 | 1 | defensive rebounds per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_assists_h5 | rates | history | 2019-20 | 1 | assists per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_tov_h5 | rates | history | 2019-20 | 1 | turnovers per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_pf_h5 | rates | history | 2019-20 | 1 | personal fouls per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_stl_h5 | rates | history | 2019-20 | 1 | steals per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_blk_h5 | rates | history | 2019-20 | 1 | blocks per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_h5 | rates | history | 2019-20 | 1 | rebound chances per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_off_h5 | rates | history | 2019-20 | 1 | offensive rebound chances per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_def_h5 | rates | history | 2019-20 | 1 | defensive rebound chances per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_touches_h5 | rates | history | 2019-20 | 1 | touches per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_passes_h5 | rates | history | 2019-20 | 1 | passes per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_secondary_ast_h5 | rates | history | 2019-20 | 1 | secondary (hockey) assists per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_ft_ast_h5 | rates | history | 2019-20 | 1 | free-throw assists per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_contested_fga_h5 | rates | history | 2019-20 | 1 | contested field-goal attempts per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_usage_pct_h5 | rates | history | 2019-20 | 1 | usage percentage (minutes-weighted) per minute (half-life 5 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_points_h15 | rates | history | 2019-20 | 1 | points per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fga_h15 | rates | history | 2019-20 | 1 | field-goal attempts per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fgm_h15 | rates | history | 2019-20 | 1 | field goals made per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fg3a_h15 | rates | history | 2019-20 | 1 | three-point attempts per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fg3m_h15 | rates | history | 2019-20 | 1 | threes made per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fta_h15 | rates | history | 2019-20 | 1 | free-throw attempts per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_ftm_h15 | rates | history | 2019-20 | 1 | free throws made per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_rebounds_h15 | rates | history | 2019-20 | 1 | rebounds per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_oreb_h15 | rates | history | 2019-20 | 1 | offensive rebounds per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_dreb_h15 | rates | history | 2019-20 | 1 | defensive rebounds per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_assists_h15 | rates | history | 2019-20 | 1 | assists per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_tov_h15 | rates | history | 2019-20 | 1 | turnovers per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_pf_h15 | rates | history | 2019-20 | 1 | personal fouls per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_stl_h15 | rates | history | 2019-20 | 1 | steals per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_blk_h15 | rates | history | 2019-20 | 1 | blocks per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_h15 | rates | history | 2019-20 | 1 | rebound chances per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_off_h15 | rates | history | 2019-20 | 1 | offensive rebound chances per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_def_h15 | rates | history | 2019-20 | 1 | defensive rebound chances per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_touches_h15 | rates | history | 2019-20 | 1 | touches per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_passes_h15 | rates | history | 2019-20 | 1 | passes per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_secondary_ast_h15 | rates | history | 2019-20 | 1 | secondary (hockey) assists per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_ft_ast_h15 | rates | history | 2019-20 | 1 | free-throw assists per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_contested_fga_h15 | rates | history | 2019-20 | 1 | contested field-goal attempts per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_usage_pct_h15 | rates | history | 2019-20 | 1 | usage percentage (minutes-weighted) per minute (half-life 15 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_points_h40 | rates | history | 2019-20 | 1 | points per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fga_h40 | rates | history | 2019-20 | 1 | field-goal attempts per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fgm_h40 | rates | history | 2019-20 | 1 | field goals made per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fg3a_h40 | rates | history | 2019-20 | 1 | three-point attempts per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fg3m_h40 | rates | history | 2019-20 | 1 | threes made per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_fta_h40 | rates | history | 2019-20 | 1 | free-throw attempts per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_ftm_h40 | rates | history | 2019-20 | 1 | free throws made per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_rebounds_h40 | rates | history | 2019-20 | 1 | rebounds per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_oreb_h40 | rates | history | 2019-20 | 1 | offensive rebounds per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_dreb_h40 | rates | history | 2019-20 | 1 | defensive rebounds per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_assists_h40 | rates | history | 2019-20 | 1 | assists per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_tov_h40 | rates | history | 2019-20 | 1 | turnovers per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_pf_h40 | rates | history | 2019-20 | 1 | personal fouls per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_stl_h40 | rates | history | 2019-20 | 1 | steals per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_blk_h40 | rates | history | 2019-20 | 1 | blocks per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_h40 | rates | history | 2019-20 | 1 | rebound chances per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_off_h40 | rates | history | 2019-20 | 1 | offensive rebound chances per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_reb_chances_def_h40 | rates | history | 2019-20 | 1 | defensive rebound chances per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_touches_h40 | rates | history | 2019-20 | 1 | touches per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_passes_h40 | rates | history | 2019-20 | 1 | passes per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_secondary_ast_h40 | rates | history | 2019-20 | 1 | secondary (hockey) assists per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_ft_ast_h40 | rates | history | 2019-20 | 1 | free-throw assists per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_contested_fga_h40 | rates | history | 2019-20 | 1 | contested field-goal attempts per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | r_usage_pct_h40 | rates | history | 2019-20 | 1 | usage percentage (minutes-weighted) per minute (half-life 40 games), ratio of decayed sums shrunk to a league prior from earlier seasons |
| player_game_base | c_fg_pct_h15 | conversion | history | 2019-20 | 1 | fg percentage: decayed made / attempted (half-life 15 games), shrunk toward the league percentage |
| player_game_base | c_fg_pct_h40 | conversion | history | 2019-20 | 1 | fg percentage: decayed made / attempted (half-life 40 games), shrunk toward the league percentage |
| player_game_base | c_fg3_pct_h15 | conversion | history | 2019-20 | 1 | fg3 percentage: decayed made / attempted (half-life 15 games), shrunk toward the league percentage |
| player_game_base | c_fg3_pct_h40 | conversion | history | 2019-20 | 1 | fg3 percentage: decayed made / attempted (half-life 40 games), shrunk toward the league percentage |
| player_game_base | c_ft_pct_h15 | conversion | history | 2019-20 | 1 | ft percentage: decayed made / attempted (half-life 15 games), shrunk toward the league percentage |
| player_game_base | c_ft_pct_h40 | conversion | history | 2019-20 | 1 | ft percentage: decayed made / attempted (half-life 40 games), shrunk toward the league percentage |
| player_game_base | pos_group | role | player_info | 2019-20 | 1 | Position group (G, F, C, GF, FC) from the static listing |
| player_game_base | m_bucket | minutes | history | 2019-20 | 1 | Trailing-minutes bucket (<12, 12-20, 20-28, 28+, -1 = no history) used to pick league priors |
| player_game_base | tm_off_rtg | team_context | team builders | 2019-20 | 1 | Offensive rating of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_def_rtg | team_context | team builders | 2019-20 | 1 | Defensive rating of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_net_rtg | team_context | team builders | 2019-20 | 1 | Net rating of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_pts | team_context | team builders | 2019-20 | 1 | Points scored of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_pts_5g | team_context | team builders | 2019-20 | 1 | Points scored, 5-game span of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_oreb | team_context | team builders | 2019-20 | 1 | Offensive rebounds of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_tov | team_context | team builders | 2019-20 | 1 | Turnovers committed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_ast | team_context | team builders | 2019-20 | 1 | Assists of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_fg3a | team_context | team builders | 2019-20 | 1 | Three-point attempts of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_fta | team_context | team builders | 2019-20 | 1 | Free-throw attempts of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_pts_allowed | team_context | team builders | 2019-20 | 1 | Points allowed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_pts_allowed_5g | team_context | team builders | 2019-20 | 1 | Points allowed, 5-game span of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_fga_allowed | team_context | team builders | 2019-20 | 1 | Field-goal attempts allowed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_fg_pct_allowed | team_context | team builders | 2019-20 | 1 | Field-goal percentage allowed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_fg3a_allowed | team_context | team builders | 2019-20 | 1 | Three-point attempts allowed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_fg3_pct_allowed | team_context | team builders | 2019-20 | 1 | Three-point percentage allowed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_fta_allowed | team_context | team builders | 2019-20 | 1 | Free-throw attempts allowed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_reb_allowed | team_context | team builders | 2019-20 | 1 | Rebounds allowed to opponents of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_reb_allowed_5g | team_context | team builders | 2019-20 | 1 | Rebounds allowed, 5-game span of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_oreb_allowed | team_context | team builders | 2019-20 | 1 | Offensive rebounds allowed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_ast_allowed | team_context | team builders | 2019-20 | 1 | Assists allowed of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_ast_allowed_5g | team_context | team builders | 2019-20 | 1 | Assists allowed, 5-game span of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_tov_forced | team_context | team builders | 2019-20 | 1 | Turnovers forced of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_win_pct_last10 | team_context | team builders | 2019-20 | 1 | Win percentage over the last 10 games of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_days_rest | schedule | team builders | 2019-20 | 1 | Days of rest before the game of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_is_b2b | schedule | team builders | 2019-20 | 1 | Second game of a back-to-back of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_games_last_7d | schedule | team builders | 2019-20 | 1 | Games in the previous 7 days of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_pace_proxy | team_context | team builders | 2019-20 | 1 | Possessions proxy (fga + 0.44 fta + tov - oreb) of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_misses | team_context | team builders | 2019-20 | 1 | Missed field goals per game (rebounds supplied by its own shooting) of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_misses_allowed | team_context | team builders | 2019-20 | 1 | Opponent missed field goals per game against this defense of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_dist_miles | schedule | team builders | 2019-20 | 1 | Travel distance to this game of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_tz_delta | schedule | team builders | 2019-20 | 1 | Time-zone change of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_body_clock_lag | schedule | team builders | 2019-20 | 1 | Body-clock lag after travel of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_road_trip_count | schedule | team builders | 2019-20 | 1 | Consecutive road games of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_game_altitude_diff | schedule | team builders | 2019-20 | 1 | Altitude difference of the venue vs the team home of the player team (pre-game EWMA, 20-game span unless stated) |
| player_game_base | tm_zone_ra_share_allowed | team_context | team shot zones | 2019-20 | 1 | share of shots allowed from the ra zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_ra_pct_allowed | team_context | team shot zones | 2019-20 | 1 | pct of shots allowed from the ra zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_paint_share_allowed | team_context | team shot zones | 2019-20 | 1 | share of shots allowed from the paint zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_paint_pct_allowed | team_context | team shot zones | 2019-20 | 1 | pct of shots allowed from the paint zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_mid_share_allowed | team_context | team shot zones | 2019-20 | 1 | share of shots allowed from the mid zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_mid_pct_allowed | team_context | team shot zones | 2019-20 | 1 | pct of shots allowed from the mid zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_c3_share_allowed | team_context | team shot zones | 2019-20 | 1 | share of shots allowed from the c3 zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_c3_pct_allowed | team_context | team shot zones | 2019-20 | 1 | pct of shots allowed from the c3 zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_atb3_share_allowed | team_context | team shot zones | 2019-20 | 1 | share of shots allowed from the atb3 zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | tm_zone_atb3_pct_allowed | team_context | team shot zones | 2019-20 | 1 | pct of shots allowed from the atb3 zone by the player team defense (20-game EWMA of earlier games) |
| player_game_base | opp_off_rtg | opponent | team builders | 2019-20 | 1 | Offensive rating of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_def_rtg | opponent | team builders | 2019-20 | 1 | Defensive rating of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_net_rtg | opponent | team builders | 2019-20 | 1 | Net rating of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_pts | opponent | team builders | 2019-20 | 1 | Points scored of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_pts_5g | opponent | team builders | 2019-20 | 1 | Points scored, 5-game span of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_oreb | opponent | team builders | 2019-20 | 1 | Offensive rebounds of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_tov | opponent | team builders | 2019-20 | 1 | Turnovers committed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_ast | opponent | team builders | 2019-20 | 1 | Assists of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_fg3a | opponent | team builders | 2019-20 | 1 | Three-point attempts of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_fta | opponent | team builders | 2019-20 | 1 | Free-throw attempts of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_pts_allowed | opponent | team builders | 2019-20 | 1 | Points allowed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_pts_allowed_5g | opponent | team builders | 2019-20 | 1 | Points allowed, 5-game span of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_fga_allowed | opponent | team builders | 2019-20 | 1 | Field-goal attempts allowed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_fg_pct_allowed | opponent | team builders | 2019-20 | 1 | Field-goal percentage allowed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_fg3a_allowed | opponent | team builders | 2019-20 | 1 | Three-point attempts allowed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_fg3_pct_allowed | opponent | team builders | 2019-20 | 1 | Three-point percentage allowed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_fta_allowed | opponent | team builders | 2019-20 | 1 | Free-throw attempts allowed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_reb_allowed | opponent | team builders | 2019-20 | 1 | Rebounds allowed to opponents of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_reb_allowed_5g | opponent | team builders | 2019-20 | 1 | Rebounds allowed, 5-game span of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_oreb_allowed | opponent | team builders | 2019-20 | 1 | Offensive rebounds allowed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_ast_allowed | opponent | team builders | 2019-20 | 1 | Assists allowed of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_ast_allowed_5g | opponent | team builders | 2019-20 | 1 | Assists allowed, 5-game span of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_tov_forced | opponent | team builders | 2019-20 | 1 | Turnovers forced of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_win_pct_last10 | opponent | team builders | 2019-20 | 1 | Win percentage over the last 10 games of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_days_rest | schedule | team builders | 2019-20 | 1 | Days of rest before the game of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_is_b2b | schedule | team builders | 2019-20 | 1 | Second game of a back-to-back of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_games_last_7d | schedule | team builders | 2019-20 | 1 | Games in the previous 7 days of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_pace_proxy | opponent | team builders | 2019-20 | 1 | Possessions proxy (fga + 0.44 fta + tov - oreb) of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_misses | opponent | team builders | 2019-20 | 1 | Missed field goals per game (rebounds supplied by its own shooting) of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_misses_allowed | opponent | team builders | 2019-20 | 1 | Opponent missed field goals per game against this defense of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_dist_miles | schedule | team builders | 2019-20 | 1 | Travel distance to this game of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_tz_delta | schedule | team builders | 2019-20 | 1 | Time-zone change of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_body_clock_lag | schedule | team builders | 2019-20 | 1 | Body-clock lag after travel of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_road_trip_count | schedule | team builders | 2019-20 | 1 | Consecutive road games of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_game_altitude_diff | schedule | team builders | 2019-20 | 1 | Altitude difference of the venue vs the team home of the opponent (pre-game EWMA, 20-game span unless stated) |
| player_game_base | opp_zone_ra_share_allowed | opponent | team shot zones | 2019-20 | 1 | share of shots allowed from the ra zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_ra_pct_allowed | opponent | team shot zones | 2019-20 | 1 | pct of shots allowed from the ra zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_paint_share_allowed | opponent | team shot zones | 2019-20 | 1 | share of shots allowed from the paint zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_paint_pct_allowed | opponent | team shot zones | 2019-20 | 1 | pct of shots allowed from the paint zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_mid_share_allowed | opponent | team shot zones | 2019-20 | 1 | share of shots allowed from the mid zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_mid_pct_allowed | opponent | team shot zones | 2019-20 | 1 | pct of shots allowed from the mid zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_c3_share_allowed | opponent | team shot zones | 2019-20 | 1 | share of shots allowed from the c3 zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_c3_pct_allowed | opponent | team shot zones | 2019-20 | 1 | pct of shots allowed from the c3 zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_atb3_share_allowed | opponent | team shot zones | 2019-20 | 1 | share of shots allowed from the atb3 zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | opp_zone_atb3_pct_allowed | opponent | team shot zones | 2019-20 | 1 | pct of shots allowed from the atb3 zone by the opponent defense (20-game EWMA of earlier games) |
| player_game_base | ctx_is_playoff | schedule | team builders | 2019-20 | 1 | Playoff game |
| player_game_base | ctx_team_game_number | schedule | team builders | 2019-20 | 1 | Number of the game in the team's season |
| player_game_base | ctx_rest_advantage | schedule | team builders | 2019-20 | 1 | Player team's days of rest minus the opponent's |
| player_game_at_T | inj_status | injury | injury reports |  | 1 | Status in the latest injury report at or before T (None = not listed, no_report = no report covers the game) |
| player_game_at_T | inj_status_code | injury | injury reports |  | 1 | Numeric status code (-1 no report, 0 none, 1 available, 2 probable, 3 questionable, 4 doubtful, 5 out) |
| player_game_at_T | inj_p_play | injury | injury reports |  | 1 | Probability of playing given status, decision time and trailing-minutes bucket, estimated from earlier seasons |
| player_game_at_T | inj_reason_class | injury | injury reports |  | 1 | Reason class from the report text (injury, illness, rest, g_league, personal, other, none) |
| player_game_at_T | inj_report_age_min | injury | injury reports |  | 1 | Minutes between the report and T (staleness of the information) |
| player_game_at_T | inj_report_missing | injury | injury reports |  | 1 | No injury report covers this game before T |
| player_game_at_T | av_miss_minutes | teammates | injury reports + history |  | 1 | Among teammates other than the player: expected share of minutes that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | av_miss_usage | teammates | injury reports + history |  | 1 | Among teammates other than the player: expected share of usage that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | av_miss_reb | teammates | injury reports + history |  | 1 | Among teammates other than the player: expected share of rebound chances that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | av_miss_ast | teammates | injury reports + history |  | 1 | Among teammates other than the player: expected share of assist creation that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | av_miss_pts | teammates | injury reports + history |  | 1 | Among teammates other than the player: expected share of scoring that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | av_n_out | teammates | injury reports + history |  | 1 | Among teammates other than the player: number with status Out that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | av_n_questionable | teammates | injury reports + history |  | 1 | Among teammates other than the player: number Questionable or Doubtful that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | av_miss_starter_minutes | teammates | injury reports + history |  | 1 | Among teammates other than the player: expected minutes of unavailable rotation players (24+ min) that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | opp_av_miss_minutes | teammates | injury reports + history |  | 1 | Among the opponent's whole roster: expected share of minutes that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | opp_av_miss_usage | teammates | injury reports + history |  | 1 | Among the opponent's whole roster: expected share of usage that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | opp_av_miss_reb | teammates | injury reports + history |  | 1 | Among the opponent's whole roster: expected share of rebound chances that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | opp_av_miss_ast | teammates | injury reports + history |  | 1 | Among the opponent's whole roster: expected share of assist creation that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | opp_av_miss_pts | teammates | injury reports + history |  | 1 | Among the opponent's whole roster: expected share of scoring that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | opp_av_n_out | teammates | injury reports + history |  | 1 | Among the opponent's whole roster: number with status Out that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | opp_av_miss_starter_minutes | teammates | injury reports + history |  | 1 | Among the opponent's whole roster: expected minutes of unavailable rotation players (24+ min) that is unavailable at T, sum of (1 - p_play) x trailing amount |
| player_game_at_T | env_team_spread | market_env | team_lines snapshots / SBR |  | 0.796 | Game spread from the player team view (negative = favourite) |
| player_game_at_T | env_total | market_env | team_lines snapshots / SBR |  | 0.796 | Game total |
| player_game_at_T | env_team_total | market_env | team_lines snapshots / SBR |  | 0.795 | Implied total of the player's team |
| player_game_at_T | env_opp_total | market_env | team_lines snapshots / SBR |  | 0.795 | Implied total of the opponent |
| player_game_at_T | env_team_win_prob | market_env | team_lines snapshots / SBR |  | 0.794 | No-vig win probability of the player's team |
| player_game_at_T | env_team_big_win_prob | market_env | team_lines snapshots / SBR |  | 0.796 | P(team wins by 15 or more), Normal(spread, sd 12) |
| player_game_at_T | env_team_big_loss_prob | market_env | team_lines snapshots / SBR |  | 0.796 | P(team loses by 15 or more) |
| player_game_at_T | env_blowout_prob | market_env | team_lines snapshots / SBR |  | 0.796 | P(|margin| >= 15) |
| player_game_at_T | env_n_books | market_env | team_lines snapshots / SBR |  | 0.796 | Books in the consensus |
| player_game_at_T | env_spread_dispersion | market_env | team_lines snapshots / SBR |  | 0.447 | Max minus min home spread across books |
| player_game_at_T | env_source | market_env | team_lines snapshots / SBR |  | 1 | 0 none, 1 our snapshot at T, 2 SBR opening line (proxy for T-240), 3 SBR closing line (proxy for T-5) |
