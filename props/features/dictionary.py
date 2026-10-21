"""
Feature dictionary: every feature column of the dataset tables with its group, source, meaning and availability by season.
A column with no matching entry is an error (tests/props/test_features_dictionary.py), so a feature cannot be added without being documented.
Key columns and label columns (prefixes y_ and lbl_) are listed separately and are never features.
"""
import re

import pandas as pd

KEYS = {'game_id', 'player_id', 'team_id', 'opp_team_id', 'season', 'game_date', 'game_type', 'is_home', 'minutes_before'}
LABEL_PREFIXES = ('y_', 'lbl_')

TEAM_DESC = {
    'off_rtg': 'offensive rating', 'def_rtg': 'defensive rating', 'net_rtg': 'net rating', 'pts': 'points scored', 'pts_5g': 'points scored, 5-game span',
    'oreb': 'offensive rebounds', 'tov': 'turnovers committed', 'ast': 'assists', 'fg3a': 'three-point attempts', 'fta': 'free-throw attempts',
    'pts_allowed': 'points allowed', 'pts_allowed_5g': 'points allowed, 5-game span', 'fga_allowed': 'field-goal attempts allowed',
    'fg_pct_allowed': 'field-goal percentage allowed', 'fg3a_allowed': 'three-point attempts allowed', 'fg3_pct_allowed': 'three-point percentage allowed',
    'fta_allowed': 'free-throw attempts allowed', 'reb_allowed': 'rebounds allowed to opponents', 'reb_allowed_5g': 'rebounds allowed, 5-game span',
    'oreb_allowed': 'offensive rebounds allowed', 'ast_allowed': 'assists allowed', 'ast_allowed_5g': 'assists allowed, 5-game span',
    'tov_forced': 'turnovers forced', 'win_pct_last10': 'win percentage over the last 10 games', 'days_rest': 'days of rest before the game',
    'is_b2b': 'second game of a back-to-back', 'games_last_7d': 'games in the previous 7 days', 'pace_proxy': 'possessions proxy (FGA + 0.44 FTA + TOV - OREB)',
    'misses': 'missed field goals per game (rebounds supplied by its own shooting)', 'misses_allowed': 'opponent missed field goals per game against this defense',
    'dist_miles': 'travel distance to this game', 'tz_delta': 'time-zone change', 'body_clock_lag': 'body-clock lag after travel',
    'road_trip_count': 'consecutive road games', 'game_altitude_diff': 'altitude difference of the venue vs the team home'}
RATE_DESC = {'points': 'points', 'fga': 'field-goal attempts', 'fgm': 'field goals made', 'fg3a': 'three-point attempts', 'fg3m': 'threes made',
             'fta': 'free-throw attempts', 'ftm': 'free throws made', 'rebounds': 'rebounds', 'oreb': 'offensive rebounds', 'dreb': 'defensive rebounds',
             'assists': 'assists', 'tov': 'turnovers', 'pf': 'personal fouls', 'stl': 'steals', 'blk': 'blocks', 'reb_chances': 'rebound chances',
             'reb_chances_off': 'offensive rebound chances', 'reb_chances_def': 'defensive rebound chances', 'touches': 'touches', 'passes': 'passes',
             'secondary_ast': 'secondary (hockey) assists', 'ft_ast': 'free-throw assists', 'contested_fga': 'contested field-goal attempts',
             'usage_pct': 'usage percentage (minutes-weighted)'}


def _hl(m):
    return f'half-life {m} games'


def describe(col):
    """Returns (group, source, description) or None."""
    if col in ('m_bucket',):
        return 'minutes', 'history', 'Trailing-minutes bucket (<12, 12-20, 20-28, 28+, -1 = no history) used to pick league priors'
    if col == 'pos_group':
        return 'role', 'player_info', 'Position group (G, F, C, GF, FC) from the static listing'
    if m := re.fullmatch(r'm_min_h(\d+)', col):
        return 'minutes', 'history', f'Decayed mean minutes per game played ({_hl(m[1])})'
    if m := re.fullmatch(r'm_games_eff_h(\d+)', col):
        return 'minutes', 'history', f'Effective number of earlier games, the sum of decay weights ({_hl(m[1])}): how much history backs the estimates'
    if col == 'm_min_sd_h15':
        return 'minutes', 'history', 'Decayed standard deviation of minutes (half-life 15 games)'
    if m := re.fullmatch(r'm_min_(blowout|close)_h40', col):
        return 'minutes', 'history', f'Decayed mean minutes in {m[1]} games (|final margin| >= 15 is a blowout), shrunk toward his overall mean (half-life 40)'
    if col == 'm_blowout_ratio_h40':
        return 'minutes', 'history', 'Minutes in blowouts divided by minutes in close games: blowout sensitivity'
    if m := re.fullmatch(r'role_p_played_h(\d+)', col):
        return 'role', 'history', f'Decayed share of his listed games in which he played ({_hl(m[1])})'
    if m := re.fullmatch(r'role_start_share_h(\d+)', col):
        return 'role', 'history', f'Decayed share of his listed games that he started ({_hl(m[1])}); uses EARLIER games only'
    rest = {'rest_days_since_played': 'Days since he last played', 'rest_returning': 'Returning from a gap of 10 or more days within a season',
            'rest_games_since_return': 'Games played since the latest return from such a gap (NaN if none)',
            'rest_games_played_season': 'Games played earlier this season'}
    if col in rest:
        return 'rest', 'history', rest[col]
    if m := re.fullmatch(r'r_(\w+?)_h(\d+)', col):
        return 'rates', 'history', f'{RATE_DESC.get(m[1], m[1])} per minute ({_hl(m[2])}), ratio of decayed sums shrunk to a league prior from earlier seasons'
    if m := re.fullmatch(r'c_(fg|fg3|ft)_pct_h(\d+)', col):
        return 'conversion', 'history', f'{m[1]} percentage: decayed made / attempted ({_hl(m[2])}), shrunk toward the league percentage'
    if m := re.fullmatch(r'(tm|opp)_zone_(ra|paint|mid|c3|atb3)_(share|pct)_allowed', col):
        who = 'the player team defense' if m[1] == 'tm' else 'the opponent defense'
        return ('team_context' if m[1] == 'tm' else 'opponent'), 'team shot zones', f'{m[3]} of shots allowed from the {m[2]} zone by {who} (20-game EWMA of earlier games)'
    if m := re.fullmatch(r'(tm|opp)_(\w+)', col):
        name = m[2]
        if name in TEAM_DESC:
            who = 'the player team' if m[1] == 'tm' else 'the opponent'
            grp = 'schedule' if name in ('days_rest', 'is_b2b', 'games_last_7d', 'dist_miles', 'tz_delta', 'body_clock_lag', 'road_trip_count', 'game_altitude_diff') \
                else ('team_context' if m[1] == 'tm' else 'opponent')
            return grp, 'team builders', f'{TEAM_DESC[name].capitalize()} of {who} (pre-game EWMA, 20-game span unless stated)'
    ctx = {'ctx_is_playoff': 'Playoff game', 'ctx_team_game_number': "Number of the game in the team's season",
           'ctx_rest_advantage': "Player team's days of rest minus the opponent's"}
    if col in ctx:
        return 'schedule', 'team builders', ctx[col]
    inj = {'inj_status': 'Status in the latest injury report at or before T (None = not listed, no_report = no report covers the game)',
           'inj_status_code': 'Numeric status code (-1 no report, 0 none, 1 available, 2 probable, 3 questionable, 4 doubtful, 5 out)',
           'inj_p_play': 'Probability of playing given status, decision time and trailing-minutes bucket, estimated from earlier seasons',
           'inj_reason_class': 'Reason class from the report text (injury, illness, rest, g_league, personal, other, none)',
           'inj_report_age_min': 'Minutes between the report and T (staleness of the information)',
           'inj_report_missing': 'No injury report covers this game before T'}
    if col in inj:
        return 'injury', 'injury reports', inj[col]
    if m := re.fullmatch(r'(opp_)?av_(miss_(minutes|usage|reb|ast|pts)|n_out|n_questionable|miss_starter_minutes)', col):
        who = "the opponent's whole roster" if m[1] else 'teammates other than the player'
        what = {'miss_minutes': 'expected share of minutes', 'miss_usage': 'expected share of usage', 'miss_reb': 'expected share of rebound chances',
                'miss_ast': 'expected share of assist creation', 'miss_pts': 'expected share of scoring', 'n_out': 'number with status Out',
                'n_questionable': 'number Questionable or Doubtful', 'miss_starter_minutes': 'expected minutes of unavailable rotation players (24+ min)'}[m[2]]
        return 'teammates', 'injury reports + history', f'Among {who}: {what} that is unavailable at T, sum of (1 - p_play) x trailing amount'
    env = {'env_team_spread': 'Game spread from the player team view (negative = favourite)', 'env_total': 'Game total',
           'env_team_total': "Implied total of the player's team", 'env_opp_total': "Implied total of the opponent",
           'env_team_win_prob': "No-vig win probability of the player's team", 'env_team_big_win_prob': 'P(team wins by 15 or more), Normal(spread, sd 12)',
           'env_team_big_loss_prob': 'P(team loses by 15 or more)', 'env_blowout_prob': 'P(|margin| >= 15)',
           'env_n_books': 'Books in the consensus', 'env_spread_dispersion': 'Max minus min home spread across books',
           'env_source': '0 none, 1 our snapshot at T, 2 SBR opening line (proxy for T-240), 3 SBR closing line (proxy for T-5)'}
    if col in env:
        return 'market_env', 'team_lines snapshots / SBR', env[col]
    return None


def is_key_or_label(col):
    return col in KEYS or col.startswith(LABEL_PREFIXES)


def build_dictionary(tables):
    """tables: {name: DataFrame}. One row per feature column per table, with availability by season."""
    rows = []
    for name, df in tables.items():
        if name == 'player_game_labels':
            continue
        season = df['season'] if 'season' in df.columns else None
        for col in df.columns:
            if is_key_or_label(col):
                continue
            info = describe(col)
            nn = df[col].notna()
            by_season = ''
            first = ''
            if season is not None:
                share = nn.groupby(season).mean()
                by_season = '; '.join(f'{s}:{v:.2f}' for s, v in share.items())
                with_data = share[share > 0]
                first = with_data.index.min() if len(with_data) else ''
            rows.append({'table': name, 'feature': col, 'group': info[0] if info else '', 'source': info[1] if info else '',
                         'description': info[2] if info else '', 'dtype': str(df[col].dtype), 'non_null_share': round(float(nn.mean()), 4),
                         'first_season_with_data': first, 'non_null_by_season': by_season})
    return pd.DataFrame(rows)


def missing_entries(tables):
    return [(n, c) for n, df in tables.items() if n != 'player_game_labels' for c in df.columns
            if not is_key_or_label(c) and describe(c) is None]
