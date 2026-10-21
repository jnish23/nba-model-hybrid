"""
Plain-English names and groups for the model's feature columns, used by the game explanations (team/explain.py).

A feature name is built from parts: an optional `diff_` (home minus away), `off_<stat>` or `def_OPP_<stat>` (the
team's own stat, or what its opponents did against it), a window (`5g`, `20g`, `trend`) and a side (`_home` / `_away`).
Example: `off_offensiveRating_trend_away` -> "Away offensive rating (recent trend)", group "Shooting efficiency".

SHAP contributions add, so summing them per group keeps the breakdown exact while the 5-game, 20-game and trend
versions of the same stat (which overlap) read as one factor.
"""
import re
from dataclasses import dataclass

MARKET = 'Market line'
SHOOTING = 'Shooting efficiency'
VOLUME = 'Shot volume / pace'
DEFENSE = 'Defense allowed'
REBOUNDING = 'Rebounding'
BALL = 'Ball control'
INJURIES = 'Injuries & roster'
SCHEDULE = 'Rest / travel / schedule'
FORM = 'Recent form'
OTHER = 'Other'

GROUP_ORDER = [MARKET, SHOOTING, VOLUME, DEFENSE, REBOUNDING, BALL, INJURIES, SCHEDULE, FORM, OTHER]

WINDOWS = {'5g': 'last 5 games', '20g': 'last 20 games', 'trend': 'recent trend'}
SIDES = {'home': 'Home', 'away': 'Away', 'diff': 'Home minus away'}

# team's own box-score stats (`off_<stat>_<window>`): stat -> (group, label)
OFF_STATS = {
    'fieldGoalPercentage': (SHOOTING, 'field-goal %'),
    'threePointerPercentage': (SHOOTING, '3-point %'),
    'freeThrowPercentage': (SHOOTING, 'free-throw %'),
    'offensiveRating': (SHOOTING, 'offensive rating'),
    'points': (SHOOTING, 'points scored'),
    'fieldGoalsAttempted': (VOLUME, 'field-goal attempts'),
    'threePointersAttempted': (VOLUME, '3-point attempts'),
    'freeThrowsAttempted': (VOLUME, 'free-throw attempts'),
    'percentageFieldGoalsAttempted3pt': (VOLUME, 'share of shots from 3'),
    'percentagePointsFastBreak': (VOLUME, 'share of points on the fast break'),
    'percentagePointsPaint': (VOLUME, 'share of points in the paint'),
    'defensiveRating': (DEFENSE, 'defensive rating'),
    'blocks': (DEFENSE, 'blocks'),
    'steals': (DEFENSE, 'steals'),
    'offensiveRebounds': (REBOUNDING, 'offensive rebounds'),
    'defensiveRebounds': (REBOUNDING, 'defensive rebounds'),
    'turnovers': (BALL, 'turnovers'),
    'assists': (BALL, 'assists'),
    'assistToTurnover': (BALL, 'assist-to-turnover ratio'),
    'percentagePointsOffTurnovers': (BALL, 'share of points off turnovers'),
    'plusMinus': (FORM, 'plus-minus'),
    'netRating': (FORM, 'net rating'),
}

# what opponents did against the team (`def_OPP_<stat>_<window>`)
DEF_STATS = {
    'PTS': (DEFENSE, 'points'), 'FGM': (DEFENSE, 'field goals made'), 'FGA': (DEFENSE, 'field-goal attempts'),
    'FG_PCT': (DEFENSE, 'field-goal %'), 'FG3M': (DEFENSE, '3-pointers made'), 'FG3A': (DEFENSE, '3-point attempts'),
    'FG3_PCT': (DEFENSE, '3-point %'), 'FTM': (DEFENSE, 'free throws made'), 'FTA': (DEFENSE, 'free-throw attempts'),
    'FT_PCT': (DEFENSE, 'free-throw %'), 'AST': (DEFENSE, 'assists'), 'BLK': (DEFENSE, 'blocks'),
    'OREB': (REBOUNDING, 'offensive rebounds'), 'DREB': (REBOUNDING, 'defensive rebounds'), 'REB': (REBOUNDING, 'rebounds'),
    'TOV': (BALL, 'turnovers'), 'STL': (BALL, 'steals'),
}

# features without a window (side stripped): base name -> (group, label)
STATIC = {
    'odds_home_spread': (MARKET, 'opening home spread'),
    'odds_away_spread': (MARKET, 'opening away spread'),
    'odds_total': (MARKET, 'opening total'),
    'odds_home_ml': (MARKET, 'opening home moneyline'),
    'odds_away_ml': (MARKET, 'opening away moneyline'),
    'days_rest': (SCHEDULE, 'days of rest'),
    'is_b2b': (SCHEDULE, 'back-to-back'),
    'games_last_7d': (SCHEDULE, 'games in the last 7 days'),
    'season_game_number': (SCHEDULE, 'games played this season'),
    'is_playoff': (SCHEDULE, 'playoff game'),
    'dist_miles': (SCHEDULE, 'travel distance'),
    'tz_delta': (SCHEDULE, 'time-zone change'),
    'body_clock_lag': (SCHEDULE, 'body-clock lag'),
    'road_trip_count': (SCHEDULE, 'games into the road trip'),
    'game_altitude_diff': (SCHEDULE, 'altitude change'),
    'win_pct_last10': (FORM, 'win % over the last 10'),
    'win_streak': (FORM, 'win streak'),
    'home_win_pct_20g': (FORM, 'home win % (last 20 games)'),
    'roster_bpm': (INJURIES, 'roster quality (BPM)'),
    'roster_bpm_ewma': (INJURIES, 'roster quality (BPM, smoothed)'),
    'inj_lost': (INJURIES, 'quality lost to injuries'),
    'inj_lost_new': (INJURIES, 'quality lost to new injuries'),
    'inj_n_out': (INJURIES, 'players ruled out'),
    'inj_missing': (INJURIES, 'injury report missing'),
    'matchup_3pa_vs_opp_3pa_5g': (VOLUME, '3-point attempts vs the opponent (last 5 games)'),
}

_WINDOWED = re.compile(r'^(off|def_OPP)_(.+)_(5g|20g|trend)$')


class UnmappedFeature(KeyError):
    """A feature name the dictionary cannot parse; add it to the tables above."""


@dataclass(frozen=True)
class FeatureInfo:
    feature: str
    group: str
    label: str
    side: str           # 'home', 'away', 'diff' or '' for market features


def _split_side(name):
    if name.startswith('diff_'):
        return name[len('diff_'):], 'diff'
    for side in ('home', 'away'):
        if name.endswith('_' + side):
            return name[:-len(side) - 1], side
    return name, ''


def describe_feature(name, strict=True):
    """FeatureInfo for a feature column. With strict=False an unknown name falls back to group 'Other' and its raw name."""
    base, side = _split_side(name)
    m = _WINDOWED.match(base)
    if m:
        kind, stat, window = m.groups()
        table = OFF_STATS if kind == 'off' else DEF_STATS
        if stat in table:
            group, text = table[stat]
            if kind == 'def_OPP':
                text = f"defense: opponents' {text}"
            return FeatureInfo(name, group, f'{SIDES[side]} {text} ({WINDOWS[window]})' if side else f'{text} ({WINDOWS[window]})', side)
    elif base in STATIC:
        group, text = STATIC[base]
        label = f'{SIDES[side]} {text}' if side else text[0].upper() + text[1:]
        return FeatureInfo(name, group, label, side)
    elif name in STATIC:                     # odds_home_spread etc. end in 'spread'/'ml' but a name could end in a side word
        group, text = STATIC[name]
        return FeatureInfo(name, group, text[0].upper() + text[1:], '')
    if strict:
        raise UnmappedFeature(name)
    return FeatureInfo(name, OTHER, name, side)
