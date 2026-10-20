"""
Model inputs from the Phase 2 dataset: one merged frame per (player, game, decision time), the feature sets, and fixed category vocabularies.

Feature sets (groups come from props/features/dictionary.py):
  base      every group except market_env
  with_env  base + market_env (game spread, total, blowout probability)
Ablations drop a group by name (exclude_groups). PLAYER PROP LINES ARE NEVER INPUTS and no column of the dataset contains one.
Outcomes (`y_*`, `lbl_*`) are never features; `FeatureSpec.check()` fails if one slips in.
"""
from dataclasses import dataclass, field

import pandas as pd

from props.features import build, dictionary

POS_GROUPS = ['G', 'F', 'C', 'GF', 'FC']
REASON_CLASSES = ['none', 'injury', 'illness', 'rest', 'g_league', 'personal', 'other']
CATEGORICAL = {'pos_group': POS_GROUPS, 'inj_reason_class': REASON_CLASSES}
NOT_FEATURES = {'game_id', 'player_id', 'team_id', 'opp_team_id', 'season', 'game_date', 'game_type', 'inj_status',   # keys, dates, and a string duplicate of inj_status_code
                'env_source'}                                                                                         # which source supplied the market environment: it encodes the era (SBR proxy before 2023-24), so it is never a model input
BASE_GROUPS = {'minutes', 'role', 'rest', 'rates', 'conversion', 'team_context', 'opponent', 'schedule', 'injury', 'teammates'}
ENV_GROUPS = {'market_env'}
BUBBLE_START = '2020-07-30'                       # games from here to the end of 2019-20 were played in the Orlando bubble
T_COLUMN = 'minutes_before'


@dataclass(frozen=True)
class FeatureSpec:
    name: str = 'base'
    exclude_groups: frozenset = field(default_factory=frozenset)
    include_decision_time: bool = True            # the pooled models see how many minutes before tip the information is from
    exclude_features: frozenset = field(default_factory=frozenset)   # single columns dropped by name (R1: season-position features)

    def groups(self):
        g = set(BASE_GROUPS) | (ENV_GROUPS if self.name == 'with_env' else set())
        return g - set(self.exclude_groups)

    def columns(self, frame_columns):
        cols = []
        for c in frame_columns:
            if c in NOT_FEATURES or dictionary.is_key_or_label(c) and c != T_COLUMN:
                continue
            if c in self.exclude_features:
                continue
            if c == T_COLUMN:
                if self.include_decision_time:
                    cols.append(c)
                continue
            info = dictionary.describe(c)
            if info is not None and info[0] in self.groups():
                cols.append(c)
        return cols

    def check(self, cols):
        bad = [c for c in cols if c.startswith(('y_', 'lbl_')) or c in NOT_FEATURES]
        if bad:
            raise ValueError(f'outcome or key columns among the features: {bad}')


def encode_categoricals(df):
    for c, vocab in CATEGORICAL.items():
        if c in df.columns:
            df[c] = pd.Categorical(df[c].fillna(vocab[0] if c != 'pos_group' else 'F'), categories=vocab)
    return df


def load_frame(out_db=build.OUT_DB):
    """All listed player-games x decision times with features, keys, dates and labels. Played rows have lbl_played == 1."""
    t = build.read_tables(out_db)
    base, labels, at_t = t['player_game_base'], t['player_game_labels'], t['player_game_at_T']
    df = at_t.merge(base, on=['game_id', 'player_id'], how='left').merge(labels, on=['game_id', 'player_id'], how='left')
    df = encode_categoricals(df)
    df['weight'] = 1.0 / df.groupby(['game_id', 'player_id'])[T_COLUMN].transform('size')       # each player-game counts once across its decision times
    return df.sort_values(['game_date', 'game_id', 'player_id', T_COLUMN]).reset_index(drop=True)
