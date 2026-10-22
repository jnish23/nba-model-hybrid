"""The model-input frame for live rows: base block + T-dependent block, with the same categorical encoding as the training frame (props.models.data.load_frame)."""
import pandas as pd

from props.models import data


def live_frame(base_live, at_t_blocks):
    """at_t_blocks: list of live_at_t outputs (one per decision time). Returns one row per (player, game, decision time) with every feature column of the training frame that does not depend on labels or the market environment."""
    at_t = pd.concat(at_t_blocks, ignore_index=True)
    df = at_t.merge(base_live, on=['game_id', 'player_id'], how='left', validate='many_to_one')
    return data.encode_categoricals(df).sort_values(['game_date', 'game_id', 'player_id', 'minutes_before']).reset_index(drop=True)
