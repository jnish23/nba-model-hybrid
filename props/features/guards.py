"""Guards for the dataset build: de-duplication that says how much it removed, and merges that state their cardinality."""
import logging

DEFAULT_MAX_SHARE = 0.02


def table_coverage(roster_games, table_games_by_name):
    """
    Game ids listed in the roster table but missing from each stat table: {table name: sorted list of game ids}, only tables with a gap.
    roster_games: iterable of game ids; table_games_by_name: {name: iterable of game ids present in that table}.
    """
    roster = {str(g) for g in roster_games}
    out = {}
    for name, games in table_games_by_name.items():
        missing = sorted(roster - {str(g) for g in games})
        if missing:
            out[name] = missing
    return out


def played_null_report(df, cols):
    """Number of PLAYED rows (lbl_played == 1) with a null in each column of `cols` that exists in df: {column: count}, only columns with a null."""
    played = df[df['lbl_played'].astype(bool)]
    out = {}
    for c in cols:
        if c in played.columns:
            n = int(played[c].isna().sum())
            if n:
                out[c] = n
    return out


def dedupe(df, keys, what, max_share=DEFAULT_MAX_SHARE):
    """drop_duplicates(keys) that logs what it removed and refuses to hide a large problem (more than max_share of the rows)."""
    out = df.drop_duplicates(keys)
    dropped = len(df) - len(out)
    if dropped:
        logging.info(f'{what}: dropped {dropped} duplicate rows on {keys} ({dropped / len(df):.3%})')
    if dropped > max_share * len(df):
        raise ValueError(f'{what}: {dropped} of {len(df)} rows are duplicates on {keys}; more than {max_share:.0%}, refusing to hide it')
    return out
