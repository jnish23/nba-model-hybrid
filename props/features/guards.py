"""Guards for the dataset build: de-duplication that says how much it removed, and merges that state their cardinality."""
import logging

DEFAULT_MAX_SHARE = 0.02


def dedupe(df, keys, what, max_share=DEFAULT_MAX_SHARE):
    """drop_duplicates(keys) that logs what it removed and refuses to hide a large problem (more than max_share of the rows)."""
    out = df.drop_duplicates(keys)
    dropped = len(df) - len(out)
    if dropped:
        logging.info(f'{what}: dropped {dropped} duplicate rows on {keys} ({dropped / len(df):.3%})')
    if dropped > max_share * len(df):
        raise ValueError(f'{what}: {dropped} of {len(df)} rows are duplicates on {keys}; more than {max_share:.0%}, refusing to hide it')
    return out
