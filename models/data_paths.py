"""Shared lookups for generated data files."""
import re
from pathlib import Path

PROCESSED_DIR = Path('data/processed')
_TRAINING_SET_RE = re.compile(r'^training_set_v1_reduced_(\d{4}-\d{2}-\d{2})\.csv$')


def latest_training_set(processed_dir=PROCESSED_DIR) -> Path:
    """
    Newest training set written by scripts/generate_training_set.py
    (training_set_v1_reduced_YYYY-MM-DD.csv). Only exact dated names match: a plain glob also
    picked up derived files such as '..._YYYY-MM-DD_no_inj.csv', which sort after the real one.
    """
    dated = [(m.group(1), p) for p in Path(processed_dir).glob('training_set_v1_reduced_*.csv')
             if (m := _TRAINING_SET_RE.match(p.name))]
    if not dated:
        raise FileNotFoundError(f"No training_set_v1_reduced_YYYY-MM-DD.csv in {processed_dir}")
    return max(dated)[1]
