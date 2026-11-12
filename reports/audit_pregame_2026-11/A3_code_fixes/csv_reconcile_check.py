"""Run from the worktree root with PYTHONPATH=. : apply the new CSV reconcile to a COPY of the production CSV
(argv[1]) using the snapshot DB read-only. Shows the 25 stale rows turn COMPLETED and nothing else changes."""
import sqlite3, sys, pandas as pd
from pathlib import Path
from core.prediction_logger import PredictionLogger, COLUMNS
copy = Path(sys.argv[1])
before = pd.read_csv(copy, dtype=str)
lg = object.__new__(PredictionLogger); lg.log_path = copy      # skip __init__: it would write to the DB
with sqlite3.connect('file:data/nba_stats.db?mode=ro', uri=True) as conn:
    lg._mirror_scores_to_csv(conn, None)
after = pd.read_csv(copy, dtype=str)
chg = before.compare(after) if len(before) == len(after) else None
print('rows', len(before), len(after), 'status PENDING before/after:', (before.status == 'PENDING').sum(), (after.status == 'PENDING').sum())
print('changed rows:', len(chg), '| changed columns:', sorted({c[0] for c in chg.columns}))
print(after[after.game_id.isin(before.loc[before.status != after.status, 'game_id'])].groupby('game_date').size().to_dict())
