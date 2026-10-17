import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DB_PATH = ROOT / 'data' / 'nba_stats.db'


def pytest_configure(config):
    config.addinivalue_line('markers', 'db: needs data/nba_stats.db (skipped in CI)')


def pytest_collection_modifyitems(config, items):
    import pytest
    if DB_PATH.exists():
        return
    skip = pytest.mark.skip(reason='data/nba_stats.db not available')
    for item in items:
        if 'db' in item.keywords:
            item.add_marker(skip)
