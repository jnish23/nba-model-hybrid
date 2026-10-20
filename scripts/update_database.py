"""Entry point: sync data/nba_stats.db from the NBA API and SBR. The implementation lives in core/update_database.py."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.update_database import main

if __name__ == "__main__":
    main()
