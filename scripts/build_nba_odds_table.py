"""
Sync SBR odds into the nba_odds table.

This file used to hold module-level copies of NBADatabaseSync.process_odds_data / sync_odds_data that still
referenced `self` (self.odds_columns, self.sportsbooks, self.db_path), so they raised NameError when called.
The maintained implementation is NBADatabaseSync.sync_odds_data (scripts/update_database.py); this is a thin
command-line wrapper around it.

Usage: python scripts/build_nba_odds_table.py --seasons 2025-26 [--db data/nba_stats.db]
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from update_database import NBADatabaseSync


def sync_odds_data(seasons, db_path='data/nba_stats.db'):
    """Sync odds for the given seasons (list of 'YYYY-YY' strings)."""
    NBADatabaseSync(db_path).sync_odds_data(list(seasons))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seasons', nargs='+', required=True)
    parser.add_argument('--db', default='data/nba_stats.db')
    args = parser.parse_args()
    sync_odds_data(args.seasons, args.db)
