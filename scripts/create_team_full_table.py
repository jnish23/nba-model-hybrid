"""
Rebuild the team_game_full table (the input of every team-form feature).

This used to be a second, hand-written copy of the table definition that left out the OPP_<stat>
columns (the opponent's actual box-score totals, joined from the opponent's own row), so the
features silently fell back to summing the overlapping C/F/G positional columns (~22% too high).
The one maintained definition lives in NBADatabaseSync.rebuild_team_game_full(); this script just runs it.

Usage: python scripts/create_team_full_table.py [--db data/nba_stats.db]
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from update_database import NBADatabaseSync

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def create_full_table(db_path='data/nba_stats.db'):
    NBADatabaseSync(db_path).rebuild_team_game_full()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default='data/nba_stats.db')
    create_full_table(parser.parse_args().db)
