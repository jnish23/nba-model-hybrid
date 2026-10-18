"""Entry point: record a timestamped odds snapshot. The implementation lives in core/snapshot_odds.py."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.snapshot_odds import main

if __name__ == "__main__":
    main()
