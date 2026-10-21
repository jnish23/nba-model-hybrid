"""
Write the props odds data-quality report.
    python scripts/report_props_quality.py                      # -> reports/props/data_quality_<today>.md
    python scripts/report_props_quality.py --out some.md --chunk-events 100
Reads data/props_odds.db (and data/nba_stats.db, read-only, for player tiers and the box-score selection check).
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props import quality  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--props-db', default='data/props_odds.db')
    ap.add_argument('--nba-db', default='data/nba_stats.db')
    ap.add_argument('--out', default=f'reports/props/data_quality_{dt.date.today()}.md')
    ap.add_argument('--chunk-events', type=int, default=150)
    args = ap.parse_args()
    nba = args.nba_db if Path(args.nba_db).exists() else None
    text = quality.build_report(args.props_db, nba, args.chunk_events)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding='utf-8')
    print(f'wrote {args.out} ({len(text.splitlines())} lines)')


if __name__ == '__main__':
    main()
