"""
Build the tables the props dashboard reads (backtest mode) into data/props_dashboard/.

    python scripts/export_props_dashboard_data.py            # all three seasons
    python scripts/export_props_dashboard_data.py --seasons 2025-26
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.betting import export  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/props_dashboard')
    ap.add_argument('--seasons', nargs='*', default=None)
    ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args()
    export.export_all(a.out, seasons=a.seasons, n_boot=a.n_boot)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
