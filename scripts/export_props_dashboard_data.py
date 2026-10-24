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
    ap.add_argument('--live', action='store_true', help='export the live paper record (data/props_live.db) to data/props_dashboard_live instead of the backtest')
    a = ap.parse_args()
    if a.live:
        from props.live import refresh
        refresh.refresh_live_export(a.out if a.out != 'data/props_dashboard' else 'data/props_dashboard_live', n_boot=a.n_boot)
        return
    export.export_all(a.out, seasons=a.seasons, n_boot=a.n_boot)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
