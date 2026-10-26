"""
Refresh the live dashboard export and publish it to the PRIVATE dashboard repository (Streamlit Community Cloud deploys from it; viewers are limited there to named people).

    python scripts/publish_props_dashboard.py              # refresh (settled bets, odds now) and publish
    python scripts/publish_props_dashboard.py --no-refresh # publish the export as it is
    python scripts/publish_props_dashboard.py --if-active  # the 10-minute chain: only while bets are live (prices before the games, provisional results after); throttled pushes

Run every morning by the "NBA Props Settle" task (after settlement) and by hand any time to see newer odds (scripts\run_props_publish.bat). Refuses to push unless the remote is private.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.live import publish, refresh  # noqa: E402

REMOTE = 'https://github.com/jnish23/nba-props-dashboard.git'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--remote', default=REMOTE)
    ap.add_argument('--clone-dir', default='data/props_private_repo')
    ap.add_argument('--no-refresh', action='store_true')
    ap.add_argument('--if-active', action='store_true', help='do nothing unless an unsettled bet is within 8 h before / 7 h after its tip; push at most every --min-interval minutes unless the bets, prices or results changed')
    ap.add_argument('--min-interval', type=float, default=30.0)
    a = ap.parse_args()
    if a.if_active and not refresh.active_now():
        print('idle: no unsettled bet near its tip')
        return
    if not a.no_refresh:
        refresh.refresh_live_export()
    print('publish:', publish.publish(a.remote, a.clone_dir, min_interval_min=a.min_interval if a.if_active else None))


if __name__ == '__main__':
    main()
