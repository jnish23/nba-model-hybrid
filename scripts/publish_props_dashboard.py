"""
Refresh the live dashboard export and publish it to the PRIVATE dashboard repository (Streamlit Community Cloud deploys from it; viewers are limited there to named people).

    python scripts/publish_props_dashboard.py              # refresh (settled bets, odds now) and publish
    python scripts/publish_props_dashboard.py --no-refresh # publish the export as it is

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
    a = ap.parse_args()
    if not a.no_refresh:
        refresh.refresh_live_export()
    print('publish:', publish.publish(a.remote, a.clone_dir))


if __name__ == '__main__':
    main()
