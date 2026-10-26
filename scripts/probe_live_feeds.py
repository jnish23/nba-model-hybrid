"""
Terminal version of the live-feed check (the checks themselves live in dashboard_props/live_probe.py so the deployable page and this script cannot drift apart).

    python scripts/probe_live_feeds.py                     # print status, size and latency of the four feeds
    python scripts/probe_live_feeds.py --save tests/fixtures/live_feed   # also keep the JSON bodies as test fixtures
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dashboard_props.live_probe import run_checks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', help='folder to write each JSON response into')
    a = ap.parse_args()
    t = run_checks(a.save)
    print(t.drop(columns=['url']).to_string(index=False))


if __name__ == '__main__':
    main()
