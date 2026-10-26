import subprocess

import pandas as pd
import pytest

from props.live import publish


def _src(tmp_path):
    root = tmp_path / 'src'
    (root / 'dashboard_props').mkdir(parents=True)
    (root / 'dashboard_props' / 'app.py').write_text('app v1')
    (root / 'dashboard_props' / 'logic.py').write_text('logic')
    (root / 'core').mkdir()
    for f in ('__init__.py', 'live_feed.py', 'live_feed_overrides.csv'):
        (root / 'core' / f).write_text(f)
    (root / 'dashboard').mkdir()
    (root / 'dashboard' / 'requirements.txt').write_text('streamlit==1.37.1\n')
    (root / '.env').write_text('ODDS_API_KEY=secret')
    for d in ('props_dashboard_live', 'props_dashboard'):
        (root / 'data' / d).mkdir(parents=True)
        (root / 'data' / d / 'bets.csv').write_text('bet_id\nb1\n')
        (root / 'data' / d / 'meta.json').write_text('{}')
    (root / 'data' / 'props_dashboard_live' / 'secrets.txt').write_text('not exported: wrong suffix')
    return root


def _bare(tmp_path):
    bare = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '-q', '--bare', '-b', 'main', str(bare)], check=True)
    return str(bare)


def test_the_bundle_has_the_app_the_requirements_and_the_data_but_nothing_else(tmp_path):
    root, dest = _src(tmp_path), tmp_path / 'dest'
    dest.mkdir()
    files = publish.build_bundle(dest, root, now=pd.Timestamp('2026-10-24T14:00:00Z'))
    assert (dest / 'requirements.txt').read_text() == 'streamlit==1.37.1\n' and (dest / 'dashboard_props' / 'app.py').read_text() == 'app v1'
    assert (dest / 'data' / 'props_dashboard_live' / 'bets.csv').exists() and not (dest / '.env').exists() and not (dest / 'data' / 'props_dashboard_live' / 'secrets.txt').exists()
    assert 'data/props_dashboard/meta.json' in files
    assert {'core/__init__.py', 'core/live_feed.py', 'core/live_feed_overrides.csv'} <= set(files) and (dest / 'core' / 'live_feed.py').read_text() == 'live_feed.py'


def test_publish_pushes_once_then_reports_unchanged_and_pushes_again_when_data_changes(tmp_path):
    root, remote, clone = _src(tmp_path), _bare(tmp_path), tmp_path / 'clone'
    assert publish.publish(remote, clone, root, require_private=False, now=pd.Timestamp('2026-10-24T14:00:00Z')) == 'pushed'
    assert publish.publish(remote, clone, root, require_private=False, now=pd.Timestamp('2026-10-24T15:00:00Z')) == 'unchanged'
    (root / 'data' / 'props_dashboard_live' / 'bets.csv').write_text('bet_id\nb1\nb2\n')
    assert publish.publish(remote, clone, root, require_private=False, now=pd.Timestamp('2026-10-24T16:00:00Z')) == 'pushed'
    log = subprocess.run(['git', '--git-dir', remote, 'log', '--format=%s'], capture_output=True, text=True).stdout.split('\n')
    assert log[:2] == ['Data through 2026-10-24T16:00:00Z', 'Data through 2026-10-24T14:00:00Z']


def test_publish_refuses_a_remote_that_is_not_confirmed_private(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'is_private', lambda remote: False)
    with pytest.raises(RuntimeError, match='not confirmed to be a private'):
        publish.publish(_bare(tmp_path), tmp_path / 'c', _src(tmp_path))


def test_throttled_publish_skips_volatile_only_changes_until_the_interval_has_passed(tmp_path):
    root, remote, clone = _src(tmp_path), _bare(tmp_path), tmp_path / 'clone'
    (root / 'data' / 'props_dashboard_live' / 'checks.csv').write_text('a\n1\n')
    assert publish.publish(remote, clone, root, require_private=False, now=pd.Timestamp('2026-10-24T14:00:00Z')) == 'pushed'
    (root / 'data' / 'props_dashboard_live' / 'meta.json').write_text('{"built": 1}')                      # build time / health tables only
    (root / 'data' / 'props_dashboard_live' / 'checks.csv').write_text('a\n2\n')
    assert publish.publish(remote, clone, root, require_private=False, now=pd.Timestamp('2026-10-24T14:10:00Z'), min_interval_min=30) == 'unchanged'
    assert (clone / 'data' / 'props_dashboard_live' / 'meta.json').read_text() == '{}'                      # the clone is back at the last pushed state
    assert publish.publish(remote, clone, root, require_private=False, now=pd.Timestamp('2026-10-24T14:40:00Z'), min_interval_min=30) == 'pushed'     # 40 min since the last push
    (root / 'data' / 'props_dashboard_live' / 'bets.csv').write_text('bet_id\nb1\nb2\n')                   # a new bet is news: pushed at once
    assert publish.publish(remote, clone, root, require_private=False, now=pd.Timestamp('2026-10-24T14:45:00Z'), min_interval_min=30) == 'pushed'
