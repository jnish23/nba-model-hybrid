import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from props.live import publish


def _src(tmp_path):
    root = tmp_path / 'src'
    (root / 'dashboard_props').mkdir(parents=True)
    (root / 'dashboard_props' / 'app.py').write_text('app v1')
    (root / 'dashboard_props' / 'logic.py').write_text('logic')
    (root / 'core').mkdir()
    for f in ('__init__.py', 'live_feed.py', 'live_feed_overrides.csv', 'props_board_logic.py'):
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


def test_the_published_app_finds_its_modules_when_only_its_own_folder_is_on_the_path(tmp_path):
    """Streamlit Cloud runs `streamlit run dashboard_props/app.py`: the script's folder is importable, the bundle root is not until app.py adds it."""
    repo = Path(__file__).resolve().parents[2]
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    publish.build_bundle(bundle, repo, live_dir=tmp_path / 'none', backtest_dir=tmp_path / 'none')
    app = bundle / 'dashboard_props' / 'app.py'
    code = ("import sys\n"
            f"app = {str(app)!r}\n"
            "sys.path.insert(0, app.rsplit('dashboard_props', 1)[0] + 'dashboard_props')\n"
            "src = open(app, encoding='utf-8').read()\n"
            "head = src[:src.index('BACKTEST_DIR =')]                 # the imports and path set-up, before the page is drawn\n"
            "ns = {'__file__': app, '__name__': 'app_head'}\n"
            "exec(compile(head, app, 'exec'), ns)\n"
            "import core.props_board_logic as m\n"
            "assert ns['F'] is not None\n"
            "print(ns['L'].__file__); print(m.__file__)\n")
    r = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path, capture_output=True, text=True)      # -I: no PYTHONPATH, no current folder on the path
    assert r.returncode == 0, r.stderr[-2000:]
    logic_file, core_file = r.stdout.split()[-2:]
    assert Path(logic_file).resolve() == (bundle / 'dashboard_props' / 'logic.py').resolve()
    assert Path(core_file).resolve() == (bundle / 'core' / 'props_board_logic.py').resolve()
