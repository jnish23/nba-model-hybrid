"""Config fallback and standalone deployment contracts for mechanical moves."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parent.parent


def test_configuration_prefers_canonical_and_preserves_legacy_fallback(tmp_path,monkeypatch):
    from props.live import config
    monkeypatch.chdir(tmp_path)
    (tmp_path/'config').mkdir()
    (tmp_path/'reports/props').mkdir(parents=True)
    canonical=tmp_path/config.CONFIG_PATH
    legacy=tmp_path/config.LEGACY_CONFIG_PATH
    canonical.write_text('{"source":"canonical"}',encoding='utf-8')
    legacy.write_text('{"source":"legacy"}',encoding='utf-8')
    assert config.load()=={'source':'canonical'}
    canonical.unlink()
    assert config.load(config.CONFIG_PATH)=={'source':'legacy'}
    for path in (config.RULE_PATH,config.LEGACY_RULE_PATH):
        (tmp_path/path).write_text('{}',encoding='utf-8')
    assert config.resolve_rule_path()==Path(config.RULE_PATH)
    (tmp_path/config.RULE_PATH).unlink()
    assert config.resolve_rule_path()==Path(config.LEGACY_RULE_PATH)


def test_relocated_frozen_rule_is_exact_and_config_changes_only_its_path():
    old=(ROOT/'reports/props/phase5_revised_rule.frozen_rule.json').read_bytes()
    new=(ROOT/'config/props_frozen_rule.json').read_bytes()
    assert old==new
    original=json.loads((ROOT/'reports/props/live_config.json').read_text(encoding='utf-8'))
    relocated=json.loads((ROOT/'config/props_live.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(new).hexdigest()==relocated['rule_sha256']
    assert relocated.pop('rule_file')=='config/props_frozen_rule.json'
    original.pop('rule_file')
    assert relocated==original


def test_generated_bundle_imports_logic_without_source_packages(tmp_path):
    from props.live.publish import build_bundle
    empty=tmp_path/'empty';empty.mkdir()
    bundle=tmp_path/'bundle';bundle.mkdir()
    build_bundle(bundle,ROOT,live_dir=empty,backtest_dir=empty,now=pd.Timestamp('2026-11-01T12:00:00Z'))
    code='''import sys,importlib.abc,pathlib
sys.path.insert(0,str(pathlib.Path.cwd()))
class ForbiddenSource(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'props','team','experiments','scripts'}:raise AssertionError(fullname)
sys.meta_path.insert(0,ForbiddenSource())
from dashboard_props import logic
from core import props_board_logic
assert logic.board is props_board_logic.board
assert pathlib.Path(props_board_logic.__file__).is_relative_to(pathlib.Path.cwd())
assert 'median' in logic.PRICE_MODES
'''
    subprocess.run([sys.executable,'-I','-c',code],cwd=bundle,check=True,capture_output=True,text=True)


def test_backtest_export_default_rule_records_pinned_hash(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from props.betting import export
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(export, 'SEASONS', {'synthetic': {'kind': 'mock'}})
    monkeypatch.setattr(export, 'labels_table', lambda _: pd.DataFrame())
    frame = pd.DataFrame({'game_id': [1], 'player_id': [2]})
    monkeypatch.setattr(export, 'build_season', lambda *a: (frame, {}, {}, {}))
    monkeypatch.setattr(export, 'reliability_and_slope', lambda *a: (pd.DataFrame({'n': [1]}), {}))
    monkeypatch.setattr(export, 'monthly_bias', lambda *a: pd.DataFrame({'n': [1]}))
    monkeypatch.setattr(export.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='offline-fixture'))
    feature_file = tmp_path / 'features.db'
    feature_file.write_bytes(b'fixture')
    export.export_all(tmp_path / 'export', features_db=feature_file, rule_path=None)
    meta = json.loads((tmp_path / 'export/meta.json').read_text())
    assert meta['rule_sha256'] == 'b597763ab3b13fd9cf295b1de802255ac87fe6e9bbf065103171db49eec2b6fb'


def test_config_write_none_resolves_default_and_creates_parent(tmp_path, monkeypatch):
    from props.live import config
    monkeypatch.chdir(tmp_path)
    cfg = {'version': 'synthetic'}
    config.write(cfg, path=None)
    assert json.loads((tmp_path / config.CONFIG_PATH).read_text()) == cfg
    custom = tmp_path / 'nested/custom.json'
    config.write(cfg, path=custom)
    assert json.loads(custom.read_text()) == cfg
    (tmp_path / config.CONFIG_PATH).unlink()
    legacy = tmp_path / config.LEGACY_CONFIG_PATH
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{}')
    config.write(cfg, path=None)
    assert json.loads(legacy.read_text()) == cfg
