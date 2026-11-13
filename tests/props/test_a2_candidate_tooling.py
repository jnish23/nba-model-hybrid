"""A2 candidate tooling (plan reports/props/audit_2026-11/A2_live_plan.md, P2): the live training script, the candidate config and the signal job's cache name. Synthetic, no database."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from props.live import config
from props.models import data, folds as folds_mod, registry

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load('train_props_live_under_test', 'scripts/train_props_live.py')
job = _load('props_signal_job_a2_under_test', 'scripts/props_signal_job.py')

DEFAULT_KEYS = ['kind', 'next_season', 'trained_through', 'policy', 'spec', 'families', 'families_from', 'refit_all', 'recalibration', 'dispersion_floor', 'fit_rows', 'calib_rows',
                'calibration_starts', 'sims', 'seed', 'features_db', 'features_db_sha256', 'rows', 'code_commit']
FAMILIES = {'fg2a': 'hier', 'ast': 'lgb'}


def _frame():
    return pd.DataFrame({'game_date': ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01'], 'season': ['2024-25'] * 4})


def _fold():
    return SimpleNamespace(fit_idx=np.array([0, 1]), calib_idx=np.array([2, 3]), cutoff_date='2025-03-01')


# ------------------------------------------------------------------------------------------------------------------------------ training script
def test_default_arguments_are_unchanged():
    a = train.parse_args(['--version', 'v', '--next-season', '2026-27'])
    assert (a.features_db, a.families_from, a.spec, a.sims, a.seed) == ('data/props_features_v3.db', 'v2_base', 'base', 4000, 0)


def test_default_manifest_has_the_original_keys_and_values(tmp_path):
    db = tmp_path / 'f.db'
    db.write_bytes(b'x')
    a = train.parse_args(['--version', 'v', '--next-season', '2026-27', '--features-db', str(db)])
    man = train.build_manifest(a, _frame(), _fold(), FAMILIES, 'abc', 'hash')
    assert list(man) == DEFAULT_KEYS                                         # no extra key (no sidecar next to the database), same order
    assert man['spec'] == 'base' and man['trained_through'] == '2025-04-01' and man['fit_rows'] == 2 and man['calib_rows'] == 2
    assert man['features_db'] == str(db) and man['features_db_sha256'] == 'hash' and man['rows'] == 4 and man['code_commit'] == 'abc'
    assert man['policy'] == 'pre-season training; monthly refits are new immutable versions' and man['sims'] == 4000 and man['seed'] == 0


def test_a_research_built_database_adds_only_its_build_record(tmp_path):
    db = tmp_path / 'f.db'
    db.write_bytes(b'x')
    Path(str(db) + '.manifest.json').write_text(json.dumps({'membership': 'pregame_all', 'max_date': '2026-07-31', 'sha256': 's', 'code_commit': 'c', 'linked_cache_sha256': 'l', 'rows': {}}))
    a = train.parse_args(['--version', 'v', '--next-season', '2026-27', '--features-db', str(db)])
    man = train.build_manifest(a, _frame(), _fold(), FAMILIES, 'abc', 'hash')
    assert list(man) == DEFAULT_KEYS + ['features_db_build']
    assert man['features_db_build'] == {'membership': 'pregame_all', 'max_date': '2026-07-31', 'sha256': 's', 'code_commit': 'c', 'linked_cache_sha256': 'l'}


def test_the_spec_flag_accepts_only_the_two_specs():
    assert train.parse_args(['--version', 'v', '--next-season', 's', '--spec', 'with_env']).spec == 'with_env'
    with pytest.raises(SystemExit):
        train.parse_args(['--version', 'v', '--next-season', 's', '--spec', 'other'])


def _fake_run(monkeypatch, tmp_path, seen):
    """Everything heavy is replaced; the real parse, manifest, registry writes and FeatureSpec construction run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data/props_models/v2_base').mkdir(parents=True)
    (tmp_path / 'data/props_models/v2_base/manifest.json').write_text(json.dumps({'families': FAMILIES}))
    (tmp_path / 'f.db').write_bytes(b'x')
    monkeypatch.setattr(train.data, 'load_frame', lambda path: _frame())
    monkeypatch.setattr(train.folds_mod, 'make_live_fold', lambda df, season: _fold())

    def fake_train(df, fold, spec, score_on_test, refit_all):
        seen['spec'] = spec
        return {'spec': spec}
    monkeypatch.setattr(train.pipeline, 'train_fold', fake_train)
    monkeypatch.setattr(train.pipeline, 'calibrate_fold', lambda *a: (SimpleNamespace(c={('points', 0): 1.0}), 'cal'))


def test_spec_with_env_reaches_the_feature_spec_and_the_manifest(tmp_path, monkeypatch):
    seen = {}
    _fake_run(monkeypatch, tmp_path, seen)
    train.main(['--version', 'v4_test', '--next-season', '2026-27', '--features-db', 'f.db', '--spec', 'with_env'])
    assert seen['spec'] == data.FeatureSpec('with_env') and seen['spec'].name == 'with_env'
    man = json.loads((tmp_path / 'data/props_models/v4_test/manifest.json').read_text())
    assert man['spec'] == 'with_env' and man['features_db'] == 'f.db' and man['dispersion_c'] == {'points:0': 1.0}
    assert (tmp_path / 'data/props_models/v4_test/2026-27/bundle.pkl').exists()


def test_the_default_call_uses_the_base_spec_and_an_existing_version_is_refused(tmp_path, monkeypatch):
    seen = {}
    _fake_run(monkeypatch, tmp_path, seen)
    args = ['--version', 'v4_test', '--next-season', '2026-27', '--features-db', 'f.db']
    train.main(args)
    assert seen['spec'] == data.FeatureSpec('base')
    assert json.loads((tmp_path / 'data/props_models/v4_test/manifest.json').read_text())['spec'] == 'base'
    seen.clear()
    with pytest.raises(FileExistsError, match='immutable'):                   # registry.write_manifest refuses before any training
        train.main(args)
    assert seen == {}


def test_a_frame_with_rows_of_the_next_season_cannot_be_trained():
    df = pd.DataFrame({'game_date': ['2026-06-13', '2026-11-01'], 'season': ['2025-26', '2026-27']})
    with pytest.raises(ValueError, match='already has rows'):
        folds_mod.make_live_fold(df, '2026-27')


# ------------------------------------------------------------------------------------------------------------------------------ candidate config
PROD_BYTES = b'{"model_version": "deployed", "keep": "this exactly"}\n'


def _world(tmp_path, monkeypatch, version='v4_test', season='2026-27', db_name='data/props_features_v4.db'):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/props_live.json').write_bytes(PROD_BYTES)
    (tmp_path / 'config/props_frozen_rule.json').write_text('{"k": 1}')
    db = tmp_path / db_name
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b'features' * 10)
    vdir = tmp_path / 'data/props_models' / version
    (vdir / season).mkdir(parents=True)
    (vdir / season / 'bundle.pkl').write_bytes(b'bundle')
    (vdir / 'manifest.json').write_text(json.dumps({'next_season': season, 'spec': 'with_env', 'features_db': db_name, 'features_db_sha256': registry.sha256_file(db)}))
    return db


def _pin(**kw):
    args = dict(model_version='v4_test', model_season='2026-27', effective_from='2026-11-20', features_db='data/props_features_v4.db', roster_rule='pregame_all', waivers=['A5'])
    args.update(kw)
    return config.pin_candidate(args.pop('model_version'), args.pop('model_season'), args.pop('effective_from'), args.pop('features_db'), args.pop('roster_rule'), **args)


def test_candidate_config_goes_to_its_own_path_and_leaves_production_byte_identical(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    cfg = _pin()
    assert (tmp_path / 'config/props_live.json').read_bytes() == PROD_BYTES
    on_disk = json.loads((tmp_path / config.CANDIDATE_CONFIG_PATH).read_text())
    assert on_disk == cfg and config.load(config.CANDIDATE_CONFIG_PATH) == cfg
    assert (cfg['model_version'], cfg['model_season'], cfg['roster_rule'], cfg['effective_from'], cfg['n_sims'], cfg['spec']) == ('v4_test', '2026-27', 'pregame_all', '2026-11-20', 20000, 'with_env')
    assert cfg['rule_file'] == 'config/props_frozen_rule.json' and cfg['features_db'] == 'data/props_features_v4.db' and cfg['features_db_bytes'] == 80
    assert cfg['model_bundle_sha256'] == registry.sha256_file(tmp_path / 'data/props_models/v4_test/2026-27/bundle.pkl')
    assert cfg['features_db_sha256'] == registry.sha256_file(tmp_path / 'data/props_features_v4.db')
    assert config.verify(cfg, check_features_db=False) is cfg


def test_a_candidate_may_not_be_written_over_the_deployed_config(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    for p in (config.CONFIG_PATH, config.LEGACY_CONFIG_PATH, tmp_path / 'config/../config/props_live.json'):
        with pytest.raises(ValueError, match='deployed'):
            _pin(path=p)
    assert (tmp_path / 'config/props_live.json').read_bytes() == PROD_BYTES
    assert not (tmp_path / config.LEGACY_CONFIG_PATH).exists()


def test_verify_detects_a_wrong_hash_on_the_candidate(tmp_path, monkeypatch):
    db = _world(tmp_path, monkeypatch)
    cfg = _pin()
    config.verify(config.load(config.CANDIDATE_CONFIG_PATH), check_features_db=True)
    bad = dict(cfg, model_bundle_sha256='0' * 64)
    with pytest.raises(config.ConfigRefused, match='model bundle'):
        config.verify(bad)
    with pytest.raises(config.ConfigRefused, match='rule file'):
        config.verify(dict(cfg, rule_sha256='0' * 64))
    with pytest.raises(config.ConfigRefused, match='features database'):
        config.verify(dict(cfg, features_db_sha256='0' * 64), check_features_db=True)
    db.write_bytes(b'changed')
    with pytest.raises(config.ConfigRefused, match='features database'):
        config.verify(cfg)


def test_waivers_must_be_given_explicitly(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    with pytest.raises(TypeError):                                           # keyword-only, no default
        config.pin_candidate('v4_test', '2026-27', '2026-11-20', 'data/props_features_v4.db', 'pregame_all')
    for bad in (None, 'A5', ['A5', 3]):
        with pytest.raises(TypeError, match='waivers'):
            _pin(waivers=bad)
    assert not (tmp_path / config.CANDIDATE_CONFIG_PATH).exists()
    assert _pin(waivers=[])['waivers'] == []
    assert _pin(waivers=('A5', 'A6'))['waivers'] == ['A5', 'A6']             # never the deployed list unless named


def test_the_candidate_must_match_the_model_it_names(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='trained for'):
        _pin(model_season='2027-28')
    (tmp_path / 'data/props_features_other.db').write_bytes(b'x')
    with pytest.raises(ValueError, match='trained on'):
        _pin(features_db='data/props_features_other.db')
    (tmp_path / 'data/props_features_v4.db').write_bytes(b'rebuilt')
    with pytest.raises(ValueError, match='hash recorded'):
        _pin()
    assert _pin(check_features_db=False)['features_db_bytes'] == 7             # the cheap path trusts the training manifest
    assert (tmp_path / 'config/props_live.json').read_bytes() == PROD_BYTES


def test_a_custom_rule_file_is_pinned(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    rule = tmp_path / 'config/props_candidate_rule.json'
    rule.write_text('{"k": 2}')
    cfg = _pin(rule_file=rule)
    assert cfg['rule_sha256'] == registry.sha256_file(rule) != registry.sha256_file(tmp_path / 'config/props_frozen_rule.json')


# ------------------------------------------------------------------------------------------------------------------------------ signal job
def test_play_table_cache_name_is_unchanged_for_the_deployed_database_and_separate_for_a_candidate():
    assert job.play_tables_cache_path('data/props_features_v3.db', '2026-27') == Path('data/processed/live_play_tables_2026-27.pkl')
    cand = job.play_tables_cache_path('data/props_features_v4_pregame_all.db', '2026-27')
    assert cand == Path('data/processed/live_play_tables_2026-27_props_features_v4_pregame_all.pkl') and cand != job.play_tables_cache_path('data/props_features_v3.db', '2026-27')


def test_the_signal_job_takes_a_config_and_a_store_path():
    assert job.config.CANDIDATE_CONFIG_PATH == 'config/props_live_candidate.json'
    src = (ROOT / 'scripts/props_signal_job.py').read_text(encoding='utf-8')
    assert "'--config'" in src and "'--db'" in src and "'--dry-run'" in src
