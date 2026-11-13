"""
Research registry root (--registry-root), --save-raw and --all-played of the a11 experiment plan (reports/props/audit_2026-11/WP6_WP7_experiment_plan.md, S3 / S4a / S4b).

No test here fits LightGBM except the one marked slow_model, which is skipped unless A11_RUN_MODEL_TESTS=1. The others replace the pipeline's fitting and simulation by stubs
(props.models.pipeline.train_fold / predict_fold / distributions) and run the real scripts, registry, fold and dispersion code.
"""
import importlib.util
import inspect
import json
import os
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from props.models import common, data, dispersion, folds as folds_mod, outcome as oc, pipeline, registry
from synthetic_models import synthetic_frame

with warnings.catch_warnings():
    warnings.simplefilter('ignore')                                                  # the marker is not registered in a conftest: this file is the only place that uses it
    slow_model = pytest.mark.slow_model
needs_model_run = pytest.mark.skipif(os.environ.get('A11_RUN_MODEL_TESTS') != '1', reason='fits LightGBM: set A11_RUN_MODEL_TESTS=1 (run at night, not while the live bot runs)')

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT, ORIGINAL_CONFIG_LOG = registry.ROOT, common.CONFIG_LOG


def load_script(name):
    spec = importlib.util.spec_from_file_location(f'{name}_a11_test', ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_script = load_script('train_props_models')
predict_script = load_script('props_phase5_predict')
dev_script = load_script('props_phase5_dev')

SEASONS = ('2019-20', '2020-21', '2021-22')
FRAME = synthetic_frame(n_players=6, days_per_season=12, seasons=SEASONS, seed=3)


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Every test works in its own folder (registry.ROOT is relative to the working directory) and never writes the real configuration log."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(common, 'CONFIG_LOG', tmp_path / 'iso_default_config_log.jsonl')
    monkeypatch.setattr(data, 'load_frame', lambda db, *a, **k: FRAME.copy())
    (tmp_path / 'features.db').write_bytes(b'not a real database, only hashed')
    return tmp_path


def fake_pmfs(m_min):
    lam = np.asarray(m_min, float) / 10.0 + 1.0
    out = {}
    for s, cap in oc.CAPS.items():
        p = poisson.pmf(np.arange(cap + 1)[None, :], lam[:, None] * (1 + 0.1 * len(s)))
        out[s] = p / p.sum(axis=1, keepdims=True)
    return out


@pytest.fixture
def stub_distributions(monkeypatch):
    calls = []

    def fake(bundle, rows, families, n_sims, seed):
        calls.append((len(rows), n_sims, seed))
        res = fake_pmfs(rows['m_min_h15'].to_numpy())
        res['minutes_mean'] = rows['m_min_h15'].to_numpy(float)
        res['sim_means'] = {}
        return res
    monkeypatch.setattr(pipeline, 'distributions', fake)
    return calls


def make_bundle(c=1.0):
    scaler = dispersion.DispersionScaler()
    for s in oc.CAPS:
        for k in range(3):
            scaler.c[(s, k)] = c
    return {'rates': SimpleNamespace(nb_k={'x': 1.0}), 'families': {'reb': 'hier'}, 'dispersion': scaler}


def played_test_rows(season):
    f = FRAME
    return f[(f['season'] == season) & (f['lbl_played'] == 1) & (f['lbl_minutes'] > 0)]


def predict_args(tmp, **kw):
    a = ['--version', 'src', '--season', '2021-22', '--out-version', 'out', '--features-db', str(tmp / 'features.db'), '--n-sims', '10', '--chunk-rows', '50']
    for k, v in kw.items():
        a += [f'--{k.replace("_", "-")}'] + ([] if v is True else [str(v)])
    return a


def quoted_prices(tmp, season='2021-22', every=3):
    rows = played_test_rows(season)[['game_id', 'player_id', 'minutes_before']].iloc[::every]
    path = tmp / f'prices_{season}.pkl'
    path.write_bytes(pickle.dumps((rows.assign(stat='points'), pd.DataFrame())))
    return path, rows


# ------------------------------------------------------------------------------------------------ defaults unchanged

def test_default_paths_are_unchanged(isolate):
    assert ORIGINAL_ROOT == Path('data/props_models') and registry.ROOT == ORIGINAL_ROOT
    assert ORIGINAL_CONFIG_LOG == Path('data/props_models/config_log.jsonl')
    assert registry.version_dir('v') == Path('data/props_models/v')
    assert registry.resolve_root(None) is registry.ROOT
    p = registry.write_manifest('vdef', {'a': 1})                                    # no root argument: lands under data/props_models of the working directory
    assert p == Path('data/props_models/vdef/manifest.json') and (isolate / 'data' / 'props_models' / 'vdef' / 'manifest.json').exists()
    with pytest.raises(FileExistsError):
        registry.write_manifest('vdef', {'a': 2})
    for fn in (registry.write_manifest, registry.update_manifest, registry.save_bundle, registry.load_bundle, registry.save_predictions, registry.load_predictions, registry.version_dir):
        assert inspect.signature(fn).parameters['root'].default is registry.ROOT


def test_registry_root_naming_the_production_root_is_refused():
    with pytest.raises(SystemExit):
        registry.resolve_root('data/props_models')
    assert registry.resolve_root('elsewhere') == Path('elsewhere')


def test_default_mode_prediction_is_byte_identical_to_the_old_algorithm(isolate, stub_distributions, monkeypatch):
    bundle = make_bundle(c=1.3)
    registry.save_bundle('src', '2021-22', bundle)                                   # default root, relative to the working directory
    prices, keys = quoted_prices(isolate)
    predict_script.main(predict_args(isolate, prices=prices, chunk_rows=7))
    # the old script's loop, inlined, on the same stubs
    played = played_test_rows('2021-22')
    rows = played.merge(keys.drop_duplicates(), on=['game_id', 'player_id', 'minutes_before'], how='inner')
    parts, mins = {s: [] for s in oc.CAPS}, []
    for lo in range(0, len(rows), 7):
        chunk = rows.iloc[lo:lo + 7]
        res = pipeline.distributions(None, chunk, None, 10, 21 + lo)
        res.pop('sim_means', None)
        mm = res.pop('minutes_mean')
        for s in oc.CAPS:
            parts[s].append(bundle['dispersion'].transform(s, res[s], mm))
        mins.append(mm)
    idx = rows[['game_id', 'player_id', 'minutes_before', 'season', 'game_date']].reset_index(drop=True)
    idx['minutes_mean'] = np.concatenate(mins)
    registry.save_predictions('expected', '2021-22', idx, {s: np.concatenate(v) for s, v in parts.items()}, root=isolate / 'ref')
    got, exp = isolate / 'data' / 'props_models' / 'out' / '2021-22', isolate / 'ref' / 'expected' / '2021-22'
    assert sorted(p.name for p in got.iterdir()) == sorted(p.name for p in exp.iterdir())          # no raw_*.npy without --save-raw
    for p in exp.iterdir():
        assert (got / p.name).read_bytes() == p.read_bytes(), p.name
    man = json.loads((isolate / 'data' / 'props_models' / 'out' / 'manifest.json').read_text())
    assert set(man) == {'source_version', 'season', 'n_sims', 'seed', 'rows', 'features_db', 'features_db_sha256', 'recalibration', 'git_commit'}
    assert man['rows'] == len(rows)
    with pytest.raises(FileExistsError):                                             # default mode still never reuses an out-version
        predict_script.main(predict_args(isolate, prices=prices))


def test_default_mode_does_not_apply_the_research_holdout_guard(isolate, stub_distributions, monkeypatch):
    # the guard is for --registry-root only: the frozen 2025-26 workflow of the repo keeps working
    frame = synthetic_frame(n_players=6, days_per_season=12, seasons=('2019-20', '2020-21', '2025-26'), seed=3)
    monkeypatch.setattr(data, 'load_frame', lambda db, *a, **k: frame.copy())
    registry.save_bundle('src', '2025-26', make_bundle())
    predict_script.main(predict_args(isolate, season='2025-26', all_played=True))
    idx, pm = registry.load_predictions('out', '2025-26')
    assert len(idx) == int(((frame['season'] == '2025-26') & (frame['lbl_played'] == 1) & (frame['lbl_minutes'] > 0)).sum()) and set(pm) == set(oc.CAPS)
    with pytest.raises(SystemExit) as e:                                             # default-mode train: the existing message, not the research one
        train_script.main(['--version', 'x', '--folds', '2025-26'])
    assert 'not development folds' in str(e.value)
    with pytest.raises(FileNotFoundError):                                           # default-mode dev run gets as far as reading the (missing) price file
        dev_script.main(['--season', '2025-26', '--p5-version', 'a', '--full-version', 'b', '--out', 'o.md'])


# ------------------------------------------------------------------------------------------------ research root: isolation

def test_research_root_writes_only_under_the_root(isolate, stub_distributions):
    research = isolate / 'research'
    registry.save_bundle('src', '2021-22', make_bundle(), root=research)
    predict_script.main(predict_args(isolate, all_played=True, registry_root=research, save_raw=True))
    assert (research / 'out' / 'manifest.json').exists() and (research / 'out' / '2021-22' / 'index.csv').exists()
    assert not (isolate / 'data').exists()                                           # nothing was created under the default root
    man = json.loads((research / 'out' / 'manifest.json').read_text())
    assert man['mode'] == 'all_played' and man['save_raw'] is True and man['registry_root'] == str(research) and man['seasons']['2021-22']['rows'] == len(played_test_rows('2021-22'))


def test_research_reads_a_named_version_from_the_research_root_only(isolate, stub_distributions):
    registry.save_bundle('src', '2021-22', make_bundle())                            # only in the production root
    with pytest.raises(FileNotFoundError, match='research root'):
        predict_script.main(predict_args(isolate, all_played=True, registry_root=isolate / 'research'))
    assert not (isolate / 'research' / 'out').exists()


def test_families_fall_back_to_the_production_root_and_the_research_root_wins(isolate):
    research = isolate / 'research'
    registry.write_manifest('v2_base', {'families': {'reb': 'hier'}})
    m, path = registry.find_manifest('v2_base', (research, registry.ROOT))
    assert m['families'] == {'reb': 'hier'} and path == registry.ROOT / 'v2_base' / 'manifest.json'
    registry.write_manifest('v2_base', {'families': {'reb': 'lgb'}}, root=research)  # the same name in the research root: that one is read first
    assert registry.find_manifest('v2_base', (research, registry.ROOT))[0]['families'] == {'reb': 'lgb'}
    with pytest.raises(FileNotFoundError):
        registry.find_manifest('missing', (research, registry.ROOT))


def test_no_overwrite_in_either_root(isolate):
    research = isolate / 'research'
    registry.write_manifest('prod_v', {'a': 1})
    registry.write_manifest('res_v', {'a': 1}, root=research)
    for name in ('prod_v', 'res_v'):                                                 # a research write may use neither a production nor a research name
        with pytest.raises(FileExistsError):
            registry.assert_new_version(name, research)
    registry.assert_new_version('fresh', research)
    with pytest.raises(FileExistsError):
        registry.write_manifest('res_v', {'a': 2}, root=research)


def test_train_in_research_mode_with_stubbed_fitting(isolate, monkeypatch):
    research = isolate / 'research'
    registry.write_manifest('v2_base', {'families': {'reb': 'hier', 'ast': 'lgb'}, 'note': 'stub: no family_table'})     # like the surviving data/props_models/v2_base
    seen = {}

    def fake_train_fold(df, fold, spec, score_on_test=True, refit_all=False, recency_half_life_days=None):
        seen['train'] = dict(score_on_test=score_on_test, refit_all=refit_all, max_season=df['season'].max())
        common.log_config('fake_fit', season=fold.test_season)
        return {'fold': fold.describe(), 'family_scores': {}}

    def fake_predict_fold(df, fold, bundle, families, n_sims=4000, seed=0):
        played = df.iloc[fold.test_idx]
        played = played[(played['lbl_played'] == 1) & (played['lbl_minutes'] > 0)]
        idx = played[['game_id', 'player_id', 'minutes_before', 'season', 'game_date']].reset_index(drop=True)
        idx['minutes_mean'] = played['m_min_h15'].to_numpy()
        scaler = make_bundle()['dispersion']
        return idx, fake_pmfs(played['m_min_h15'].to_numpy()), 'cal', played, scaler

    monkeypatch.setattr(pipeline, 'train_fold', fake_train_fold)
    monkeypatch.setattr(pipeline, 'predict_fold', fake_predict_fold)
    monkeypatch.setattr(train_script.report, 'smoke_table', lambda played, pmfs: pd.DataFrame())
    monkeypatch.setattr(train_script.report, 'calibration_table', lambda played, pmfs, cal: pd.DataFrame())
    train_script.main(['--version', 'r0', '--features-db', str(isolate / 'features.db'), '--folds', '2021-22', '--families-from', 'v2_base', '--refit-all', '--registry-root', str(research)])
    man = json.loads((research / 'r0' / 'manifest.json').read_text())
    assert man['families'] == {'reb': 'hier', 'ast': 'lgb'} and man['families_from'] == 'v2_base' and man['research_mode'] is True
    assert man['family_table'] == [{'component': 'reb', 'chosen': 'hier'}, {'component': 'ast', 'chosen': 'lgb'}]
    assert man['families_from_manifest'].replace('\\', '/').endswith('data/props_models/v2_base/manifest.json')
    assert (research / 'r0' / '2021-22' / 'bundle.pkl').exists() and (research / 'r0' / '2021-22' / 'pred_points.npy').exists() and (research / 'r0' / 'training_report.md').exists()
    assert seen['train'] == {'score_on_test': False, 'refit_all': True, 'max_season': '2021-22'}
    log = [json.loads(line)['kind'] for line in (research / 'config_log.jsonl').read_text().splitlines()]
    assert 'fake_fit' in log and 'dispersion_scaler' in log                          # this run's configuration entries are under its own root
    assert not (isolate / 'iso_default_config_log.jsonl').exists()
    assert sorted(p.name for p in (isolate / 'data' / 'props_models').iterdir()) == ['v2_base']        # production root untouched
    with pytest.raises(FileExistsError):                                              # and the version cannot be written again
        train_script.main(['--version', 'r0', '--features-db', str(isolate / 'features.db'), '--folds', '2021-22', '--families-from', 'v2_base', '--registry-root', str(research)])


def test_research_frame_never_holds_holdout_rows(isolate, stub_distributions, monkeypatch):
    late = FRAME.assign(season='2025-26', game_date='2026-01-15')
    monkeypatch.setattr(data, 'load_frame', lambda db, *a, **k: pd.concat([FRAME, late], ignore_index=True))
    seen = []
    real = folds_mod.make_fold
    monkeypatch.setattr(folds_mod, 'make_fold', lambda frame, s, *a, **k: (seen.append(frame['season'].max()), real(frame, s, *a, **k))[1])
    registry.save_bundle('src', '2021-22', make_bundle(), root=isolate / 'research')
    predict_script.main(predict_args(isolate, all_played=True, registry_root=isolate / 'research'))
    assert seen == ['2021-22']


# ------------------------------------------------------------------------------------------------ research mode refuses the holdout

@pytest.mark.parametrize('season', ['2025-26', '2026-27'])
def test_research_mode_refuses_the_holdout_in_every_script(isolate, season):
    research = isolate / 'research'
    with pytest.raises(SystemExit, match='refuses'):
        train_script.main(['--version', 'x', '--folds', '2021-22', season, '--families-from', 'v2_base', '--registry-root', str(research)])
    with pytest.raises(SystemExit, match='refuses'):
        predict_script.main(predict_args(isolate, season=season, all_played=True, registry_root=research))
    with pytest.raises(SystemExit, match='refuses'):
        dev_script.main(['--season', season, '--p5-version', 'a', '--full-version', 'b', '--out', 'o.md', '--registry-root', str(research)])
    with pytest.raises(SystemExit, match='refuses'):
        dev_script.main(['--season', '2023-24', '--carry-season', season, '--p5-version', 'a', '--full-version', 'b', '--out', 'o.md', '--registry-root', str(research)])
    assert not research.exists() and not (isolate / 'data').exists()                 # refused before anything was written


def test_research_mode_accepts_the_seen_season_guard_only_for_the_holdout():
    registry.refuse_holdout(['2021-22', '2024-25', None])
    with pytest.raises(SystemExit):
        registry.refuse_holdout(['2024-25', '2025-26'])


def test_dev_script_reads_research_versions_from_the_research_root_only(isolate):
    research = isolate / 'research'
    idx = pd.DataFrame({'game_id': ['0000000001'], 'player_id': [1]})
    registry.save_predictions('p5', '2023-24', idx, {'points': np.ones((1, 61)) / 61}, root=research)
    with pytest.raises(FileNotFoundError):                                           # the price file is read first
        dev_script.main(['--season', '2023-24', '--p5-version', 'p5', '--full-version', 'f', '--out', 'o.md', '--registry-root', str(research), '--prices', 'nope.pkl'])
    (isolate / 'features_dev.db').write_bytes(b'')
    import sqlite3
    with sqlite3.connect(isolate / 'features_dev.db') as c:
        c.execute('create table player_game_labels (game_id text, player_id int, lbl_played int, lbl_minutes real, y_points real, y_rebounds real, y_assists real, y_threes real, y_pra real)')
        c.execute("insert into player_game_labels values ('0000000001', 1, 1, 30.0, 10, 5, 5, 1, 20)")
    (isolate / 'prices.pkl').write_bytes(pickle.dumps((pd.DataFrame(), pd.DataFrame())))
    registry.save_predictions('f', '2023-24', idx, {'points': np.ones((1, 61)) / 61})        # --full-version exists only in the production root
    with pytest.raises(FileNotFoundError):
        dev_script.main(['--season', '2023-24', '--p5-version', 'p5', '--full-version', 'f', '--out', 'o.md', '--registry-root', str(research), '--prices', str(isolate / 'prices.pkl'),
                         '--features-db', str(isolate / 'features_dev.db')])
    rows, pm = dev_script.load_rows_light('p5', '2023-24', str(isolate / 'features_dev.db'), root=research)
    assert len(rows) == 1 and rows['lbl_played'].iloc[0] == 1 and 'points' in pm


# ------------------------------------------------------------------------------------------------ --save-raw and --all-played

def test_save_raw_equals_the_dispersed_pmfs_when_the_factor_is_one(isolate, stub_distributions):
    research = isolate / 'research'
    registry.save_bundle('src', '2021-22', make_bundle(c=1.0), root=research)
    predict_script.main(predict_args(isolate, all_played=True, registry_root=research, save_raw=True, chunk_rows=7))
    idx, pred = registry.load_predictions('out', '2021-22', root=research)
    raw = registry.load_raw_predictions('out', '2021-22', root=research)
    assert set(raw) == set(pred) == set(oc.CAPS)
    for s in oc.CAPS:
        assert raw[s].shape == pred[s].shape == (len(idx), oc.CAPS[s] + 1) and raw[s].dtype == np.float32
        assert np.array_equal(raw[s], pred[s])
    assert sorted(p.name for p in (research / 'out' / '2021-22').glob('raw_*.npy')) == sorted(f'raw_{s}.npy' for s in oc.CAPS)
    assert [c for c in registry.load_predictions('out', '2021-22', root=research)[0].columns][:5] == ['game_id', 'player_id', 'minutes_before', 'season', 'game_date']


def test_save_raw_rows_align_with_the_index_and_differ_from_dispersed_when_c_above_one(isolate, stub_distributions):
    research = isolate / 'research'
    registry.save_bundle('src', '2021-22', make_bundle(c=1.5), root=research)
    predict_script.main(predict_args(isolate, all_played=True, registry_root=research, save_raw=True, chunk_rows=11))
    idx, pred = registry.load_predictions('out', '2021-22', root=research)
    raw = registry.load_raw_predictions('out', '2021-22', root=research)
    m = idx.merge(FRAME[['game_id', 'player_id', 'minutes_before', 'm_min_h15']], on=['game_id', 'player_id', 'minutes_before'], how='left')
    assert m['m_min_h15'].notna().all()
    expected = fake_pmfs(m['m_min_h15'].to_numpy())                                  # the raw pmf of row i is the pre-dispersion pmf of the row at index.csv line i
    for s in oc.CAPS:
        assert np.allclose(raw[s], expected[s], atol=1e-6)
        assert not np.allclose(raw[s], pred[s])
        assert np.allclose(pred[s].sum(axis=1), 1.0, atol=1e-4) and np.allclose(raw[s].sum(axis=1), 1.0, atol=1e-5)
        # the dispersed pmf is what the scaler makes of the raw one, row for row
    scaler = make_bundle(c=1.5)['dispersion']
    assert np.allclose(pred['points'], scaler.transform('points', raw['points'], idx['minutes_mean'].to_numpy()), atol=1e-6)


def test_all_played_row_set_is_the_played_test_rows(isolate, stub_distributions):
    research = isolate / 'research'
    registry.save_bundle('src', '2021-22', make_bundle(), root=research)
    predict_script.main(predict_args(isolate, all_played=True, registry_root=research, chunk_rows=13))
    idx, pred = registry.load_predictions('out', '2021-22', root=research)
    exp = played_test_rows('2021-22')
    assert len(idx) == len(exp) and len(idx) > 0
    assert (idx['game_id'].to_numpy() == exp['game_id'].to_numpy()).all() and (idx['player_id'].to_numpy() == exp['player_id'].to_numpy()).all()
    assert (idx['minutes_before'].to_numpy() == exp['minutes_before'].to_numpy()).all()
    assert set(idx['minutes_before']) == {240, 60, 5} and (idx['season'] == '2021-22').all()      # every decision time, only the test season
    assert not (research / 'out' / '2021-22' / 'raw_points.npy').exists()            # --save-raw not given: no raw arrays
    n_unplayed = int(((FRAME['season'] == '2021-22') & ~((FRAME['lbl_played'] == 1) & (FRAME['lbl_minutes'] > 0))).sum())
    assert n_unplayed > 0 and len(idx) == int((FRAME['season'] == '2021-22').sum()) - n_unplayed   # the unplayed rows really are left out
    assert [c[1] for c in stub_distributions].count(10) == len(stub_distributions)   # --n-sims is passed through


def test_quoted_mode_is_a_subset_of_the_all_played_rows(isolate, stub_distributions):
    research = isolate / 'research'
    registry.save_bundle('src', '2021-22', make_bundle(), root=research)
    prices, keys = quoted_prices(isolate)
    predict_script.main(predict_args(isolate, prices=prices, registry_root=research))
    idx, _ = registry.load_predictions('out', '2021-22', root=research)
    assert len(idx) == len(keys.drop_duplicates()) and len(idx) < len(played_test_rows('2021-22'))
    with pytest.raises(SystemExit):
        predict_script.main(predict_args(isolate, prices=prices, all_played=True, registry_root=research))      # both a price file and --all-played
    with pytest.raises(SystemExit):
        predict_script.main(['--version', 'src', '--season', '2021-22', '--out-version', 'o2'])                    # neither


def test_research_version_collects_one_manifest_entry_per_season(isolate, stub_distributions):
    research = isolate / 'research'
    for s in ('2020-21', '2021-22'):
        registry.save_bundle('src', s, make_bundle(), root=research)
        predict_script.main(predict_args(isolate, season=s, all_played=True, registry_root=research, save_raw=True))
    man = json.loads((research / 'out' / 'manifest.json').read_text())
    assert set(man['seasons']) == {'2020-21', '2021-22'} and man['seasons']['2020-21']['rows'] == len(played_test_rows('2020-21'))
    assert (research / 'out' / '2020-21' / 'raw_ra.npy').exists() and (research / 'out' / '2021-22' / 'raw_ra.npy').exists()
    with pytest.raises(FileExistsError):                                             # no season is written twice
        predict_script.main(predict_args(isolate, season='2021-22', all_played=True, registry_root=research, save_raw=True))
    other = SimpleNamespace(out_version='out', version='src', season='2022-23', n_sims=99, seed=21, features_db='x', all_played=True, save_raw=True, prices=None, chunk_rows=50)
    with pytest.raises(SystemExit, match='different settings'):                      # a new season with other settings under the same version name
        predict_script.write_manifest(other, research, True, 5, 'sha', None)
    registry.write_manifest('out2', {'x': 1})
    with pytest.raises(FileExistsError):                                             # a production-root name is never reused by a research version
        predict_script.main(predict_args(isolate, all_played=True, registry_root=research, out_version='out2'))


# ------------------------------------------------------------------------------------------------ real fitting: deferred to the night

@slow_model
@needs_model_run
def test_train_and_predict_in_research_mode_with_real_models(isolate, monkeypatch):
    """Fits LightGBM on a small synthetic league: run with A11_RUN_MODEL_TESTS=1 when the live bot is not running."""
    frame = synthetic_frame()
    monkeypatch.setattr(data, 'load_frame', lambda db, *a, **k: frame.copy())
    research = isolate / 'research'
    registry.write_manifest('v2_base', {'families': {'fg2a': 'hier', 'fg3a': 'hier', 'fta': 'hier', 'reb': 'hier', 'ast': 'hier', 'p2': 'hier', 'p3': 'hier', 'pft': 'hier'}})
    monkeypatch.setattr(train_script.report, 'smoke_table', lambda played, pmfs: pd.DataFrame())
    monkeypatch.setattr(train_script.report, 'calibration_table', lambda played, pmfs, cal: pd.DataFrame())
    train_script.main(['--version', 'r0', '--features-db', str(isolate / 'features.db'), '--folds', '2021-22', '--families-from', 'v2_base', '--refit-all', '--sims', '100', '--registry-root', str(research)])
    assert (research / 'r0' / '2021-22' / 'bundle.pkl').exists() and (research / 'config_log.jsonl').exists()
    assert not (isolate / 'iso_default_config_log.jsonl').exists()
    predict_script.main(['--version', 'r0', '--season', '2021-22', '--out-version', 'r0_raw', '--features-db', str(isolate / 'features.db'), '--n-sims', '200', '--all-played', '--save-raw',
                         '--registry-root', str(research)])
    idx, pred = registry.load_predictions('r0_raw', '2021-22', root=research)
    raw = registry.load_raw_predictions('r0_raw', '2021-22', root=research)
    assert len(idx) == len(played_test_rows('2021-22')) and raw['points'].shape == pred['points'].shape
