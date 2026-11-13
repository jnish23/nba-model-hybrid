"""
Props model registry: versioned, separate from the team registry (models/saved is never touched). Layout:

    data/props_models/<version>/manifest.json          configuration, folds, family selection, git commit, row counts
    data/props_models/<version>/<test_season>/bundle.pkl   fitted availability, minutes, rate models, calibrator for that fold
    data/props_models/<version>/<test_season>/pred_<stat>.npy + index.csv    predictive pmfs for the fold's played test rows
    data/props_models/<version>/<test_season>/raw_<stat>.npy                 (optional, props_phase5_predict --save-raw) the same rows' pmfs BEFORE the dispersion widening

A version is immutable once written; a new configuration gets a new version name.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path('data/props_models')


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def version_dir(version, root=ROOT):
    return Path(root) / version


# ---------------------------------------------------------------------------------------------------------------- research registry root
# A script run with --registry-root is in RESEARCH mode (plan reports/props/audit_2026-11/WP6_WP7_experiment_plan.md A.1, E.0.4, G6). The rules, in one place:
#   writes      new versions, manifests, bundles, predictions and the config-log entries of the run go under the research root only;
#   no overwrite a version that exists under the research root OR under the production root ROOT cannot be written again (one name, one meaning);
#   reads       a named version (--version, --p5-version, --full-version, --carry-version) is read from the research root ONLY, never from the production root;
#   families    --families-from looks in the research root first and falls back to the production root (where the frozen v2_base manifest lives);
#   holdout     no season from HOLDOUT_SEASON on may be trained, predicted or scored (plan A.1).
HOLDOUT_SEASON = '2025-26'


def resolve_root(registry_root=None):
    """None -> the production root (ROOT). A given root is research mode; naming the production root itself is refused (it would switch the holdout guard on while writing into production)."""
    if registry_root is None:
        return ROOT
    root = Path(registry_root)
    if root.resolve() == ROOT.resolve():
        raise SystemExit(f'--registry-root {registry_root} is the production registry {ROOT}: leave the flag out to work there')
    return root


def refuse_holdout(seasons, what='season'):
    """Research mode only: refuse the 2025-26 holdout and anything after it (a fold for a later season would train on it)."""
    bad = sorted({str(s) for s in seasons if s is not None and str(s) >= HOLDOUT_SEASON})
    if bad:
        raise SystemExit(f'{what} {bad}: research mode (--registry-root) refuses {HOLDOUT_SEASON} and later seasons (plan A.1: the holdout is not trained, predicted or scored)')


def version_exists(version, root=ROOT):
    d = version_dir(version, root)
    return d.is_dir() and any(d.iterdir())


def assert_new_version(version, root, production_root=ROOT):
    """Refuse a version name that is already used under `root` or under the production root: a research version never shares a name with a production one."""
    for r in dict.fromkeys([Path(root), Path(production_root)]):
        if version_exists(version, r):
            raise FileExistsError(f'version {version!r} already exists under {r}: a registry version is immutable; use a new version name')


def find_manifest(version, roots):
    """The first manifest.json of `version` found along `roots` (in order); returns (manifest dict, path). Used for --families-from: research root first, production second."""
    for r in roots:
        p = version_dir(version, r) / 'manifest.json'
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8')), p
    raise FileNotFoundError(f'no manifest.json for version {version!r} under ' + ' or '.join(str(r) for r in roots))


def write_manifest(version, info, root=ROOT, overwrite=False):
    d = version_dir(version, root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'manifest.json'
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} exists: a registry version is immutable; use a new version name')
    path.write_text(json.dumps({**info, 'git_commit': _git_commit()}, indent=2, default=str), encoding='utf-8')
    return path


def update_manifest(version, patch, root=ROOT):
    path = version_dir(version, root) / 'manifest.json'
    m = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    m.update(patch)
    path.write_text(json.dumps(m, indent=2, default=str), encoding='utf-8')


def save_bundle(version, season, bundle, root=ROOT):
    d = version_dir(version, root) / season
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / 'bundle.pkl')


def load_bundle(version, season, root=ROOT):
    return joblib.load(version_dir(version, root) / season / 'bundle.pkl')


def save_predictions(version, season, index_df, pmfs, root=ROOT):
    d = version_dir(version, root) / season
    d.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(d / 'index.csv', index=False)
    for stat, arr in pmfs.items():
        np.save(d / f'pred_{stat}.npy', arr.astype(np.float32))


def load_predictions(version, season, root=ROOT):
    d = version_dir(version, root) / season
    idx = pd.read_csv(d / 'index.csv', dtype={'game_id': str})
    return idx, {p.stem.replace('pred_', ''): np.load(p) for p in sorted(d.glob('pred_*.npy'))}


def save_raw_predictions(version, season, pmfs, root=ROOT):
    """The PRE-dispersion pmfs (before DispersionScaler.transform) as raw_<stat>.npy: row i of every array is row i of index.csv, the same rows as pred_<stat>.npy."""
    d = version_dir(version, root) / season
    d.mkdir(parents=True, exist_ok=True)
    for stat, arr in pmfs.items():
        np.save(d / f'raw_{stat}.npy', arr.astype(np.float32))


def load_raw_predictions(version, season, root=ROOT):
    d = version_dir(version, root) / season
    return {p.stem.replace('raw_', '', 1): np.load(p) for p in sorted(d.glob('raw_*.npy'))}


def sha256_file(path, chunk=1 << 22):
    """Content hash of a data file (the dataset a version was trained on), recorded in the manifest so a rebuilt dataset cannot silently stand behind an old version."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()
