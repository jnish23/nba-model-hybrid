"""
Props model registry: versioned, separate from the team registry (models/saved is never touched). Layout:

    data/props_models/<version>/manifest.json          configuration, folds, family selection, git commit, row counts
    data/props_models/<version>/<test_season>/bundle.pkl   fitted availability, minutes, rate models, calibrator for that fold
    data/props_models/<version>/<test_season>/pred_<stat>.npy + index.csv    predictive pmfs for the fold's played test rows

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


def sha256_file(path, chunk=1 << 22):
    """Content hash of a data file (the dataset a version was trained on), recorded in the manifest so a rebuilt dataset cannot silently stand behind an old version."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()
