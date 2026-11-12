"""Proves the deployed set is untouched by the calibrator code: (1) load_bundle works and model_set_hash equals the manifest's, for the hash
version the manifest recorded; (2) predict_markets on a slice of the (rebuilt) training set gives outputs identical to a saved pre-change run.
  ... check_deployed_unchanged.py save   -> writes scratch/pre_change_predict_markets.csv  (run BEFORE editing team/)
  ... check_deployed_unchanged.py check  -> recomputes and compares bit for bit
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from team import model_registry as mr                       # noqa: E402
from team.market_models import predict_markets              # noqa: E402

HERE = Path(__file__).resolve().parent
SCRATCH = HERE / 'scratch'
SUFFIX = mr.DEPLOYED_SUFFIX


def frame():
    from experiments.lib import load_games
    m, g = load_games()
    g = g.dropna(subset=['odds_total', 'odds_home_spread']).sort_values('gameDate').tail(1500).reset_index(drop=True)
    return g


def main(mode):
    man = mr.read_manifest(SUFFIX)
    b = mr.load_bundle(SUFFIX)
    ver = man.get('hash_version', 1)
    info = {'suffix': SUFFIX, 'manifest_model_set_hash': man['model_set_hash'], 'recomputed_model_set_hash': mr.model_set_hash(SUFFIX, mr.MODEL_DIR, ver),
            'bundle_model_set_hash': b.model_set_hash, 'hash_version': ver, 'files_hashed_in_manifest': sorted(man['files']),
            'files_present_with_suffix': sorted(p.name for p in mr.MODEL_DIR.glob(f'*{SUFFIX}*')), 'has_calibrator_total_attr': hasattr(b, 'calibrator_total')}
    info['hash_equals_manifest'] = info['manifest_model_set_hash'] == info['recomputed_model_set_hash'] == info['bundle_model_set_hash']
    out = predict_markets(b, frame())
    SCRATCH.mkdir(exist_ok=True)
    p = SCRATCH / 'pre_change_predict_markets.csv'
    if mode == 'save':
        out.to_csv(p, index=False, float_format='%.17g')
        info['saved_rows'] = int(len(out))
    else:
        old = pd.read_csv(p)
        new = pd.read_csv(__import__('io').StringIO(out.to_csv(index=False, float_format='%.17g')))
        info['rows_compared'] = int(len(new))
        info['predict_markets_identical_to_pre_change'] = bool(old.shape == new.shape and np.array_equal(old.to_numpy(), new.to_numpy(), equal_nan=True))
        info['max_abs_diff'] = float(np.nanmax(np.abs(old.to_numpy() - new.to_numpy())))
        info['total_std_distinct_values'] = int(new['total_std'].nunique())
    info['pickle_sha256'] = {q.name: hashlib.sha256(q.read_bytes()).hexdigest()[:16] for q in sorted(mr.MODEL_DIR.glob(f'*{SUFFIX}*'))}
    (HERE / f'deployed_check_{mode}.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ('pickle_sha256',)}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'check')
