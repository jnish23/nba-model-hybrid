"""Proves the default training output is byte for byte what it was before the opt-in existed.
Extracts team/ and core/ from git HEAD (the pre-change code) into a scratch folder, trains the same synthetic history with the old code and with the
working-tree code (tuner='grid', deterministic), and compares every saved file. Run from the worktree root:
  C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe reports/audit_pregame_2026-11/A6_totals_calibration/check_default_training_identical.py
"""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCRATCH = HERE / 'scratch' / 'default_training'

CHILD = r'''
import sys, json
from pathlib import Path
code_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(code_root)); sys.path.insert(1, sys.argv[3])
import matplotlib; matplotlib.use('Agg')
from team.train_baseline import BaselineModel
import team.train_baseline as tb
assert Path(tb.__file__).resolve().is_relative_to(code_root.resolve()), tb.__file__
from synthetic_history import training_csv
out_dir.mkdir(parents=True, exist_ok=True)
csv = out_dir / 'train.csv'
training_csv(csv)
m = BaselineModel(csv, model_dir=str(out_dir / 'saved'))
m.train_and_save_all(use_odds=True, save_suffix='_t', time_budget=2, calib_holdout_days=90, total_target='level', tuner='grid')
'''


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = SCRATCH / 'old_code'
    if old.exists():
        import shutil
        shutil.rmtree(old)
    old.mkdir(parents=True)
    tar = subprocess.run(['git', 'archive', 'HEAD', 'team', 'core'], cwd=ROOT, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(old)
    (SCRATCH / 'child.py').write_text(CHILD)
    tests_dir = str(ROOT / 'tests' / 'team')
    res = {}
    for tag, code_root in (('old_head', old), ('new_worktree', ROOT)):
        out = SCRATCH / tag
        r = subprocess.run([sys.executable, str(SCRATCH / 'child.py'), str(code_root), str(out), tests_dir], cwd=str(code_root), capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-2000:])
            raise SystemExit(f'{tag} training failed')
        res[tag] = {p.name: sha(p) for p in sorted((out / 'saved').iterdir())}
    names = sorted(set(res['old_head']) | set(res['new_worktree']))
    rows = {n: {'old': res['old_head'].get(n), 'new': res['new_worktree'].get(n), 'identical': res['old_head'].get(n) == res['new_worktree'].get(n)} for n in names}
    # the manifest carries a timestamp and a git commit, so compare its model_set_hash and file hashes instead of its bytes
    mo = json.loads((SCRATCH / 'old_head' / 'saved' / 'model_manifest_t.json').read_text())
    mn = json.loads((SCRATCH / 'new_worktree' / 'saved' / 'model_manifest_t.json').read_text())
    rows['model_manifest_t.json']['identical'] = (mo['model_set_hash'] == mn['model_set_hash'] and mo['files'] == mn['files'] and mo['training_set_hash'] == mn['training_set_hash'])
    out = {'files': rows, 'all_identical': all(v['identical'] for v in rows.values()), 'model_set_hash_old': mo['model_set_hash'], 'model_set_hash_new': mn['model_set_hash']}
    (HERE / 'default_training_identical.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v['identical'] for k, v in rows.items()}, indent=1), '\nall identical:', out['all_identical'], '\nmodel_set_hash', mo['model_set_hash'][:16], mn['model_set_hash'][:16])


if __name__ == '__main__':
    main()
