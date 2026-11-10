"""Exact model/export comparisons against the public frozen recovery fixtures."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def compare(expected,actual):
    checks=[]
    def record(name,passed,**details):
        checks.append({'check':name,'passed':bool(passed),**details})
    for name in ('team_games.csv','team_features.csv','team_predictions.csv','props_features.csv'):
        record(name,sha(expected/name)==sha(actual/name),expected_sha256=sha(expected/name),actual_sha256=sha(actual/name))
    with np.load(expected/'props_probabilities.npz') as old,np.load(actual/'props_probabilities.npz') as new:
        record('probability_array_keys',set(old.files)==set(new.files))
        for key in old.files:
            a,b=old[key],new[key]
            record('probability_array/'+key,a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),
                   dtype=str(b.dtype),shape=b.shape,expected_array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                   actual_array_sha256=hashlib.sha256(b.tobytes()).hexdigest())
    timestamp_exceptions={'props_export/meta.json','props_deployment/data/props_dashboard_live/meta.json'}
    def data_file(rel):
        a,b=expected/rel,actual/rel
        if rel in timestamp_exceptions:
            x=json.loads(a.read_text(encoding='utf-8'));y=json.loads(b.read_text(encoding='utf-8'))
            old_stamp=x.pop('built_at_utc');new_stamp=y.pop('built_at_utc')
            record(rel,x==y,metadata_difference={'field':'built_at_utc','expected':old_stamp,'actual':new_stamp},
                   expected_sha256=sha(a),actual_sha256=sha(b))
        else:
            record(rel,sha(a)==sha(b),expected_sha256=sha(a),actual_sha256=sha(b))
    for folder in ('team_export','props_export','props_deployment/data'):
        old={p.relative_to(expected).as_posix() for p in (expected/folder).rglob('*') if p.is_file()}
        new={p.relative_to(actual).as_posix() for p in (actual/folder).rglob('*') if p.is_file()}
        record(folder+'/file_set',old==new,missing=sorted(old-new),extra=sorted(new-old))
        for rel in sorted(old & new):data_file(rel)
    bundle=Path('props_deployment')
    for rel in ['dashboard_props/app.py','core/__init__.py','core/live_feed.py','core/live_feed_overrides.csv',
                'requirements.txt','.streamlit/config.toml','README.md','PUBLISHED.json']:
        data_file((bundle/rel).as_posix())
    old_logic=(expected/bundle/'dashboard_props/logic.py').read_text(encoding='utf-8')
    new_logic=(actual/bundle/'core/props_board_logic.py').read_text(encoding='utf-8')
    record('moved_board_logic_ast',ast.dump(ast.parse(old_logic),include_attributes=False)==ast.dump(ast.parse(new_logic),include_attributes=False))
    shim=(actual/bundle/'dashboard_props/logic.py').read_text(encoding='utf-8')
    record('standalone_shim_targets_bundled_logic','from core.props_board_logic import *' in shim)
    return {'passed':all(x['passed'] for x in checks),'checks':checks,
            'policy':'Exact bytes and array dtype/shape/bytes; only two explicit built_at_utc fields differ. Generated logic relocation is compared by exact AST and standalone import checks. No numerical tolerance.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected',type=Path,required=True)
    parser.add_argument('--actual',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.expected,args.actual)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks'])}))
    raise SystemExit(not result['passed'])
