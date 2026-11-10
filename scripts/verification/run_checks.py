"""Launch recovery checks against restored copies with production/network guards."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import shutil
import hashlib

SOURCE=Path(__file__).resolve().parents[2]
REPORT=SOURCE/'reports/phase2_verification'
RUN='milestone1-20261107-0600'
RESTORE=Path('D:/nba_restore_checks')/RUN
ENV=Path('D:/nba_model_verification/check_env')
TOOLS=Path(__file__).parent
ROOTS={'models':SOURCE}
TMP_BASE=Path(os.environ.get('NBA_VERIFY_TMP',str(SOURCE.parent/'verification-tmp')))

def protected_roots():
    """The production checkouts the verification guard keeps read-only. There is no default: name them explicitly."""
    value = os.environ.get('NBA_VERIFY_PROTECTED_ROOTS')
    if not value:
        raise SystemExit('Set NBA_VERIFY_PROTECTED_ROOTS to the production checkouts the guard must keep read-only (semicolon separated)')
    return value

def setup():
    REPORT.mkdir(parents=True,exist_ok=True)
    if not (ENV/'Scripts/python.exe').exists():
        venv.EnvBuilder(with_pip=False).create(ENV)
        # Independent interpreter prefix; existing pinned dependencies are read-only inputs.
        deps=[p for p in sys.path if p.lower().endswith('site-packages')]
        (ENV/'Lib/site-packages/milestone_dependencies.pth').write_text('\n'.join(deps)+'\n')
        (ENV/'Lib/site-packages/milestone_guard.pth').write_text(str(TOOLS)+'\n')
    probe_env=os.environ.copy()
    probe_env['NBA_VERIFY_PROTECTED_ROOTS']=protected_roots()
    probe_env['PYTHONDONTWRITEBYTECODE']='1'
    code="import importlib.metadata,json,sys; print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}}))"
    probe=subprocess.run([str(ENV/'Scripts/python.exe'),'-I','-c',code],env=probe_env,capture_output=True,text=True,check=True)
    environment=json.loads(probe.stdout)
    environment.update(isolated_prefix=str(ENV),dependency_policy='Read-only existing runtime dependencies; actual child-prefix metadata; no downloads or upgrades')
    (REPORT/'environment.json').write_text(json.dumps(environment,indent=2))


def run(owner, action):
    setup()
    root=ROOTS[owner] if action in ('tests','targeted') else RESTORE/owner
    # Keep Git quarantine object paths below Windows MAX_PATH; NTFS supports lock tests.
    tmp=TMP_BASE/owner/action
    tmp.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for k in list(env):
        if any(s in k.upper() for s in ('API_KEY','SECRET','TOKEN','POLYMARKET_','ODDS_API','LIVE_BOT_')):
            env.pop(k,None)
    env.update(ODDS_API_KEY='offline-disabled',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(TOOLS)+os.pathsep+str(root),TEMP=str(tmp),TMP=str(tmp),
        GIT_ALLOW_PROTOCOL='file',GIT_TERMINAL_PROMPT='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        MPLCONFIGDIR=str(tmp/'mpl'),MILESTONE_RESTORE=str(RESTORE),MILESTONE_REPORT=str(REPORT),
        NBA_VERIFY_PROTECTED_ROOTS=protected_roots())
    python=str(ENV/'Scripts/python.exe')
    if action=='db' and owner=='models':
        legacy=root/'data/props_features.db';active=root/'data/props_features_v3.db'
        if not legacy.exists() and active.exists():
            shutil.copyfile(active,legacy)
            def h(p):
                x=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(8<<20),b''):x.update(b)
                return x.hexdigest()
            assert h(active)==h(legacy)
            (REPORT/'db_fixture_mapping.json').write_text(json.dumps({'source':str(active),'test_only_copy':str(legacy),
                'sha256':h(legacy),'reason':'Legacy tests use props_features.db; explicit byte copy of restored active v3 dataset, not regeneration'},indent=2))
    if action in ('tests','targeted','db','historical'):
        selected=['tests'] if action in ('tests','db') else ['tests/props/test_live_publish.py']
        cmd=[python,'-m','pytest',*selected,'-q','-ra','-m','db' if action=='db' else 'not db','-p','no:cacheprovider','-p','failure_details',
             '--basetemp',str(tmp/'pytest'),'--junitxml',str(REPORT/f'{owner}_{action}.xml')]
    elif action=='smoke':
        cmd=[python,str(TOOLS/'recovery_smoke.py'),owner]
    else:raise ValueError(action)
    with (REPORT/f'{owner}_{action}.log').open('w',encoding='utf-8') as f:
        result=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps({'owner':owner,'action':action,'exit_code':result.returncode}),flush=True)
    return result.returncode

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['tests','targeted','db','smoke'])
    ap.add_argument('--report',type=Path)
    ap.add_argument('--restore',type=Path)
    ap.add_argument('--models-root',type=Path)
    ap.add_argument('--temp-root',type=Path)
    ap.add_argument('--tools',type=Path)
    ap.add_argument('--env',type=Path)
    a=ap.parse_args()
    if a.report:REPORT=a.report
    if a.restore:RESTORE=a.restore
    if a.models_root:ROOTS['models']=a.models_root
    if a.temp_root:TMP_BASE=a.temp_root
    if a.tools:TOOLS=a.tools
    if a.env:ENV=a.env
    raise SystemExit(run('models',a.action))
