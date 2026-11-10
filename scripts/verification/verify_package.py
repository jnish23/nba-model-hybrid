"""Build/install the public wheel offline in a fresh prefix, then verify imports/data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def protected_roots():
    """The production checkouts the verification guard keeps read-only. There is no default: name them explicitly."""
    value = os.environ.get('NBA_VERIFY_PROTECTED_ROOTS')
    if not value:
        raise SystemExit('Set NBA_VERIFY_PROTECTED_ROOTS to the production checkouts the guard must keep read-only (semicolon separated)')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True, help='Existing read-only site-packages')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--build-dependencies', type=Path, required=True, help='Existing local setuptools/wheel site-packages')
    args = parser.parse_args()
    stage = args.workspace / 'source'
    prefix = args.workspace / 'environment'
    if args.workspace.exists():
        raise SystemExit('Refusing to overwrite an offline package workspace')
    stage.mkdir(parents=True)
    for package in ('core', 'team', 'props'):
        shutil.copytree(ROOT/package, stage/package, ignore=shutil.ignore_patterns('__pycache__', 'polymarket'))
    for name in ('pyproject.toml', 'PACKAGING.md'):
        shutil.copyfile(ROOT/name, stage/name)
    venv.EnvBuilder(with_pip=False).create(prefix)
    site = prefix / 'Lib/site-packages'
    (site/'dependencies.pth').write_text(str(args.dependencies.resolve())+'\n'+str(args.build_dependencies.resolve())+'\n')
    (site/'verification_guard.pth').write_text(str(ROOT/'scripts/verification')+'\n')
    python = prefix/'Scripts/python.exe'
    env = os.environ.copy()
    for name in list(env):
        if any(word in name.upper() for word in ('API_KEY', 'SECRET', 'TOKEN', 'LIVE_BOT_', 'ODDS_API', 'POLYMARKET_')):
            env.pop(name, None)
    env.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1',
               NBA_VERIFY_PROTECTED_ROOTS=protected_roots(),
               ODDS_API_KEY='offline-disabled')
    commands = [[str(python), '-m', 'pip', 'wheel', '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir',
                 '--wheel-dir', str(args.workspace/'wheelhouse'), str(stage)]]
    def run(command):
        result = subprocess.run(command, cwd=args.workspace, env=env, capture_output=True, text=True)
        with (args.report/'package_build.log').open('a') as log:
            log.write(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError('Offline packaging failed; see package_build.log')
        return result.stdout
    for command in commands:
        run(command)
    wheel = next((args.workspace/'wheelhouse').glob('*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        files = archive.namelist()
        assert 'core/live_feed_overrides.csv' in files
        assert archive.read('core/live_feed_overrides.csv') == (ROOT/'core/live_feed_overrides.csv').read_bytes()
        assert all(name.split('/')[0] in {'core','team','props'} or '.dist-info/' in name for name in files)
        assert not any(name.startswith('props/polymarket/') or name.endswith(('.pkl','.db','.jsonl')) for name in files)
    run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', str(wheel)])
    code = """import importlib.metadata, importlib.resources, json, pathlib, sys
import core, team, props
from core import live_feed, props_board_logic
from team import market_models, model_registry
from props.models import outcome
site = pathlib.Path(sys.prefix) / 'Lib/site-packages'
origins = {m.__name__:str(pathlib.Path(m.__file__).resolve()) for m in (core,team,props,live_feed,market_models,outcome)}
assert all(pathlib.Path(p).is_relative_to(site) for p in origins.values()), origins
model_registry.alias_legacy_pickle_modules()
import models.market_models as legacy
assert legacy is market_models
assert importlib.resources.files('core').joinpath('live_feed_overrides.csv').is_file()
assert importlib.metadata.version('nba-pregame-models') == '0.1.0'
print(json.dumps({'version': '0.1.0', 'origins':origins, 'sys_path':sys.path, 'package_data':str(live_feed.OVERRIDES)}))
"""
    imports = json.loads(run([str(python), '-I', '-c', code]))
    result = {'passed':True, 'wheel':str(wheel), 'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),
              'wheel_files':files, 'imports':imports, 'installed_prefix':str(prefix),
              'network':'denied', 'dependencies':'reused read-only; no dependency installation/upgrade',
              'production_writes':'denied', 'source_checkout_on_import_path':False}
    (args.report/'package_contract.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed':True,'wheel':str(wheel),'files':len(files)}))


if __name__ == '__main__':
    main()
