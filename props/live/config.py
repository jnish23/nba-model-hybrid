"""
The pinned live configuration (reports/props/live_config.json) and the guard that refuses to run when a pinned file has changed. A monthly refit is a new model version, which is a
reviewed edit of this file (version, bundle hash, effective date); nothing at run time chooses a model or a rule.
"""
import json
import subprocess
from pathlib import Path

from props.models import registry

CONFIG_PATH = 'config/props_live.json'
LEGACY_CONFIG_PATH = 'reports/props/live_config.json'
RULE_PATH = 'config/props_frozen_rule.json'
LEGACY_RULE_PATH = 'reports/props/phase5_revised_rule.frozen_rule.json'


def resolve_config_path(path=None):
    if path is not None and str(path) != CONFIG_PATH:
        return Path(path)
    canonical = Path(CONFIG_PATH)
    return canonical if canonical.exists() or not Path(LEGACY_CONFIG_PATH).exists() else Path(LEGACY_CONFIG_PATH)


def resolve_rule_path(path=None):
    if path is not None:
        return Path(path)
    canonical = Path(RULE_PATH)
    return canonical if canonical.exists() or not Path(LEGACY_RULE_PATH).exists() else Path(LEGACY_RULE_PATH)


class ConfigRefused(RuntimeError):
    pass


def load(path=None):
    return json.loads(resolve_config_path(path).read_text(encoding='utf-8'))


def bundle_path(cfg):
    return registry.version_dir(cfg['model_version']) / cfg['model_season'] / 'bundle.pkl'


def verify(cfg, check_features_db=False):
    """Raises ConfigRefused naming every mismatch. The features database (3 GB) is hashed only when asked (the nightly health check); size and name are always checked."""
    problems = []
    rule = Path(cfg['rule_file'])
    if not rule.exists() or registry.sha256_file(rule) != cfg['rule_sha256']:
        problems.append(f"rule file {cfg['rule_file']} does not match the pinned hash")
    b = bundle_path(cfg)
    if not b.exists() or registry.sha256_file(b) != cfg['model_bundle_sha256']:
        problems.append(f"model bundle {b} does not match the pinned hash")
    f = Path(cfg['features_db'])
    if not f.exists() or f.stat().st_size != cfg['features_db_bytes']:
        problems.append(f"features database {f} is missing or its size changed")
    elif check_features_db and registry.sha256_file(f) != cfg['features_db_sha256']:
        problems.append(f'features database {f} does not match the pinned hash')
    if problems:
        raise ConfigRefused('; '.join(problems))
    return cfg


def code_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def write(cfg, path=None):
    """Write a reviewed configuration to its resolved canonical or legacy target."""
    target = resolve_config_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=1), encoding='utf-8')


def pin(model_version, model_season, effective_from, path=None):
    """Write the pinned configuration for a model version (a reviewed, committed act)."""
    rule_file = resolve_rule_path().as_posix()
    f = Path('data/props_features_v3.db')
    man = json.loads((registry.version_dir(model_version) / 'manifest.json').read_text(encoding='utf-8'))
    cfg = {'rule_file': rule_file, 'rule_sha256': registry.sha256_file(rule_file), 'model_version': model_version, 'model_season': model_season,
           'model_bundle_sha256': registry.sha256_file(bundle_path({'model_version': model_version, 'model_season': model_season})),
           'features_db': f.as_posix(), 'features_db_bytes': f.stat().st_size, 'features_db_sha256': man['features_db_sha256'], 'roster_rule': 'v2', 'n_sims': 20000,
           'effective_from': effective_from,
           'waivers': ['A1 (October coverage ~60%)', 'A4 (teammate-availability inputs cannot be reproduced before the game)', 'A5', 'A6'],
           'note': 'Paper record only. The parity gap is disclosed in reports/props/d1_parity_report.md and in every live report.'}
    write(cfg, path=path)
    return cfg


CANDIDATE_CONFIG_PATH = 'config/props_live_candidate.json'


def pin_candidate(model_version, model_season, effective_from, features_db, roster_rule, *, waivers, rule_file=None, n_sims=20000, path=CANDIDATE_CONFIG_PATH,
                  check_features_db=True, note=None):
    """
    Write a CANDIDATE configuration (A2 plan, P2) to its own file; the deployed config/props_live.json is never touched (a path that resolves to it or to its legacy copy is refused).
    The model is read from its training manifest: its season and features database must match the arguments, and the pinned features hash is the one recorded at training (re-hashed
    from the file when check_features_db). `waivers` has no default: name exactly the waivers that still apply to this candidate (an empty list is allowed, None is not).
    rule_file defaults to the frozen rule; pass the candidate rule file once one exists. Returns the written configuration; `load(path)` and `verify(cfg)` work on it unchanged.
    """
    if isinstance(waivers, str) or not isinstance(waivers, (list, tuple)) or not all(isinstance(w, str) for w in waivers):
        raise TypeError('waivers must be an explicit list of strings (an empty list means no waiver)')
    target = Path(path)
    if target.resolve() in {Path(CONFIG_PATH).resolve(), Path(LEGACY_CONFIG_PATH).resolve()}:
        raise ValueError(f'{path} is the deployed live configuration; a candidate is written to its own file (default {CANDIDATE_CONFIG_PATH})')
    man = json.loads((registry.version_dir(model_version) / 'manifest.json').read_text(encoding='utf-8'))
    f = Path(features_db)
    if man.get('next_season') != model_season:
        raise ValueError(f"model {model_version} was trained for {man.get('next_season')}, not {model_season}")
    if Path(man['features_db']).as_posix() != f.as_posix():
        raise ValueError(f"model {model_version} was trained on {man['features_db']}, not {features_db}")
    if check_features_db and registry.sha256_file(f) != man['features_db_sha256']:
        raise ValueError(f'features database {f} does not match the hash recorded when {model_version} was trained')
    rule = (Path(rule_file) if rule_file is not None else resolve_rule_path()).as_posix()
    cfg = {'rule_file': rule, 'rule_sha256': registry.sha256_file(rule), 'model_version': model_version, 'model_season': model_season,
           'model_bundle_sha256': registry.sha256_file(bundle_path({'model_version': model_version, 'model_season': model_season})),
           'features_db': f.as_posix(), 'features_db_bytes': f.stat().st_size, 'features_db_sha256': man['features_db_sha256'], 'roster_rule': roster_rule, 'n_sims': n_sims,
           'effective_from': effective_from, 'spec': man.get('spec', 'base'), 'waivers': list(waivers),
           'note': note or 'CANDIDATE, not deployed. Not read by the scheduled jobs; reached only through scripts/props_signal_job.py --config.'}
    write(cfg, path=target)
    return cfg
