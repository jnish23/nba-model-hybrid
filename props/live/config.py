"""
The pinned live configuration (reports/props/live_config.json) and the guard that refuses to run when a pinned file has changed. A monthly refit is a new model version, which is a
reviewed edit of this file (version, bundle hash, effective date); nothing at run time chooses a model or a rule.
"""
import json
import subprocess
from pathlib import Path

from props.models import registry

CONFIG_PATH = 'reports/props/live_config.json'


class ConfigRefused(RuntimeError):
    pass


def load(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding='utf-8'))


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


def pin(model_version, model_season, effective_from, path=CONFIG_PATH):
    """Write the pinned configuration for a model version (a reviewed, committed act)."""
    rule_file = 'reports/props/phase5_revised_rule.frozen_rule.json'
    f = Path('data/props_features_v3.db')
    man = json.loads((registry.version_dir(model_version) / 'manifest.json').read_text(encoding='utf-8'))
    cfg = {'rule_file': rule_file, 'rule_sha256': registry.sha256_file(rule_file), 'model_version': model_version, 'model_season': model_season,
           'model_bundle_sha256': registry.sha256_file(bundle_path({'model_version': model_version, 'model_season': model_season})),
           'features_db': f.as_posix(), 'features_db_bytes': f.stat().st_size, 'features_db_sha256': man['features_db_sha256'], 'roster_rule': 'v2', 'n_sims': 20000,
           'effective_from': effective_from,
           'waivers': ['A1 (October coverage ~60%)', 'A4 (teammate-availability inputs cannot be reproduced before the game)', 'A5', 'A6'],
           'note': 'Paper record only. The parity gap is disclosed in reports/props/d1_parity_report.md and in every live report.'}
    Path(path).write_text(json.dumps(cfg, indent=1), encoding='utf-8')
    return cfg
