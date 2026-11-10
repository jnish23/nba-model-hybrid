"""
core/ is the base layer: core never imports team or props, and team never imports props.
props may reuse exactly ONE team module, the team model's pre-game feature builder (team.team_features_v2), so opponent and schedule features
come from code that already has train/serve parity tests. Nothing else in team/ (models, registry, betting, training) may be imported by props.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNED_LAYERS = {'dashboard', 'dashboard_props', 'experiments', 'scripts', 'reports', 'live_bot', 'polymarket_venue', 'polymarket_props'}
FORBIDDEN = {'core': {'team', 'props'} | BANNED_LAYERS, 'team': {'props'} | BANNED_LAYERS, 'props': BANNED_LAYERS}
PROPS_MAY_IMPORT_FROM_TEAM = {'team.team_features_v2'}


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module
            if node.module == 'team':                         # 'from team import x' names a module of team
                yield from (f'team.{a.name}' for a in node.names)


def _imported_roots(path):
    return (m.split('.')[0] for m in _imported_modules(path))


def test_package_dependency_direction():
    bad = []
    for pkg, banned in FORBIDDEN.items():
        for path in (ROOT / pkg).rglob('*.py'):
            bad += [f'{path.relative_to(ROOT)} imports {m}' for m in set(_imported_roots(path)) & banned]
    for path in (ROOT / 'props').rglob('*.py'):
        bad += [f'{path.relative_to(ROOT)} imports {m}' for m in set(_imported_modules(path))
                if (m == 'team' or m.startswith('team.')) and m not in PROPS_MAY_IMPORT_FROM_TEAM and m != 'team']
    assert not bad, bad


def test_nothing_imports_the_old_models_package():
    """`models/` now only holds saved artifacts (models/saved); the code moved to core/ and team/."""
    bad = []
    for pkg in ('core', 'team', 'props', 'scripts', 'dashboard', 'tests'):
        for path in (ROOT / pkg).rglob('*.py'):
            bad += [str(path.relative_to(ROOT)) for m in set(_imported_roots(path)) if m == 'models']
    assert not bad, bad


def test_legacy_model_class_paths_resolve_to_current_classes():
    import importlib
    from team import model_registry
    model_registry.alias_legacy_pickle_modules()
    current=importlib.import_module('team.market_models')
    legacy=importlib.import_module('models.market_models')
    for name in ('MarginStackCalibrator','ResidualSpreadModel','HeteroSigma'):
        assert getattr(legacy,name) is getattr(current,name)


def test_wheel_package_contract_is_public_and_explicit():
    import tomllib
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert metadata['project']['name'] == 'nba-pregame-models'
    assert metadata['project']['version'] == '0.1.0'
    find = metadata['tool']['setuptools']['packages']['find']
    assert find['include'] == ['core', 'core.*', 'team', 'team.*', 'props', 'props.*']
    assert 'props.polymarket' in find['exclude']
    assert metadata['tool']['setuptools']['package-data']['core'] == ['live_feed_overrides.csv']
    assert metadata['tool']['setuptools']['include-package-data'] is False


def test_public_checkout_contains_no_private_packages_or_imports():
    private = {'live_bot', 'polymarket_venue', 'polymarket_props'}
    assert not any((ROOT / name).exists() for name in private)
    assert not (ROOT / 'props/polymarket').exists()
    bad = []
    for package in ('core', 'team', 'props', 'dashboard', 'dashboard_props'):
        for path in (ROOT / package).rglob('*.py'):
            bad += [f'{path.relative_to(ROOT)} imports {name}'
                    for name in set(_imported_roots(path)) & private]
    assert not bad, bad
