"""
core/ is the base layer: core never imports team or props, and team never imports props.
props may reuse exactly ONE team module, the team model's pre-game feature builder (team.team_features_v2), so opponent and schedule features
come from code that already has train/serve parity tests. Nothing else in team/ (models, registry, betting, training) may be imported by props.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = {'core': {'team', 'props'}, 'team': {'props'}}
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
