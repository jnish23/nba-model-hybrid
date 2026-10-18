"""core/ is the base layer: core never imports team or props, team never imports props, props never imports team."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = {'core': {'team', 'props'}, 'team': {'props'}, 'props': {'team'}}


def _imported_roots(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module.split('.')[0]


def test_package_dependency_direction():
    bad = []
    for pkg, banned in FORBIDDEN.items():
        for path in (ROOT / pkg).rglob('*.py'):
            bad += [f'{path.relative_to(ROOT)} imports {m}' for m in set(_imported_roots(path)) & banned]
    assert not bad, bad


def test_nothing_imports_the_old_models_package():
    """`models/` now only holds saved artifacts (models/saved); the code moved to core/ and team/."""
    bad = []
    for pkg in ('core', 'team', 'props', 'scripts', 'dashboard', 'tests'):
        for path in (ROOT / pkg).rglob('*.py'):
            bad += [str(path.relative_to(ROOT)) for m in set(_imported_roots(path)) if m == 'models']
    assert not bad, bad
