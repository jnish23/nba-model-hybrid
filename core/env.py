"""
Minimal .env loader (no dependency). Used so scheduled jobs and scripts can read secrets such as ODDS_API_KEY from a
gitignored .env file. Values are never logged or printed here; variables already set in the real environment win.

Search order: the project root's .env, then (for a git worktree) the main checkout's .env.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _main_checkout(root):
    """If `root` is a linked git worktree, the main working tree's directory; else None."""
    git = root / '.git'
    if not git.is_file():
        return None
    text = git.read_text(encoding='utf-8', errors='ignore').strip()
    if not text.startswith('gitdir:'):
        return None
    gitdir = Path(text.split(':', 1)[1].strip())          # <main>/.git/worktrees/<name>
    return gitdir.parent.parent.parent if gitdir.parent.name == 'worktrees' else None


def candidate_files(root=ROOT):
    out = [Path(root) / '.env']
    main = _main_checkout(Path(root))
    if main is not None:
        out.append(main / '.env')
    return out


def parse_env_text(text):
    """KEY=VALUE lines; '#' comments, blank lines, optional 'export ', optional single/double quotes, CRLF tolerated."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export '):].lstrip()
        key, _, value = line.partition('=')
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        elif ' #' in value:
            value = value.split(' #', 1)[0].rstrip()
        if key.strip():
            out[key.strip()] = value
    return out


def load_env(root=ROOT, override=False):
    """Load the first .env found into os.environ. Returns the NAMES it set (never values)."""
    for path in candidate_files(root):
        if path.is_file():
            names = []
            for k, v in parse_env_text(path.read_text(encoding='utf-8-sig')).items():
                if override or k not in os.environ:
                    os.environ[k] = v
                    names.append(k)
            return names
    return []
