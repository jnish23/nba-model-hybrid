"""
Build the Phase 2 dataset (data/props_features.db: player_game_base, player_game_labels, player_game_at_T) and the feature dictionary.
    python scripts/build_props_features.py                      # uses the cached linked injury reports
    python scripts/build_props_features.py --refresh-reports    # re-parse new report PDFs and re-link (needed after more downloads)
Writes reports/props/phase2_feature_dictionary.csv and .md.

Research builds (WP6a, reports/props/audit_2026-11/WP6_WP7_experiment_plan.md B.6): passing --membership (box, v2nq or pregame_all) makes the build a research
build, which must name a NEW --out-db (never data/props_features_v3.db or data/props_features.db, never an existing file) and a --dictionary-dir outside
reports/props, reads the existing linked-report cache only (no --refresh-reports), and writes <out-db>.manifest.json (sha256 of the DB and of the cache, code
commit, membership). Without --membership the script behaves exactly as before.
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.features import build, dictionary, pregame_roster  # noqa: E402

PROTECTED_DB_NAMES = ('props_features_v3.db', 'props_features.db')
DEFAULT_DICTIONARY_DIR = 'reports/props'


def research_guard(args):
    """Raise SystemExit when a research build (--membership given) could overwrite a production file. Returns nothing when the arguments are safe."""
    if args.membership is None:
        return
    out = Path(args.out_db)
    if out.name in PROTECTED_DB_NAMES or out.resolve() == Path(build.OUT_DB).resolve():
        raise SystemExit(f'--membership builds may not write {out} (protected name {PROTECTED_DB_NAMES}); name a new file')
    if out.exists():
        raise SystemExit(f'--out-db {out} exists; a research build never replaces a file')
    if args.dictionary_dir is None:
        raise SystemExit('--membership builds need an explicit --dictionary-dir (the default would overwrite reports/props/phase2_feature_dictionary.*)')
    if Path(args.dictionary_dir).resolve() == Path(DEFAULT_DICTIONARY_DIR).resolve():
        raise SystemExit('--dictionary-dir may not be reports/props for a --membership build')
    if args.refresh_reports:
        raise SystemExit('--refresh-reports rewrites the parse caches the live jobs read; not allowed with --membership')
    if not Path(args.linked_cache).exists():
        raise SystemExit(f'linked-report cache {args.linked_cache} not found; a research build never re-parses the reports')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def code_commit():
    try:
        c = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent, timeout=30).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parent.parent, timeout=60).stdout.strip()
        return c + ('-dirty' if dirty else '')
    except Exception:                                    # noqa: BLE001 - provenance only
        return 'unknown'


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/nba_stats.db')
    ap.add_argument('--props-db', default='data/props_odds.db')
    ap.add_argument('--out-db', default=str(build.OUT_DB))
    ap.add_argument('--refresh-reports', action='store_true')
    ap.add_argument('--max-date', default=None, help='only history up to this ET date (for tests and quick looks)')
    ap.add_argument('--dictionary-dir', default=None, help=f'default {DEFAULT_DICTIONARY_DIR}; required (and elsewhere) with --membership')
    ap.add_argument('--membership', default=None, choices=pregame_roster.MEMBERSHIPS,
                    help='research build: teammate set of av_* (box = today, v2nq, pregame_all); see the module docstring for the guards')
    ap.add_argument('--linked-cache', default=str(build.LINKED_CACHE), help='linked injury-report cache read by a --membership build')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    research_guard(args)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.membership is None:
        base, labels, at_t = build.build_all(args.db, args.props_db, args.out_db, max_date=args.max_date, refresh_reports=args.refresh_reports)
    else:
        import pandas as pd
        reports = pd.read_pickle(args.linked_cache)
        base, labels, at_t = build.build_all(args.db, args.props_db, args.out_db, max_date=args.max_date, membership=args.membership, reports=reports)
    tables = {'player_game_base': base, 'player_game_labels': labels, 'player_game_at_T': at_t}
    missing = dictionary.missing_entries(tables)
    if missing:
        raise SystemExit(f'features without a dictionary entry: {missing}')
    d = dictionary.build_dictionary(tables)
    out = Path(args.dictionary_dir or DEFAULT_DICTIONARY_DIR)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / 'phase2_feature_dictionary.csv', index=False)
    summary = d.groupby(['table', 'group']).size().rename('features').reset_index()
    from props.quality import md_table
    (out / 'phase2_feature_dictionary.md').write_text(
        '# Phase 2 feature dictionary\n\n' + f'{len(d)} features. Full detail (with availability by season) in phase2_feature_dictionary.csv.\n\n'
        + md_table(summary) + '\n' + md_table(d[['table', 'feature', 'group', 'source', 'first_season_with_data', 'non_null_share', 'description']]),
        encoding='utf-8')
    if args.membership is not None:
        man = {'out_db': str(args.out_db), 'sha256': sha256(args.out_db), 'membership': args.membership, 'code_commit': code_commit(),
               'linked_cache': str(args.linked_cache), 'linked_cache_sha256': sha256(args.linked_cache), 'nba_db': str(args.db), 'props_db': str(args.props_db),
               'max_date': args.max_date, 'rows': {k: int(len(v)) for k, v in tables.items()}}
        Path(str(args.out_db) + '.manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
    print(f'{len(d)} features documented; tables written to {args.out_db}')


if __name__ == '__main__':
    main()
