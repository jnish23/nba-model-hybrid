"""
Build the Phase 2 dataset (data/props_features.db: player_game_base, player_game_labels, player_game_at_T) and the feature dictionary.
    python scripts/build_props_features.py                      # uses the cached linked injury reports
    python scripts/build_props_features.py --refresh-reports    # re-parse new report PDFs and re-link (needed after more downloads)
Writes reports/props/phase2_feature_dictionary.csv and .md.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.features import build, dictionary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/nba_stats.db')
    ap.add_argument('--props-db', default='data/props_odds.db')
    ap.add_argument('--out-db', default=str(build.OUT_DB))
    ap.add_argument('--refresh-reports', action='store_true')
    ap.add_argument('--max-date', default=None, help='only history up to this ET date (for tests and quick looks)')
    ap.add_argument('--dictionary-dir', default='reports/props')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    base, labels, at_t = build.build_all(args.db, args.props_db, args.out_db, max_date=args.max_date, refresh_reports=args.refresh_reports)
    tables = {'player_game_base': base, 'player_game_labels': labels, 'player_game_at_T': at_t}
    missing = dictionary.missing_entries(tables)
    if missing:
        raise SystemExit(f'features without a dictionary entry: {missing}')
    d = dictionary.build_dictionary(tables)
    out = Path(args.dictionary_dir)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / 'phase2_feature_dictionary.csv', index=False)
    summary = d.groupby(['table', 'group']).size().rename('features').reset_index()
    from props.quality import md_table
    (out / 'phase2_feature_dictionary.md').write_text(
        '# Phase 2 feature dictionary\n\n' + f'{len(d)} features. Full detail (with availability by season) in phase2_feature_dictionary.csv.\n\n'
        + md_table(summary) + '\n' + md_table(d[['table', 'feature', 'group', 'source', 'first_season_with_data', 'non_null_share', 'description']]),
        encoding='utf-8')
    print(f'{len(d)} features documented; tables written to {args.out_db}')


if __name__ == '__main__':
    main()
