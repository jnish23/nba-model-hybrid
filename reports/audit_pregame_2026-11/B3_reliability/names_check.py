"""Old fuzzy team-name lookup (verbatim from the pre-B3 run_inference_v2.get_todays_games) versus the new exact lookup,
over every team name in nba_odds, nba_odds_opening and odds_snapshots (and the abbreviation columns) of data/nba_stats.db."""
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')
import run_inference_v2 as rm

conn = sqlite3.connect('data/nba_stats.db')
team_info = pd.read_sql("SELECT DISTINCT teamId, teamName, teamTricode FROM team_basic_stats", conn)

name_map = {}
for _, t in team_info.iterrows():
    name_map[t['teamName']] = t['teamId']
    name_map[t['teamTricode']] = t['teamId']
    simple = t['teamName'].split(' ')[-1]
    name_map[simple] = t['teamId']
name_map['LA Clippers'] = 1610612746
name_map['L.A. Clippers'] = 1610612746
name_map['LA Lakers'] = 1610612747
name_map['L.A. Lakers'] = 1610612747
name_map['Clippers'] = 1610612746
name_map['Lakers'] = 1610612747


def old_map_id(name):
    if not isinstance(name, str):
        return None
    if name in name_map:
        return name_map[name]
    for k, v in name_map.items():
        if k in name or name in k:
            return v
    return None


lookup = rm.build_team_lookup(team_info)
names = {}
for t in ('nba_odds', 'nba_odds_opening'):
    for col in ('home_team', 'away_team'):
        for n, in conn.execute(f"SELECT DISTINCT {col} FROM {t}"):
            names.setdefault(n, set()).add(f'{t}.{col}')
for col in ('home_team', 'away_team'):
    for n, in conn.execute(f"SELECT DISTINCT {col} FROM odds_snapshots"):
        names.setdefault(n, set()).add(f'odds_snapshots.{col}')
rows = []
for n in sorted(names, key=str):
    o, nw = old_map_id(n), rm.map_team_id(n, lookup)
    rows.append((n, o, nw, o == nw))
df = pd.DataFrame(rows, columns=['name', 'old_id', 'new_id', 'same'])
print(df.to_string(index=False))
print('distinct names:', len(df), 'same:', int(df.same.sum()), 'different:', int((~df.same).sum()))

# abbreviations in the three tables: every one must be a tricode or an SBR abbreviation the job maps
abbrs = set()
for t, cols in (('nba_odds', ('home_team_abbr', 'away_team_abbr')), ('nba_odds_opening', ('home_team_abbr', 'away_team_abbr')),
                ('odds_snapshots', ('home_team_abbr', 'away_team_abbr'))):
    for c in cols:
        abbrs |= {a for a, in conn.execute(f"SELECT DISTINCT {c} FROM {t}")}
tri = set(team_info.teamTricode)
mapped = {rm.SBR_ABBREVIATIONS.get(a, a) for a in abbrs}
print('abbreviations seen:', len(abbrs), 'after SBR_ABBREVIATIONS all tricodes:', mapped <= tri, 'unmapped:', sorted(mapped - tri))
print('as team names through the new lookup:', {a: rm.map_team_id(a, lookup) for a in sorted(abbrs) if rm.map_team_id(a, lookup) is None})

# What the old fuzzy code mapped that the new one refuses (illustrative variants)
for v in ['Portland', 'Trailblazers', 'Los Angeles', 'Philadelphia Sixers', 'NY Knicks', 'Hawks of Atlanta', 'Utah', 'San Antonio', 'LA', 'Charlotte']:
    print(f'{v!r:24} old={old_map_id(v)} new={rm.map_team_id(v, lookup)}')
