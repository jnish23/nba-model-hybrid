"""
Phase 2 data report, computed from data/props_features.db (nothing typed by hand):
    python scripts/report_props_features.py --out reports/props/phase2_dataset_report.md
Sections: size, feature groups, availability, injury information (staleness, status vs outcome, probability calibration), market environment
provenance and agreement, and a plain sanity check that the features carry signal (rank correlations; NOT a model evaluation).
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.features import build, dictionary  # noqa: E402
from props.quality import md_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(build.OUT_DB))
    ap.add_argument('--props-db', default='data/props_odds.db')
    ap.add_argument('--out', default='reports/props/phase2_dataset_report.md')
    args = ap.parse_args()
    t = build.read_tables(args.dataset)
    base, labels, at_t = t['player_game_base'], t['player_game_labels'], t['player_game_at_T']
    d = dictionary.build_dictionary(t)
    md = ['# Phase 2 dataset report\n']

    # 1. size
    per = base.merge(labels[['game_id', 'player_id', 'lbl_played']], on=['game_id', 'player_id']).groupby('season').agg(
        player_games=('player_id', 'size'), played=('lbl_played', 'sum'), games=('game_id', 'nunique'), players=('player_id', 'nunique'))
    per['played_share'] = per['played'] / per['player_games']
    md += ['## 1. Size\n', f'{len(base):,} player-games (every listed player, played or not), {len(at_t):,} rows at the three decision times '
           '(T-240, T-60, T-5 minutes before the reference tip), tables joined on (player_id, game_id [, minutes_before]).\n', md_table(per.reset_index())]

    # 2. features
    g = d.groupby(['table', 'group']).size().rename('features').reset_index()
    md += ['## 2. Features by group\n', f'{len(d)} features in total; each documented in `phase2_feature_dictionary.csv` with availability by season.\n', md_table(g)]

    # 3. availability
    nulls = base.groupby('season')[['m_min_h15', 'r_points_h15', 'c_fg3_pct_h15', 'tm_off_rtg', 'opp_zone_ra_share_allowed']].apply(lambda x: x.isna().mean())
    md += ['## 3. Missing values by season (share null)\n',
           'Players with no earlier game have no minutes history (NaN, not zero); their per-minute rates fall back to the league prior, so those are never null.\n',
           md_table(nulls.reset_index())]

    # 4. injury information
    x = at_t.merge(labels[['game_id', 'player_id', 'lbl_played']], on=['game_id', 'player_id']).merge(
        base[['game_id', 'player_id', 'season', 'm_bucket']], on=['game_id', 'player_id'])
    age = x.pivot_table(index='season', columns='minutes_before', values='inj_report_age_min', aggfunc='median').round(0)
    age.columns = [f'median age at T-{c} (min)' for c in age.columns]
    miss = x.groupby('season')['inj_report_missing'].mean().rename('share with no report covering the game')
    md += ['## 4. Injury information\n', '### Staleness of the report used (minutes before T). The archive is hourly from 2021-22 and sparse before; production will see '
           'fresher reports than the 2019-21 training rows did.\n', md_table(age.join(miss).reset_index()),
           '### P(played) by reported status and decision time (all seasons, realized)\n',
           md_table(x.pivot_table(index='inj_status', columns='minutes_before', values='lbl_played', aggfunc='mean').round(3).reset_index()),
           '### Status mix at T-60\n', md_table(x[x.minutes_before == 60]['inj_status'].value_counts(normalize=True).round(3).rename('share').reset_index())]
    cal = x[(x.minutes_before == 60) & (x.season >= '2023-24')].copy()
    cal['decile'] = pd.qcut(cal['inj_p_play'], 10, duplicates='drop')
    c = cal.groupby('decile', observed=True).agg(predicted=('inj_p_play', 'mean'), realized=('lbl_played', 'mean'), n=('lbl_played', 'size')).round(3)
    c.index = c.index.astype(str)
    md += ['### Calibration of the play probability (T-60, 2023-24 onward, by decile)\n', md_table(c.reset_index()),
           '_The probability for a season is estimated from earlier seasons only (checked by a test that shuffles the labels)._\n']

    # 5. market environment
    src = at_t.merge(base[['game_id', 'player_id', 'season']], on=['game_id', 'player_id']).groupby(['season', 'minutes_before'])['env_source'].agg(
        lambda s: s.value_counts(normalize=True).round(2).to_dict())
    md += ['## 5. Market game environment: where it comes from\n', '0 none, 1 our snapshot at T, 2 SBR opening line (proxy for T-240), 3 SBR closing line (proxy for T-5).\n',
           md_table(src.rename('env_source shares').reset_index())]
    try:
        from props.features import market_env
        snap = market_env.snapshot_env(args.props_db)
        from core.odds_clean import load_clean_odds
        proxy = market_env.sbr_proxy(load_clean_odds(Path('data/nba_stats.db')))
        m = snap[snap.minutes_before == 5].merge(proxy[proxy.minutes_before == 5], on='game_id', suffixes=('_s', '_c')).dropna(subset=['home_spread_s', 'home_spread_c'])
        m2 = snap[snap.minutes_before == 240].merge(proxy[proxy.minutes_before == 240], on='game_id', suffixes=('_s', '_o')).dropna(subset=['home_spread_s', 'home_spread_o'])
        md += [f'Proxy check on {len(m):,} games with both: the T-5 snapshot spread differs from the SBR closing spread by {(m.home_spread_s - m.home_spread_c).abs().mean():.2f} points '
               f'on average (total: {(m.total_s - m.total_c).abs().mean():.2f}); the T-240 snapshot differs from the SBR opening spread by '
               f'{(m2.home_spread_s - m2.home_spread_o).abs().mean():.2f} points. The closing proxy is close; the opening proxy is a looser stand-in for T-240.\n']
    except Exception as e:                                                       # the report must not fail on an unavailable store
        md += [f'_Proxy check unavailable ({type(e).__name__})._\n']

    # 6. sanity
    s = base.merge(labels[['game_id', 'player_id', 'lbl_played', 'lbl_minutes', 'y_points', 'y_rebounds', 'y_assists', 'y_threes']], on=['game_id', 'player_id'])
    s = s[s.lbl_played == 1]
    rows = []
    for feat, target in (('m_min_h15', 'lbl_minutes'), ('role_start_share_h15', 'lbl_minutes')):
        rows.append((feat, target, s[feat].corr(s[target], method='spearman')))
    s['exp_points'] = s['r_points_h15'] * s['m_min_h15']
    s['exp_reb'] = s['r_rebounds_h15'] * s['m_min_h15']
    s['exp_ast'] = s['r_assists_h15'] * s['m_min_h15']
    s['exp_3pm'] = s['r_fg3m_h15'] * s['m_min_h15']
    for feat, target in (('exp_points', 'y_points'), ('exp_reb', 'y_rebounds'), ('exp_ast', 'y_assists'), ('exp_3pm', 'y_threes')):
        rows.append((f'{feat} = rate x minutes', target, s[feat].corr(s[target], method='spearman')))
    md += ['## 6. Do the features carry signal? (Spearman rank correlation with the realized value, players who played)\n',
           'A plain sanity check that nothing is broken or reversed, **not** a model evaluation (that is Phase 4).\n',
           md_table(pd.DataFrame(rows, columns=['feature', 'target', 'spearman']).round(3))]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(md), encoding='utf-8')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
