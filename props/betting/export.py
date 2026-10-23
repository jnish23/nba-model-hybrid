"""
Tables for the props dashboard (backtest mode): the frozen revised rule's qualified bets for each season with the three pricing views, names, times, price by book, line history,
a compact distribution per bet, CLV, season summaries (with the placebo), reliability and monthly forecast bias. Pure data preparation; no Streamlit.

The rule is read from reports/props/phase5_revised_rule.frozen_rule.json and applied EXACTLY as in the scoring (no recalibration, best price, earliest time with positive EV).
"""
import json
import pickle
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from core.timeutil import et_game_date
from dashboard_props import logic
from props.betting import evaluate as be, rules
from props.models import evaluate as ev, market_bench as mb, outcome as oc, registry
from props.models.pipeline import observed

SEASONS = {
    '2023-24': {'p5': 'v3_dev_p5', 'full': 'v3_dev', 'start': '2023-10-01', 'end': '2024-06-30', 'kind': 'development (in-sample for k and the rule)'},
    '2024-25': {'p5': 'v3_seen_p5', 'full': 'v3_seen_2024-25', 'start': '2024-10-01', 'end': '2025-06-30', 'kind': 'development (in-sample for k and the rule)'},
    '2025-26': {'p5': 'v3_final_p5', 'full': 'v3_final_2025-26', 'start': '2025-10-01', 'end': '2026-06-30', 'kind': 'final holdout (scored once)'},
}
PLACEBO_SEEDS = (11, 12, 13, 14)
FLAGGED = {'stands': 'stands', 'early_exit_flagged_books': 'flagged', 'early_exit_all_books': 'all'}
KEY = ['game_id', 'player_id', 'stat', 'minutes_before', 'line']


def sha256(path):
    return registry.sha256_file(path)


def labels_table(db):
    with sqlite3.connect(db) as conn:
        lab = pd.read_sql('SELECT game_id, player_id, lbl_played, lbl_minutes, y_points, y_rebounds, y_assists, y_threes, y_pra FROM player_game_labels', conn)
    lab['game_id'] = lab['game_id'].astype(str)
    return lab


def names_and_times(props_db, start, end):
    with sqlite3.connect(f'file:{props_db}?mode=ro', uri=True, timeout=120) as conn:
        games = pd.read_sql("SELECT game_id, max(home_team) AS home_team, max(away_team) AS away_team FROM prop_snapshots WHERE game_id IS NOT NULL "
                            "AND substr(commence_time_utc, 1, 10) BETWEEN ? AND ? GROUP BY game_id", conn, params=[start, end])
        players = pd.read_sql("SELECT game_id, player_id, max(player_name_raw) AS player_name FROM prop_snapshots WHERE player_id IS NOT NULL AND player_id > 0 "
                              "AND substr(commence_time_utc, 1, 10) BETWEEN ? AND ? GROUP BY game_id, player_id", conn, params=[start, end])
        tips = pd.read_sql("SELECT game_id, tip_ref_utc FROM event_ref WHERE game_id IS NOT NULL", conn).drop_duplicates('game_id', keep='last')
    return games, players, tips


def market_level(q):
    g = q.groupby(KEY)
    return g.agg(over_med=('over_dec', 'median'), under_med=('under_dec', 'median'), over_best=('over_dec', 'max'), under_best=('under_dec', 'max'), n_quotes=('book', 'nunique')).reset_index()


def clv_per_bet(sel, cons):
    """Consensus fair probability of the bet side at T-5 minus at the bet time, only where the consensus line did not move; NaN for T-5 bets and moved lines. Also the line move in the bettor's favour."""
    close = cons[cons['minutes_before'] == 5][['game_id', 'player_id', 'stat', 'consensus_line', 'consensus_fair_over']].rename(
        columns={'consensus_line': 'close_line', 'consensus_fair_over': 'close_fair_over'})
    b = sel[['game_id', 'player_id', 'stat', 'minutes_before', 'line', 'best_side', 'consensus_fair_over']].merge(close, on=['game_id', 'player_id', 'stat'], how='left', validate='many_to_one')
    over = (b['best_side'] == 'over').to_numpy()
    fair_bet = np.where(over, b['consensus_fair_over'], 1 - b['consensus_fair_over'])
    fair_close = np.where(over, b['close_fair_over'], 1 - b['close_fair_over'])
    has = b['close_line'].notna().to_numpy() & (b['minutes_before'].to_numpy() > 5)
    same = has & (b['close_line'].to_numpy() == b['line'].to_numpy())
    clv = np.where(same, fair_close - fair_bet, np.nan)
    move = np.where(has & ~same, np.where(over, 1.0, -1.0) * (b['close_line'].to_numpy() - b['line'].to_numpy()), np.nan)
    return clv, move, b['close_line'].to_numpy()


def _pmf_json(pmf):
    cum = np.cumsum(pmf)
    n = int(np.searchsorted(cum, 0.9999)) + 1
    return [round(float(x), 4) for x in pmf[:max(n, 3)]]


def build_season(season, cfg, rule, props_db, labels, n_boot=1000):
    cons, bets = pickle.load(open(f'data/props_betting/prices_{season}.pkl', 'rb'))
    idx, pm = registry.load_predictions(cfg['p5'], season)
    q = be.attach_outcomes(be.quote_table(bets, idx, pm), labels)
    q = q[q['stat'].isin(rule['stats'])].reset_index(drop=True)
    k = rule['k_by_stat']
    prim = rule['primary']
    q = rules.add_ev(q, k)
    sel = be.settle_all(rules.select(q, prim['e'], prim['time'], prim['min_books']))
    for long, short in FLAGGED.items():
        sel[f'status_{short}'] = sel.pop(f'status_{long}')
        sel = sel.drop(columns=[f'profit_{long}'])
    m = market_level(q)
    sel = sel.merge(m, on=KEY, how='left', validate='many_to_one')
    over = (sel['best_side'] == 'over').to_numpy()
    sel['side'] = sel['best_side']
    sel['dec_best'] = sel['dec_price']
    sel['dec_median'] = np.where(over, sel['over_med'], sel['under_med'])
    sel['dec_flat'] = logic.FLAT_110
    sel['ev_best'] = sel['best_ev']
    for name, o, u in (('median', sel['over_med'], sel['under_med']), ('flat', logic.FLAT_110, logic.FLAT_110)):
        eo, eu = rules.ev_over_under(sel['p_star'], sel['push'], o, u)
        sel[f'ev_{name}'] = np.where(over, eo, eu)
    sel['p_over_model'], sel['p_over_star'], sel['p_over_market'] = sel['p_mod'], sel['p_star'], sel['consensus_fair_over']
    sel['clv_prob'], sel['line_move_favour'], sel['close_line'] = clv_per_bet(sel, cons)
    games, players, tips = names_and_times(props_db, cfg['start'], cfg['end'])
    sel = sel.merge(games, on='game_id', how='left', validate='many_to_one').merge(players, on=['game_id', 'player_id'], how='left', validate='many_to_one') \
             .merge(tips, on='game_id', how='left', validate='many_to_one')
    sel['game_date'] = [et_game_date(t) if isinstance(t, str) else None for t in sel['tip_ref_utc']]
    sel['placed_at_utc'] = (pd.to_datetime(sel['tip_ref_utc'], utc=True) - pd.to_timedelta(sel['minutes_before'], unit='m')).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    sel['matchup'] = sel['away_team'].astype(str) + ' @ ' + sel['home_team'].astype(str)
    sel['season'], sel['data_kind'] = season, cfg['kind']
    sel['bet_id'] = season + '|' + sel['game_id'] + '|' + sel['player_id'].astype(str) + '|' + sel['stat']
    sel['minutes_actual'] = sel['lbl_minutes']
    # line history from the consensus at the three decision times
    hist = cons.pivot_table(index=['game_id', 'player_id', 'stat'], columns='minutes_before', values=['consensus_line', 'consensus_fair_over'], aggfunc='first')
    hist.columns = [f"{'line' if a == 'consensus_line' else 'mkt'}_{b}" for a, b in hist.columns]
    sel = sel.merge(hist.reset_index(), on=['game_id', 'player_id', 'stat'], how='left', validate='many_to_one')
    # distributions and prices by book, keyed by bet id
    pmf = {bid: _pmf_json(pm[st][i]) for bid, st, i in zip(sel['bet_id'], sel['stat'], sel['i'])}
    qk = q.merge(sel[KEY + ['bet_id', 'side']], on=KEY, how='inner')
    qk['px'] = np.where(qk['side'] == 'over', qk['over_dec'], qk['under_dec'])
    books = {bid: {b: round(float(p), 3) for b, p in zip(g['book'], g['px'])} for bid, g in qk.groupby('bet_id')}
    # season summary with the placebo
    summ = summarize(season, cfg, sel, q, cons, k, prim, n_boot)
    keep = ['bet_id', 'season', 'data_kind', 'game_id', 'game_date', 'tip_ref_utc', 'placed_at_utc', 'minutes_before', 'matchup', 'home_team', 'away_team', 'player_id', 'player_name', 'stat', 'side',
            'line', 'book', 'dec_best', 'dec_median', 'dec_flat', 'ev_best', 'ev_median', 'ev_flat', 'p_over_model', 'p_over_star', 'p_over_market', 'push', 'n_at_line', 'n_quotes',
            'expected_minutes', 'minutes_actual', 'y', 'played', 'status_stands', 'status_flagged', 'status_all', 'clv_prob', 'line_move_favour', 'close_line',
            'line_240', 'line_60', 'line_5', 'mkt_240', 'mkt_60', 'mkt_5']
    return sel[[c for c in keep if c in sel.columns]], pmf, books, summ


def summarize(season, cfg, sel, q, cons, k, prim, n_boot):
    row = {'season': season, 'kind': cfg['kind'], 'bets': int(len(sel)), 'player_games': int(sel.groupby(['game_id', 'player_id']).ngroups)}
    for mode in logic.PRICE_MODES:
        for tr in logic.TREATMENTS:
            r = logic.roi_summary(sel, mode, tr, n_boot)
            row.update({f'roi_{mode}_{tr}': r['roi'], f'roi_{mode}_{tr}_lo': r['lo'], f'roi_{mode}_{tr}_hi': r['hi']})
    r = logic.roi_summary(sel, 'median', 'stands', n_boot)
    row.update({'settled_bets': r['bets'], 'win_rate': r['win_rate']})
    c = sel['clv_prob'].dropna()
    cp, clo, chi = ev.cluster_bootstrap(c.to_numpy(), sel.loc[c.index, 'game_id'], n_boot=n_boot) if len(c) > 50 else (np.nan, np.nan, np.nan)
    row.update({'clv_real_pts': 100 * cp, 'clv_real_lo_pts': 100 * clo, 'clv_n': int(len(c))})
    pl = []
    for sd in PLACEBO_SEEDS:
        qp = rules.add_ev(be.placebo_quotes(q, sd), k)
        sp = be.settle_all(rules.select(qp, prim['e'], prim['time'], prim['min_books']))
        clv, _, _ = clv_per_bet(sp, cons)
        cc = pd.Series(clv).dropna()
        pcp, plo, phi = ev.cluster_bootstrap(cc.to_numpy(), sp.loc[cc.index, 'game_id'], n_boot=300) if len(cc) > 50 else (np.nan,) * 3
        stake = np.isin(sp['status_stands'], ['win', 'loss']).astype(float)
        prof = np.where(sp['status_stands'] == 'win', sp['dec_price'] - 1, np.where(sp['status_stands'] == 'loss', -1.0, 0.0))
        pl.append({'clv': 100 * pcp, 'clv_hi': 100 * phi, 'roi': prof.sum() / max(stake.sum(), 1)})
        del qp, sp
    pl = pd.DataFrame(pl)
    row.update({'clv_placebo_mean_pts': pl['clv'].mean(), 'clv_placebo_max_hi_pts': pl['clv_hi'].max(), 'placebo_roi': pl['roi'].mean(),
                'clv_excess_pts': row['clv_real_pts'] - pl['clv'].mean(), 'clv_excess_lower_pts': row['clv_real_lo_pts'] - pl['clv_hi'].max()})
    return row


def reliability_and_slope(season, cfg, labels, rule, n_boot=1000):
    cons, _ = pickle.load(open(f'data/props_betting/prices_{season}.pkl', 'rb'))
    idx, pm = registry.load_predictions(cfg['p5'], season)
    rows = idx.merge(labels, on=['game_id', 'player_id'], how='left', validate='many_to_one')
    cmp = mb.compare(rows, pm, cons.rename(columns={'stat': 'market'}))
    four = cmp[cmp['stat'].isin(rule['stats'])]
    slope = mb.information_slope(four, n_boot=n_boot).query("stat == 'all'").iloc[0]
    rel = mb.reliability_at_book_line(four).assign(season=season)
    return rel, {'slope4': slope['slope'], 'slope4_lo': slope['slope_lo'], 'slope4_hi': slope['slope_hi']}


def monthly_bias(season, cfg, stats):
    rows, pm = ev.load_rows_light(cfg['full'], season)
    out = []
    month = pd.to_datetime(rows['game_date']).dt.strftime('%Y-%m')
    for stat in stats:
        y = observed(rows, stat)
        mu = oc.pmf_mean(pm[stat])
        d = pd.DataFrame({'month': month, 'y': y, 'mu': mu}).dropna()
        g = d.groupby('month').agg(n=('y', 'size'), observed=('y', 'sum'), predicted=('mu', 'sum')).reset_index()
        g['ratio'] = g['observed'] / g['predicted']
        out.append(g.assign(season=season, stat=stat))
    return pd.concat(out, ignore_index=True)


def export_all(out_dir, props_db='data/props_odds.db', features_db='data/props_features_v3.db', rule_path='reports/props/phase5_revised_rule.frozen_rule.json', n_boot=1000, seasons=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rule = json.loads(Path(rule_path).read_text(encoding='utf-8'))
    labels = labels_table(features_db)
    bets_all, pmf_all, books_all, summ, rel_all, health = [], {}, {}, [], [], []
    for season, cfg in SEASONS.items():
        if seasons and season not in seasons:
            continue
        sel, pmf, books, s = build_season(season, cfg, rule, props_db, labels, n_boot)
        rel, sl = reliability_and_slope(season, cfg, labels, rule, n_boot)
        s.update(sl)
        bets_all.append(sel); pmf_all.update(pmf); books_all.update(books); summ.append(s); rel_all.append(rel)
        health.append(monthly_bias(season, cfg, rule['stats'] + ['points']))
        print(f'{season}: {len(sel)} bets, {sel.groupby(["game_id", "player_id"]).ngroups} player-games', flush=True)
    bets = pd.concat(bets_all, ignore_index=True)
    bets.to_csv(out / 'bets.csv', index=False)
    (out / 'pmf.json').write_text(json.dumps(pmf_all, separators=(',', ':')), encoding='utf-8')
    (out / 'books.json').write_text(json.dumps(books_all, separators=(',', ':')), encoding='utf-8')
    pd.DataFrame(summ).to_csv(out / 'season_summary.csv', index=False)
    pd.concat(rel_all, ignore_index=True).to_csv(out / 'reliability.csv', index=False)
    pd.concat(health, ignore_index=True).to_csv(out / 'bias_monthly.csv', index=False)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = {'mode': 'backtest', 'rule': rule, 'rule_sha256': sha256(rule_path), 'code_commit': head, 'features_db_sha256': sha256(features_db),
            'built_at_utc': pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), 'seasons': {s: c['kind'] for s, c in SEASONS.items() if not seasons or s in seasons},
            'pricing': {'median': 'median over the bettable books of their quotes at the consensus line', 'best': 'best quote among the bettable books', 'flat': 'flat -110 (1.909)'}}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return bets
