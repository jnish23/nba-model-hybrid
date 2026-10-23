"""
From due decision slots to signals and first-qualification bets. Nothing here chooses a rule: features come from props.live (the training functions), probabilities from the saved live
fold, quotes from props.betting.prices, selection and EV from props.betting.rules with the frozen rule file's k and thresholds.
"""
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dashboard_props import logic
from props.betting import evaluate as be, export, rules
from props.live import at_t as A, roster
from props.live.base import build_live_base
from props.live.frame import live_frame
from props.live.predict import predict_rows
from props.models import outcome as oc


@dataclass
class Context:
    hist: pd.DataFrame
    pos: dict
    reports: pd.DataFrame
    tips: pd.Series
    tables: dict
    bundle: dict
    rule: dict
    nba_db: str
    n_sims: int
    model_version: str
    provenance: dict            # model_sha256, features_db_sha256, rule_sha256, roster_rule, code_commit
    roster_rule: str = 'v2'


@dataclass
class SlotItem:
    game: pd.Series              # game_id, game_date, season, home_team_id, away_team_id, home_team, away_team, tip_ref_utc
    slot: int                    # nominal minutes before the reference tip
    snapshot_ts_utc: str
    snapshot_minutes: float
    event_id: str
    cons: pd.DataFrame
    bets: pd.DataFrame
    names: dict                  # player_id -> name


def slot_seed(game_id, slot):
    return int(hashlib.sha1(f'{game_id}|{slot}'.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def build_frames(ctx, date, items):
    """The model-input frame of every item of one date (one base build for the date)."""
    games = pd.DataFrame([i.game for i in items]).drop_duplicates('game_id')
    games = games.assign(game_date=date)
    h = ctx.hist[ctx.hist['game_date'] < date]
    season = str(games['season'].iloc[0])
    per = []
    extras = []
    for it in items:
        rep = A.asof_reports(ctx.reports, ctx.tips, it.snapshot_minutes)
        rp = A.report_players(rep, ctx.tips, it.snapshot_minutes)
        rp = rp[rp['game_id'] == it.game['game_id']]
        qp = it.cons[['game_id', 'player_id']].drop_duplicates()
        per.append((it, rep, rp, qp))
        extras += [rp, qp]
    base_all, unresolved = build_live_base(ctx.hist, ctx.pos, games, pd.concat(extras, ignore_index=True), date, ctx.nba_db,
                                           rosters=roster.previous_game_rosters(h, date, season))
    blocks, frames = [], {}
    for it, rep, rp, qp in per:
        g1 = games[games['game_id'] == it.game['game_id']]
        members = roster.rule_members(ctx.roster_rule, g1, ctx.hist, date, rp, qp)
        base_T = base_all.merge(members, on=['game_id', 'player_id'], how='inner', validate='one_to_one')
        blocks.append(A.live_at_t(base_T, rep, ctx.tips, it.slot, ctx.tables[it.slot], asof_minutes=it.snapshot_minutes))
    frame = live_frame(base_all, blocks)
    return frame, unresolved


def evaluate_item(ctx, it, frame, existing):
    """Signals (every evaluated rule-stat market) and the new bets of one slot. existing: set of (game_id, player_id, stat) already in the bet record."""
    rule = ctx.rule
    gid = it.game['game_id']
    rows = frame[(frame['game_id'] == gid) & (frame['minutes_before'] == it.slot)]
    quoted = it.cons[['game_id', 'player_id']].drop_duplicates()
    rows = rows.merge(quoted, on=['game_id', 'player_id'], how='inner').sort_values(['player_id']).reset_index(drop=True)
    missing = sorted(set(quoted['player_id']) - set(rows['player_id']))
    detail = {'quoted_players': int(quoted['player_id'].nunique()), 'players_without_features': [int(p) for p in missing]}
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), detail
    mm, pm = predict_rows(ctx.bundle, rows, ctx.n_sims, slot_seed(gid, it.slot))
    idx = rows[['game_id', 'player_id', 'minutes_before', 'season', 'game_date']].assign(minutes_mean=mm).reset_index(drop=True)
    q = be.quote_table(it.bets, idx, pm)
    q = q[q['stat'].isin(rule['stats'])].reset_index(drop=True)
    if q.empty:
        return pd.DataFrame(), pd.DataFrame(), detail
    q = rules.add_ev(q, rule['k_by_stat'])
    prim = rule['primary']
    sel = rules.select(q, prim['e'], 'earliest', prim['min_books'])
    m = export.market_level(q)
    key = ['game_id', 'player_id', 'stat', 'minutes_before', 'line']
    best = q.sort_values('best_ev', ascending=False).drop_duplicates(['game_id', 'player_id', 'stat']).merge(m, on=key, how='left', validate='many_to_one')
    over = (best['best_side'] == 'over').to_numpy()
    best['dec_median'] = np.where(over, best['over_med'], best['under_med'])
    eo, eu = rules.ev_over_under(best['p_star'], best['push'], best['over_med'], best['under_med'])
    best['ev_median'] = np.where(over, eo, eu)
    eo, eu = rules.ev_over_under(best['p_star'], best['push'], logic.FLAT_110, logic.FLAT_110)
    best['ev_flat'] = np.where(over, eo, eu)
    qual_keys = set(map(tuple, sel[['game_id', 'player_id', 'stat']].itertuples(index=False, name=None)))
    k3 = list(map(tuple, best[['game_id', 'player_id', 'stat']].itertuples(index=False, name=None)))
    best['qualified'] = [int(k in qual_keys) for k in k3]
    best['already_bet'] = [int(k in existing) for k in k3]
    created = ctx.provenance
    sig = pd.DataFrame({
        'signal_id': [f"{it.game['season']}|{gid}|{p}|{s}|{it.slot}" for p, s in zip(best['player_id'], best['stat'])], 'season': it.game['season'], 'game_id': gid,
        'game_date': it.game['game_date'], 'tip_ref_utc': it.game['tip_ref_utc'], 'slot_minutes': it.slot, 'snapshot_ts_utc': it.snapshot_ts_utc, 'snapshot_minutes': it.snapshot_minutes,
        'player_id': best['player_id'].astype(int), 'player_name': best['player_id'].map(it.names), 'stat': best['stat'], 'line': best['line'], 'n_at_line': best['n_at_line'],
        'p_market': best['consensus_fair_over'], 'p_model': best['p_mod'], 'p_star': best['p_star'], 'push': best['push'], 'side': best['best_side'], 'book': best['book'],
        'dec_best': best['dec_price'], 'dec_median': best['dec_median'], 'ev_best': best['best_ev'], 'ev_median': best['ev_median'], 'ev_flat': best['ev_flat'],
        'expected_minutes': best['expected_minutes'], 'qualified': best['qualified'], 'already_bet': best['already_bet'], 'model_version': ctx.model_version})
    new = sel[[tuple(r) not in existing for r in sel[['game_id', 'player_id', 'stat']].itertuples(index=False, name=None)]]
    new = new.merge(m, on=key, how='left', validate='many_to_one')
    if new.empty:
        return sig, pd.DataFrame(), detail
    over = (new['best_side'] == 'over').to_numpy()
    dec_med = np.where(over, new['over_med'], new['under_med'])
    eo, eu = rules.ev_over_under(new['p_star'], new['push'], new['over_med'], new['under_med'])
    ev_med = np.where(over, eo, eu)
    eo, eu = rules.ev_over_under(new['p_star'], new['push'], logic.FLAT_110, logic.FLAT_110)
    ev_flat = np.where(over, eo, eu)
    books = {}
    for (g, p, s), grp in q.groupby(['game_id', 'player_id', 'stat']):
        books[(g, p, s)] = grp
    out = []
    tip = pd.to_datetime(it.game['tip_ref_utc'], utc=True)
    for j, r in enumerate(new.itertuples(index=False)):
        grp = books[(r.game_id, r.player_id, r.stat)]
        grp = grp[grp['line'] == r.line]
        px = grp['over_dec'] if r.best_side == 'over' else grp['under_dec']
        out.append({
            'bet_id': f"{it.game['season']}|{gid}|{int(r.player_id)}|{r.stat}", 'season': it.game['season'], 'game_id': gid, 'game_date': it.game['game_date'],
            'tip_ref_utc': it.game['tip_ref_utc'], 'placed_at_utc': (tip - pd.Timedelta(minutes=it.slot)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'minutes_before': it.slot,
            'matchup': f"{it.game['away_team']} @ {it.game['home_team']}", 'home_team': it.game['home_team'], 'away_team': it.game['away_team'], 'player_id': int(r.player_id),
            'player_name': it.names.get(int(r.player_id)), 'stat': r.stat, 'side': r.best_side, 'line': float(r.line), 'book': r.book, 'dec_best': float(r.dec_price),
            'dec_median': float(dec_med[j]), 'dec_flat': float(logic.FLAT_110), 'ev_best': float(r.best_ev), 'ev_median': float(ev_med[j]), 'ev_flat': float(ev_flat[j]),
            'p_over_model': float(r.p_mod), 'p_over_star': float(r.p_star), 'p_over_market': float(r.consensus_fair_over), 'push': float(r.push), 'n_at_line': int(r.n_at_line),
            'n_quotes': int(r.n_quotes), 'expected_minutes': float(r.expected_minutes), 'snapshot_ts_utc': it.snapshot_ts_utc, 'snapshot_minutes': it.snapshot_minutes,
            'quote_age_min': float(r.quote_age_min), 'pmf_json': _json(pm[r.stat][int(r.i)]),
            'books_json': _json_books(grp['book'], px), 'model_version': ctx.model_version, 'model_sha256': created['model_sha256'], 'features_db_sha256': created['features_db_sha256'],
            'rule_sha256': created['rule_sha256'], 'roster_rule': created['roster_rule'], 'code_commit': created['code_commit']})
    return sig, pd.DataFrame(out), detail


def _json(pmf):
    return json.dumps(export._pmf_json(pmf))


def _json_books(books, px):
    return json.dumps({b: round(float(p), 3) for b, p in zip(books, px)})
