"""
Selection rules for simulated betting (reports/props/phase5_plan.md, section 4). Nothing here is fitted to a result; k and the thresholds come from the development season
and are frozen before any later season is looked at.

For a quote at the consensus line:  p_mkt = consensus de-vigged P(over);  p_mod = the model's P(over | no push);  push = the model's P(push) at that line.
Shrunk probability  p* = p_mkt + k * (p_mod - p_mkt)  (k = information slope, floor 0, cap K_CAP).
EV per unit staked (a push refunds):  over = (1 - push) * (p* * (dec_over - 1) - (1 - p*)),  under = (1 - push) * ((1 - p*) * (dec_under - 1) - p*).
"""
import numpy as np
import pandas as pd

from core import devig
from props.betting import prices

K_CAP = 0.5
TIMES = (240, 60, 5)
P_CLIP = (0.01, 0.99)                     # p* clip of the C2-C4 forms (reports/props/audit_2026-11/WP6_WP7_experiment_plan.md, C.4); the rule file's p_star_clip is used when present
SUPPORTED_CANDIDATES = ('C0', 'C1', 'C2', 'C3', 'C4')       # the forms props.betting.a11_rules.candidate_rule can write


def shrunk(p_mkt, p_mod, k):
    return np.asarray(p_mkt, float) + np.asarray(k, float) * (np.asarray(p_mod, float) - np.asarray(p_mkt, float))


def ev_over_under(p_star, push, dec_over, dec_under):
    p_star, push = np.asarray(p_star, float), np.asarray(push, float)
    ev_o = (1.0 - push) * (p_star * (np.asarray(dec_over, float) - 1.0) - (1.0 - p_star))
    ev_u = (1.0 - push) * ((1.0 - p_star) * (np.asarray(dec_under, float) - 1.0) - p_star)
    return ev_o, ev_u


def clip_k(slope):
    return float(np.clip(slope, 0.0, K_CAP))


def add_ev(q, k_by_stat):
    """q: one row per bettable quote with p_mkt (consensus_fair_over), p_mod, push, over_dec, under_dec, stat. Adds p_star, ev_over, ev_under, best_side, best_ev."""
    k = q['stat'].map(k_by_stat).fillna(0.0).to_numpy(float)
    q = q.copy()
    q['p_star'] = shrunk(q['consensus_fair_over'], q['p_mod'], k)
    q['ev_over'], q['ev_under'] = ev_over_under(q['p_star'], q['push'], q['over_dec'], q['under_dec'])
    q['best_side'] = np.where(q['ev_over'] >= q['ev_under'], 'over', 'under')
    q['best_ev'] = np.maximum(q['ev_over'], q['ev_under'])
    q['dec_price'] = np.where(q['best_side'] == 'over', q['over_dec'], q['under_dec'])
    return q


def _map(q, d):
    return q['stat'].map(d or {}).fillna(0.0).to_numpy(float)          # a stat without a parameter gets 0 (as add_ev's fillna(0) for k); rule_spec refuses such a rule file for the new forms


def p_star(cand, q, params, clip=P_CLIP):
    """
    p* (probability of the OVER) for every quote row of q (columns stat, consensus_fair_over, p_mod, minutes_before), candidate forms of plan C.4 (the research and live paths both call this).
    params: C0/C1 {'b'}; C2 {'b', 'a'}; C3 {'b', 'a_pos', 'a_neg'} (a_pos where g = p_mod - p_mkt >= 0, a_neg where g < 0, so g == 0 takes a_pos); C4 {'b', 'a', 'c'}; P0 {'a'}; P00 {}.
    """
    pm = q['consensus_fair_over'].to_numpy(float)
    g = q['p_mod'].to_numpy(float) - pm
    lo, hi = clip
    if cand in ('C0', 'C1'):
        return shrunk(pm, q['p_mod'].to_numpy(float), _map(q, params['b']))
    if cand == 'C1_E4':
        key = q['stat'].astype(str) + '|' + q['minutes_before'].astype(int).astype(str)
        return shrunk(pm, q['p_mod'].to_numpy(float), key.map(params['b_slot']).fillna(0.0).to_numpy(float))
    if cand == 'C2':
        return np.clip(pm + _map(q, params['a']) + _map(q, params['b']) * g, lo, hi)
    if cand == 'C3':
        a = np.where(g >= 0, _map(q, params['a_pos']), _map(q, params['a_neg']))
        return np.clip(pm + a + _map(q, params['b']) * g, lo, hi)
    if cand == 'C4':
        c = float(params['c'])
        return np.clip(pm + _map(q, params['a']) + _map(q, params['b']) * np.clip(g, -c, c), lo, hi)
    if cand == 'P0':
        return np.clip(pm + _map(q, params['a']), lo, hi)
    if cand == 'P00':
        return pm.copy()
    raise ValueError(cand)


def add_ev_pstar(q, ps):
    """add_ev with p* given instead of k: the same EV formula (ev_over_under), side choice, best_ev and dec_price, for both sides of every quote."""
    q = q.copy()
    q['p_star'] = np.asarray(ps, float)
    q['ev_over'], q['ev_under'] = ev_over_under(q['p_star'], q['push'], q['over_dec'], q['under_dec'])
    q['best_side'] = np.where(q['ev_over'] >= q['ev_under'], 'over', 'under')
    q['best_ev'] = np.maximum(q['ev_over'], q['ev_under'])
    q['dec_price'] = np.where(q['best_side'] == 'over', q['over_dec'], q['under_dec'])
    return q


class RuleFileError(ValueError):
    """A rule file whose optional keys are inconsistent or unknown. Raised at load, never worked around (a silent fallback to the k-only form would record bets under another rule)."""


def _finite(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
        raise RuleFileError(f'{what} must be a finite number, got {x!r}')
    return float(x)


def rule_spec(rule):
    """
    Validates a rule dict (a loaded rule file) and resolves its behaviour: dict(candidate, params, clip, pmarket, cap).
    A rule WITHOUT the optional keys (candidate, intercept_by_stat, p_star_clip, pmarket, cap_per_player_game) is the frozen k-only form: candidate None, pmarket the current method (multiplicative
    de-vig, mean over all books), cap None; it is not validated further, so the frozen rule behaves exactly as before. With `candidate` set (C0-C4) every stat of rule['stats'] needs its parameters.
    Raises RuleFileError for an unknown candidate or pmarket variant, a missing or non-numeric parameter, intercept_by_stat or p_star_clip without a candidate, a bad cap.
    """
    cand = rule.get('candidate')
    if cand is None:
        if 'intercept_by_stat' in rule or 'p_star_clip' in rule:
            raise RuleFileError('intercept_by_stat / p_star_clip given without a candidate: the rule file does not say which form to use')
        params, clip = None, P_CLIP
    else:
        if cand not in SUPPORTED_CANDIDATES:
            raise RuleFileError(f'unknown candidate {cand!r} (supported: {", ".join(SUPPORTED_CANDIDATES)})')
        stats = list(rule.get('stats') or [])
        if not stats:
            raise RuleFileError('rule file has no stats')
        kb = rule.get('k_by_stat') or {}
        miss = [s for s in stats if s not in kb]
        if miss:
            raise RuleFileError(f'k_by_stat has no entry for {miss}')
        params = {'b': {s: _finite(kb[s], f'k_by_stat[{s}]') for s in stats}}
        icpt = rule.get('intercept_by_stat')
        if cand in ('C2', 'C3', 'C4'):
            if not isinstance(icpt, dict):
                raise RuleFileError(f'candidate {cand} needs intercept_by_stat')
            miss = [s for s in stats if s not in icpt]
            if miss:
                raise RuleFileError(f'intercept_by_stat has no entry for {miss}')
            if cand == 'C3':
                for s in stats:
                    if not isinstance(icpt[s], dict) or {'a_pos', 'a_neg'} - set(icpt[s]):
                        raise RuleFileError(f'candidate C3 needs intercept_by_stat[{s}] = {{a_pos, a_neg}}')
                params['a_pos'] = {s: _finite(icpt[s]['a_pos'], f'intercept_by_stat[{s}].a_pos') for s in stats}
                params['a_neg'] = {s: _finite(icpt[s]['a_neg'], f'intercept_by_stat[{s}].a_neg') for s in stats}
            else:
                params['a'] = {s: _finite(icpt[s], f'intercept_by_stat[{s}]') for s in stats}
            if cand == 'C4':
                params['c'] = _finite(rule.get('c'), 'c')
            pc = rule.get('p_star_clip')
            if not (isinstance(pc, (list, tuple)) and len(pc) == 2):
                raise RuleFileError(f'candidate {cand} needs p_star_clip = [lo, hi]')
            clip = (_finite(pc[0], 'p_star_clip[0]'), _finite(pc[1], 'p_star_clip[1]'))
            if not 0.0 < clip[0] < clip[1] < 1.0:
                raise RuleFileError(f'p_star_clip {list(clip)} must satisfy 0 < lo < hi < 1')
        else:
            clip = P_CLIP                                                       # C0 / C1 are not clipped (plan C.4)
    pm = rule.get('pmarket')
    if pm is None:
        pmarket = prices.CURRENT_PMARKET
    else:
        if not isinstance(pm, dict) or {'devig_method', 'consensus_set'} - set(pm):
            raise RuleFileError(f'pmarket must be {{devig_method, consensus_set}}, got {pm!r}')
        if pm['devig_method'] not in devig.METHODS:
            raise RuleFileError(f"unknown pmarket devig_method {pm['devig_method']!r} (known: {', '.join(devig.METHODS)})")
        if pm['consensus_set'] not in prices.CONSENSUS_SETS:
            raise RuleFileError(f"unknown pmarket consensus_set {pm['consensus_set']!r} (known: {', '.join(prices.CONSENSUS_SETS)})")
        pmarket = (str(pm['devig_method']), str(pm['consensus_set']))
    cap = rule.get('cap_per_player_game')
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise RuleFileError(f'cap_per_player_game must be an integer >= 1 (or absent for no cap), got {cap!r}')
    return {'candidate': cand, 'params': params, 'clip': clip, 'pmarket': pmarket, 'cap': cap}


def add_ev_rule(q, rule):
    """add_ev for any rule file: the frozen k-only form calls add_ev unchanged; a rule with a candidate computes p* with p_star and then the same EV code (add_ev_pstar)."""
    spec = rule_spec(rule)
    if spec['candidate'] is None:
        return add_ev(q, rule['k_by_stat'])
    return add_ev_pstar(q, p_star(spec['candidate'], q, spec['params'], spec['clip']))


def pstar_of(rule, stat, p_mkt, p_mod):
    """p* of one market under a rule file (the single-market form of add_ev_rule, for the 'odds now' view)."""
    q = pd.DataFrame({'stat': [stat], 'consensus_fair_over': [float(p_mkt)], 'p_mod': [float(p_mod)], 'minutes_before': [0]})
    spec = rule_spec(rule)
    if spec['candidate'] is None:
        return float(shrunk(p_mkt, p_mod, (rule['k_by_stat'] or {}).get(stat, 0.0)))
    return float(p_star(spec['candidate'], q, spec['params'], spec['clip'])[0])


def select(q, e, time_mode='earliest', min_books=3):
    """
    One bet per (game, player, stat). Among bettable quotes with at least min_books agreeing books, at each decision time take the quote with the highest EV; the bet is
    placed at the first allowed time whose best EV is at least e ('earliest') or only at the given time (240/60/5). Both sides of one market are never both bet.
    """
    q = q[(q['n_at_line'] >= min_books) & (q['best_ev'] >= e)]
    if time_mode != 'earliest':
        q = q[q['minutes_before'] == int(time_mode)]
    q = q.sort_values(['game_id', 'player_id', 'stat', 'minutes_before', 'best_ev'], ascending=[True, True, True, False, False])
    first_t = q.groupby(['game_id', 'player_id', 'stat'])['minutes_before'].transform('max')            # larger minutes_before = earlier in time
    q = q[q['minutes_before'] == first_t]
    return q.drop_duplicates(['game_id', 'player_id', 'stat']).reset_index(drop=True)


def cap_per_player_game(sel, n):
    """At most n bets per player-game (the highest-EV ones): points, pr, pa and pra on one player are largely the same bet."""
    s = sel.sort_values('best_ev', ascending=False)
    s = s.groupby(['game_id', 'player_id'], sort=False).head(n)
    return s.sort_index().reset_index(drop=True)


def cap_causal(new, n, taken=None):
    """
    The causal per-player-game cap for ONE decision slot. new: the markets that qualify at this slot and are not yet bet (one row per market; columns game_id, player_id, stat, best_ev).
    taken: {(game_id, player_id): bets already in the record}. Returns a boolean mask over `new`: a player-game keeps up to n - taken of its markets, highest best_ev first; ties go to the
    stat that sorts first alphabetically (a market is one row, so (best_ev, stat) is a total order). A player-game with n or more bets already gets none.
    """
    keep = np.zeros(len(new), bool)
    if not len(new):
        return keep
    taken = taken or {}
    d = pd.DataFrame({'g': new['game_id'].astype(str).to_numpy(), 'p': new['player_id'].astype(int).to_numpy(), 'stat': new['stat'].astype(str).to_numpy(),
                      'ev': new['best_ev'].to_numpy(float), 'pos': np.arange(len(new))})
    d = d.sort_values(['g', 'p', 'ev', 'stat'], ascending=[True, True, False, True], kind='mergesort')
    rank = d.groupby(['g', 'p'], sort=False).cumcount().to_numpy()
    room = np.array([n - int(taken.get((g, p), 0)) for g, p in zip(d['g'], d['p'])])
    keep[d['pos'].to_numpy()] = rank < room
    return keep


def cap_per_player_game_causal(sel, n):
    """
    cap_per_player_game as the live job can apply it: slots in time order (largest minutes_before first), at each slot cap_causal against the bets kept at earlier slots. sel: the selected bets
    of rules.select (one per market, at the market's earliest qualifying slot). A market never waits for a later, better one, and a player-game with a bet takes no later bet.
    """
    keep = np.zeros(len(sel), bool)
    taken = {}
    mb = sel['minutes_before'].to_numpy()
    for t in sorted(set(mb), reverse=True):
        idx = np.flatnonzero(mb == t)
        sub = sel.iloc[idx]
        m = cap_causal(sub, n, taken)
        keep[idx] = m
        for g, p in zip(sub['game_id'].astype(str).to_numpy()[m], sub['player_id'].astype(int).to_numpy()[m]):
            taken[(g, p)] = taken.get((g, p), 0) + 1
    return sel[keep].reset_index(drop=True)


def cap_per_game(sel, n):
    """At most n bets per game (the highest-EV ones)."""
    s = sel.sort_values('best_ev', ascending=False).groupby('game_id', sort=False).head(n)
    return s.sort_index().reset_index(drop=True)
