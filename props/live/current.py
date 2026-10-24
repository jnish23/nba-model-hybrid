"""
"Odds now" for the pending bets: the latest snapshot's consensus and book prices for each bet's market, and what the recorded bet is worth at those prices. The model's distribution stored
with the bet (pmf_json) is re-read at the current line, so a moved line is handled; the market probability is the current consensus. Information only: the recorded bet never changes.
"""
import json

import numpy as np
import pandas as pd

from props.betting import rules
from props.live import events, quotes
from props.models import outcome as oc

COLS = ['bet_id', 'as_of_utc', 'snapshot_minutes_now', 'line_now', 'line_moved', 'n_at_line_now', 'p_market_now', 'dec_median_now', 'dec_best_now', 'best_book_now', 'ev_median_now',
        'ev_best_now', 'ev_flat_now', 'still_qualifies']


def bet_now(bet, mkt, rule, k):
    """
    bet: a recorded bet (dict-like with side, line, pmf). mkt: the current bettable quotes of its market (rows with line, book, over_dec, under_dec,
    consensus_fair_over, n_at_line). Returns the dict of COLS (without bet_id / timestamps) or None when the market has no quote now.
    """
    if mkt is None or mkt.empty:
        return None
    line = float(mkt['line'].iloc[0])        # every row of the bettable-quote frame sits on the consensus line
    p_mkt = float(mkt['consensus_fair_over'].iloc[0])
    pmf = np.asarray(bet['pmf'], dtype=float)[None, :]
    p_mod = float(oc.prob_over_no_push(pmf, np.array([line]))[0])
    push = float(oc.prob_push(pmf, np.array([line]))[0])
    p_star = float(rules.shrunk(p_mkt, p_mod, k))
    over = bet['side'] == 'over'
    dec = (mkt['over_dec'] if over else mkt['under_dec']).to_numpy(float)
    med = float(np.median(dec))
    i = int(np.argmax(dec))
    ev = {}
    for name, px in (('median', med), ('best', float(dec[i]))):
        eo, eu = rules.ev_over_under(p_star, push, px if over else 1.0, 1.0 if over else px)      # only the bet side's price enters
        ev[name] = float(eo if over else eu)
    flat = 1.0 + 100.0 / 110.0
    eo, eu = rules.ev_over_under(p_star, push, flat, flat)
    n_at = int(mkt['n_at_line'].iloc[0])
    return {'line_now': line, 'line_moved': int(line != float(bet['line'])), 'n_at_line_now': n_at, 'p_market_now': p_mkt, 'dec_median_now': med, 'dec_best_now': float(dec[i]),
            'best_book_now': str(mkt['book'].iloc[i]), 'ev_median_now': ev['median'], 'ev_best_now': ev['best'], 'ev_flat_now': float(eo if over else eu),
            'still_qualifies': int(ev['best'] >= rule['primary']['e'] and n_at >= rule['primary']['min_books'])}


def current_odds(pending, schedule, odds_db, now, rule):
    """
    pending: recorded bets with a game that has not started and no result: columns bet_id, game_id, player_id, stat, side, line, p_over_star, push, pmf_json.
    schedule: season schedule (for the event -> game match). Returns a frame with COLS, one row per bet that has a quote in the latest snapshot of its event.
    """
    from props import schedule as S
    if pending.empty:
        return pd.DataFrame(columns=COLS)
    ev = events.events_in_window(odds_db, now, hours_back=0.0, hours_ahead=14.0)
    if ev.empty:
        return pd.DataFrame(columns=COLS)
    matched = S.match_games(ev, schedule, S.team_id_map('data/nba_stats.db'))
    rows = []
    for g in matched.itertuples(index=False):
        if not isinstance(g.game_id, str):
            continue
        mine = pending[pending['game_id'] == g.game_id]
        if mine.empty:
            continue
        snaps = quotes.event_snapshots(odds_db, g.event_id_vendor, since_utc=events.iso(now - pd.Timedelta(hours=8)))
        sm = pd.Series({ts: quotes.minutes_before_tip(ts, g.tip_ref_utc) for ts in snaps['snapshot_ts_utc']}, dtype=float)
        sm = sm[sm >= quotes.MIN_MINUTES_TO_TIP]
        if sm.empty:
            continue
        ts = sm.idxmin()                                                         # the most recent snapshot that is still before the tip
        cons, bets, _ = quotes.slot_quotes(odds_db, g.event_id_vendor, g.game_id, ts, 0)
        if bets is None or bets.empty:
            continue
        for b in mine.itertuples(index=False):
            mkt = bets[(bets['player_id'] == b.player_id) & (bets['stat'] == b.stat)]
            r = bet_now({'side': b.side, 'line': b.line, 'pmf': json.loads(b.pmf_json)}, mkt, rule, rule['k_by_stat'][b.stat])
            if r:
                rows.append({'bet_id': b.bet_id, 'as_of_utc': ts, 'snapshot_minutes_now': float(sm[ts]), **r})
    return pd.DataFrame(rows, columns=COLS)
