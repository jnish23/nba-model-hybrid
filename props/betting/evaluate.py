"""
Assembling and scoring simulated bets (Phase 5). Intervals are game-clustered bootstraps (props/models/evaluate.cluster_bootstrap).
"""
import numpy as np
import pandas as pd

from props.betting import settle
from props.models import evaluate as ev, outcome as oc, recalibrate
from props.models.pipeline import observed

LABEL_COLS = ['y_points', 'y_rebounds', 'y_assists', 'y_threes', 'y_pra']


def y_of(df, stat):
    """Realized value of any of the eight stats from a frame that has the label columns."""
    return observed(df, stat)


def r2b_ratio_by_date(full_rows, full_pmfs, carry=None):
    """Per stat: {(date, group) -> R2b ratio}, from the full-season forecasts (completed games of the same group before the date; regular-season rows also see the previous
    regular season's last game dates via `carry` = (rows, pmfs) of the previous fold)."""
    out = {}
    groups = recalibrate.game_group(full_rows['game_id'])
    d = pd.to_datetime(full_rows['game_date']).dt.normalize().to_numpy()
    for stat in full_pmfs:
        c = None
        if carry is not None:
            c = (carry[0]['game_date'], observed(carry[0], stat), oc.pmf_mean(carry[1][stat]), recalibrate.game_group(carry[0]['game_id']))
        r = recalibrate.rolling_ratio_gamedays(full_rows['game_date'], observed(full_rows, stat), oc.pmf_mean(full_pmfs[stat]), c, groups=groups)
        out[stat] = pd.Series(r, index=pd.MultiIndex.from_arrays([d, groups])).groupby(level=[0, 1]).first()
    return out


def apply_r2b(idx, pmfs, ratios):
    """Scale the high-draw pmfs by the ratio of their date and game group (1.0 where there is none)."""
    mi = pd.MultiIndex.from_arrays([pd.to_datetime(idx['game_date']).dt.normalize().to_numpy(), recalibrate.game_group(idx['game_id'])])
    new = {}
    for stat, pm in pmfs.items():
        r = ratios[stat].reindex(mi).fillna(1.0).to_numpy(float)
        new[stat] = recalibrate.scale_pmf(pm, r)
    return new


def quote_table(bets, idx, pmfs):
    """Join each bettable quote to the model pmf of its player-game-time and compute p_mod (P(over | no push)) and the push probability at the quote's own line."""
    key = idx[['game_id', 'player_id', 'minutes_before']].copy()
    key['i'] = np.arange(len(idx))
    q = bets.merge(key, on=['game_id', 'player_id', 'minutes_before'], how='inner', validate='many_to_one')
    q['p_mod'] = np.nan
    q['push'] = np.nan
    for stat, g in q.groupby('stat'):
        i = g['i'].to_numpy()
        line = g['line'].to_numpy(float)
        q.loc[g.index, 'p_mod'] = oc.prob_over_no_push(pmfs[stat][i], line)
        q.loc[g.index, 'push'] = oc.prob_push(pmfs[stat][i], line)
    q['expected_minutes'] = idx['minutes_mean'].to_numpy()[q['i'].to_numpy()]
    return q.rename(columns={'fair_over': 'book_fair_over'})


def attach_outcomes(bets, labels):
    """labels: frame with game_id, player_id, lbl_played, lbl_minutes and the y_* columns (one row per player-game)."""
    lab = labels[['game_id', 'player_id', 'lbl_played', 'lbl_minutes'] + LABEL_COLS].drop_duplicates(['game_id', 'player_id'])
    b = bets.merge(lab, on=['game_id', 'player_id'], how='left', validate='many_to_one')
    y = np.full(len(b), np.nan)
    for stat, g in b.groupby('stat'):
        y[g.index.to_numpy()] = y_of(g, stat)
    b['y'] = y
    b['played'] = (b['lbl_played'] == 1) & (b['lbl_minutes'] > 0)
    return b


def settle_all(b):
    """Adds profit_<treatment> and status_<treatment> for each bounding treatment (b needs best_side, line, dec_price, y, played, lbl_minutes, expected_minutes, book)."""
    b = b.copy()
    for t in settle.TREATMENTS:
        b[f'profit_{t}'], b[f'status_{t}'] = settle.settle(b['best_side'].to_numpy(), b['line'], b['dec_price'], b['y'], b['played'], b['lbl_minutes'], b['expected_minutes'],
                                                           b['book'].to_numpy(), t)
    return b


def roi_table(b, label, n_boot=ev.N_BOOT, alpha_adj=None):
    """ROI per unit staked (voids and pushes are not stakes) for each settlement treatment, with game-clustered intervals."""
    out = []
    for t in settle.TREATMENTS:
        prof, st = b[f'profit_{t}'].to_numpy(float), b[f'status_{t}'].to_numpy()
        stake = np.isin(st, ['win', 'loss']).astype(float)
        if stake.sum() < 50:
            out.append({'rule': label, 'treatment': t, 'bets': int(stake.sum())})
            continue
        p, lo, hi = ev.cluster_bootstrap(prof, b['game_id'], den=stake, n_boot=n_boot)
        row = {'rule': label, 'treatment': t, 'bets': int(stake.sum()), 'wins': int((st == 'win').sum()), 'win_rate': float((st == 'win').sum() / stake.sum()),
               'profit_units': float(prof.sum()), 'roi': p, 'lo': lo, 'hi': hi, 'mean_odds': float(b.loc[stake > 0, 'dec_price'].mean())}
        if alpha_adj:                                                       # family-wise interval (Bonferroni over the tried rules)
            _, row['lo_adj'], row['hi_adj'] = ev.cluster_bootstrap(prof, b['game_id'], den=stake, n_boot=n_boot, alpha=alpha_adj)
        out.append(row)
    return pd.DataFrame(out)


def clv_table(selected, cons, n_boot=ev.N_BOOT):
    """
    Closing-line value to T-5 for bets placed at T-240 or T-60 (a T-5 bet has no later reference). The market-move measure: the T-5 consensus fair probability of the bet side minus the
    consensus fair probability of that side at the bet time, defined only where the consensus line did not move (otherwise the two probabilities are at different lines);
    the share of bets whose line moved, and how the line moved relative to the bet side, are reported separately.
    """
    close = cons[cons['minutes_before'] == 5][['game_id', 'player_id', 'stat', 'consensus_line', 'consensus_fair_over']].rename(
        columns={'consensus_line': 'close_line', 'consensus_fair_over': 'close_fair_over'})
    b = selected[selected['minutes_before'] > 5].merge(close, on=['game_id', 'player_id', 'stat'], how='left')
    has = b['close_line'].notna()
    same = has & (b['close_line'] == b['line'])
    side_sign = np.where(b['best_side'] == 'over', 1.0, -1.0)
    fair_at_bet = np.where(b['best_side'] == 'over', b['consensus_fair_over'], 1.0 - b['consensus_fair_over'])
    fair_close = np.where(b['best_side'] == 'over', b['close_fair_over'], 1.0 - b['close_fair_over'])
    b['clv_prob'] = np.where(same, fair_close - fair_at_bet, np.nan)
    b['line_move_in_favour'] = np.where(has & ~same, side_sign * (b['close_line'] - b['line']), np.nan)        # points the line moved in the bettor's favour: a higher closing line favours an over bet (the over at the lower bet line was the better deal), a lower one an under bet
    rows = []
    if same.sum() > 50:
        p, lo, hi = ev.cluster_bootstrap(b.loc[same, 'clv_prob'].to_numpy(), b.loc[same, 'game_id'], n_boot=n_boot)
        rows.append({'measure': 'consensus fair probability change to T-5, line unchanged', 'n': int(same.sum()), 'mean': p, 'lo': lo, 'hi': hi})
    moved = has & ~same
    if moved.sum() > 50:
        p, lo, hi = ev.cluster_bootstrap(b.loc[moved, 'line_move_in_favour'].to_numpy(), b.loc[moved, 'game_id'], n_boot=n_boot)
        rows.append({'measure': 'line moved by T-5: points moved in the bettor\'s favour', 'n': int(moved.sum()), 'mean': p, 'lo': lo, 'hi': hi})
    return pd.DataFrame(rows), {'bets_before_T5': int(len(b)), 'with_T5_quote': int(has.sum()), 'line_unchanged': int(same.sum()), 'line_moved': int(moved.sum())}


def placebo_quotes(q, seed):
    """Placebo model: each market's (p_mod, push) is replaced by those of a random OTHER market of the same stat (a permutation within stat over markets). Same k, same selection rule,
    same prices; the probabilities carry no information about this market. Its CLV and ROI are the zero point for the real model's."""
    mk = ['game_id', 'player_id', 'stat', 'minutes_before']
    m = q.groupby(mk, sort=False)[['p_mod', 'push']].first().reset_index()
    rng = np.random.default_rng(seed)
    for stat, g in m.groupby('stat'):
        pos = g.index.to_numpy()
        m.loc[pos, ['p_mod', 'push']] = m.loc[rng.permutation(pos), ['p_mod', 'push']].to_numpy()
    return q.drop(columns=['p_mod', 'push']).merge(m[mk + ['p_mod', 'push']], on=mk, how='left', validate='many_to_one')


def exposure(sel):
    """How many of the bets are the same player-game: distinct player-games and the distribution of bets per player-game."""
    per = sel.groupby(['game_id', 'player_id']).size()
    return {'bets': int(len(sel)), 'distinct_player_games': int(len(per)), 'mean_bets_per_player_game': float(per.mean()), 'max_bets_per_player_game': int(per.max())}
