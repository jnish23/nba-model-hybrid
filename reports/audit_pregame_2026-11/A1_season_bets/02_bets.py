"""Step 2: reconstruct the recommended bets and settle them.
Observed bets (picks table of data/raw/daily_inference.log, Nov 1+) are taken as printed. Games before the log starts
(Oct 20-31) are RECOMPUTED with team/betting.py from the logged predictions. Writes bets.csv, recon.json, recon_games.csv."""
import json
from scipy.optimize import brentq
from common import *
from team.betting import get_ml_bet, get_spread_bet, get_total_bet, cover_probability, calculate_implied_prob

G = pd.read_csv(OUT / 'games.csv', dtype={'game_id': str}, parse_dates=['tip_et', 'run_et'])
META = json.load(open(ROOT / 'models/saved/calibration_meta_with_odds_cand_20261019.json'))
SP_STD, TOT_STD = META['spread_std'], META['total_std']
THR = 0.05


def inv_implied(p):
    return -100 * p / (1 - p) if p > 0.5 else 100 * (1 - p) / p


def snap_price(r, kind, side_pos):
    """price of the side at the matched book's run-time snapshot (NaN when unavailable / the book's lines do not match the logged ones)"""
    if pd.isna(r.bet_book) or r.bet_book_match < 3:
        return np.nan
    return {('sp', 1): r.bk_home_spread_odds, ('sp', -1): r.bk_away_spread_odds, ('tot', 1): r.bk_over_odds, ('tot', -1): r.bk_under_odds}[(kind, side_pos)]


# ---------------------------------------------------------------- price / sigma identification on the logged days
L = G[G.in_log].copy()
spread_chk = []
for r in L.itertuples():
    pe = r.spread_pred + r.odds_home_spread
    cov = cover_probability(abs(pe), SP_STD, line=r.odds_home_spread)
    be = cov - r.lg_spread_edge                                       # break-even probability the run used for the better side
    s = 1 if pe > 0 else -1
    px = snap_price(r, 'sp', s)
    spread_chk.append(dict(game_id=r.game_id, sp_be_logged=be, sp_price_logged=inv_implied(be),
                           sp_price_snap=px, sp_snap_exact=(abs(implied(px) - be) < 1e-3) if pd.notna(px) else False))
spread_chk = pd.DataFrame(spread_chk)
L = L.merge(spread_chk, on='game_id')
# total sigma per game (one equation, price taken from the snapshot of the matched book when the spread price confirmed that snapshot)
sig = []
for r in L.itertuples():
    pe = r.total_pred - r.odds_total
    s = 1 if pe > 0 else -1
    px = snap_price(r, 'tot', s) if r.sp_snap_exact else np.nan
    if pd.isna(px):
        sig.append((np.nan, np.nan)); continue
    tgt = implied(px) + r.lg_total_edge
    try:
        sg = brentq(lambda z: cover_probability(abs(pe), z, line=r.odds_total) - tgt, 3, 200)
    except Exception:
        sg = np.nan
    sig.append((sg, px))
L['tot_sigma_implied'] = [a for a, b in sig]
L['tot_price_used_for_sigma'] = [b for a, b in sig]
# only games with enough point edge pin sigma down: |edge| >= 0.5 pt (sigma is not identified near 0)
ident = L[(L.total_pred - L.odds_total).abs() >= 1.0]
SIG_TOT = float(ident.tot_sigma_implied.median())
SIG_RANGE = (float(ident.tot_sigma_implied.quantile(.10)), float(ident.tot_sigma_implied.quantile(.90)))
print('implied total sigma (|pe|>=1, price confirmed): n=%d median %.2f p10 %.2f p90 %.2f ; meta total_std %.2f' % (len(ident), SIG_TOT, *SIG_RANGE, TOT_STD))
L[['game_id', 'sp_be_logged', 'sp_price_logged', 'sp_price_snap', 'sp_snap_exact', 'tot_sigma_implied']].to_csv(OUT / 'recon_price_sigma.csv', index=False)

# ---------------------------------------------------------------- independent recompute for EVERY game (decisions and edges)
def recompute(r, price_mode, sig_tot):
    """price_mode 'snap' (matched-book snapshot price when available else -110) or '110'."""
    out = {}
    ml, mle = get_ml_bet({'ml_prob': r.ml_prob, 'odds_home_ml': r.odds_home_ml, 'odds_away_ml': r.odds_away_ml})
    out.update(rc_ml_bet=ml, rc_ml_edge=mle)
    if pd.notna(r.spread_pred) and pd.notna(r.odds_home_spread):
        pe = r.spread_pred + r.odds_home_spread
        s = 1 if pe > 0 else -1
        px = snap_price(r, 'sp', s) if price_mode == 'snap' else np.nan
        sb, se = get_spread_bet(r.spread_pred, r.odds_home_spread, SP_STD, THR, price=-110,
                                home_price=px if s == 1 else None, away_price=px if s == -1 else None)
        out.update(rc_spread_bet=sb, rc_spread_edge=se, rc_spread_price=px if pd.notna(px) else -110)
        pe_t = r.total_pred - r.odds_total
        s = 1 if pe_t > 0 else -1
        px = snap_price(r, 'tot', s) if price_mode == 'snap' else np.nan
        tb, te = get_total_bet(r.total_pred, r.odds_total, sig_tot, THR, price=-110,
                               over_price=px if s == 1 else None, under_price=px if s == -1 else None)
        out.update(rc_total_bet=tb, rc_total_edge=te, rc_total_price=px if pd.notna(px) else -110)
    return out


recs = []
for r in G.itertuples():
    d = dict(game_id=r.game_id)
    for mode, sg, tag in [('snap', SIG_TOT, 'A'), ('110', TOT_STD, 'B')]:
        for k, v in recompute(r, mode, sg).items():
            d[f'{k}_{tag}'] = v
    recs.append(d)
R = G.merge(pd.DataFrame(recs), on='game_id')
R.to_csv(OUT / 'recon_games.csv', index=False)

summ = {'meta_spread_std': SP_STD, 'meta_total_std': TOT_STD, 'implied_total_sigma_median': SIG_TOT, 'implied_total_sigma_p10_p90': SIG_RANGE,
        'n_ident_sigma': int(len(ident)), 'spread_snap_price_exact_share_logged_days': float(L.sp_snap_exact.mean()),
        'spread_snap_price_exact_share_when_match_ge3': float(L[L.bet_book_match >= 3].sp_snap_exact.mean())}
Rl = R[R.in_log]
nn = lambda s: s.fillna('none')
for mk in ['ml', 'spread', 'total']:
    obs = nn(Rl[f'lg_{mk}_bet'])
    for tag in 'AB':
        rc = nn(Rl[f'rc_{mk}_bet_{tag}'])
        summ[f'{mk}_{tag}_logged_bets'] = int((obs != 'none').sum())
        summ[f'{mk}_{tag}_recomputed_bets'] = int((rc != 'none').sum())
        summ[f'{mk}_{tag}_identical_decision'] = int((obs == rc).sum())
        summ[f'{mk}_{tag}_n'] = int(len(Rl))
        summ[f'{mk}_{tag}_edge_maxabs_diff'] = float((Rl[f'lg_{mk}_edge'] - Rl[f'rc_{mk}_edge_{tag}']).abs().max())
        summ[f'{mk}_{tag}_edge_meanabs_diff'] = float((Rl[f'lg_{mk}_edge'] - Rl[f'rc_{mk}_edge_{tag}']).abs().mean())
        summ[f'{mk}_{tag}_edge_corr'] = float(np.corrcoef(Rl[f'lg_{mk}_edge'], Rl[f'rc_{mk}_edge_{tag}'])[0, 1])
json.dump(summ, open(OUT / 'recon.json', 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))

# ---------------------------------------------------------------- build the bet list
rows = []
for r in R.itertuples():
    status = 'pending' if pd.isna(r.home_score) else 'settled'
    src_obs = bool(r.in_log)
    sigma_flag = None
    # ----- ML (exact either way: no sigma, both prices logged)
    ml_side = r.lg_ml_bet if src_obs else r.rc_ml_bet_A
    ml_edge = r.lg_ml_edge if src_obs else r.rc_ml_edge_A
    # ----- spread
    if src_obs:
        sp_side, sp_edge = r.lg_spread_bet, r.lg_spread_edge
    else:
        sp_side, sp_edge = r.rc_spread_bet_A, r.rc_spread_edge_A
    # ----- total
    if src_obs:
        tt_side, tt_edge = r.lg_total_bet, r.lg_total_edge
    else:
        tt_side, tt_edge = r.rc_total_bet_A, r.rc_total_edge_A
    for mk, side, edge in [('ML', ml_side, ml_edge), ('Spread', sp_side, sp_edge), ('Total', tt_side, tt_edge)]:
        if not isinstance(side, str):
            continue
        rows.append(dict(game_id=r.game_id, game_date=r.game_date, matchup=r.matchup, home_team=r.home_team, market=mk, side=side,
                         edge=edge, source='log' if src_obs else 'recomputed', status=status))
B = pd.DataFrame(rows)
B = B.merge(R, on=['game_id', 'game_date', 'home_team'], suffixes=('', '_g'), how='left')

# ---------------------------------------------------------------- price, line, settlement
Lp = L.set_index('game_id')
out = []
for b in B.itertuples():
    d = dict(game_id=b.game_id)
    s = 1 if b.side in ('home', 'over') else -1
    if b.market == 'ML':
        d['line'] = np.nan
        d['price'] = b.odds_home_ml if b.side == 'home' else b.odds_away_ml
        d['price_source'] = 'logged moneyline'
        won = (b.home_score > b.away_score) if b.side == 'home' else (b.away_score > b.home_score)
        push = False
    elif b.market == 'Spread':
        d['line'] = b.odds_home_spread if s == 1 else -b.odds_home_spread           # the line of the side bet
        if b.source == 'log':
            px_log = Lp.loc[b.game_id, 'sp_price_logged']
            d['price'] = round(px_log)                                                     # break-even implied by the printed edge, exact to the cent
            d['price_source'] = 'recovered from logged edge'
        else:
            px = snap_price(b, 'sp', s)
            d['price'], d['price_source'] = (px, 'snapshot, matched book') if pd.notna(px) else (-110.0, '-110 assumed')
        margin = b.home_score - b.away_score
        v = margin + b.odds_home_spread
        push = (v == 0)
        won = (v > 0) if s == 1 else (v < 0)
    else:
        d['line'] = b.odds_total
        if b.source == 'log':
            px = snap_price(b, 'tot', s)
            ok = Lp.loc[b.game_id, 'sp_snap_exact'] if b.game_id in Lp.index else False
            if pd.notna(px) and ok:
                d['price'], d['price_source'] = px, 'snapshot, matched book (confirmed by spread price)'
            elif pd.notna(px):
                d['price'], d['price_source'] = px, 'snapshot, matched book (unconfirmed)'
            else:
                d['price'], d['price_source'] = -110.0, '-110 assumed'
        else:
            px = snap_price(b, 'tot', s)
            d['price'], d['price_source'] = (px, 'snapshot, matched book') if pd.notna(px) else (-110.0, '-110 assumed')
        tot = b.home_score + b.away_score
        v = tot - b.odds_total
        push = (v == 0)
        won = (v > 0) if s == 1 else (v < 0)
    if b.status == 'settled':
        d['result'] = 'P' if push else ('W' if won else 'L')
        d['profit'] = 0.0 if push else (float(payout(d['price'])) if won else -1.0)
    else:
        d['result'], d['profit'] = 'pending', np.nan
    out.append(d)
O = pd.DataFrame(out)
B = pd.concat([B.reset_index(drop=True), O.drop(columns='game_id')], axis=1)
B.to_pickle(OUT / '_bets_stage2.pkl')
print(B.groupby(['market', 'source', 'status']).size())
