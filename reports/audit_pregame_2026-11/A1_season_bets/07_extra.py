"""Step 7: (a) how fragile are the recomputed (Oct 20-31) bets to the sigma / price assumptions, (b) scoring drift against the lines
(over/under context), (c) coverage table. Writes tables_extra.md."""
import json
from common import *
from team.betting import get_spread_bet, get_total_bet, get_ml_bet

R = pd.read_csv(OUT / 'recon_games.csv', dtype={'game_id': str})
B = pd.read_csv(OUT / 'bets.csv', dtype={'game_id': str})
rc = json.load(open(OUT / 'recon.json'))
SP = rc['meta_spread_std']
SIG_LO, SIG_MID, SIG_HI = rc['implied_total_sigma_p10_p90'][0], rc['implied_total_sigma_median'], rc['implied_total_sigma_p10_p90'][1]
md = ["## T1b. Fragility of the recomputed bets (Oct 20-31, 87 completed games; recomputation = betting.py on the logged predictions)\n"]


def snap_price(r, kind, s):
    if pd.isna(r.bet_book) or r.bet_book_match < 3:
        return None
    v = {('sp', 1): r.bk_home_spread_odds, ('sp', -1): r.bk_away_spread_odds, ('tot', 1): r.bk_over_odds, ('tot', -1): r.bk_under_odds}[(kind, s)]
    return None if pd.isna(v) else v


oct_ = R[(~R.in_log) & R.home_score.notna() & R.odds_total.notna()].copy()
rows = []
for r in oct_.itertuples():
    sc = {}
    s = 1 if r.spread_pred + r.odds_home_spread > 0 else -1
    px = snap_price(r, 'sp', s)
    sc['spread'] = {}
    for tag, price in [('snap', px), ('-110', None)]:
        side, e = get_spread_bet(r.spread_pred, r.odds_home_spread, SP, 0.05, home_price=price if s == 1 else None, away_price=price if s == -1 else None)
        sc['spread'][tag] = (side, e)
    s = 1 if r.total_pred - r.odds_total > 0 else -1
    px = snap_price(r, 'tot', s)
    sc['total'] = {}
    for sg_tag, sg in [('lo', SIG_LO), ('mid', SIG_MID), ('hi', SIG_HI), ('meta', 18.58954304144289)]:
        for ptag, price in [('snap', px), ('-110', None)]:
            side, e = get_total_bet(r.total_pred, r.odds_total, sg, 0.05, over_price=price if s == 1 else None, under_price=price if s == -1 else None)
            sc['total'][f'{sg_tag}/{ptag}'] = (side, e)
    rows.append((r.game_id, sc))
for mk in ['spread', 'total']:
    base_tag = 'snap' if mk == 'spread' else 'mid/snap'
    base = {g: sc[mk][base_tag][0] for g, sc in rows}
    anybet = {g: {v[0] for v in sc[mk].values() if v[0]} for g, sc in rows}
    firm = [g for g, sc in rows if all(v[0] == sc[mk][base_tag][0] and v[0] for v in sc[mk].values())]
    base_bets = [g for g, b in base.items() if b]
    flipped_in = [g for g in anybet if anybet[g] and not base[g]]
    flipped_out = [g for g in base_bets if not firm.count(g)]
    md.append(f"- {mk}: {len(base_bets)} recomputed bets under the base assumptions; {len(firm)} are recommended under EVERY scenario tried "
              f"({'snapshot price vs -110' if mk=='spread' else 'sigma p10 17.9 / median 18.9 / p90 20.3 / meta 18.6 x snapshot price vs -110'}); "
              f"{len(base_bets) - len(firm)} drop out under at least one scenario; {len(flipped_in)} games that are not bets in the base case would be bets under another scenario.")
    bm = B[(B.market == mk.capitalize()) & (B.source == 'recomputed') & (B.status == 'settled')]
    f = bm[bm.game_id.isin(firm)]
    nf = bm[~bm.game_id.isin(firm)]
    md.append(f"  - firm bets: {len(f)}, W-L {(f.result=='W').sum()}-{(f.result=='L').sum()}, units {f.profit.sum():+.2f}; borderline bets: {len(nf)}, W-L {(nf.result=='W').sum()}-{(nf.result=='L').sum()}, units {nf.profit.sum():+.2f}")
    md.append(f"  - games that would enter under another scenario: " +
              ', '.join(f"{g}" for g in flipped_in) if flipped_in else "  - no game would enter under another scenario")

# ---- Nov validation: same scenario machinery against the observed picks
nov = R[R.in_log & R.home_score.notna()]
tot = 0
md.append("\nValidation on the 72 completed games whose picks were printed in the log (Nov 1-10): decisions of the base recomputation (snapshot price, sigma = implied median 18.9) versus the printed picks: " +
          f"spread {int((nov.lg_spread_bet.fillna('n') == nov.rc_spread_bet_A.fillna('n')).sum())}/{len(nov)} identical, total {int((nov.lg_total_bet.fillna('n') == nov.rc_total_bet_A.fillna('n')).sum())}/{len(nov)}, "
          f"ML {int((nov.lg_ml_bet.fillna('n') == nov.rc_ml_bet_A.fillna('n')).sum())}/{len(nov)}. With a flat -110 price and the meta sigma (18.59) instead: spread "
          f"{int((nov.lg_spread_bet.fillna('n') == nov.rc_spread_bet_B.fillna('n')).sum())}/{len(nov)}, total {int((nov.lg_total_bet.fillna('n') == nov.rc_total_bet_B.fillna('n')).sum())}/{len(nov)}. "
          f"(sigma 18.9 was fitted on these same days, so the total agreement is not an out-of-sample test.)")

# ---- scoring drift context
G = R[R.home_score.notna()].copy()
G['tot'] = G.home_score + G.away_score
G['margin'] = G.home_score - G.away_score
md.append("\n## Scoring drift against the lines (context for over/under results)\n")
md.append("| quantity | mean | 95% CI (day bootstrap) | games |\n|---|---|---|---|")
for nm, x in [('total points minus model total', G.tot - G.total_pred), ('total points minus run-time total line', G.tot - G.odds_total),
              ('total points minus closing total (median)', (G.tot - G.snap_close_total)), ('home margin minus model spread', G.margin - G.spread_pred),
              ('home margin minus (-run-time home spread)', G.margin + G.odds_home_spread)]:
    ok = x.notna()
    m, lo, hi, _ = day_boot(G.game_date[ok], x[ok])
    md.append(f"| {nm} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {int(ok.sum())} |")
over = (G.tot > G.odds_total)
under = (G.tot < G.odds_total)
md.append(f"\nAll {int(G.odds_total.notna().sum())} completed games with a total line: over {int(over.sum())}, under {int(under.sum())}, push {int((G.tot == G.odds_total).sum())}. "
          f"Blanket over at -110: ROI {100*(over.sum()*0.9091 - under.sum())/(over.sum()+under.sum()):+.1f}%; blanket under: {100*(under.sum()*0.9091 - over.sum())/(over.sum()+under.sum()):+.1f}%.")
(OUT / 'tables_extra.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))

# ---- is the claimed win probability (break-even + edge) contradicted by the results? exact Poisson-binomial lower tail
S = B[B.status == 'settled'].copy()
S['claimed'] = implied(S.price) + S.edge
md2 = ["\n## Claimed win probability versus results (exact Poisson-binomial, one-sided lower tail P(wins <= observed))\n",
       "| market | bets | expected wins if the claimed probabilities are right | observed wins | P(wins <= observed) |", "|---|---|---|---|---|"]
for m in ['ML', 'Spread', 'Total', 'ALL']:
    d = S if m == 'ALL' else S[S.market == m]
    pmf = np.array([1.0])
    for p in d.claimed:
        pmf = np.convolve(pmf, [1 - p, p])
    w = int((d.result == 'W').sum())
    md2.append(f"| {m} | {len(d)} | {d.claimed.sum():.1f} | {w} | {pmf[:w + 1].sum():.3f} |")
(OUT / 'tables_extra.md').write_text(((OUT / 'tables_extra.md').read_text(encoding='utf-8')) + '\n'.join(md2), encoding='utf-8')
print('\n'.join(md2))
