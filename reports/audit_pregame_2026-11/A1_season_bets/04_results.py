"""Step 4: settled results (task 2), CLV (task 3, bets), own-view/move decomposition (task 4), pre-declared splits (task 5).
Writes tables_results.md and results.json. All intervals: day bootstrap (resample game dates), 1000 draws, seed 42."""
import json
from common import *

B = pd.read_csv(OUT / 'bets.csv', dtype={'game_id': str})
S = B[B.status == 'settled'].copy()
S['win'] = (S.result == 'W').astype(float)
S['graded'] = (S.result != 'P').astype(float)          # pushes are refunded: not staked, not counted in ROI or hit rate
S['stake'] = S.graded
S['mk'] = S.market
S['grp_side'] = np.where(S.side.isin(['home', 'over']), 'home/over', 'away/under')
md = []
RES = {}


def line(name, d, extra=''):
    n = len(d)
    w, l, p = (d.result == 'W').sum(), (d.result == 'L').sum(), (d.result == 'P').sum()
    if n == 0:
        return f"| {name} | 0 | - | - | - | - |"
    roi, lo, hi, dr = day_boot(d.game_date, d.profit, d.stake)
    hr, hlo, hhi, _ = day_boot(d.game_date, d.win, d.stake)
    be = float(np.mean(implied(d.price)))
    return (f"| {name} | {n} | {w}-{l}-{p} | {d.profit.sum():+.2f} | {100*roi:+.1f}% [{100*lo:+.1f}, {100*hi:+.1f}] | "
            f"{100*hr:.1f}% [{100*hlo:.1f}, {100*hhi:.1f}] | {100*be:.1f}% |{extra}")


hdr = "| set | bets | W-L-P | units | ROI (day-boot 95% CI) | hit rate (CI) | mean break-even |\n|---|---|---|---|---|---|---|"
md.append("## T2. Settled results (1 unit flat; pushes refunded)\n")
for title, mask in [('All reconstructed bets (primary)', S.index == S.index),
                    ('Observed in the log only (Nov 1-10)', S.source == 'log'),
                    ('Recomputed only (Oct 20-31)', S.source == 'recomputed'),
                    ('Excluding the two special runs (Oct 23 18:59 ET run, Oct 24 Odds-API-open games)', S.special_run.isna())]:
    sub = S[mask]
    md.append(f"\n### {title}\n\n{hdr}")
    md.append(line('ALL markets', sub))
    for m in ['ML', 'Spread', 'Total']:
        md.append(line(m, sub[sub.market == m]))
    for m, sides in [('ML', ['home', 'away']), ('Spread', ['home', 'away']), ('Total', ['over', 'under'])]:
        for sd in sides:
            md.append(line(f'{m} {sd}', sub[(sub.market == m) & (sub.side == sd)]))
    for m in ['ML', 'Spread']:
        for fd in ['fav', 'dog']:
            md.append(line(f'{m} {fd}', sub[(sub.market == m) & (sub.fav_dog == fd)]))
    if title.startswith('All'):
        RES['all'] = {m: dict(n=int((sub.market == m).sum())) for m in ['ML', 'Spread', 'Total']}

# calibration of the claimed edge
md.append("\n### Claimed edge versus realised (all reconstructed bets)\n\n| market | bets | mean claimed win prob | realised hit rate (graded) | claimed edge (mean) | realised hit minus mean break-even |\n|---|---|---|---|---|---|")
for m in ['ML', 'Spread', 'Total']:
    d = S[S.market == m]
    be = implied(d.price)
    claimed = be + d.edge
    hr = d.win.sum() / d.stake.sum()
    md.append(f"| {m} | {len(d)} | {100*claimed.mean():.1f}% | {100*hr:.1f}% | {100*d.edge.mean():+.1f} pts | {100*(hr - (be*d.stake).sum()/d.stake.sum()):+.1f} pts |")

# ------------------------------------------------------------------ CLV
md.append("\n## T3a. Closing-line value of the bets\n\nCLV in points (spread, total) and in de-vigged probability points (ML, fair-to-fair). 'toward' = the line moved in the bet's favour between the bet "
          "and the close (CLV>0), 'against' = CLV<0. Close = last pre-game snapshot (cross-book median); bets whose close is more than 90 minutes before tip are dropped.\n")
md.append("| market | basis | bets with close | mean CLV (CI) | toward / against / none | share toward [CI] | exact sign test p (ties dropped) | sd |\n|---|---|---|---|---|---|---|---|")
for m in ['ML', 'Spread', 'Total']:
    for col, name in [('clv', 'cross-book median close'), ('clv_book', 'same-book close')]:
        d = S[(S.market == m) & S[col].notna()]
        if not len(d):
            continue
        mean, lo, hi, dr = day_boot(d.game_date, d[col])
        sc = 100 if m == 'ML' else 1
        unit = ' pp' if m == 'ML' else ' pts'
        t, a, z = (d[col] > 1e-9).sum(), (d[col] < -1e-9).sum(), (d[col].abs() <= 1e-9).sum()
        ps, plo, phi, _ = day_boot(d.game_date, (d[col] > 1e-9).astype(float))
        from scipy.stats import binomtest
        pv = binomtest(int(t), int(t + a), 0.5).pvalue if (t + a) else np.nan
        md.append(f"| {m} | {name} | {len(d)} | {sc*mean:+.2f}{unit} [{sc*lo:+.2f}, {sc*hi:+.2f}] | {t} / {a} / {z} | {100*ps:.0f}% [{100*plo:.0f}, {100*phi:.0f}] | {pv:.3f} | {sc*d[col].std():.2f} |")
d = S[(S.market == 'ML') & S.clv_prob_price.notna()]
mean, lo, hi, _ = day_boot(d.game_date, d.clv_prob_price)
md.append(f"| ML | close fair prob minus the VIGGED break-even actually taken | {len(d)} | {100*mean:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}] | | | | |")
d = S[(S.market.isin(['Spread', 'Total'])) & S.clv.notna()]
mean, lo, hi, _ = day_boot(d.game_date, d.clv)
md.append(f"| Spread+Total pooled | cross-book median close | {len(d)} | {mean:+.2f} pts [{lo:+.2f}, {hi:+.2f}] | | | | |")
# 11 am, run-time lines cannot be a CLV base: the bet line IS the run-time line. report sample coverage
md.append("\nCLV and results of the sample that has a valid close vs without (selection check):\n")
md.append("| market | bets | with close | ROI with close | ROI without close |\n|---|---|---|---|---|")
for m in ['ML', 'Spread', 'Total']:
    d = S[S.market == m]
    a, b = d[d.clv.notna()], d[d.clv.isna()]
    md.append(f"| {m} | {len(d)} | {len(a)} | {100*a.profit.sum()/max(a.stake.sum(),1):+.1f}% | {100*b.profit.sum()/max(b.stake.sum(),1):+.1f}% |")

# ------------------------------------------------------------------ decomposition
md.append("\n## T4. Own view versus line move (all reconstructed bets with an opening line)\n\nown view = side-signed (model - opening line the model read); move = side-signed "
          "(opening - run-time line), in points (ML: probability points). edge_points = own + move. Opening = the model's own input from prediction_explanations "
          "('explanations') or the snapshot proxy for the 14 Oct 20-21 games. 'fade' = move > 0 (the line moved away from the bet side between the open and the run).\n")
md.append("| market | bets | mean edge | mean own | mean move | share fading the move | share following | share no move | own-dominated (own >= move) |\n|---|---|---|---|---|---|---|---|---|")
for m in ['ML', 'Spread', 'Total', 'ALL']:
    d = S if m == 'ALL' else S[S.market == m]
    d = d[d.own_view.notna()]
    sc = 100 if m == 'ML' else 1
    if m == 'ALL':
        md.append(f"| ALL | {len(d)} | - | - | - | {100*(d.fade>0).mean():.0f}% | {100*(d.fade<0).mean():.0f}% | {100*(d.fade==0).mean():.0f}% | {100*d.own_dom.mean():.0f}% |")
    else:
        md.append(f"| {m} | {len(d)} | {sc*d.edge_points.mean():.2f} | {sc*d.own_view.mean():.2f} | {sc*d.move.mean():.2f} | {100*(d.fade>0).mean():.0f}% | {100*(d.fade<0).mean():.0f}% | {100*(d.fade==0).mean():.0f}% | {100*d.own_dom.mean():.0f}% |")
shares = []
for m, d_ in [('ML', S[(S.market == 'ML') & S.fade.notna()]), ('Spread', S[(S.market == 'Spread') & S.fade.notna()]), ('Total', S[(S.market == 'Total') & S.fade.notna()]), ('ALL', S[S.fade.notna()])]:
    p_, lo_, hi_, _ = day_boot(d_.game_date, (d_.fade > 0).astype(float))
    shares.append(f"{m} {100*p_:.0f}% [{100*lo_:.0f}, {100*hi_:.0f}] (n={len(d_)})")
md.append("\nShare of bets that fade the open-to-run move, day-bootstrap 95% CI: " + '; '.join(shares))
md.append("\nResults split by which part dominates:\n\n" + hdr)
for lab, mask in [('own-view-dominated', S.own_dom == 1), ('move-dominated', S.own_dom == 0), ('fades the move (move>0)', S.fade > 0),
                  ('follows or no move (move<=0)', S.fade <= 0)]:
    md.append(line(lab, S[mask]))
for m in ['ML', 'Spread', 'Total']:
    for lab, mask in [('own-dominated', S.own_dom == 1), ('move-dominated', S.own_dom == 0)]:
        md.append(line(f'{m} {lab}', S[(S.market == m) & mask]))
# proxy quality
md.append("\nQuality of the snapshot-opening proxy against the model's own opening input (games with both):\n")
md.append("| market | n | mean (proxy - true) | MAE | share exact | corr(own proxy, own true) | same own/move-dominated class |\n|---|---|---|---|---|---|---|")
G = pd.read_csv(OUT / 'games.csv', dtype={'game_id': str})
g = G[G.ex_open_home_spread.notna() & G.snap_open_spread.notna() & G.home_score.notna()]
for nm, a, b in [('spread (pts)', g.snap_open_spread, g.ex_open_home_spread), ('total (pts)', g.snap_open_total, g.ex_open_total)]:
    dd = a - b
    md.append(f"| {nm} | {len(g)} | {dd.mean():+.2f} | {dd.abs().mean():.2f} | {100*(dd==0).mean():.0f}% | | |")
for m in ['Spread', 'Total']:
    d = S[(S.market == m) & S.own_view.notna() & S.own_view_snapproxy.notna() & (S.open_source == 'explanations')]
    same = ((d.own_view >= d.move) == (d.own_view_snapproxy >= d.move_snapproxy)).mean()
    md.append(f"| {m} bets (decomposition) | {len(d)} | {(d.own_view_snapproxy-d.own_view).mean():+.2f} | {(d.own_view_snapproxy-d.own_view).abs().mean():.2f} | | {np.corrcoef(d.own_view, d.own_view_snapproxy)[0,1]:.3f} | {100*same:.0f}% |")

# ------------------------------------------------------------------ splits (declared in RESULT.md BEFORE looking at outcomes; same list here)
S['edge_hi'] = np.where(S.edge >= 0.07, 'edge>=7%', 'edge 5-7%')
S['move_dom'] = np.where(S.own_dom.isna(), None, np.where(S.own_dom == 1, 'own-dominated', 'move-dominated'))
S['inj'] = np.where(S.new_inj.isna(), None, np.where(S.new_inj == 1, 'new injury absence', 'no new absence'))
S['b2b'] = np.where(S.b2b_home.isna(), None, np.where((S.b2b_home + S.b2b_away) > 0, 'either team on a back-to-back', 'no back-to-back'))
S['period'] = np.where(S.first_two_weeks == 1, 'first two weeks (<= Nov 2)', 'later (Nov 4+)')
S['fd2'] = S.fav_dog.where(S.fav_dog.isin(['fav', 'dog']))
S['sd2'] = S.grp_side
SPLITS = [('1 market', 'mk', None), ('2 side', 'sd2', ('home/over', 'away/under')), ('2b favourite vs underdog (ML+spread)', 'fd2', ('fav', 'dog')),
          ('3 edge size', 'edge_hi', ('edge>=7%', 'edge 5-7%')), ('4 own vs move', 'move_dom', ('own-dominated', 'move-dominated')),
          ('5 new injury absence', 'inj', ('new injury absence', 'no new absence')),
          ('6 back-to-back', 'b2b', ('either team on a back-to-back', 'no back-to-back')), ('7 first two weeks vs later', 'period', ('first two weeks (<= Nov 2)', 'later (Nov 4+)'))]
md.append("\n## T5. Pre-declared splits (all reconstructed settled bets, 3 markets pooled in units)\n\nCLV in the table is pooled over spread and total bets only (points). "
          "'diff' = first level minus second level, day-bootstrap.\n")
md.append("| split | level | bets | W-L-P | hit rate | ROI [95% CI] | CLV pts spread+total (n) |\n|---|---|---|---|---|---|---|")
tests = []
for nm, col, pair in SPLITS:
    levels = sorted(S[col].dropna().unique()) if pair is None else list(pair)
    for lv in levels:
        d = S[S[col] == lv]
        roi, lo, hi, _ = day_boot(d.game_date, d.profit, d.stake) if len(d) else (np.nan,) * 4
        hr = d.win.sum() / d.stake.sum() if d.stake.sum() else np.nan
        dc = d[d.market.isin(['Spread', 'Total']) & d.clv.notna()]
        cm = f"{dc.clv.mean():+.2f} ({len(dc)})" if len(dc) else "-"
        w, l, p = (d.result == 'W').sum(), (d.result == 'L').sum(), (d.result == 'P').sum()
        md.append(f"| {nm} | {lv} | {len(d)} | {w}-{l}-{p} | {100*hr:.1f}% | {100*roi:+.1f}% [{100*lo:+.1f}, {100*hi:+.1f}] | {cm} |")
    if pair is None:
        continue
    a, b = S[S[col] == pair[0]], S[S[col] == pair[1]]
    if len(a) < 3 or len(b) < 3:
        continue
    keyd = pd.concat([a, b])
    in_a = np.r_[np.ones(len(a)), np.zeros(len(b))]
    jb = joint_boot(keyd.game_date, {'ra': (keyd.profit * in_a, keyd.stake * in_a), 'rb': (keyd.profit * (1 - in_a), keyd.stake * (1 - in_a))})
    diff = jb['ra'][1] - jb['rb'][1]
    pt = jb['ra'][0] - jb['rb'][0]
    lo, hi = ci(diff)
    tests.append(dict(split=nm, outcome='ROI', n_a=len(a), n_b=len(b), diff=pt, lo=lo, hi=hi, p=boot_p(diff)))
    ka = keyd[keyd.market.isin(['Spread', 'Total']) & keyd.clv.notna()]
    ina = np.where(ka.index.isin(a.index), 1.0, 0.0)
    if ina.sum() >= 3 and (1 - ina).sum() >= 3:
        jb = joint_boot(ka.game_date, {'a': (ka.clv * ina, ina), 'b': (ka.clv * (1 - ina), 1 - ina)})
        diff = jb['a'][1] - jb['b'][1]
        lo, hi = ci(diff)
        tests.append(dict(split=nm, outcome='CLV pts', n_a=int(ina.sum()), n_b=int((1 - ina).sum()), diff=jb['a'][0] - jb['b'][0], lo=lo, hi=hi, p=boot_p(diff)))
T = pd.DataFrame(tests)
# markets: ROI different from 0 per market (3 tests) join the family
mk_tests = []
for m in ['ML', 'Spread', 'Total']:
    d = S[S.market == m]
    roi, lo, hi, dr = day_boot(d.game_date, d.profit, d.stake)
    mk_tests.append(dict(split='1 market', outcome=f'ROI {m} vs 0', n_a=len(d), n_b=np.nan, diff=roi, lo=lo, hi=hi, p=boot_p(dr)))
    dc = d[d.clv.notna()]
    mean, lo, hi, dr = day_boot(dc.game_date, dc.clv)
    mk_tests.append(dict(split='1 market', outcome=f'CLV {m} vs 0', n_a=len(dc), n_b=np.nan, diff=mean, lo=lo, hi=hi, p=boot_p(dr)))
T = pd.concat([pd.DataFrame(mk_tests), T], ignore_index=True)
T['p_holm'] = holm(T.p.values)
T['p_bonf'] = np.minimum(1, T.p * T.p.notna().sum())
md.append(f"\n### Family of {T.p.notna().sum()} tests, Holm-adjusted\n\n| split | outcome | n a | n b | diff or level (ROI as fraction of stake; CLV pts / ML prob) | 95% CI | raw p | Holm p |\n|---|---|---|---|---|---|---|---|")
for r in T.itertuples():
    md.append(f"| {r.split} | {r.outcome} | {r.n_a} | {'' if pd.isna(r.n_b) else int(r.n_b)} | {r.diff:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p:.3f} | {r.p_holm:.3f} |")
T.to_csv(OUT / 'split_tests.csv', index=False)
md.append(f"\nSurvivors at Holm 0.05: {int((T.p_holm < 0.05).sum())} of {T.p.notna().sum()}.")

# ------------------------------------------------------------------ power
md.append("\n## T5b. Power\n\nRequired number of independent bets for a two-sided 5% test at 80% power: n = ((1.96+0.84) * sd / delta)^2. ROI sd per unit staked is taken from the settled bets "
          "(about 1.0 at near-even prices); CLV sd from the observed per-bet CLV.\n")
sd_roi = S.profit[S.graded == 1].std()
n_roi = ((1.96 + 0.8416) * sd_roi / 0.02) ** 2
md.append(f"- ROI: per-bet sd {sd_roi:.3f} units; a 2-point ROI edge (delta 0.02) needs about {n_roi:,.0f} bets. We have {int(S.graded.sum())} graded bets; the 95% CI half-width on ROI is about {100*1.96*sd_roi/np.sqrt(S.graded.sum()):.0f} points.")
for m, nm in [('Spread', 'spread'), ('Total', 'total')]:
    d = S[(S.market == m) & S.clv.notna()]
    sd = d.clv.std()
    md.append(f"- CLV {nm}: sd {sd:.2f} pts over {len(d)} bets; a 0.3-point CLV edge needs about {((1.96+0.8416)*sd/0.3)**2:,.0f} bets (we have {len(d)}).")
d = S[S.market.isin(['Spread', 'Total']) & S.clv.notna()]
sd = d.clv.std()
md.append(f"- CLV spread+total pooled: sd {sd:.2f}; 0.3 points needs about {((1.96+0.8416)*sd/0.3)**2:,.0f} bets (we have {len(d)}).")
RES['sd_roi'] = float(sd_roi)
(OUT / 'tables_results.md').write_text('\n'.join(md), encoding='utf-8')
json.dump(RES, open(OUT / 'results.json', 'w'), indent=1, default=float)
print('\n'.join(md))
