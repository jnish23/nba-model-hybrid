"""Step 5: model versus open / 11 am / run-time / close on ALL completed games (task 3), and the check of lead finding P1 (task 4).
Lines come from odds_snapshots (cross-book medians). Intervals: paired day bootstrap of the mean difference (1000 draws, seed 42)."""
from common import *
from team.betting import points_threshold

G = pd.read_csv(OUT / 'games.csv', dtype={'game_id': str})
G = G[G.home_score.notna()].copy()
G['margin'] = G.home_score - G.away_score
G['tot'] = G.home_score + G.away_score
G['home_win'] = (G.margin > 0).astype(float)
G['close_ok'] = G.snap_close_mins_to_tip <= 90
md = []


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def table(title, need, fn_model, fn_line, name, fmt=3, scale=1.0):
    md.append(f"\n### {title}\n")
    s = G.dropna(subset=need).copy()
    md.append(f"games with all lines: {len(s)} (days {s.game_date.nunique()})\n")
    md.append(f"| comparison | model {name} | line {name} | diff (model - line) [95% CI] | bias of model (outcome - pred) | bias of line |\n|---|---|---|---|---|---|")
    return s


# --------------------------------------------------------------------------------- ML log loss
cols = {'open': 'snap_open_p_home', '11 am': 'snap_11_p_home', 'run-time (logged DK price, de-vigged)': None, 'close': 'snap_close_p_home'}
G['logged_p_home'] = devig2(G.odds_home_ml, G.odds_away_ml)
cols['run-time (logged DK price, de-vigged)'] = 'logged_p_home'
need = ['snap_open_p_home', 'snap_11_p_home', 'snap_close_p_home', 'logged_p_home']
s = G.dropna(subset=need)
s = s[s.close_ok]
md.append(f"## T3b. Model versus the lines, all completed games\n\n### Moneyline log loss (lines de-vigged per book, cross-book median)\n\nGames with open, 11 am, run-time and a close within 90 minutes of tip: {len(s)} ({s.game_date.nunique()} days).\n")
md.append("| benchmark | model log loss | line log loss | diff model - line [95% CI] | mean(home win - p) model | line |\n|---|---|---|---|---|---|")
ml_rows = []
for nm, col in cols.items():
    lm, lb = ll(s.ml_prob.values, s.home_win.values), ll(s[col].values, s.home_win.values)
    jb = joint_boot(s.game_date, {'d': (lm - lb, None), 'm': (lm, None), 'b': (lb, None)})
    lo, hi = ci(jb['d'][1])
    md.append(f"| {nm} | {jb['m'][0]:.4f} | {jb['b'][0]:.4f} | {jb['d'][0]:+.4f} [{lo:+.4f}, {hi:+.4f}] | {(s.home_win-s.ml_prob).mean():+.3f} | {(s.home_win-s[col]).mean():+.3f} |")

# --------------------------------------------------------------------------------- spread / total MAE
for mk, pred, line_cols, outcome, sign in [('Spread (home margin; line-implied margin = -home spread)', 'spread_pred',
                                             {'open': 'snap_open_spread', '11 am': 'snap_11_spread', 'run-time (logged)': 'odds_home_spread', 'close': 'snap_close_spread'}, 'margin', -1),
                                            ('Total', 'total_pred',
                                             {'open': 'snap_open_total', '11 am': 'snap_11_total', 'run-time (logged)': 'odds_total', 'close': 'snap_close_total'}, 'tot', 1)]:
    need = [pred] + list(line_cols.values())
    s = G.dropna(subset=need)
    s = s[s.close_ok]
    md.append(f"\n### {mk}: MAE\n\nGames with all lines and a close within 90 minutes of tip: {len(s)} ({s.game_date.nunique()} days).\n")
    md.append("| benchmark | model MAE | line MAE | diff model - line [95% CI] | model bias (outcome - pred) | line bias |\n|---|---|---|---|---|---|")
    for nm, col in line_cols.items():
        lp = sign * s[col]
        em, el = (s[outcome] - s[pred]).abs(), (s[outcome] - lp).abs()
        jb = joint_boot(s.game_date, {'d': (em - el, None), 'm': (em, None), 'b': (el, None)})
        lo, hi = ci(jb['d'][1])
        md.append(f"| {nm} | {jb['m'][0]:.3f} | {jb['b'][0]:.3f} | {jb['d'][0]:+.3f} [{lo:+.3f}, {hi:+.3f}] | {(s[outcome]-s[pred]).mean():+.2f} | {(s[outcome]-lp).mean():+.2f} |")

# --------------------------------------------------------------------------------- P1
md.append("\n## P1 check: is the edge versus the 11 am line mostly the open-to-11 am line move?\n")
thr_sp, thr_tt = points_threshold(14.286969463981409), points_threshold(18.58954304144289)
md.append(f"5%-rule thresholds at -110: spread {thr_sp:.2f} pts (std 14.29), total {thr_tt:.2f} pts (std 18.59). 'larger edge' = |model - 11 am line| >= that threshold.\n")
md.append("| basis | market | games | corr(edge vs 11 am, open - 11 am) | sd(own = model - open) | sd(move) | corr(own, move) | larger-edge games | of which fade the move | follow | no move |\n|---|---|---|---|---|---|---|---|---|---|---|")
for basis, o_sp, c_sp, o_tt, c_tt, mask in [
        ('snapshots: open, 11 am (all games with the three lines)', 'snap_open_spread', 'snap_11_spread', 'snap_open_total', 'snap_11_total', G.snap_11_spread.notna()),
        ("model's own opening input (explanations) and the logged run-time line", 'ex_open_home_spread', 'odds_home_spread', 'ex_open_total', 'odds_total', G.ex_open_home_spread.notna())]:
    for mk, o, c_, pred, sgn, thr in [('spread', o_sp, c_sp, 'spread_pred', 1, thr_sp), ('total', o_tt, c_tt, 'total_pred', -1, thr_tt)]:
        s = G[mask].dropna(subset=[o, c_, pred])
        if mk == 'spread':
            edge = s[pred] + s[c_]                  # home-side point edge vs the later line
            own = s[pred] + s[o]
            move = s[c_] - s[o]                     # side-signed for home: later minus open
        else:
            edge = s[pred] - s[c_]
            own = s[pred] - s[o]
            move = s[o] - s[c_]
        big = edge.abs() >= thr
        side = np.sign(edge)
        sm = side * move                            # >0: bet side's line moved away from it since the open (fade)
        md.append(f"| {basis} | {mk} | {len(s)} | {np.corrcoef(edge, move)[0,1]:.2f} | {own.std():.2f} | {move.std():.2f} | {np.corrcoef(own, move)[0,1]:.2f} | {int(big.sum())} | "
                  f"{(sm[big] > 0).sum()} ({100*(sm[big]>0).mean():.0f}%) | {(sm[big] < 0).sum()} | {(sm[big] == 0).sum()} |")
(OUT / 'tables_model_vs_lines.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))
