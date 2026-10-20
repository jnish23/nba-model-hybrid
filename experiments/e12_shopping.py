"""E12 - line shopping and consensus (see PREREGISTRATION.md). Bets = the baseline 5%-edge selections (made against the line
the model saw); only where they are settled changes. `odds_snapshots` has no rows in this database, so the multi-book source is
the per-book columns of nba_odds_opening (cleaned with the odds_clean.py outlier filters). python -m experiments.e12_shopping"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import metrics as M                                                   # noqa: E402
from experiments.lib import OUT, load_games, load_run                                   # noqa: E402
from models import odds_clean as oc                                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAME = 'E12_line_shopping'
KEYS = (-7.0, -3.0, 3.0, 7.0)


def book_matrices(df):
    """Cleaned per-book quotes (rows = opening-table rows, columns = books), using the odds_clean outlier rules."""
    def cols(prefix):
        return df[[f'{prefix}_{b}' for b in oc.BOOKS if f'{prefix}_{b}' in df.columns]].apply(pd.to_numeric, errors='coerce')
    hs, as_ = cols('home_spread'), cols('away_spread')
    as_.columns = hs.columns
    hs = hs.where((hs.abs() <= oc.MAX_ABS_SPREAD) & ((hs + as_).abs() < 0.01))
    hs = oc._filter_to_consensus(hs, oc.SPREAD_TOL)
    a_line = -hs
    tot = cols('total')
    tot = tot.where((tot >= oc.TOTAL_RANGE[0]) & (tot <= oc.TOTAL_RANGE[1]))
    tot = oc._filter_to_consensus(tot, oc.TOTAL_TOL)

    def price(prefix, like):
        p = cols(prefix)
        p.columns = like.columns
        p = p.where((p.abs() >= 100) & (p.abs() <= 1000))
        return p.where(like.notna())
    hs_odds, as_odds = price('home_spread_odds', hs), price('away_spread_odds', hs)
    ov, un = price('over_odds', tot), price('under_odds', tot)
    ml_h, ml_a = cols('home_ml').apply(oc._clean_price), cols('away_ml').apply(oc._clean_price)
    ml_a.columns = ml_h.columns
    both = ml_h.notna() & ml_a.notna()
    ml_h, ml_a = ml_h.where(both), ml_a.where(both)
    fair = pd.DataFrame(oc.american_to_prob(ml_h), index=ml_h.index, columns=ml_h.columns)
    fair = fair / (fair + pd.DataFrame(oc.american_to_prob(ml_a), index=ml_a.index, columns=ml_a.columns))
    keep = oc._filter_to_consensus(fair, oc.PROB_TOL).notna()
    return {'hs': hs, 'as': a_line, 'hs_odds': hs_odds, 'as_odds': as_odds, 'tot': tot, 'over': ov, 'under': un,
            'ml_h': ml_h.where(keep), 'ml_a': ml_a.where(keep)}


def load_multibook():
    with sqlite3.connect(ROOT / 'data' / 'nba_stats.db') as conn:
        opening = oc._load_table(conn, 'nba_odds_opening')
        games = oc._game_results(conn)
        try:
            n_snap = conn.execute('SELECT COUNT(*) FROM odds_snapshots').fetchone()[0]
        except sqlite3.OperationalError:
            n_snap = 0
    oc._fill_team_ids([opening])
    key = ['gameDate', 'teamId_home', 'teamId_away']
    opening = opening[(opening['gameDate'] >= '2019-10-01') & ~opening['status'].str.contains('Postponed', na=False)]
    opening = opening.drop_duplicates(subset=key, keep='last').reset_index(drop=True)
    mats = book_matrices(opening)
    ids = opening[key].merge(games[key + ['gameId']], on=key, how='left')['gameId']
    valid = (ids.notna() & ~ids.duplicated(keep='last')).to_numpy()
    ids = ids.astype(str).str.zfill(10)
    out = {}
    for k, m in mats.items():
        m = m.loc[valid].copy()
        m.index = ids[valid].values
        out[k] = m
    return out, n_snap


def payout(price):
    price = np.asarray(price, float)
    return np.where(price < 0, 100.0 / -price, price / 100.0)


def settle(result, price_payout):
    """result: >0 win, <0 loss, ==0 push (profit 0). price_payout in units won per unit staked."""
    return np.where(result > 0, price_payout, np.where(result < 0, -1.0, 0.0)) * 100


def main():
    base, _ = load_run('baseline')
    mats, n_snap = load_multibook()
    bets = M.make_bets(base, thr=0.05)
    nb = {k: v.notna().sum(axis=1) for k, v in mats.items() if k in ('hs', 'tot', 'ml_h')}
    lines = [f'# {NAME}\n', f'`odds_snapshots` rows in the database: **{n_snap}** -> no timestamped multi-book snapshots exist; the multi-book source is the '
             f'per-book columns of `nba_odds_opening` (opening quotes, up to 6 books). Coverage below is therefore the opening-line coverage.\n']
    # coverage by season
    d = base[['gameId', 'seasonYear']].copy()
    for k, lab in (('hs', 'spread'), ('tot', 'total'), ('ml_h', 'moneyline')):
        d[lab] = d['gameId'].map(nb[k]).fillna(0)
    cov = d.groupby('seasonYear').agg(games=('gameId', 'size'), **{f'{lab} >=2 books': (lab, lambda s: (s >= 2).mean()) for lab in ('spread', 'total', 'moneyline')},
                                      **{f'{lab} mean books': (lab, 'mean') for lab in ('spread', 'total', 'moneyline')})
    lines += ['## Coverage (share of test games with >= 2 books quoting the market; mean number of books)\n', cov.round(2).to_markdown(), '']
    rows = []
    out = {}
    # ---------------- spread / total bets
    sp = bets[bets['market'] == 'spread'].copy()
    t = bets[bets['market'] == 'total'].copy()
    a = base.set_index('gameId')
    def shop(frame, market):
        res = []
        for r in frame.itertuples():
            gid = r.gameId
            g = a.loc[gid]
            if market == 'spread':
                home = r.side == 'home'
                lines_ = mats['hs' if home else 'as'].loc[gid]
                odds_ = mats['hs_odds' if home else 'as_odds'].loc[gid]
                if lines_.dropna().empty:
                    res.append(None); continue
                best_line = lines_.max()
                bi = lines_[lines_ == best_line].index
                best_price = odds_[bi].max() if odds_[bi].notna().any() else -110.0
                med_line = lines_.median()
                h0 = r.line if home else -r.line           # handicap of the bet at the DK line the model saw
                margin = g['target_point_diff']
                # result in terms of OUR side: cover if (our margin + handicap) > 0
                our = (lambda h: margin + h) if home else (lambda h: -margin + h)
                res.append(dict(h0=h0, h1=best_line, hmed=med_line, nbooks=int(lines_.notna().sum()), price1=best_price,
                                r0=our(h0), r1=our(best_line), rmed=our(med_line)))
            else:
                over = r.side == 'over'
                tots = mats['tot'].loc[gid]
                odds_ = mats['over' if over else 'under'].loc[gid]
                if tots.dropna().empty:
                    res.append(None); continue
                best = tots.min() if over else tots.max()
                bi = tots[tots == best].index
                best_price = odds_[bi].max() if odds_[bi].notna().any() else -110.0
                tot_actual = g['target_total_pts']
                our = (lambda L: tot_actual - L) if over else (lambda L: L - tot_actual)
                res.append(dict(h0=r.line, h1=best, hmed=tots.median(), nbooks=int(tots.notna().sum()), price1=best_price,
                                r0=our(r.line), r1=our(best), rmed=our(tots.median())))
        return res
    for frame, mk in ((sp, 'spread'), (t, 'total')):
        recs = shop(frame, mk)
        ok = [i for i, x in enumerate(recs) if x is not None and x['nbooks'] >= 1]
        z = pd.DataFrame([recs[i] for i in ok], index=frame.index[ok])
        f = frame.loc[z.index]
        z['p0'] = settle(z['r0'], 100 / 110)
        z['pmed'] = settle(z['rmed'], 100 / 110)
        z['p1'] = settle(z['r1'], 100 / 110)
        z['p1q'] = settle(z['r1'], payout(z['price1']))
        z['improve'] = (z['h1'] - z['h0']) if mk == 'spread' else np.where(f['side'] == 'over', z['h0'] - z['h1'], z['h1'] - z['h0'])
        z['changed'] = np.sign(z['r1']) != np.sign(z['r0'])
        z['gameDate'] = f['gameDate']
        z['profit'], z['won'] = z['p0'], z['p0'] > 0
        out[mk] = z
        def roi(c):
            nz = z[c][z['r0'] != 0] if False else z[c]
            return float(nz.sum() / (100 * len(nz)) * 100)
        z0 = z.assign(profit=z['p0'], won=z['p0'] > 0)
        z1 = z.assign(profit=z['p1'], won=z['p1'] > 0)
        z1q = z.assign(profit=z['p1q'], won=z['p1q'] > 0)
        zm = z.assign(profit=z['pmed'], won=z['pmed'] > 0)
        d1 = M.roi_diff_ci(z1, z0)
        d1q = M.roi_diff_ci(z1q, z0)
        rows.append({'market': mk, 'bets with data': len(z), 'bets where best line is better': int((z['improve'] > 0).sum()),
                     'mean improvement (pts) all bets': round(float(z['improve'].mean()), 3),
                     'ROI DK line': round(roi('p0'), 2), 'ROI consensus median line': round(roi('pmed'), 2), 'ROI best line @ -110': round(roi('p1'), 2),
                     'best line vs DK diff [95% CI]': f"{d1['roi_diff']:+.2f} [{d1['ci'][0]:+.2f}, {d1['ci'][1]:+.2f}]",
                     'ROI best line @ quoted price': round(roi('p1q'), 2),
                     'quoted-price diff [95% CI]': f"{d1q['roi_diff']:+.2f} [{d1q['ci'][0]:+.2f}, {d1q['ci'][1]:+.2f}]"})
        out[mk + '_d'] = d1
    lines += ['## Spread / total bets settled at the best available line across books\n', pd.DataFrame(rows).to_markdown(index=False), '']
    # ---------------- key numbers
    z = out['spread'].copy()
    def crosses(r):
        if r['h1'] <= r['h0']:
            return 'no improvement'
        for s in KEYS:
            if r['h0'] < s <= r['h1']:
                return f'crosses {int(abs(s))}'
        return 'other half-point shift'
    z['bucket'] = z.apply(crosses, axis=1)
    kn = z.groupby('bucket').agg(bets=('p0', 'size'), mean_improvement_pts=('improve', 'mean'), result_changes=('changed', 'mean'),
                                 roi_dk=('p0', lambda s: s.sum() / (100 * len(s)) * 100), roi_best_line=('p1', lambda s: s.sum() / (100 * len(s)) * 100))
    kn['roi_gain'] = kn['roi_best_line'] - kn['roi_dk']
    # expected value of the improvement implied by the "flat value per half point" assumption: ~ 2.5% win-prob per point? report realised instead
    lines += ['## Realised price improvement by key number (spread bets; improvement measured in points of handicap)\n', kn.round(3).to_markdown(), '',
              f"Total bets: improvement distribution (points): {out['total']['improve'].describe().round(3).to_dict()}", '']
    # ---------------- moneyline
    ml = bets[bets['market'] == 'ml'].copy()
    mrows = []
    zs = []
    for r in ml.itertuples():
        g = a.loc[r.gameId]
        home = r.side == 'home'
        pr = mats['ml_h' if home else 'ml_a'].loc[r.gameId]
        if pr.dropna().empty:
            continue
        pay = payout(pr.dropna())
        best_pay, med_pay = pay.max(), np.median(pay)
        win = (g['target_home_win'] == 1) if home else (g['target_home_win'] == 0)
        zs.append({'gameId': r.gameId, 'gameDate': r.gameDate, 'dk': payout(r.price) * 100 if win else -100.0, 'med': med_pay * 100 if win else -100.0,
                   'best': best_pay * 100 if win else -100.0})
    zml = pd.DataFrame(zs)
    d_ml = M.roi_diff_ci(zml.assign(profit=zml['best'], won=zml['best'] > 0), zml.assign(profit=zml['dk'], won=zml['dk'] > 0))
    mrows.append({'market': 'ml', 'bets with data': len(zml), 'ROI DK price': round(float(zml['dk'].mean()), 2), 'ROI median price': round(float(zml['med'].mean()), 2),
                  'ROI best price': round(float(zml['best'].mean()), 2), 'best vs DK diff [95% CI]': f"{d_ml['roi_diff']:+.2f} [{d_ml['ci'][0]:+.2f}, {d_ml['ci'][1]:+.2f}]"})
    lines += ['## Moneyline bets settled at the best price across books\n', pd.DataFrame(mrows).to_markdown(index=False), '']
    # ---------------- control: bet both sides of every game at the best price
    ids = [g for g in base['gameId'] if g in mats['ml_h'].index]
    ctl = []
    for gid in ids:
        h, aw = mats['ml_h'].loc[gid].dropna(), mats['ml_a'].loc[gid].dropna()
        if h.empty or aw.empty:
            continue
        hw = a.loc[gid, 'target_home_win'] == 1
        ctl.append((payout(h).max() * 100 if hw else -100.0) + (payout(aw).max() * 100 if not hw else -100.0))
    ctl_roi = float(np.sum(ctl) / (200 * len(ctl)) * 100)
    sp_ctl = []
    for gid in base['gameId']:
        if gid not in mats['hs'].index:
            continue
        hl, al = mats['hs'].loc[gid].dropna(), mats['as'].loc[gid].dropna()
        if hl.empty or al.empty:
            continue
        m_ = a.loc[gid, 'target_point_diff']
        sp_ctl.append(settle(np.array([m_ + hl.max()]), 100 / 110)[0] + settle(np.array([-m_ + al.max()]), 100 / 110)[0])
    sp_roi = float(np.sum(sp_ctl) / (200 * len(sp_ctl)) * 100)
    lines += ['## Control: bet BOTH sides of every game at the best available price/line (an efficient market should lose here)\n',
              f'- moneyline, both sides at best price: ROI {ctl_roi:+.2f}% over {len(ctl)} games',
              f'- spread, both sides at the best line (-110): ROI {sp_roi:+.2f}% over {len(sp_ctl)} games', '']
    # ---------------- consensus as the market reference
    _, df = load_games()
    odds = oc.load_clean_odds(ROOT / 'data' / 'nba_stats.db')
    odds['gameId'] = odds['gameId'].astype(str).str.zfill(10)
    j = base.merge(odds[['gameId', 'open_home_prob', 'close_home_prob']], on='gameId', how='left', suffixes=('', '_o')).dropna(subset=['open_home_prob', 'close_home_prob'])
    y = j['target_home_win'].to_numpy(float)
    ll = lambda p: float(-(y * np.log(M.clip_p(p)) + (1 - y) * np.log(1 - M.clip_p(p))).mean())
    lines += [f'## Consensus de-vigged probability as the reference ({len(j)} games with open and close consensus)\n',
              f"- log loss: model {ll(j['ml_prob']):.4f} | consensus open {ll(j['open_home_prob']):.4f} | consensus close {ll(j['close_home_prob']):.4f}",
              f"- mean |model - open consensus| = {np.abs(j['ml_prob'] - j['open_home_prob']).mean():.3f}; share of games with model-open gap > 5 pts: {(np.abs(j['ml_prob'] - j['open_home_prob']) > .05).mean():.1%}", '']
    sp_row, ml_row = rows[0], mrows[0]
    dec = {'spread': False, 'total': False, 'ml': False}
    for mk, r in (('spread', rows[0]), ('total', rows[1])):
        d = out[mk + '_d']
        dec[mk] = bool(d['ci'][0] > 0 and sp_roi <= 0)
    dec['ml'] = bool(d_ml['ci'][0] > 0 and ctl_roi <= 0)
    lines += ['## Decision (pre-registered: adopt best-price settlement as a reporting option only if ROI improves with a CI excluding zero AND the both-sides control has ROI <= 0)\n'] + \
        [f'- {k}: **{"ADOPT as reporting option" if v else "REJECT"}**' for k, v in dec.items()]
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'decision': dec, 'control_ml_roi': ctl_roi, 'control_spread_roi': sp_roi, 'odds_snapshots_rows': n_snap}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
