"""
Read-only monitor of the props paper-trading record (data/props_live.db) for the forward test (reports/props/forward_test_preregistration.md).
Opens every database with mode=ro and writes nothing to data/. Prints markdown, or with --out writes a dated markdown file in that folder.

  python scripts/props_forward_monitor.py --db data/props_live.db --since 2026-11-20
  python scripts/props_forward_monitor.py --db data/props_live.db --nba-db data/nba_stats.db --out reports/props/monitor

--since is the registration effective date: bets with game_date before it are shown as "pre-registration record, descriptive only"; the forward test and the decision rule use the rest.
Without --since the whole record is shown as one descriptive block. --nba-db (optional) supplies box scores for outcomes of non-bet signals (market-level slope) and game margins.
"""
import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from props.live import monitor as mon  # noqa: E402

FROZEN_STATS = ('rebounds', 'assists', 'threes', 'ra')
# DRAFT design constants (see the pre-registration); the owner confirms them there.
HORIZON_STAKES = 8800
HORIZON_DATE = '2027-04-12'
ALT_ROI = 0.03
CP_FLOOR = 0.10
SECONDARY_MIN_STAKES = 300
N_BOOT = 2000


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Read-only database access
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def open_ro(path):
    p = Path(path).resolve()
    if not p.exists():
        raise SystemExit(f'database not found: {p}')
    conn = sqlite3.connect(f'file:{p.as_posix()}?mode=ro', uri=True, timeout=60)
    conn.execute('PRAGMA query_only = ON')
    return conn


def has_table(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)).fetchone() is not None


def load_live(conn):
    bets = pd.read_sql('SELECT * FROM prop_bets', conn).drop(columns=['pmf_json', 'books_json'], errors='ignore')
    results = pd.read_sql('SELECT * FROM prop_results', conn)
    signals = pd.read_sql('SELECT * FROM prop_signals', conn)
    fills = pd.read_sql('SELECT * FROM prop_fill_checks', conn) if has_table(conn, 'prop_fill_checks') else None
    return bets, results, signals, fills


def _in_clause(ids):
    return ','.join('?' * len(ids)), list(ids)


def load_box(nba_conn, game_ids):
    """One scan of player_basic_stats (no index on gameId) for the games in question."""
    ph, params = _in_clause(sorted(set(game_ids)))
    box = pd.read_sql(f'SELECT gameId AS game_id, personId AS player_id, minutes, points, rebounds, assists, threePointersMade AS threes FROM player_basic_stats WHERE gameId IN ({ph})',
                      nba_conn, params=params)
    box['game_id'] = box['game_id'].astype(str)
    return box


def load_margins(nba_conn, game_ids):
    ph, params = _in_clause(sorted(set(game_ids)))
    t = pd.read_sql(f'SELECT gameId AS game_id, plusMinus FROM team_basic_stats WHERE gameId IN ({ph})', nba_conn, params=params)
    t['game_id'] = t['game_id'].astype(str)
    return t.groupby('game_id')['plusMinus'].apply(lambda s: float(s.abs().max())).rename('margin').reset_index()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Markdown helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def pct(x, d=1):
    return '' if x is None or not np.isfinite(x) else f'{100 * x:+.{d}f}%'


def pct_ci(p, lo, hi, d=1):
    return '' if not np.isfinite(p) else f'{100 * p:+.{d}f}% [{100 * lo:+.{d}f}, {100 * hi:+.{d}f}]'


def num(x, d=3):
    return '' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


def table(rows, header):
    if not rows:
        return '_no rows_\n'
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out) + '\n'


def segment_md(seg, label):
    rows = []
    for r in seg.itertuples(index=False):
        clv = f'{r.clv_mean:+.2f} (n {r.clv_n}, se {r.clv_se:.2f})' if np.isfinite(r.clv_mean) else ''
        rows.append([getattr(r, label) if isinstance(label, str) else ' / '.join(str(getattr(r, c)) for c in label), r.n_stakes, ('n<50' if r.low_n else ''),
                     num(r.win_rate, 3), num(r.mean_p_market, 3), num(r.mean_p_star, 3), pct_ci(r.roi_median, r.roi_median_lo, r.roi_median_hi),
                     pct_ci(r.roi_best, r.roi_best_lo, r.roi_best_hi), pct(r.roi_flat), clv])
    return table(rows, ['segment', 'stakes', 'flag', 'win rate', 'mean p mkt', 'mean p*', 'ROI median [95% CI]', 'ROI best [95% CI]', 'ROI flat', 'CLV pts'])


def seg_section(prep, by, title):
    label = by if isinstance(by, str) else by
    seg = mon.segment_table(prep, by, n_boot=N_BOOT)
    return f'#### {title}\n\n' + segment_md(seg, label) + '\n'


def slope_md(sl):
    rows = [[r.stat, r.n, r.n_dates, f'{r.over_bias:+.3f}', f'{r.slope_origin:+.3f} [{r.so_lo:+.3f}, {r.so_hi:+.3f}]', f'{r.slope_icpt:+.3f} [{r.si_lo:+.3f}, {r.si_hi:+.3f}]',
             f'{r.intercept:+.3f} [{r.a_lo:+.3f}, {r.a_hi:+.3f}]'] for r in sl.itertuples(index=False)]
    return table(rows, ['stat', 'n', 'dates', 'mean(hit - p_mkt)', 'slope through origin', 'slope with intercept', 'intercept'])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# One period
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def headline(prep):
    st = prep['stake'].to_numpy()
    out = {'n_bets': len(prep), 'n_stakes': int(st.sum()), 'voids': int((prep['status_stands'] == 'void').sum()), 'pushes': int((prep['status_stands'] == 'push').sum())}
    for name in ('median', 'best'):
        out[name] = mon.cluster_bootstrap_roi(prep[f'profit_{name}'], st, prep['game_date'], n_boot=N_BOOT)
    out['flat'] = float(prep['profit_flat'].sum() / st.sum()) if st.sum() else np.nan
    out['clv'] = mon.cluster_bootstrap_mean(prep['clv_points'], prep['game_date'], n_boot=N_BOOT)
    return out


def headline_md(h):
    m, b, c = h['median'], h['best'], h['clv']
    rows = [['Bets settled (rows in prop_results)', h['n_bets']], ['Stakes (win or loss)', h['n_stakes']], ['Voids / pushes (not stakes)', f"{h['voids']} / {h['pushes']}"],
            ['Game dates with stakes', m['n_clusters']],
            ['ROI at MEDIAN price (primary) [95% date-clustered CI]', pct_ci(m['point'], m['lo'], m['hi']) + f"  P(ROI<=0) = {m['p_le0']:.3f}"],
            ['ROI at BEST price (secondary) [95% CI]', pct_ci(b['point'], b['lo'], b['hi']) + f"  P(ROI<=0) = {b['p_le0']:.3f}"],
            ['ROI flat -110', pct(h['flat'])],
            ['Mean CLV, points (n with CLV; clustered SE)', f"{c['point']:+.2f} (n {c['n']}; se {c['se']:.2f})" if np.isfinite(c['point']) else '']]
    return table(rows, ['measure', 'value'])


def calibration_md(cal):
    if not cal.get('n'):
        return '_no settled stakes_\n'
    rows = [['Actual wins / expected under p*', f"{cal['actual_wins']:.0f} / {cal['expected_wins_pstar']:.1f}  (diff {cal['diff']:+.1f}, clustered SE {cal['diff_se']:.1f}, z {cal['z']:+.2f})"],
            ['Win rate vs mean p* vs mean market fair p (bet side)', f"{cal['win_rate']:.3f} / {cal['mean_p_star']:.3f} / {cal['mean_p_market']:.3f}"],
            ['Brier, bet side: p* / p_model / p_market', f"{cal['brier_pstar']:.4f} / {cal['brier_pmodel']:.4f} / {cal['brier_pmarket']:.4f}"]]
    return table(rows, ['measure', 'value'])


def breakeven_md(be):
    if not be.get('n'):
        return '_no bets with CLV_\n'
    rows = []
    for name in ('median', 'best', 'flat'):
        rows.append([name, f"{be[f'mean_price_{name}']:.3f}", f"{be[f'breakeven_clv_{name}_points']:+.2f}", f"{be['clv_mean_points']:+.2f}", pct(be[f'clv_implied_roi_{name}'], 2),
                     pct(be.get(f'realised_roi_{name}_same_bets', np.nan), 2) if name != 'flat' else ''])
    return table(rows, ['price', 'mean decimal price', 'break-even CLV (pts)', 'mean CLV earned (pts)', 'ROI implied by CLV', 'realised ROI, same bets']) + \
        f"\n{be['n']} stakes with a CLV value. Break-even CLV = mean(1/price - fair probability of the bet side at the bet time), in points.\n"


def fill_md(fs):
    lines = [f"- bets with a fill check: {fs['n']}"]
    for k, v in fs.items():
        if k == 'n':
            continue
        lines.append(f'- {k}: {v}')
    return '\n'.join(lines) + '\n'


def period_report(title, prep, signals, outcomes, margins, fills, level='###'):
    parts = [f'{level} {title}\n']
    if prep.empty or prep['stake'].sum() == 0:
        return parts[0] + '\n_no settled stakes in this period_\n'
    d = mon.add_segments(prep)
    if margins is not None:
        d = d.merge(margins, on='game_id', how='left')
        d['blowout'] = np.where(d['margin'].isna(), 'unknown', np.where(d['margin'] >= 20, 'margin 20+', 'margin <20'))
    h = headline(prep)
    parts.append(headline_md(h))
    parts.append('\n**Segments** (n<50 stakes are flagged; intervals are date-clustered)\n')
    for by, t in (('side', 'By side'), ('slot', 'By slot (the T-5 slot first ran 2026-11-12)'), ('stat', 'By stat'), (['stat', 'side'], 'By stat x side'),
                  ('model_version', 'By model version'), ('ev_bucket', 'By claimed EV at best price (does claimed EV track realised ROI?)'), ('minutes_bucket', 'By expected minutes'),
                  ('week', 'By week (Monday)')):
        parts.append(seg_section(d, by, t))
    parts.append('#### Information slope on the bets (hit_over - p_market on p_model - p_market)\n\n' + slope_md(mon.information_slope(mon.slope_frame_from_bets(d), n_boot=N_BOOT)))
    if outcomes is not None and signals is not None and len(signals):
        mk = mon.market_frame(signals, outcomes, stats=FROZEN_STATS)
        mk = mk[mk['game_date'].isin(set(d['game_date']))] if len(d) else mk
        parts.append('\n#### Information slope on every evaluated market (earliest slot, all signals incl. non-qualified, outcomes from the box scores)\n\n'
                     'The "mean(hit - p_mkt)" column is the market-wide over-bias; a through-origin slope is biased when it is not zero.\n\n' + slope_md(mon.information_slope(mk, n_boot=N_BOOT)))
    else:
        parts.append('\n_Market-level slope skipped: needs --nba-db for outcomes of non-bet signals._\n')
    parts.append('\n#### Calibration of p* on settled stakes\n\n' + calibration_md(mon.calibration(prep)))
    parts.append('\n#### CLV against break-even\n\n' + breakeven_md(mon.clv_breakeven(prep)))
    if fills is not None:
        parts.append('\n#### Fill study (prop_fill_checks)\n\n' + fill_md(mon.fill_summary(prep, fills)))
    parts.append('\n#### Ex-post descriptive splits (never used for decisions)\n')
    parts.append(seg_section(d, 'book', 'By book of the best price'))
    if margins is not None:
        parts.append(seg_section(d, ['blowout', 'side'], 'By game margin (20+) x side'))
    return '\n'.join(parts)


def design_md(prep):
    p = prep
    rows = []
    for hw in (0.02, 0.015):
        r = mon.required_dates(p['profit_median'], p['stake'], p['game_date'], hw)
        rows.append([f'+/-{100 * hw:.1f} pts', f"{r['dates']:.0f}", f"{r['stakes']:.0f}", f"{r['mean_stakes_per_date']:.1f}", f"{100 * r['sd_resid_per_date'] / r['mean_stakes_per_date']:.1f}"])
    r0 = mon.required_dates(p['profit_median'], p['stake'], p['game_date'], 0.02)
    head = (f"From this record's {r0['n_dates_now']} game dates and {int(p['stake'].sum())} stakes: median-price ROI {pct(r0['roi'], 2)}, half-width now +/-{100 * r0['half_width_now']:.1f} pts, "
            f"{r0['mean_stakes_per_date']:.1f} stakes per date, residual SD per date {r0['sd_resid_per_date']:.1f} units "
            f"({100 * r0['sd_resid_per_date'] / r0['mean_stakes_per_date']:.0f} units per 100 stakes).\n\n")
    return head + table(rows, ['target 95% half-width', 'game dates needed', 'stakes needed', 'stakes per date now', 'resid SD per date, per 100 stakes (units)'])


def sequential_md(prep, since, today):
    f = prep[prep['stake'] > 0]
    s = mon.sequential_status(f['profit_median'], f['stake'], f['game_date'], HORIZON_STAKES, HORIZON_DATE, today, ALT_ROI, CP_FLOOR, n_boot=N_BOOT * 2)
    rows = [['Status', f"**{s['status']}**"], ['Reason', s['reason']], ['Stakes / horizon', f"{s['n_stakes']} / {s['horizon_stakes']} (to {s['horizon_date']} if earlier)"],
            ['Game dates', s['n_dates']], ['Median-price ROI [95% CI]', pct_ci(s['roi'], s['lo'], s['hi'], 2) if np.isfinite(s['roi']) else ''],
            ['Conditional power at last look', num(s['cp'], 3)]]
    for lk in s['looks']:
        rows.append([f"Look at {lk['fraction']:.2f}", f"through {lk['cutoff']}: {lk['n_stakes']} stakes, ROI {pct(lk['roi'], 2)}, SE {100 * lk['se']:.2f} pts, conditional power {lk['cp']:.3f}"])
    return table(rows, ['item', 'value'])


def integrity_md(bets):
    seg = mon.config_segments(bets)
    rows = [[r.model_version, str(r.model_sha256)[:12], str(r.rule_sha256)[:12], r.roster_rule, r.n_bets, r.first_date, r.last_date, r.slots] for r in seg.itertuples(index=False)]
    note = '' if len(seg) <= 1 else '\n**More than one configuration: a model, rule or roster-rule change starts a new numbered segment (see the pre-registration).**\n'
    return table(rows, ['model_version', 'model sha', 'rule sha', 'roster rule', 'bets', 'first date', 'last date', 'slots']) + note


def secondary_md(prep):
    d = prep[(prep['side'] == 'under') & (prep['expected_minutes'].astype(float) < 20)]
    if d.empty or d['stake'].sum() == 0:
        return '_no qualifying bets_\n'
    h = headline(d)
    flag = 'below' if h['n_stakes'] < SECONDARY_MIN_STAKES else 'at or above'
    return headline_md(h) + f"\nThe pre-registered minimum is {SECONDARY_MIN_STAKES} stakes; this sample is {flag} it.\n"


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_report(db, nba_db=None, since=None, today=None):
    conn = open_ro(db)
    try:
        bets, results, signals, fills = load_live(conn)
    finally:
        conn.close()
    prep = mon.prepare_bets(bets, results)
    outcomes = margins = None
    if nba_db:
        nconn = open_ro(nba_db)
        try:
            box = load_box(nconn, signals['game_id'].astype(str).unique())
            outcomes = mon.outcomes_from_box(box, signals.assign(game_id=signals['game_id'].astype(str)))
            margins = load_margins(nconn, prep['game_id'].astype(str).unique())
        finally:
            nconn.close()
        signals = signals.assign(game_id=signals['game_id'].astype(str))
    today = today or datetime.now().strftime('%Y-%m-%d')
    out = [f'# Props forward-test monitor, {today}\n',
           f'Database `{db}` (read-only). Bets {len(bets)} ({bets["game_date"].min()} to {bets["game_date"].max()}), settled {len(prep)}, signals {len(signals)}. '
           f'Model versions: {", ".join(f"{k} ({v})" for k, v in bets["model_version"].value_counts().sort_index().items())}. '
           f'Minutes before tip of the bets: {", ".join(f"T-{k}: {v}" for k, v in bets["minutes_before"].value_counts().sort_index(ascending=False).items())}.\n',
           '**Reading guide.** Primary metric: ROI at the median bettable price. The best-price ROI is value only if a limit order at that price fills (see the fill study when present). '
           'CLV is the consensus fair-probability move of the bet side from the bet to the T-5 snapshot (about 10 minutes before tip), in points. Intervals are 95% bootstraps over game dates.\n']
    if since:
        pre = prep[prep['game_date'] < since]
        fwd = prep[prep['game_date'] >= since]
        sig_pre = signals[signals['game_date'] < since]
        sig_fwd = signals[signals['game_date'] >= since]
        fills_pre = fills[fills['bet_id'].isin(set(pre['bet_id']))] if fills is not None else None
        fills_fwd = fills[fills['bet_id'].isin(set(fwd['bet_id']))] if fills is not None else None
        out.append(f'## A. Forward test: game_date >= {since}\n')
        out.append('### Configuration integrity (bets placed in the forward test)\n')
        out.append(integrity_md(bets[bets['game_date'] >= since]) if (bets['game_date'] >= since).any() else '_no bets yet_\n')
        if fwd.empty:
            out.append('_No settled forward-test bets yet._\n')
        else:
            out.append(period_report('Forward-test record', fwd, sig_fwd, outcomes, margins, fills_fwd, '###'))
            out.append('\n### Decision rule (DRAFT pre-registration)\n')
            out.append(sequential_md(fwd, since, today))
            out.append('\n### Sample-size arithmetic on the forward record\n')
            out.append(design_md(fwd))
            out.append('\n### Secondary hypothesis (exploratory): unders with expected minutes < 20\n')
            out.append(secondary_md(fwd))
        out.append(f'\n## B. Pre-registration record (game_date < {since}), descriptive only\n')
        out.append('Reasons it is not part of the test: rows restored from an export after the 2026-10-31 data loss, no T-5 slot before 2026-11-12, and inspected in the audit that wrote the rules.\n')
        out.append(period_report('Pre-registration record', pre, sig_pre, outcomes, margins, fills_pre, '###'))
        out.append('\n### Sample-size arithmetic on the pre-registration record\n')
        out.append(design_md(pre) if pre['stake'].sum() else '_no stakes_\n')
        out.append('\n### Secondary hypothesis candidate on the pre-registration record (descriptive only)\n')
        out.append(secondary_md(pre))
    else:
        out.append('## Full record (no --since given: descriptive only)\n')
        out.append('### Configurations in the record\n')
        out.append(integrity_md(bets))
        out.append(period_report('All settled bets', prep, signals, outcomes, margins, fills, '###'))
        out.append('\n### Sample-size arithmetic on the full record\n')
        out.append(design_md(prep))
        out.append('\n### Secondary hypothesis candidate: unders with expected minutes < 20 (post-hoc, descriptive here)\n')
        out.append(secondary_md(prep))
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default='data/props_live.db')
    ap.add_argument('--nba-db', default=None, help='optional data/nba_stats.db for outcomes of non-bet signals and game margins')
    ap.add_argument('--since', default=None, help='registration effective date YYYY-MM-DD')
    ap.add_argument('--out', default=None, help='folder for a dated markdown report (default: print to stdout)')
    ap.add_argument('--today', default=None, help='override today (YYYY-MM-DD), for tests')
    a = ap.parse_args(argv)
    text = build_report(a.db, a.nba_db, a.since, a.today)
    if a.out:
        out_dir = Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'props_forward_monitor_{a.today or datetime.now().strftime("%Y-%m-%d")}.md'
        path.write_text(text, encoding='utf-8')
        print(f'wrote {path}')
    else:
        sys.stdout.reconfigure(encoding='utf-8')
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
