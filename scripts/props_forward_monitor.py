"""
Read-only monitor of the props paper-trading record (data/props_live.db) for the forward test (reports/props/forward_test_preregistration.md).
Opens every database with mode=ro and writes nothing to data/. Prints markdown, or with --out writes a dated markdown file in that folder.

  python scripts/props_forward_monitor.py --db data/props_live.db --since 2026-11-20
  python scripts/props_forward_monitor.py --db data/props_live.db --nba-db data/nba_stats.db --out reports/props/monitor
  python scripts/props_forward_monitor.py --segment 2 --db data/props_live.db --since 2026-11-14 --nba-db data/nba_stats.db --schedule-csv data/raw/nba_schedule/schedule_2026-27.csv

--segment 1 (default) is the signed segment 1 rule; --segment 2 is the segment 2 draft rule (game-date looks, Z boundaries; needs --since E2). The parameters of each are the SEGMENTS table below.

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

# One table of decision parameters per forward-test segment (the pre-registration of each segment is the authority; the owner confirms the values there).
SEGMENTS = {
    1: {                                    # reports/props/forward_test_preregistration.md, signed 2026-11-12
        'rule': 'stakes',                   # stake horizon, futility-only interim looks at fractions of it, final test on the bootstrap lower bound
        'effective_date': None,             # given with --since
        'horizon_stakes': 8800,
        'horizon_date': '2027-04-12',
        'alt_roi': 0.03,
        'cp_floor': 0.10,
        'looks': (1 / 3, 2 / 3),
        'min_dates': 30,
        'secondary_min_stakes': 300,
        'n_boot': 2000,                     # descriptive tables; the decision rule uses 2 x n_boot = 4,000 with seed 20261112
    },
    2: {                                    # reports/props/forward_test_preregistration_segment2.md, DRAFT
        'rule': 'dates',                    # game-date horizon, three looks with early success, Z boundaries, futility at looks 1 and 2
        'effective_date': None,             # E2 is the first game date on which all three slots run under the candidate: must be given with --since (proposed 2026-11-14, not assumed)
        'horizon_date': '2027-04-12',
        'look_dates': (50, 100),            # game dates with at least one settled stake
        'boundaries': (3.471, 2.454, 2.004),  # look 1, look 2, final: PASS when Z is above
        'min_dates': 30,
        'alt_roi': 0.03,
        'cp_floor': 0.10,
        'n_boot': 4000,                     # primary and secondary intervals and the paired difference
        'seed': 20261112,
        'descriptive_n_boot': 2000,
        'primary': 'uncapped recorded bets, median-price ROI, Z = ROI / analytic date-clustered SE',
        'cap_n': 1,                         # secondary: rules.cap_per_player_game_causal(recorded bets, 1)
        'expected_dates_fallback': 150,     # segment 2 section 6: "about 150 game dates remain"; used only when neither --expected-dates nor a schedule file is available
        'schedule_csv': 'data/raw/nba_schedule/schedule_2026-27.csv',   # props.schedule cache (column et_game_date); never fetched from the network here
        'settle_grace_days': 3,             # an unsettled bet older than this is listed as stale (postponed or unsettled) and no longer holds a look back
        'registered': {'model_version': 'v4_live_2026-27c1', 'model_sha256': '4cc85ee601c261ee0c82c71f649638b8d93243816c8f403653041563e750fdf0',
                       'rule_sha256': 'fd026afd076e728cf38ea8a1aaf2321cce21ad1837256dd55b6a670878dd941d', 'roster_rule': 'pregame_all'},
    },
}
HORIZON_STAKES = SEGMENTS[1]['horizon_stakes']
HORIZON_DATE = SEGMENTS[1]['horizon_date']
ALT_ROI = SEGMENTS[1]['alt_roi']
CP_FLOOR = SEGMENTS[1]['cp_floor']
SECONDARY_MIN_STAKES = SEGMENTS[1]['secondary_min_stakes']
N_BOOT = SEGMENTS[1]['n_boot']


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

def load_inputs(db, nba_db):
    """Everything the report reads, all through read-only connections: bets, results, signals, fills, the settled/prepared bets, box-score outcomes and game margins (the last two need nba_db)."""
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
    return bets, results, signals, fills, prep, outcomes, margins


def build_report(db, nba_db=None, since=None, today=None, segment=1, expected_dates=None, schedule_csv=None):
    if segment == 2:
        return build_report_segment2(db, nba_db, since, today, expected_dates, schedule_csv)
    bets, results, signals, fills, prep, outcomes, margins = load_inputs(db, nba_db)
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


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Segment 2 (reports/props/forward_test_preregistration_segment2.md, sections 1, 6, 7, 11)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def expected_game_dates(since, horizon_date, override=None, schedule_csv=None, fallback=150, explicit_csv=False):
    """
    Expected number of game dates from `since` through `horizon_date` (the denominator of the futility rule's information fraction). Order: --expected-dates; the cached NBA schedule file
    (distinct et_game_date in the window); else the constant of the pre-registration (about 150). Returns (value, text saying which source was used). Never touches the network.
    """
    if override:
        return float(override), f'--expected-dates {override} (given on the command line)'
    p = Path(schedule_csv) if schedule_csv else None
    if p is not None and not p.is_absolute():
        p = ROOT / p
    if p is not None and p.exists():
        dates = pd.read_csv(p, usecols=['et_game_date'])['et_game_date'].astype(str).str[:10]
        n = int(dates[(dates >= since) & (dates <= horizon_date)].nunique())
        if n > 0:
            return float(n), f'NBA schedule file `{p.as_posix()}`: {n} distinct game dates from {since} through {horizon_date} (schedule dates; a date without any settled bet will not count towards the looks)'
    if explicit_csv:
        raise SystemExit(f'schedule file not found or without dates in the window: {p}')
    return float(fallback), (f'FALLBACK constant {fallback} from the pre-registration (section 6, "about {fallback} game dates remain"): no --expected-dates and no usable schedule file; '
                             f'it affects only the futility rule\'s information fraction, never the PASS boundaries')


def _roi_ci(p, lo, hi):
    return pct_ci(p, lo, hi, 2) if p is not None and np.isfinite(p) else ''


def decision2_md(s, spec, exp_dates, exp_source, pending, stale):
    b = spec['boundaries']
    rows = [['Status', f"**{s['status']}**"], ['Reason', s['reason']], ['Decided at', s['decided_at'] or '(not decided)'],
            ['Game dates with a settled stake / stakes (test data to date)', f"{s['n_dates']} / {s['n_stakes']}"],
            ['Median-price ROI to date, SE, Z (descriptive once decided)', f"{pct(s['roi'], 2)}, SE {100 * s['se']:.2f} pts, Z {num(s['z'], 3)}" if np.isfinite(s['roi']) else ''],
            ['Expected game dates, E2 through ' + spec['horizon_date'], f'{exp_dates:.0f}'], ['Source of the expected dates', exp_source],
            ['Boundaries (look 1, look 2, final)', f'Z above {b[0]}, {b[1]}, {b[2]}; at least {spec["min_dates"]} game dates; futility at looks 1 and 2 only (conditional power < {spec["cp_floor"]:.0%} at true ROI {spec["alt_roi"]:+.1%})']]
    out = table(rows, ['item', 'value'])
    lrows = []
    reached = {lk['look'] for lk in s['looks']}
    for lk in s['looks']:
        if 'z' not in lk:
            lrows.append([lk['look'], num(b[lk['look'] - 1], 3), lk['n_dates_target'], lk['cutoff'], '', '', '', '', 'WAITING for settlement of ' + ', '.join(lk['waiting']), '', ''])
            continue
        res = 'PASS' if lk['passed'] else ('below boundary' if lk['look'] == 'final' else ('futile' if (np.isfinite(lk['cp']) and lk['cp'] < spec['cp_floor']) else 'continue'))
        lrows.append([lk['look'], num(lk['boundary'], 3), lk['n_dates'], lk['cutoff'], lk['n_stakes'], pct(lk['roi'], 2), f"{100 * lk['se']:.2f}", num(lk['z'], 3), res, num(lk['t'], 3), num(lk['cp'], 3)])
    for i, nl in enumerate(spec['look_dates']):
        if (i + 1) not in reached and s['status'] in ('CONTINUE', 'SEGMENT-ENDED'):
            lrows.append([i + 1, num(b[i], 3), nl, '(not reached)', '', '', '', '', f'{s["n_dates"]} of {nl} game dates', '', ''])
    out += '\n**Looks** (each uses only the data through its own cut-off date; stakes per look are shown because dates differ in size and the boundaries are not adjusted for it)\n\n'
    out += table(lrows, ['look', 'boundary Z', 'game dates', 'through', 'stakes', 'ROI median', 'SE (pts)', 'Z', 'result', 'info fraction t', 'cond. power'])
    if s['status'] in ('PASS', 'STOP-FUTILITY') and s['decided_at'] != 'final':
        out += (f"\nThe confirmatory test ended at {s['decided_at']}. Every run recomputes that outcome from the first {s['looks'][-1]['n_dates']} game dates only (never later data); "
                f'the record below keeps running and is descriptive.\n')
    if pending:
        out += f"\nGame dates with recorded bets not yet settled (hold back any look at or after them): {', '.join(sorted(pending))}.\n"
    if len(stale):
        out += f"\nUnsettled bets older than {spec['settle_grace_days']} days (postponed or not settled; excluded, listed for the final report): {len(stale)} bets on {', '.join(sorted(set(stale['game_date'])))}.\n"
    return out + '\n'


def secondary2_md(test_prep, capped_ids, slices, spec):
    """Capped subset and uncapped record at each reached look and to date: ROI at median and best price (same 4,000 date resamples for both records), paired differences, CLV."""
    rows = []
    for label, cut in slices:
        a = test_prep[test_prep['game_date'] <= cut]
        c = a[a['bet_id'].isin(capped_ids)]
        med = mon.paired_roi_comparison(a, c, spec['n_boot'], spec['seed'], profit='profit_median')
        best = mon.paired_roi_comparison(a, c, spec['n_boot'], spec['seed'], profit='profit_best')
        if not med.get('n_dates'):
            continue
        clv_a = mon.cluster_bootstrap_mean(a['clv_points'], a['game_date'], n_boot=200, seed=0)
        clv_c = mon.cluster_bootstrap_mean(c['clv_points'], c['game_date'], n_boot=200, seed=0)
        clv = lambda r: f"{r['point']:+.2f} (n {r['n']}, se {r['se']:.2f})" if np.isfinite(r['point']) else ''
        rows.append([f'{label} (through {cut})', 'uncapped (primary)', med['n_dates'], int(a['stake'].sum()), _roi_ci(med['roi_a'], med['lo_a'], med['hi_a']), '',
                     _roi_ci(best['roi_a'], best['lo_a'], best['hi_a']), '', clv(clv_a)])
        rows.append(['', 'capped subset', med['n_dates'], int(c['stake'].sum()), _roi_ci(med['roi_b'], med['lo_b'], med['hi_b']), _roi_ci(med['diff'], med['diff_lo'], med['diff_hi']),
                     _roi_ci(best['roi_b'], best['lo_b'], best['hi_b']), _roi_ci(best['diff'], best['diff_lo'], best['diff_hi']), clv(clv_c)])
    head = ['look', 'record', 'dates', 'stakes', 'ROI median [95% CI]', 'diff vs uncapped, median [95% CI]', 'ROI best [95% CI]', 'diff vs uncapped, best [95% CI]', 'CLV pts']
    return table(rows, head) + (f"\nIntervals: date-clustered bootstrap, {spec['n_boot']} resamples, seed {spec['seed']}; the uncapped record, the capped subset and the paired difference use the same resamples "
                                f"(date axis = dates of the uncapped record with a stake). Never decisive.\n")


def build_report_segment2(db, nba_db=None, since=None, today=None, expected_dates=None, schedule_csv=None):
    spec = SEGMENTS[2]
    if not since:
        raise SystemExit('--segment 2 needs --since E2 (the effective date; the document proposes 2026-11-14 but E2 is the first game date under the full configuration, so it is not assumed)')
    bets, results, signals, fills, prep, outcomes, margins = load_inputs(db, nba_db)
    today = today or datetime.now().strftime('%Y-%m-%d')
    reg = spec['registered']
    fwd_bets = bets[bets['game_date'] >= since]
    integ = mon.segment_integrity(fwd_bets, reg['rule_sha256'], reg['roster_rule'])
    end = integ['end_date']
    test_bets = fwd_bets[fwd_bets['game_date'] < end] if end else fwd_bets
    test_prep = prep[prep['bet_id'].isin(set(test_bets['bet_id']))]
    exp_n, exp_src = expected_game_dates(since, spec['horizon_date'], expected_dates, schedule_csv or spec['schedule_csv'], spec['expected_dates_fallback'], explicit_csv=bool(schedule_csv))
    grace_cut = (pd.Timestamp(today) - pd.Timedelta(days=spec['settle_grace_days'])).strftime('%Y-%m-%d')
    unsettled = test_bets[~test_bets['bet_id'].isin(set(results['bet_id']))]
    pending = sorted(set(unsettled.loc[unsettled['game_date'] >= grace_cut, 'game_date']))
    stale = unsettled[unsettled['game_date'] < grace_cut]
    f = test_prep[test_prep['stake'] > 0]
    s = mon.group_sequential_status(f['profit_median'], f['stake'], f['game_date'], spec['look_dates'], spec['boundaries'], exp_n, alt_roi=spec['alt_roi'], cp_floor=spec['cp_floor'],
                                    min_dates=spec['min_dates'], horizon_date=spec['horizon_date'], today=today, pending_dates=pending)
    if end and s['status'] == 'CONTINUE':
        s['status'], s['reason'] = 'SEGMENT-ENDED', f"{integ['reason']}; no look had terminated the test before it"
    capped_ids = mon.capped_bet_ids(test_bets, spec['cap_n']) if len(test_bets) else set()
    out = [f'# Props forward-test monitor, segment 2, {today}\n',
           f'Database `{db}` (read-only). Bets {len(bets)} ({bets["game_date"].min()} to {bets["game_date"].max()}), settled {len(prep)}, signals {len(signals)}. '
           f'Model versions: {", ".join(f"{k} ({v})" for k, v in bets["model_version"].value_counts().sort_index().items())}. '
           f'Minutes before tip of the bets: {", ".join(f"T-{k}: {v}" for k, v in bets["minutes_before"].value_counts().sort_index(ascending=False).items())}.\n',
           '**Reading guide.** Segment 2 (DRAFT pre-registration). Primary: median-price ROI of the UNCAPPED recorded bets from E2, decided by Z = ROI / date-clustered SE at three looks counted in game dates. '
           'The capped subset (`rules.cap_per_player_game_causal(recorded bets, 1)`), best-price ROI and CLV are secondary and never decisive. Intervals are 95% bootstraps over game dates.\n',
           f'## A. Forward test: game_date >= {since} (E2)\n', '### Configuration integrity (bets placed in the forward test)\n']
    if len(fwd_bets):
        seg = integ['table']
        rows = [[r.model_version, str(r.model_sha256)[:12], str(r.rule_sha256)[:12], r.roster_rule, r.n_bets, r.first_date, r.last_date, r.slots] for r in seg.itertuples(index=False)]
        out.append(table(rows, ['model_version', 'model sha', 'rule sha', 'roster rule', 'bets', 'first date', 'last date', 'slots']))
        out.append(f"\nRegistered: model_version {reg['model_version']}, model sha256 {reg['model_sha256'][:12]}, rule sha256 {reg['rule_sha256'][:12]}, roster rule {reg['roster_rule']}.\n")
        out.append(f"\n**{integ['reason']}.**\n" if (integ['ended'] or integ['n_model_rows'] > 1) else '\nOne configuration.\n')
        if (seg['model_version'] != reg['model_version']).any() or (seg['model_sha256'].astype(str) != reg['model_sha256']).any():
            out.append('\nNote: at least one (model_version, model sha256) differs from the registered one; a refit by the same recipe is allowed (section 11), anything else ends the segment: check the retrain log.\n')
    else:
        out.append('_no bets yet_\n')
    out.append('\n### Decision rule (segment 2, DRAFT pre-registration)\n')
    out.append(decision2_md(s, spec, exp_n, exp_src, pending, stale))
    if test_prep.empty or test_prep['stake'].sum() == 0:
        out.append('_No settled forward-test bets yet._\n')
    else:
        slices = [(f"look {lk['look']}", lk['cutoff']) for lk in s['looks'] if 'z' in lk and lk['look'] != 'final'] + ([('final', spec['horizon_date'])] if any(lk['look'] == 'final' for lk in s['looks']) else [])
        slices.append(('to date (descriptive)', test_prep['game_date'].max()))
        out.append('### Secondary (never decisive): capped subset, paired difference, best price, CLV\n')
        out.append('Capped subset: all recorded bets of the test window go through `rules.cap_per_player_game_causal(sel, 1)` with the columns game_id, player_id, stat, minutes_before as stored and best_ev = ev_best '
                   f'(EV at the best price); {len(capped_ids)} of {len(test_bets)} recorded bets are kept. Slots run T-240, T-60, T-5; within one slot the highest ev_best wins, ties go to the stat that sorts first.\n')
        out.append(secondary2_md(test_prep, capped_ids, slices, spec))
        out.append('\n' + period_report('Forward-test record, uncapped, all game dates to date (descriptive; after a PASS or STOP the outcome above is not revisited)', test_prep, signals[signals['game_date'] >= since],
                                         outcomes, margins, fills[fills['bet_id'].isin(set(test_prep['bet_id']))] if fills is not None else None, '###'))
    if end:
        late = prep[prep['bet_id'].isin(set(fwd_bets[fwd_bets['game_date'] >= end]['bet_id']))]
        out.append(f'\n### After the segment ended ({end} on): {len(late)} settled bets, descriptive only\n')
    out.append(f'\n## B. Record before E2 (game_date < {since}), descriptive only\n')
    out.append('Segment 1 and earlier: never counts towards segment 2.\n')
    out.append(period_report('Record before E2', prep[prep['game_date'] < since], signals[signals['game_date'] < since], outcomes, margins,
                             fills[fills['bet_id'].isin(set(prep[prep['game_date'] < since]['bet_id']))] if fills is not None else None, '###'))
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default='data/props_live.db')
    ap.add_argument('--nba-db', default=None, help='optional data/nba_stats.db for outcomes of non-bet signals and game margins')
    ap.add_argument('--since', default=None, help='registration effective date YYYY-MM-DD')
    ap.add_argument('--out', default=None, help='folder for a dated markdown report (default: print to stdout)')
    ap.add_argument('--today', default=None, help='override today (YYYY-MM-DD), for tests')
    ap.add_argument('--segment', type=int, choices=(1, 2), default=1, help='which segment\'s decision rule to report (default 1, the signed segment 1 rule)')
    ap.add_argument('--expected-dates', type=int, default=None, help='segment 2: expected number of game dates from E2 through the horizon (information fraction of the futility rule)')
    ap.add_argument('--schedule-csv', default=None, help='segment 2: cached NBA schedule (props.schedule format, column et_game_date) to count the expected game dates')
    a = ap.parse_args(argv)
    if a.segment == 1 and (a.expected_dates or a.schedule_csv):
        ap.error('--expected-dates and --schedule-csv apply to --segment 2 only')
    text = build_report(a.db, a.nba_db, a.since, a.today, segment=a.segment, expected_dates=a.expected_dates, schedule_csv=a.schedule_csv)
    if a.out:
        out_dir = Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = 'props_forward_monitor' if a.segment == 1 else 'props_forward_monitor_seg2'
        path = out_dir / f'{stem}_{a.today or datetime.now().strftime("%Y-%m-%d")}.md'
        path.write_text(text, encoding='utf-8')
        print(f'wrote {path}')
    else:
        sys.stdout.reconfigure(encoding='utf-8')
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
