"""
The dashboard's betting record: which bets the live run RECOMMENDED, and how they settled.
Pure pandas, no Streamlit (so the hosted app stays thin and this is unit-tested: tests/team/test_dashboard_record.py).

Where a recommendation comes from, per game and market (ml / spread / total):
  1. 'run'    the bet_recommendations_log row of the FIRST inference run at which the game had a line for that market. A later run
              (a re-run for late lines, a manual re-run) never replaces it: a bet is what the run that first had a line said.
  2. 'printed' / 'recomputed'  rows written by scripts/maintenance/backfill_bet_recommendations.py for games before that table existed
              (their provenance lives in the small bet_recommendations_provenance table and arrives here as prov_<market> columns).
  3. 'page'   no row at all: the page recomputes the bet with the live rule from the stored prediction, the current line, the side's
              own moneyline price (-110 for spread and total, which are not stored) and the deployed model set's saved sigmas.
All three use the same rule (team/betting.py): bet when the model's win probability beats the price's break-even by more than 5%.

Grading is push-aware (a push returns the stake, is shown in a W-L-P record and is neither a win nor a loss) and uses the logged line and the side's own price.
Columns added to the predictions frame, per market m: rec_<m>_<field> with fields in BASIS_FIELDS (a = home / over side, b = away / under side).
"""
import numpy as np
import pandas as pd

from team.betting import (DEFAULT_PROB_EDGE, DEFAULT_SPREAD_STD, DEFAULT_TOTAL_STD, STANDARD_PRICE, calculate_implied_prob, calculate_payout,
                          get_spread_bet, get_total_bet, implied_prob_edge, ml_side_edges)

MARKETS = ('ml', 'spread', 'total')
MARKET_OF_BET_TYPE = {'Moneyline': 'ml', 'Spread': 'spread', 'Totals': 'total'}
BASIS_FIELDS = ['side', 'edge', 'line', 'pred', 'sigma', 'pa', 'pb', 'own_view', 'line_move', 'run_ts', 'src', 'close_line']
LOGGED_SOURCES = ('run', 'printed')          # the recommendation was written down (by a run, or printed in the daily log)
SOURCE_LABELS = {'run': 'logged by the live run', 'printed': 'printed in the daily log', 'recomputed': 'recomputed (backfill)', 'page': 'recomputed by this page'}

# What it takes for a game to "have a line" in a market, and where the market's inputs sit in a bet_recommendations_log row.
# a = home (ML, spread) / over (total), b = away / under.
_LOG_SPEC = {
    'ml': dict(has=['curr_odds_home_ml', 'curr_odds_away_ml'], line=None, pa='curr_odds_home_ml', pb='curr_odds_away_ml', pred='ml_prob', sigma=None,
               close='close_p_home'),
    'spread': dict(has=['curr_odds_home_spread'], line='curr_odds_home_spread', pa='curr_price_home_spread_odds', pb='curr_price_away_spread_odds',
                   pred='spread_pred', sigma='spread_std', close='close_home_spread'),
    'total': dict(has=['curr_odds_total'], line='curr_odds_total', pa='curr_price_over_odds', pb='curr_price_under_odds',
                  pred='total_pred', sigma='total_std', close='close_total'),
}
# The same inputs in a predictions_log_v2 row (the page's recomputation); spread / total prices are not stored there
_PRED_SPEC = {
    'ml': dict(line=None, pa='odds_home_ml', pb='odds_away_ml', pred='ml_prob'),
    'spread': dict(line='odds_home_spread', pa=None, pb=None, pred='spread_pred'),
    'total': dict(line='odds_total', pa=None, pb=None, pred='total_pred'),
}
SIDES = {'ml': ('home', 'away'), 'spread': ('home', 'away'), 'total': ('over', 'under')}


def canonical_ids(s):
    return pd.Series(s).astype(str).str.zfill(10)


# ----------------------------------------------------------------------------------------------
# Sigmas of the deployed model set

def sigmas_from_meta(meta):
    """{'spread_std', 'total_std', 'source'} from a calibration_meta*.json dict; the betting.py defaults (flagged) when a number is missing."""
    out, approx = {}, []
    for key, default in (('spread_std', DEFAULT_SPREAD_STD), ('total_std', DEFAULT_TOTAL_STD)):
        v = (meta or {}).get(key)
        if v is None or not np.isfinite(float(v)) or float(v) <= 0:
            out[key], _ = float(default), approx.append(key)
        else:
            out[key] = float(v)
    out['approximate'] = approx
    return out


# ----------------------------------------------------------------------------------------------
# Grading

def grade(market, side, line, home_score, away_score):
    """'W' / 'L' / 'P' (push) of a bet at `line` (home spread for the spread, the total for the total; unused for the moneyline), None if the bet or the score is missing."""
    if side is None or (isinstance(side, float) and np.isnan(side)) or pd.isna(home_score) or pd.isna(away_score):
        return None
    hs, as_ = float(home_score), float(away_score)
    if market == 'ml':
        d = hs - as_ if side == 'home' else as_ - hs
    elif market == 'spread':
        if pd.isna(line):
            return None
        d = (hs + float(line) - as_) if side == 'home' else (as_ - hs - float(line))
    else:
        if pd.isna(line):
            return None
        d = (hs + as_ - float(line)) if side == 'over' else (float(line) - hs - as_)
    return 'W' if d > 0 else ('L' if d < 0 else 'P')


def bet_profit(result, price, stake=100):
    """Profit of a graded bet in dollars on `stake`: the payout at the bet's OWN price if it won, -stake if it lost, 0 for a push. A missing price is -110."""
    if result is None:
        return 0.0
    if result == 'P':
        return 0.0
    if result == 'L':
        return -float(stake)
    return calculate_payout(STANDARD_PRICE if pd.isna(price) else price, stake)


def points_to_prob_threshold(points_edge, sigma, line=None):
    """The what-if slider is in points. The probability edge those points stand for at -110: on a whole-number line a push is possible, so the
    same number of points is a slightly different probability than on a half-point line (team.betting.cover_probability). `line=None` = half-point formula."""
    return implied_prob_edge(points_edge, sigma, STANDARD_PRICE, line=line)


def _devig_home(p_home_ml, p_away_ml):
    ph, pa = calculate_implied_prob(p_home_ml), calculate_implied_prob(p_away_ml)
    return ph / (ph + pa)


def clv(market, side, line, pa, pb, close):
    """Closing-line value of a bet, positive = the number moved toward the bet after it was taken. Spread and total: points (a better line than the close).
    Moneyline: probability points of the side (close fair probability minus the fair probability of the prices bet at; both de-vigged). NaN if unknown."""
    if side is None or pd.isna(close):
        return np.nan
    if market == 'spread':
        return float(line - close) if side == 'home' else float(close - line)
    if market == 'total':
        return float(close - line) if side == 'over' else float(line - close)
    if pd.isna(pa) or pd.isna(pb):
        return np.nan
    fair_bet = _devig_home(pa, pb)
    return float(100 * ((close - fair_bet) if side == 'home' else ((1 - close) - (1 - fair_bet))))


# ----------------------------------------------------------------------------------------------
# The recommendation of each game: first run that had a line

def first_line_runs(log):
    """
    {market: DataFrame indexed by game_id} of the bet_recommendations_log rows that count as THE recommendation: for each game and market,
    the row of the earliest run (smallest run_ts_utc) at which the market had a current line (and for the moneyline both prices).
    A game that was re-run later keeps its first recommendation, whether that was a bet or no bet. Empty frames when the log is empty or absent.
    """
    out = {m: pd.DataFrame() for m in MARKETS}
    if log is None or len(log) == 0 or 'game_id' not in log.columns:
        return out
    d = log.copy()
    d['game_id'] = canonical_ids(d['game_id']).to_numpy()
    d = d.sort_values(['game_id', 'run_ts_utc'], kind='stable')
    for m in MARKETS:
        cols = [c for c in _LOG_SPEC[m]['has'] if c in d.columns]
        if len(cols) < len(_LOG_SPEC[m]['has']):
            continue
        have = d[d[cols].notna().all(axis=1)]
        out[m] = have.drop_duplicates('game_id', keep='first').set_index('game_id')
    return out


def _logged_basis(first, m, gids):
    """The wide basis columns (BASIS_FIELDS) of market m for the games in `gids` that have a first-run row; NaN rows for the others."""
    spec = _LOG_SPEC[m]
    idx = pd.Index(gids)
    rows = first[m].reindex(idx) if len(first[m]) else pd.DataFrame(index=idx)
    get = lambda c: pd.to_numeric(rows[c], errors='coerce').to_numpy(float) if c and c in rows.columns else np.full(len(idx), np.nan)
    nan = np.full(len(idx), np.nan)
    prov = rows['prov_' + m].to_numpy(object) if 'prov_' + m in rows.columns else np.full(len(idx), None, dtype=object)
    present = rows['run_ts_utc'].notna().to_numpy() if 'run_ts_utc' in rows.columns else np.zeros(len(idx), bool)
    src = np.where(present, np.where(pd.isna(prov), 'run', prov), None)
    side = rows[f'{m}_bet'].to_numpy(object) if f'{m}_bet' in rows.columns else np.full(len(idx), None, dtype=object)
    return pd.DataFrame({
        'side': np.where(pd.isna(side), None, side), 'edge': get(f'{m}_edge'), 'line': get(spec['line']), 'pred': get(spec['pred']),
        'sigma': get(spec['sigma']), 'pa': get(spec['pa']), 'pb': get(spec['pb']), 'own_view': get(f'{m}_own_view'),
        'line_move': get(f'{m}_line_move'), 'run_ts': rows['run_ts_utc'].to_numpy(object) if 'run_ts_utc' in rows.columns else np.full(len(idx), None, dtype=object),
        'src': src, 'close_line': get(spec['close'])}, index=idx)


def _page_basis(df, m, sigmas):
    """What the live rule says about each row of the predictions frame (current line, stored prediction, deployed sigma), for games with no logged row."""
    spec = _PRED_SPEC[m]
    col = lambda c: pd.to_numeric(df[c], errors='coerce').to_numpy(float) if c and c in df.columns else np.full(len(df), np.nan)
    sigma = {'ml': np.nan, 'spread': sigmas['spread_std'], 'total': sigmas['total_std']}[m]
    b = pd.DataFrame({'side': None, 'edge': np.nan, 'line': col(spec['line']), 'pred': col(spec['pred']), 'sigma': sigma, 'pa': col(spec['pa']),
                      'pb': col(spec['pb']), 'own_view': np.nan, 'line_move': np.nan, 'run_ts': None, 'src': 'page', 'close_line': np.nan}, index=df.index)
    d = _decide_from_basis(b, m, DEFAULT_PROB_EDGE, None)
    b['side'], b['edge'] = d['side'].to_numpy(object), d['edge'].to_numpy(float)
    return b


def attach_recommendations(df, log=None, sigmas=None):
    """
    `df` (the dashboard's predictions frame, one row per game) plus rec_<market>_<field> columns: the recommendation of each game and market
    (see the module docstring for where it comes from). `log` = bet_recommendations_log rows (+ prov_* and close_* columns when the export has them) or None.
    """
    sigmas = sigmas or sigmas_from_meta(None)
    out = df.copy()
    gids = canonical_ids(out['game_id']).to_numpy()
    first = first_line_runs(log)
    for m in MARKETS:
        logged = _logged_basis(first, m, gids)
        logged.index = out.index
        page = _page_basis(out, m, sigmas)
        has = logged['src'].notna()
        basis = page.copy()
        basis.loc[has] = logged.loc[has]
        # a logged row without its own sigma (a NULL) still needs one for the what-if
        if m != 'ml':
            basis['sigma'] = basis['sigma'].fillna(sigmas['spread_std'] if m == 'spread' else sigmas['total_std'])
        for f in BASIS_FIELDS:
            out[f'rec_{m}_{f}'] = basis[f].to_numpy(object) if f in ('side', 'run_ts', 'src') else basis[f].to_numpy(float)
    return out


# ----------------------------------------------------------------------------------------------
# Decisions and results

def market_basis(df, m):
    """The rec_<m>_* columns of `df` under their short names."""
    return pd.DataFrame({f: df[f'rec_{m}_{f}'] for f in BASIS_FIELDS}, index=df.index)


def _decide_from_basis(b, m, threshold, points_edge):
    """The live rule applied to a basis frame: per row (side or None, edge of the better side). `points_edge` (what-if slider) overrides `threshold` with a per-game
    probability threshold that knows whether the row's line is a whole number."""
    sides, edges = [], []
    for r in b.itertuples(index=False):
        if m == 'ml':
            if pd.isna(r.pred) or pd.isna(r.pa) or pd.isna(r.pb):
                sides.append(None); edges.append(0.0); continue
            he, ae = ml_side_edges(r.pred, r.pa, r.pb)
            side, edge = ('home', he) if he >= ae else ('away', ae)
            sides.append(side if edge > threshold else None); edges.append(edge)
            continue
        thr = threshold if points_edge is None or pd.isna(r.sigma) or pd.isna(r.line) else points_to_prob_threshold(points_edge, r.sigma, r.line)
        if pd.isna(r.pred) or pd.isna(r.line) or pd.isna(r.sigma):
            sides.append(None); edges.append(0.0); continue
        pa = None if pd.isna(r.pa) else r.pa
        pb = None if pd.isna(r.pb) else r.pb
        if m == 'spread':
            side, edge = get_spread_bet(r.pred, r.line, r.sigma, thr, home_price=pa, away_price=pb)
        else:
            side, edge = get_total_bet(r.pred, r.line, r.sigma, thr, over_price=pa, under_price=pb)
        sides.append(side); edges.append(edge)
    return pd.DataFrame({'side': sides, 'edge': edges}, index=b.index)


def decide(df, m, threshold=DEFAULT_PROB_EDGE, points_edge=None, recorded=True):
    """
    The bets of market `m` for every row of `df` (a frame with the rec_* columns): DataFrame (index of df) with side (None = no bet), edge, line, price, price_assumed,
    src, run_ts, own_view, line_move, clv.
    recorded=True: the recommendations as written (logged, or recomputed with the live rule at 5%). recorded=False: the what-if, the same rule with another
    minimum edge (`threshold`, probability) or, for spread / total, `points_edge` (points, converted per game with its line and sigma), on each game's recorded
    line, prediction, sigma and prices.
    """
    b = market_basis(df, m)
    if not recorded:
        d = _decide_from_basis(b, m, threshold, points_edge)
        b['side'], b['edge'] = d['side'].to_numpy(object), d['edge'].to_numpy(float)
    side = b['side'].astype(object).where(b['side'].notna(), None)
    has_bet = side.notna().to_numpy()
    price = np.where(side == SIDES[m][0], b['pa'], b['pb']).astype(float)
    price = np.where(has_bet, price, np.nan)
    assumed = has_bet & np.isnan(price)
    if m != 'ml':
        price = np.where(assumed, STANDARD_PRICE, price)           # spread / total prices are stored only from the run that logs them: -110 otherwise
    out = pd.DataFrame({'side': side, 'edge': b['edge'], 'line': b['line'], 'price': price, 'price_assumed': assumed, 'src': b['src'],
                        'run_ts': b['run_ts'], 'own_view': b['own_view'], 'line_move': b['line_move']}, index=df.index)
    out['clv'] = [clv(m, s, ln, pa, pb, c) if s is not None and not pd.isna(s) else np.nan
                  for s, ln, pa, pb, c in zip(out['side'], b['line'], b['pa'], b['pb'], b['close_line'])]
    return out


def settle(df, m, decisions):
    """Series (index of df) of 'W' / 'L' / 'P' / None for the bets in `decisions` (the result of decide) against the scores in `df`."""
    return pd.Series([grade(m, s if isinstance(s, str) else None, ln, hs, as_) for s, ln, hs, as_ in
                      zip(decisions['side'], decisions['line'], df['home_score'], df['away_score'])], index=df.index, dtype=object)


# ----------------------------------------------------------------------------------------------
# Record summaries

def summarize(bets):
    """
    Record of a frame of settled bets with columns result ('W' / 'L' / 'P'), profit (dollars on a $100 stake), clv, src:
    bets, wins, losses, pushes, units (profit / 100), roi (profit / stake of the bets that did not push, %), mean_clv, n_clv, and the number of bets from each source.
    """
    if bets is None or len(bets) == 0:
        return dict(bets=0, wins=0, losses=0, pushes=0, units=0.0, roi=0.0, mean_clv=np.nan, n_clv=0, by_src={k: 0 for k in SOURCE_LABELS})
    w, l, p = (int((bets['result'] == k).sum()) for k in ('W', 'L', 'P'))
    profit = float(bets['profit'].sum())
    c = pd.to_numeric(bets['clv'], errors='coerce')
    return dict(bets=len(bets), wins=w, losses=l, pushes=p, units=profit / 100, roi=(profit / (100 * (w + l)) * 100) if (w + l) else 0.0,
                mean_clv=float(c.mean()) if c.notna().any() else np.nan, n_clv=int(c.notna().sum()),
                by_src={k: int((bets['src'] == k).sum()) for k in SOURCE_LABELS})


def record_text(rec):
    """'7-11' or '11-8-1' (pushes only when there are some)."""
    return f"{rec['wins']}-{rec['losses']}" + (f"-{rec['pushes']}" if rec['pushes'] else '')


def settled_bets(df, m, decisions=None):
    """The settled bets of market m in `df` at the recorded decisions: one row per bet with the game, side, line, price, result, profit, clv, src."""
    dec = decide(df, m) if decisions is None else decisions
    res = settle(df, m, dec)
    ok = dec['side'].notna() & res.notna()
    b = dec[ok].copy()
    b['result'] = res[ok]
    b['profit'] = [bet_profit(r, p) for r, p in zip(b['result'], b['price'])]
    for c in ('game_id', 'game_date', 'matchup'):
        if c in df.columns:
            b[c] = df.loc[b.index, c]
    return b


def source_sentence(by_src):
    """'12 logged by the live run, 40 recomputed by this page' from summarize()['by_src'] (zero entries left out)."""
    parts = [f"{n} {SOURCE_LABELS[k]}" for k, n in by_src.items() if n]
    return ', '.join(parts) if parts else 'no bets'


def logged_share(by_src):
    """(bets written down, bets recomputed) from summarize()['by_src']."""
    logged = sum(by_src.get(k, 0) for k in LOGGED_SOURCES)
    return logged, sum(by_src.values()) - logged


# ----------------------------------------------------------------------------------------------
# Labels for the schedule table

def bet_label(m, side, line, price, edge, home=None, away=None, assumed=False):
    """'BET HOME -3.5 @ -110 (edge 6.2%)' for a recommended bet; '' when there is none."""
    if side is None or pd.isna(side):
        return ''
    p = '' if pd.isna(price) else f" @ {price:+.0f}{'*' if assumed else ''}"
    e = '' if pd.isna(edge) else f" (edge {100 * edge:.1f}%)"
    if m == 'ml':
        team = {'home': home, 'away': away}[side]
        return f"BET {team or side.upper()}{p}{e}"
    if m == 'spread':
        ln = line if side == 'home' else -line
        team = {'home': home, 'away': away}[side]
        return f"BET {team or side.upper()} {ln:+g}{p}{e}"
    return f"BET {side.upper()} {line:g}{p}{e}"


def override_pick_sides(games):
    """`games` as made by live_scores.pick_sides, with ml_side / spread_side / total_side replaced by the RECOMMENDED bets (None = no bet) and the line of the
    bet used where there is one. Needs the rec_* columns."""
    g = games.copy()
    for m, col in (('ml', 'ml_side'), ('spread', 'spread_side'), ('total', 'total_side')):
        dec = decide(g, m)
        g[col] = dec['side'].where(dec['side'].notna(), None)
        if m == 'spread':
            g['odds_home_spread'] = np.where(dec['side'].notna(), dec['line'], g['odds_home_spread'])
        elif m == 'total':
            g['odds_total'] = np.where(dec['side'].notna(), dec['line'], g['odds_total'])
    return g


# ----------------------------------------------------------------------------------------------
# Reading the log (local database) and the deployed sigmas; used by the app and by scripts/export_dashboard_data.py

LOG_TABLE = 'bet_recommendations_log'                 # core/prediction_logger.BET_LOG_TABLE (not imported: the hosted app has no use for that module)
PROVENANCE_TABLE = 'bet_recommendations_provenance'   # scripts/maintenance/backfill_bet_recommendations.py
PROVENANCE_COLUMNS = ['prov_ml', 'prov_spread', 'prov_total', 'open_source', 'note']
REC_FILE_NAME = 'bet_recommendations.csv'
RECORD_META_FILE_NAME = 'record_meta.json'


def read_log_from_db(conn):
    """bet_recommendations_log of an open sqlite connection with the provenance columns joined on (game_id, run_ts_utc); an empty frame when the table does not exist yet."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if LOG_TABLE not in tables:
        return pd.DataFrame()
    if PROVENANCE_TABLE in tables:
        cols = ', '.join(f'p.{c} AS {c}' for c in PROVENANCE_COLUMNS)
        sql = f"SELECT r.*, {cols} FROM {LOG_TABLE} r LEFT JOIN {PROVENANCE_TABLE} p ON p.game_id = r.game_id AND p.run_ts_utc = r.run_ts_utc"
    else:
        sql = f"SELECT * FROM {LOG_TABLE}"
    return pd.read_sql_query(sql, conn)


def deployed_meta_path(root):
    """Path of the deployed model set's calibration_meta<suffix>.json in a checkout, found by reading DEPLOYED_SUFFIX out of team/model_registry.py as text
    (importing that module would need joblib, which the hosted app does not have). None if it cannot be found."""
    import re
    from pathlib import Path
    try:
        text = (Path(root) / 'team' / 'model_registry.py').read_text(encoding='utf-8')
        suffix = re.search(r"^DEPLOYED_SUFFIX\s*=\s*['\"]([^'\"]+)['\"]", text, re.M).group(1)
        path = Path(root) / 'models' / 'saved' / f'calibration_meta{suffix}.json'
        return path if path.exists() else None
    except Exception:
        return None
