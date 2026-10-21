"""
Why does the model's number differ from the betting line? Exact SHAP breakdowns for the spread and total.

Every breakdown starts at the CURRENT line (what a bet is graded against) and ends at the model's number, so the pieces add up to
the edge shown on the dashboard:

    total   current line -> line movement since the open (open - current)
                         -> how much the model trusts the opening line (baseline + the line's own contribution - open)
                         -> one step per feature group (feature_groups.py) -> model total
    spread  current line (as home margin = -home spread) -> line movement since the open
                         -> baseline (the residual model's average miss vs the opening line)
                         -> one step per feature group -> model spread (home margin)

LightGBM's own `predict(X, pred_contrib=True)` gives exact per-feature contributions (they sum to the raw prediction, no extra
package, milliseconds per game). Contributions add, so summing them per group keeps the identity exact.

explain_markets() returns one long table (see COLUMNS); scripts/run_inference_v2.py stores it, scripts/export_dashboard_data.py
publishes it and dashboard/app.py draws it. This module imports only numpy/pandas so the dashboard can use the readers below.
"""
import logging
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from team.feature_groups import GROUP_ORDER, describe_feature

SPREAD, TOTAL = 'spread', 'total'
MARKETS = (SPREAD, TOTAL)
TABLE = 'prediction_explanations'
COLUMNS = ['game_id', 'prediction_date', 'market', 'kind', 'feature', 'label', 'group', 'value', 'contribution', 'pctile', 'severity']

# kind = 'feature' (one SHAP row per model feature), 'piece' (a step of the breakdown) or 'flag' (a warning / context line)
LINE_MOVE_FLAG_PTS = 3.0          # the line moved more than this since the open
EARLY_SEASON_GAMES = 5            # rolling stats mostly reflect last season up to this game
OFFSEASON_REST_DAYS = 20          # days_rest this large = first game after the offseason
TOLERANCE = 1e-6                  # pieces must add up to the model's prediction within this many points
EDGE_BANDS = [0.0, 2.0, 4.0, 6.0, 8.0, np.inf]
BREAK_EVEN = 0.524                # at -110
_OOS_FILE = re.compile(r'^oos_predictions_(\d{4}-\d{2})\.csv$')
REPORTS_DIR = Path(__file__).resolve().parent.parent / 'reports'


# ------------------------------------------------------------------------------------------------ contributions

def shap_contributions(model, X):
    """(contributions per feature, bias) of a fitted LightGBM sklearn model; each row sums to model.predict(X)."""
    contrib = np.asarray(model.predict(X, pred_contrib=True), float)
    return contrib[:, :-1], contrib[:, -1]


def build_reference(df, features):
    """Sorted values per feature from a slice of the training set, for 'where does this value rank' percentiles."""
    ref = {}
    for f in features:
        if f in df.columns:
            v = pd.to_numeric(df[f], errors='coerce').dropna().to_numpy(float)
            if len(v):
                ref[f] = np.sort(v)
    return ref


def _pctile(reference, feature, value):
    arr = (reference or {}).get(feature)
    if arr is None or not np.isfinite(value):
        return np.nan
    return float(np.searchsorted(arr, value, side='left') + np.searchsorted(arr, value, side='right')) / (2 * len(arr))


# ------------------------------------------------------------------------------------------------ history

def load_oos_history(reports_dir=REPORTS_DIR):
    """
    Every out-of-sample backtest bet's market, |edge vs the opening line| and result, from reports/oos_predictions_<season>.csv
    (the model set trained before each season, scored on it). Pushes are dropped. Empty frame if the files are missing.
    """
    parts = []
    for p in sorted(Path(reports_dir).glob('oos_predictions_*.csv')):
        if not _OOS_FILE.match(p.name):
            continue
        d = pd.read_csv(p)
        done = d['home_score'].notna() & d['away_score'].notna()
        d = d[done]
        margin = d['home_score'] - d['away_score']
        total = d['home_score'] + d['away_score']
        edge_s = d['spread_pred'] + d['odds_home_spread']                # > 0: model likes the home side
        cover = margin + d['odds_home_spread']                           # > 0: home covers
        edge_t = d['total_pred'] - d['odds_total']                       # > 0: model likes the over
        over = total - d['odds_total']                                   # > 0: over hits
        for market, edge, res in ((SPREAD, edge_s, cover), (TOTAL, edge_t, over)):
            keep = edge.notna() & res.notna() & (res != 0) & (edge != 0)
            parts.append(pd.DataFrame({'market': market, 'abs_edge': edge[keep].abs(), 'won': (np.sign(edge[keep]) == np.sign(res[keep])),
                                       'season': p.stem.replace('oos_predictions_', '')}))
    if not parts:
        return pd.DataFrame(columns=['market', 'abs_edge', 'won', 'season'])
    return pd.concat(parts, ignore_index=True)


def history_for_edge(history, market, abs_edge):
    """(win rate, n, band label, first season, last season) of past bets whose |edge| is in the same band as `abs_edge`, or None."""
    if history is None or history.empty or not np.isfinite(abs_edge):
        return None
    for lo, hi in zip(EDGE_BANDS[:-1], EDGE_BANDS[1:]):
        if lo <= abs_edge < hi:
            h = history[(history['market'] == market) & (history['abs_edge'] >= lo) & (history['abs_edge'] < hi)]
            if len(h) < 30:
                return None
            label = f"{lo:g}+ pts" if np.isinf(hi) else f"{lo:g}-{hi:g} pts"
            return float(h['won'].mean()), int(len(h)), label, h['season'].min(), h['season'].max()
    return None


# ------------------------------------------------------------------------------------------------ the breakdown

def _group_sums(features, contributions, skip=()):
    """{group: summed contribution} for one game, excluding the features in `skip`."""
    out = {}
    for f, c in zip(features, contributions):
        if f in skip:
            continue
        g = describe_feature(f, strict=False).group
        out[g] = out.get(g, 0.0) + float(c)
    return out


def _piece(game_id, pdate, market, feature, label, value, contribution=np.nan, group=''):
    return dict(game_id=game_id, prediction_date=pdate, market=market, kind='piece', feature=feature, label=label, group=group,
                value=value, contribution=contribution, pctile=np.nan, severity='')


def _flag(game_id, pdate, market, code, label, value=np.nan, severity='warn'):
    return dict(game_id=game_id, prediction_date=pdate, market=market, kind='flag', feature=code, label=label, group='',
                value=value, contribution=np.nan, pctile=np.nan, severity=severity)


def _market_rows(market, game_id, pdate, features, X, i, contrib, bias, open_line, current_line, model_pred, line_feature, reference):
    """Feature rows and pieces of one game and market. Returns (rows, edge) or ([], nan) if the pieces do not add up."""
    rows = []
    c = contrib[i]
    for f, v, cf in zip(features, X.iloc[i].to_numpy(float), c):
        info = describe_feature(f, strict=False)
        rows.append(dict(game_id=game_id, prediction_date=pdate, market=market, kind='feature', feature=f, label=info.label, group=info.group,
                         value=v, contribution=float(cf), pctile=_pctile(reference, f, v), severity=''))
    if market == TOTAL:
        start, movement = current_line, open_line - current_line
        k = features.index(line_feature) if line_feature in features else None
        trust = bias[i] + (c[k] if k is not None else 0.0) - open_line
        groups = _group_sums(features, c, skip=(line_feature,))
        steps = [('movement', 'Line movement since the open', movement),
                 ('trust', 'How much the model trusts the opening line', trust)]
    else:
        start, movement = -current_line, current_line - open_line              # home margin the market implies, then the move since the open
        groups = _group_sums(features, c)
        steps = [('movement', 'Line movement since the open', movement),
                 ('baseline', "Model's average miss vs the opening line", bias[i])]
    running = start
    rows.append(_piece(game_id, pdate, market, 'line', 'Current line', start))
    for code, label, delta in steps:
        running += delta
        rows.append(_piece(game_id, pdate, market, code, label, running, delta))
    for g in [g for g in GROUP_ORDER if g in groups] + [g for g in groups if g not in GROUP_ORDER]:
        running += groups[g]
        rows.append(_piece(game_id, pdate, market, f'group:{g}', g, running, groups[g], group=g))
    rows.append(_piece(game_id, pdate, market, 'prediction', 'Model', running))
    if not np.isfinite(running) or abs(running - model_pred) > TOLERANCE * max(1.0, abs(model_pred)):
        logging.error(f"Explanation for game {game_id} ({market}) does not add up: pieces {running:.6f} vs model {model_pred:.6f}; dropped")
        return [], np.nan
    return rows, running - start


def _num(g, key):
    v = pd.to_numeric(g.get(key), errors='coerce')
    return float(v) if v is not None and pd.notna(v) else np.nan


def _teams(sides):
    return 'both teams' if len(sides) == 2 else f'the {sides[0]} team'


def _game_flags(game_id, pdate, frame, i, edges, moves, history):
    out = []
    for market, move in moves.items():
        if np.isfinite(move) and abs(move) > LINE_MOVE_FLAG_PTS:
            what = 'total' if market == TOTAL else 'spread'
            out.append(_flag(game_id, pdate, market, 'line_moved',
                             f"The {what} has moved {abs(move):.1f} points since the open. The model was built on the opening line, so part of this "
                             f"edge is the market moving, not the model finding something.", abs(move)))
    g = frame.iloc[i]
    played = [p for p in (_num(g, f'season_game_number_{s}') for s in ('home', 'away')) if np.isfinite(p)]
    if played and min(played) <= EARLY_SEASON_GAMES:
        out.append(_flag(game_id, pdate, 'all', 'early_season',
                         f"Early season: a team is in its game {int(min(played))}, so the rolling stats mostly reflect last season's roster.",
                         min(played)))
    missing = [s for s in ('home', 'away') if _num(g, f'inj_missing_{s}') == 1]
    if missing:
        out.append(_flag(game_id, pdate, 'all', 'injury_missing',
                         f"No injury report found for {_teams(missing)}: the model assumed nobody is out.", len(missing)))
    long_rest = [s for s in ('home', 'away') if _num(g, f'days_rest_{s}') >= OFFSEASON_REST_DAYS]
    if long_rest:
        out.append(_flag(game_id, pdate, 'all', 'offseason_rest',
                         f"Offseason-length rest for {_teams(long_rest)}: rest and form numbers are not comparable to mid-season games.",
                         max(_num(g, f'days_rest_{s}') for s in long_rest)))
    for market, edge in edges.items():
        h = history_for_edge(history, market, edge)
        if h:
            rate, n, band, first, last = h
            out.append(_flag(game_id, pdate, market, 'history',
                             f"Context: {market} edges of {band} vs the opening line won {rate:.1%} of {n} out-of-sample bets "
                             f"({first} to {last}); break-even at -110 is {BREAK_EVEN:.1%}.", rate, severity='info'))
    return out


def explain_markets(bundle, frame, current_lines, align=None, reference=None, history=None, prediction_date=None):
    """
    Explanation rows (COLUMNS) for the spread and total of every row of `frame`.

    frame          the live frame (or a training-set slice) with a gameId column and the opening lines as odds_* columns
    current_lines  DataFrame, one row per frame row (same order): `spread_line` (current home spread) and `total_line`
    align          like predict_markets: align(frame, name, cols) selects a model's columns
    reference      build_reference() output, for the league-wide percentile of each value
    history        load_oos_history() output, for the 'edges this size won X%' context line
    Games or markets without a current line, or whose pieces do not add up to the model's number, get no rows for that market.
    """
    def cols(name, key):
        c = bundle.features[key]
        return align(frame, name, c) if align is not None else frame[c]

    n = len(frame)
    ids = frame['gameId'].astype(str).tolist() if 'gameId' in frame.columns else [str(i) for i in frame.index]
    pdate = prediction_date
    spread_line = pd.to_numeric(pd.Series(np.asarray(current_lines['spread_line']), dtype=float)).to_numpy()
    total_line = pd.to_numeric(pd.Series(np.asarray(current_lines['total_line']), dtype=float)).to_numpy()

    X_sp, X_tot = cols('Spread', 'spread'), cols('Total', 'total')
    open_spread = frame['odds_home_spread'].to_numpy(float) if 'odds_home_spread' in frame.columns else np.full(n, np.nan)
    open_total = frame['odds_total'].to_numpy(float) if 'odds_total' in frame.columns else np.full(n, np.nan)

    sp_model = getattr(bundle.spread, 'model', bundle.spread)           # ResidualSpreadModel wraps the LightGBM regressor
    sp_contrib, sp_bias = shap_contributions(sp_model, X_sp)
    if not hasattr(bundle.spread, 'predict_residual'):
        raise NotImplementedError('explanations need the market-aware (residual) spread model; this model set has a plain regressor')
    sp_pred = bundle.spread.predict(X_sp, line=open_spread)
    tot_model = getattr(bundle.total, 'model', bundle.total)
    tot_contrib, tot_bias = shap_contributions(tot_model, X_tot)
    if hasattr(bundle.total, 'predict_residual'):
        raise NotImplementedError('explanations for the residual total model are not built (it is not the deployed total model)')
    tot_pred = bundle.total.predict(X_tot)

    sp_feats, tot_feats = list(X_sp.columns), list(X_tot.columns)
    rows = []
    for i in range(n):
        edges, moves, per_game = {}, {}, []
        if np.isfinite(spread_line[i]) and np.isfinite(open_spread[i]) and np.isfinite(sp_pred[i]):
            r, edge = _market_rows(SPREAD, ids[i], pdate, sp_feats, X_sp, i, sp_contrib, sp_bias, open_spread[i], spread_line[i], sp_pred[i],
                                   None, reference)
            if r:
                per_game += r
                edges[SPREAD] = abs(sp_pred[i] + open_spread[i])             # history bands use the edge vs the OPENING line
                moves[SPREAD] = spread_line[i] - open_spread[i]
        if np.isfinite(total_line[i]) and np.isfinite(open_total[i]) and np.isfinite(tot_pred[i]):
            r, edge = _market_rows(TOTAL, ids[i], pdate, tot_feats, X_tot, i, tot_contrib, tot_bias, open_total[i], total_line[i], tot_pred[i],
                                   'odds_total', reference)
            if r:
                per_game += r
                edges[TOTAL] = abs(tot_pred[i] - open_total[i])
                moves[TOTAL] = total_line[i] - open_total[i]
        if per_game:
            rows += per_game + _game_flags(ids[i], pdate, frame, i, edges, moves, history)
    return pd.DataFrame(rows, columns=COLUMNS)


# ------------------------------------------------------------------------------------------------ storage

def normalize_game_ids(expl):
    """Canonical 10-character NBA game ids, as in predictions_log_v2 (a CSV round trip drops the leading zeros)."""
    out = expl.copy()
    out['game_id'] = out['game_id'].astype(str).map(lambda g: g.zfill(10) if g.isdigit() else g)
    return out


def save_explanations(db_path, expl):
    """Replace the stored rows of the games in `expl` (a re-run refreshes them) and add the new ones. Other games' rows stay."""
    if expl.empty:
        return 0
    expl = normalize_game_ids(expl).reindex(columns=COLUMNS)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} (game_id TEXT, prediction_date TEXT, market TEXT, kind TEXT, feature TEXT, label TEXT, "
                     f"\"group\" TEXT, value REAL, contribution REAL, pctile REAL, severity TEXT)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_game ON {TABLE} (game_id)")
        conn.executemany(f"DELETE FROM {TABLE} WHERE game_id = ?", [(g,) for g in expl['game_id'].unique()])
        expl.to_sql(TABLE, conn, if_exists='append', index=False)
    return len(expl)


def load_explanations(db_path):
    """All stored explanation rows; an empty frame (with COLUMNS) if the table does not exist yet."""
    try:
        with sqlite3.connect(db_path) as conn:
            return normalize_game_ids(pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)).reindex(columns=COLUMNS)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame(columns=COLUMNS)


# ------------------------------------------------------------------------------------------------ readers (dashboard)

def game_pieces(expl, game_id, market):
    """The breakdown steps of one game and market, in display order: line, movement, trust/baseline, groups by size, model."""
    p = expl[(expl['game_id'].astype(str) == str(game_id)) & (expl['market'] == market) & (expl['kind'] == 'piece')]
    if p.empty:
        return p
    head = p[p['feature'].isin(['line', 'movement', 'trust', 'baseline'])]
    head = head.set_index('feature').loc[[c for c in ['line', 'movement', 'trust', 'baseline'] if c in set(head['feature'])]].reset_index()
    groups = p[p['feature'].str.startswith('group:')]
    groups = groups.reindex(groups['contribution'].abs().sort_values(ascending=False).index)
    return pd.concat([head, groups, p[p['feature'] == 'prediction']], ignore_index=True)


def game_flags(expl, game_id, market):
    f = expl[(expl['game_id'].astype(str) == str(game_id)) & (expl['kind'] == 'flag') & expl['market'].isin([market, 'all'])]
    return f.sort_values('severity', ascending=False)      # 'warn' before 'info'


def game_features(expl, game_id, market):
    f = expl[(expl['game_id'].astype(str) == str(game_id)) & (expl['market'] == market) & (expl['kind'] == 'feature')]
    return f.reindex(f['contribution'].abs().sort_values(ascending=False).index)


def edge_of(pieces):
    """Model number minus current line, from game_pieces output (what the bars add up to)."""
    if pieces.empty:
        return np.nan
    return float(pieces.loc[pieces['feature'] == 'prediction', 'value'].iloc[0] - pieces.loc[pieces['feature'] == 'line', 'value'].iloc[0])


def chart_steps(pieces, top=5):
    """
    Bars of the breakdown chart from game_pieces output: [(label, amount, measure)] where measure is 'absolute' (the line, the model) or
    'relative' (a step). The `top` biggest groups are shown, the rest are summed into 'Everything else', so the steps add up to the edge.
    """
    if pieces.empty:
        return []
    steps = [(r.label, float(r.value), 'absolute') for r in pieces[pieces['feature'] == 'line'].itertuples()]
    steps += [(r.label, float(r.contribution), 'relative') for r in pieces[pieces['feature'].isin(['movement', 'trust', 'baseline'])].itertuples()]
    groups = pieces[pieces['feature'].str.startswith('group:')]
    steps += [(r.label, float(r.contribution), 'relative') for r in groups.head(top).itertuples()]
    rest = groups.iloc[top:]
    if len(rest):
        steps.append((f'Everything else ({len(rest)} groups)' if len(rest) > 1 else f'Everything else ({rest.iloc[0].label})',
                      float(rest['contribution'].sum()), 'relative'))
    steps += [(r.label, float(r.value), 'absolute') for r in pieces[pieces['feature'] == 'prediction'].itertuples()]
    return steps


LEAN_STRONG_PTS = 3.0      # |model - line| at or above this reads "Lean", below it "Slight lean"


def lean_text(edge):
    return 'Lean' if abs(edge) >= LEAN_STRONG_PTS else 'Slight lean'


def summary_sentence(pieces, market, home_team=None, away_team=None):
    """
    Markdown headline: the side the model likes with a lean strength, the model number vs the current line, and how much of the gap is
    the line moving since the open. Total: 'Lean OVER'. Spread: 'Lean <team> <spread>' (the team the model has covering).
    """
    if pieces.empty:
        return ''
    line = float(pieces.loc[pieces['feature'] == 'line', 'value'].iloc[0])
    model = float(pieces.loc[pieces['feature'] == 'prediction', 'value'].iloc[0])
    edge = model - line
    move = pieces.loc[pieces['feature'] == 'movement', 'contribution']
    move = float(move.iloc[0]) if len(move) else 0.0
    home, away = home_team or 'Home team', away_team or 'Away team'
    if market == TOTAL:
        side = 'OVER' if edge > 0 else 'UNDER'
        head = f"**{lean_text(edge)} {side} {line:.1f}.** Model total {model:.1f} vs line {line:.1f} ({edge:+.1f})."
    else:
        def margin_text(team, other, m):
            return f"{team} win by {m:.1f}" if m > 0 else (f"{other} win by {-m:.1f}" if m < 0 else "a tie")
        team, spread = (home, -line) if edge > 0 else (away, line)           # line = home margin the market implies = -home spread
        head = (f"**{lean_text(edge)} {team} {spread:+.1f}.** Model: {margin_text(home, away, model)}. "
                f"Market: {margin_text(home, away, line)}. That is {abs(edge):.1f} points toward {team}.")
    if abs(move) < 0.05:
        return head + " The line has not moved since the open."
    same = np.sign(move) == np.sign(edge)
    return head + (f" {abs(move):.1f} of that is the line moving since the open." if same
                   else f" The line moving since the open works against it by {abs(move):.1f}.")
