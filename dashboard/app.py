import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import textwrap

# Add root to sys.path to allow importing models
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from team.betting import (calculate_implied_prob, calculate_bet_ev,
                            settle_bet, STANDARD_PRICE, get_spread_bet, get_total_bet, points_threshold,
                            DEFAULT_SPREAD_STD, DEFAULT_TOTAL_STD, DEFAULT_PROB_EDGE, implied_prob_edge,
                            cover_probability)
from core.odds_rerun import STATUS_FILE_NAME, read_status, rerun_message
from core.season_utils import date_bounds, nba_today, season_date_range, season_for_date, season_options
from team.explain import (COLUMNS as EXPLANATION_COLUMNS, MONEYLINE, SPREAD, TOTAL, chart_steps, driver_stats, game_features, game_flags,
                          game_pieces, load_explanations, normalize_game_ids, outcome_sentence, summary_sentence)
from core import live_feed
from dashboard import live_scores, bet_record

# Set page config
st.set_page_config(
    page_title="NBA Model Dashboard",
    page_icon="🏀",
    layout="wide"
)

# Helper for formatted metrics
def display_subgroup_metrics(df, subgroup_col, result_col, title, group_order=None):
    """
    Displays metrics for unique values in subgroup_col.
    df: DataFrame with bet results
    subgroup_col: Column name defining the group (e.g. 'bet_subgroup')
    result_col: Column containing bet profit/loss (e.g. 'bet_result' or calculated profit)
    group_order: optional explicit list of group labels to show (in that order)
    """
    if subgroup_col not in df.columns or df.empty:
        return

    if group_order is not None:
        groups = [g for g in group_order if g in df[subgroup_col].values]
    else:
        groups = df[subgroup_col].unique()
        groups = [g for g in groups if pd.notna(g) and g != '']
        groups = sorted(groups)
        # Custom sort for specific terms
        if 'Favorite' in groups and 'Underdog' in groups:
            groups = ['Favorite', 'Underdog']
        if 'Home' in groups and 'Away' in groups:
            groups = ['Home', 'Away']
        if 'Over' in groups and 'Under' in groups:
            groups = ['Over', 'Under']

    if not groups:
        st.write("-")
        return

    cols = st.columns(len(groups))
    
    for i, group in enumerate(groups):
        sub_df = df[df[subgroup_col] == group]
        if sub_df.empty:
            cols[i].metric(f"{title}: {group}", "0 bets", "–")
            continue
            
        bets = len(sub_df)
        # Case A: Simulation DF (has 'bet_result' as float profit)
        if df[result_col].dtype.kind in 'fc': # float or complex
             profit = sub_df[result_col].sum()
             roi = (profit / (bets * 100)) * 100
             wins = len(sub_df[sub_df[result_col] > 0])
             losses = len(sub_df[sub_df[result_col] < 0])
             pushes = len(sub_df[sub_df[result_col] == 0])
        else:
             st.error("Invalid column type for metrics")
             return

        cols[i].metric(f"{group}", f"{wins}-{losses}-{pushes}", f"{roi:.1f}%")


# Constants
# NBA_DASHBOARD_DB overrides the DB path (e.g. point it at a missing file to test the hosted data path)
DB_PATH = Path(os.environ.get("NBA_DASHBOARD_DB", Path(__file__).resolve().parent.parent / "data" / "nba_stats.db"))
# Published daily by scripts/export_dashboard_data.py --publish (used when the DB is not available,
# e.g. on Streamlit Community Cloud or in a fresh clone)
REMOTE_DATA_URL = "https://raw.githubusercontent.com/jnish23/nba-model-hybrid/dashboard-data/predictions_log.csv"
SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshot" / "predictions_log.csv"
# Why the model likes each game (team/explain.py), published next to the predictions
# NBA_DASHBOARD_DATA_BASE points the published files at another folder or URL (e.g. a local export, to try the hosted path)
DATA_BASE_URL = os.environ.get("NBA_DASHBOARD_DATA_BASE", "https://raw.githubusercontent.com/jnish23/nba-model-hybrid/dashboard-data/")
EXPLANATIONS_URL = DATA_BASE_URL + "explanations.csv"
HISTORY_MANIFEST_URL = DATA_BASE_URL + "explanations_manifest.json"
HISTORY_DIR_URL = DATA_BASE_URL + "explanations_history/"
# How long loaded data is reused. GitHub's raw file CDN holds a file for about 5 minutes itself, so going lower than DATA_TTL gains nothing;
# the sidebar's Refresh button clears these caches at once. The history files (about 3 MB) only change when a game finishes.
DATA_TTL = 300
HISTORY_TTL = 1800
EXPLANATIONS_SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshot" / "explanations.csv"


def _read_predictions():
    """predictions_log_v2 from the local DB, else the published data branch, else the bundled snapshot.
    Returns (DataFrame, source label)."""
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query("SELECT * FROM predictions_log_v2", conn), "local database"
    try:
        return pd.read_csv(REMOTE_DATA_URL, dtype={'game_id': str}), "published data (updated daily)"
    except Exception:
        pass
    if SNAPSHOT_PATH.exists():
        return pd.read_csv(SNAPSHOT_PATH, dtype={'game_id': str}), "bundled snapshot"
    raise FileNotFoundError(f"No data: {DB_PATH}, {REMOTE_DATA_URL} and {SNAPSHOT_PATH} are all unavailable")


# The recorded bets (dashboard/bet_record.py): bet_recommendations_log rows and the deployed model set's sigmas, from the same three places as the
# predictions: the local database, the published data branch, the bundled snapshot. Both are optional: without them the page recomputes the bets.
RECS_URL = DATA_BASE_URL + bet_record.REC_FILE_NAME
RECORD_META_URL = DATA_BASE_URL + bet_record.RECORD_META_FILE_NAME
RECS_SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshot" / bet_record.REC_FILE_NAME
RECORD_META_SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshot" / bet_record.RECORD_META_FILE_NAME


def _read_recommendations():
    """bet_recommendations_log rows (+ provenance and closing-line columns when exported). Returns (DataFrame or None, source label). Never raises:
    None means the page recomputes every bet. With a local database only the local table counts (no table yet = None), like the predictions."""
    try:
        if DB_PATH.exists():
            with sqlite3.connect(DB_PATH) as conn:
                log = bet_record.read_log_from_db(conn)
            return (log, "local database") if len(log) else (None, "none logged yet")
    except Exception:
        return None, "unreadable"
    for src, label in ((RECS_URL, "published data"), (RECS_SNAPSHOT_PATH, "bundled snapshot")):
        try:
            if src == RECS_SNAPSHOT_PATH and not src.exists():
                continue
            log = pd.read_csv(src, dtype={'game_id': str})
            if len(log):
                return log, label
        except Exception:
            continue
    return None, "none logged yet"


def _read_sigmas():
    """The deployed model set's spread / total sigma: {'spread_std', 'total_std', 'approximate'} and where it came from. The repo's own calibration_meta file
    first when the database is local, else the published record_meta.json, the bundled one, then the repo file; betting.py's defaults (flagged) as a last resort."""
    def from_repo():
        path = bet_record.deployed_meta_path(root_path)
        return (json.loads(path.read_text(encoding='utf-8')), f"model files ({path.name})") if path else None

    def from_url():
        from urllib.request import urlopen
        with urlopen(RECORD_META_URL, timeout=10) as f:
            return json.loads(f.read().decode('utf-8')), "published data"

    def from_snapshot():
        return (json.loads(RECORD_META_SNAPSHOT_PATH.read_text(encoding='utf-8')), "bundled snapshot") if RECORD_META_SNAPSHOT_PATH.exists() else None

    order = [from_repo, from_url, from_snapshot] if DB_PATH.exists() else [from_url, from_snapshot, from_repo]
    for read in order:
        try:
            found = read()
        except Exception:
            continue
        if found:
            return bet_record.sigmas_from_meta(found[0]), found[1]
    return bet_record.sigmas_from_meta(None), "defaults in team/betting.py (approximate)"


@st.cache_data(ttl=DATA_TTL)
def load_record_context():
    """The recorded bets' inputs: the log (or None), the sigmas, and where each came from. Never raises."""
    log, log_source = _read_recommendations()
    sigmas, sigma_source = _read_sigmas()
    return {'log': log, 'log_source': log_source, 'sigmas': sigmas, 'sigma_source': sigma_source}


NO_ODDS = "No odds found"
LIVE_REFRESH_S = 30


@st.cache_data(ttl=LIVE_REFRESH_S, show_spinner=False)
def live_scoreboard(game_date):
    """ESPN scoreboard of one ET date (core/live_feed.py), shared by every viewer: at most one request per 30 s."""
    return live_feed.scoreboard(game_date)


@st.fragment(run_every=LIVE_REFRESH_S)
def render_live_scores(games):
    """Score, period and clock of today's started games next to the recommended bets. Display only; nothing shown before the first tip."""
    try:
        board = live_feed.attach_nba_ids(live_scoreboard(str(games['game_date'].iloc[0])), games[['game_id', 'game_date', 'home_team', 'away_team']])
        if board.empty:
            st.caption("Live scores unavailable right now.")
            return
        games_with_sides = bet_record.override_pick_sides(live_scores.pick_sides(games))      # the recommended bets only
        view = live_scores.live_view(games_with_sides, board)
    except Exception as e:                       # a live-score problem must never take the schedule down with it
        live_feed.log.warning('live scores panel: %s: %s', type(e).__name__, e)
        st.caption("Live scores unavailable right now.")
        return
    if view.empty:
        return
    render_day_record(live_scores.day_record(games_with_sides, view))
    st.markdown("**Live scores (live, not final)**")
    st.caption(f"Updates every {LIVE_REFRESH_S} s (last {pd.Timestamp.now(tz='America/New_York'):%H:%M:%S} ET), scores from ESPN. Each recommended bet is shown against the score right now (green / red once it has won / lost): "
               "who leads for the moneyline bet, the current margin against the spread bet, the combined score against the total bet. A market with no recommended bet shows only the score. "
               "Results in the log are graded after the nightly update.")
    st.dataframe(colour_live_scores(view), hide_index=True, width='stretch')


def render_day_record(record):
    """Today's W-L (and pushes) of the recommended ML, Spread and Total bets in the table below, with the bets still open; counts exactly what the table colours green / red."""
    st.markdown("**Today's record (recommended bets)**")
    for col, (name, label) in zip(st.columns(3), (('ML', 'Moneyline'), ('Spread', 'Spread'), ('Total', 'Total'))):
        r = record[name]
        col.metric(label, f"{r['won']}-{r['lost']}" + (f"-{r['push']}" if r['push'] else ''), f"{r['open']} open", delta_color='off')


def colour_live_scores(view):
    """Green / red on the ML, Spread and Total cells whose pick is settled won / lost (dashboard/live_scores.py pick_result); the result columns are not shown."""
    shade = {'won': 'background-color: rgba(46, 160, 67, 0.30)', 'lost': 'background-color: rgba(218, 54, 51, 0.30)'}
    results = view[list(live_scores.RESULT_COLS.values())]
    shown = view.drop(columns=results.columns)
    return shown.style.apply(lambda col: [shade.get(r, '') for r in results[live_scores.RESULT_COLS[col.name]]] if col.name in live_scores.RESULT_COLS else [''] * len(col))


def _read_rerun_status():
    """status.json (core/odds_rerun.py): which games still lack lines and when the next re-run is. From the local data folder, else the
    published data branch; None if there is none."""
    if DB_PATH.exists():
        return read_status(DB_PATH.parent / 'rerun_status.json')
    try:
        from urllib.request import urlopen
        with urlopen(DATA_BASE_URL + STATUS_FILE_NAME, timeout=10) as f:
            return json.loads(f.read().decode('utf-8'))
    except Exception:
        return None


def render_rerun_status(status, game_date, now=None):
    """A line above the table when some of the day's games have no betting lines yet (nothing when all have them)."""
    message = rerun_message(status, game_date, now)
    if message:
        st.info(message)


def _read_explanations():
    """Explanation rows from the local DB, else the published data branch, else the bundled snapshot; an empty frame if none has any."""
    if DB_PATH.exists():
        found = load_explanations(DB_PATH)
        if not found.empty:
            return found
    try:
        return pd.read_csv(EXPLANATIONS_URL, dtype={'game_id': str})
    except Exception:
        pass
    if EXPLANATIONS_SNAPSHOT_PATH.exists():
        return pd.read_csv(EXPLANATIONS_SNAPSHOT_PATH, dtype={'game_id': str})
    return pd.DataFrame(columns=EXPLANATION_COLUMNS)


def _read_explanation_history():
    """Explanations of every explained game (past ones keep their biggest inputs only when published): the local DB, else the published
    monthly files listed in the manifest; an empty frame if there are none."""
    if DB_PATH.exists():
        found = load_explanations(DB_PATH)
        if not found.empty:
            return found
    try:
        from urllib.request import urlopen
        with urlopen(HISTORY_MANIFEST_URL, timeout=20) as f:
            manifest = json.loads(f.read().decode('utf-8'))
        parts = [pd.read_csv(HISTORY_DIR_URL + shard['file'], dtype={'game_id': str}) for shard in manifest['shards']]
        if parts:
            return pd.concat(parts, ignore_index=True)
    except Exception:
        pass
    return pd.DataFrame(columns=EXPLANATION_COLUMNS)


@st.cache_data(ttl=HISTORY_TTL)
def load_explanation_history():
    """All explanation rows with canonical game ids, for settled games and the drivers tab. Never raises."""
    try:
        return normalize_game_ids(_read_explanation_history()).reindex(columns=EXPLANATION_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)


@st.cache_data(ttl=DATA_TTL)
def load_explanation_data():
    """Explanation rows with canonical game ids. Never raises: the panel just says 'No explanation available'."""
    try:
        return normalize_game_ids(_read_explanations()).reindex(columns=EXPLANATION_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)


def explanation_chart(steps, percent=False):
    """Horizontal waterfall from the current line to the model's number (team.explain.chart_steps). percent=True: steps are win-probability points."""
    labels = ['<br>'.join(textwrap.wrap(label, 24)) for label, _, _ in steps]     # wrapped so the bars keep room on a phone
    values = [value for _, value, _ in steps]
    measures = ['total' if m == 'absolute' and i > 0 else m for i, (_, _, m) in enumerate(steps)]
    levels, running = [], 0.0
    for value, measure in zip(values, measures):
        running = value if measure != 'relative' else running + value
        levels.append(running)
    pad = max(1.0, 0.15 * (max(levels) - min(levels)))
    unit = '%' if percent else ''
    fig = go.Figure(go.Waterfall(
        orientation='h', y=labels, x=values, measure=measures,
        text=[f"{v:.1f}{unit}" if m != 'relative' else f"{v:+.1f}" for v, m in zip(values, measures)], textposition='outside', cliponaxis=False,
        increasing={'marker': {'color': '#2E86C1'}}, decreasing={'marker': {'color': '#E67E22'}},
        totals={'marker': {'color': '#7F8C8D'}}, connector={'line': {'color': '#B0B0B0'}}))
    fig.update_xaxes(range=[min(levels) - pad, max(levels) + pad], title='home win probability (%)' if percent else 'points')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=70 + 46 * len(steps), margin=dict(l=0, r=40, t=10, b=0), showlegend=False)
    return fig


LOGGED_TOLERANCE = {TOTAL: 0.06, SPREAD: 0.06, MONEYLINE: 0.005}          # the log rounds today's spread and total to 0.1


def logged_number(game, market):
    key = {TOTAL: 'total_pred', SPREAD: 'spread_pred', MONEYLINE: 'ml_prob'}[market]
    v = game.get(key) if hasattr(game, 'get') else None
    return float(v) if v is not None and pd.notna(v) else np.nan


def render_explanation(expl, game, settled=False):
    """
    The breakdown panel for one game row (Today's Games, or a settled game from the log): a Total, Spread and Moneyline tab.
    settled=True words the start of the breakdown as the line the game had and adds the result.
    """
    gid = str(game['game_id']).zfill(10)
    st.markdown(f"#### Breakdown: {game['matchup']}")
    st.caption(("Starts at the line the game had" if settled else "Starts at today's betting line") + " and ends at the model's number. Each bar is a group of "
               "related inputs: blue pushes the model's number up, orange pushes it down. Moneyline bars are win-probability points for the home team.")
    tabs = st.tabs(["Total", "Spread", "Moneyline"])
    for tab, market in zip(tabs, (TOTAL, SPREAD, MONEYLINE)):
        with tab:
            pieces = game_pieces(expl, gid, market)
            if pieces.empty:
                st.info("No explanation available")
                continue
            st.markdown(summary_sentence(pieces, market, game.get('home_team'), game.get('away_team')))
            if settled:
                result = outcome_sentence(pieces, market, game.get('home_team'), game.get('away_team'), game.get('home_score'), game.get('away_score'))
                if result:
                    st.markdown(result)
            explained = float(pieces.loc[pieces['feature'] == 'prediction', 'value'].iloc[0])
            logged = logged_number(game, market)
            if np.isfinite(logged) and abs(logged - explained) > LOGGED_TOLERANCE[market]:
                fmt = (lambda v: f"{100 * v:.1f}%") if market == MONEYLINE else (lambda v: f"{v:.1f}")
                st.caption(f"Note: the prediction log has {fmt(logged)} for this game. This breakdown re-scores it with the model set that was "
                           f"out-of-sample for its season, which gives {fmt(explained)}, and explains that number. The log is only rewritten when predictions "
                           f"are regenerated.")
            for flag in game_flags(expl, gid, market).itertuples():
                (st.warning if flag.severity == 'warn' else st.info)(flag.label)
            st.plotly_chart(explanation_chart(chart_steps(pieces, scale=100.0 if market == MONEYLINE else 1.0), percent=market == MONEYLINE),
                            width='stretch', key=f"why_{gid}_{market}_{'s' if settled else 't'}")
            with st.expander("Feature detail"):
                st.caption("Every input of the model, biggest push first. The chart sums these by group; the line's own push is shown there as "
                           "'how much the model trusts the opening line'."
                           + (" Moneyline contributions are in log-odds and include the part that flows through the spread forecast." if market == MONEYLINE else "")
                           + (" Past games list the 15 biggest inputs." if settled else ""))
                detail = game_features(expl, gid, market)
                st.dataframe(pd.DataFrame({
                    'Factor': detail['label'], 'Group': detail['group'], 'This game': detail['value'].round(2),
                    'League percentile': (100 * detail['pctile']).round(0),
                    ('Contribution (log-odds)' if market == MONEYLINE else 'Contribution (pts)'): detail['contribution'].round(3 if market == MONEYLINE else 2)}),
                    hide_index=True, width='stretch',
                    column_config={'League percentile': st.column_config.NumberColumn(
                        help="Where this game's value ranks among recent seasons' games (50 = typical, 95 = higher than 95% of them).", format="%d")})


@st.cache_data(ttl=DATA_TTL)
def load_data():
    """Loads prediction data (see _read_predictions for the sources)."""
    try:
        df, source = _read_predictions()
        # Sort by game_date, then by prediction_date so the LATEST prediction is first
        df = df.sort_values(by=['game_date', 'prediction_date'], ascending=[False, False])
        df = df.drop_duplicates(subset=['game_id'], keep='first')

        # Convert date columns to datetime
        df['game_date'] = pd.to_datetime(df['game_date']).dt.date
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        
        # Ensure numeric columns are actually numeric
        numeric_cols = ['home_score', 'away_score', 'ml_prob', 'odds_home_ml', 'odds_away_ml', 'spread_pred', 'total_pred']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # What the live run recommended (rec_<market>_* columns): the logged bet where there is one, else the live rule recomputed
        ctx = load_record_context()
        df = bet_record.attach_recommendations(df, ctx['log'], ctx['sigmas'])
        return df, source
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), "unavailable"

# Helper functions moved to team.betting
# calculate_implied_prob is now imported


def simulate_strategies(df, threshold, bet_type, recorded=True, points_edge=None):
    """
    The settled bets of one market for the completed games of `df`, one row per game with the bet columns (bet_outcome 'Win' / 'Loss' / 'Push' / 'No Bet',
    bet_result = profit in dollars on a $100 stake, subgroup labels, bet_side / bet_line / bet_price / bet_edge / bet_src / bet_clv).
    bet_type: 'Moneyline', 'Spread', 'Totals'.
    recorded=True: the RECOMMENDED bets, as the live run wrote them down (or, for games with no log row, recomputed with the live rule; see dashboard/bet_record.py),
    graded at the logged line and the side's own price, a push as a push. `threshold` is ignored.
    recorded=False: a what-if, the same rule with another minimum edge: `threshold` (probability points) or, for spread / totals, `points_edge` (points, converted per
    game with its own line and sigma, so whole-number lines are right).
    """
    m = bet_record.MARKET_OF_BET_TYPE[bet_type]
    if f'rec_{m}_side' not in df.columns:                        # a frame that did not come through load_data()
        df = bet_record.attach_recommendations(df, None, None)
    sim_df = df[df['status'] == 'COMPLETED'].copy()
    dec = bet_record.decide(sim_df, m, threshold, points_edge, recorded)
    res = bet_record.settle(sim_df, m, dec)
    placed = dec['side'].notna() & res.notna()

    sim_df['bet_result'] = 0.0
    sim_df['bet_subgroup'] = None     # Favorite/Underdog/Over/Under
    sim_df['bet_home_away'] = None    # Home/Away (ML & Spread only)
    sim_df['bet_intersection'] = None # Home Fav / Home Dog / Away Fav / Away Dog
    sim_df['bet_side'] = None
    for col in ('bet_line', 'bet_price', 'bet_edge', 'bet_clv'):
        sim_df[col] = np.nan
    sim_df['bet_src'] = None
    sim_df['bet_price_assumed'] = False
    sim_df['bet_outcome'] = np.where(placed, res.map({'W': 'Win', 'L': 'Loss', 'P': 'Push'}), 'No Bet')
    for idx in sim_df.index[placed]:
        side, price, line = dec.at[idx, 'side'], dec.at[idx, 'price'], dec.at[idx, 'line']
        sim_df.at[idx, 'bet_result'] = bet_record.bet_profit(res[idx], price)
        for col, val in (('bet_side', side), ('bet_line', line), ('bet_price', price), ('bet_edge', dec.at[idx, 'edge']), ('bet_src', dec.at[idx, 'src']),
                         ('bet_clv', dec.at[idx, 'clv']), ('bet_price_assumed', bool(dec.at[idx, 'price_assumed']))):
            sim_df.at[idx, col] = val
        if m == 'total':
            sim_df.at[idx, 'bet_subgroup'] = 'Over' if side == 'over' else 'Under'
            continue
        is_fav = (price < 0) if m == 'ml' else ((line < 0) if side == 'home' else ((-line) < 0))
        label = 'Home' if side == 'home' else 'Away'
        sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if is_fav else 'Underdog'
        sim_df.at[idx, 'bet_home_away'] = label
        sim_df.at[idx, 'bet_intersection'] = f"{label} {'Fav' if is_fav else 'Dog'}"

    # Cumulative ROI
    # Fix: Sort by date ASCENDING to calculate cumulative sum correctly (Past -> Future)
    sim_df = sim_df.sort_values(by='game_date', ascending=True)
    sim_df['cumulative_profit'] = sim_df['bet_result'].cumsum()

    # Sort back to DESCENDING for display (Latest -> Past)
    sim_df = sim_df.sort_values(by='game_date', ascending=False)

    return sim_df


def record_of(sim_df):
    """bet_record.summarize() of the bets of a simulate_strategies() frame (record, units, ROI, mean CLV, where the bets came from)."""
    placed = sim_df[sim_df['bet_outcome'] != 'No Bet']
    return bet_record.summarize(pd.DataFrame({
        'result': placed['bet_outcome'].map({'Win': 'W', 'Loss': 'L', 'Push': 'P'}), 'profit': placed['bet_result'], 'clv': placed['bet_clv'], 'src': placed['bet_src']}))


def _team_model_record(sim_df, team_name, is_totals=False):
    """ (bets, wins, losses, pushes, roi) for model bets involving team_name in a simulate_strategies() result. """
    if is_totals:
        team_bets = sim_df[((sim_df['home_team'] == team_name) | (sim_df['away_team'] == team_name)) & (sim_df['bet_outcome'] != 'No Bet')]
    else:
        team_bets_home = sim_df[(sim_df['home_team'] == team_name) & (sim_df['bet_home_away'] == 'Home')]
        team_bets_away = sim_df[(sim_df['away_team'] == team_name) & (sim_df['bet_home_away'] == 'Away')]
        team_bets = pd.concat([team_bets_home, team_bets_away])

    bets = len(team_bets)
    if bets == 0:
        return 0, 0, 0, 0, 0.0
    wins, losses, pushes = (int((team_bets['bet_outcome'] == k).sum()) for k in ('Win', 'Loss', 'Push'))
    profit = team_bets['bet_result'].sum()
    roi = (profit / ((wins + losses) * 100)) * 100 if wins + losses else 0.0
    return bets, wins, losses, pushes, roi


@st.cache_data(ttl=600)
def build_league_summary(df, threshold, recorded=True):
    """ Builds a single league-wide summary table for all teams with Actual and Model records. recorded=True: the model columns are the recommended bets. """
    teams = sorted(set(df['home_team'].dropna().unique().tolist() + df['away_team'].dropna().unique().tolist()))

    completed = df[df['status'] == 'COMPLETED'].copy()
    for c in ['home_score', 'away_score', 'odds_home_spread', 'odds_total']:
        if c in completed.columns:
            completed[c] = pd.to_numeric(completed[c], errors='coerce')

    sim_ml = simulate_strategies(df, threshold, 'Moneyline', recorded)
    sim_sp = simulate_strategies(df, threshold, 'Spread', recorded)
    sim_tl = simulate_strategies(df, threshold, 'Totals', recorded)

    rows = []
    for team in teams:
        home_games = completed[completed['home_team'] == team]
        away_games = completed[completed['away_team'] == team]
        gp = len(home_games) + len(away_games)
        if gp == 0:
            continue

        home_su_w = (home_games['home_score'] > home_games['away_score']).sum()
        away_su_w = (away_games['away_score'] > away_games['home_score']).sum()
        su_wins = home_su_w + away_su_w

        # Home cover margin (home_score + home line - away_score); pushes (0) and missing lines (NaN) excluded
        home_margin_h = home_games['home_score'] + home_games['odds_home_spread'] - home_games['away_score']
        home_margin_a = away_games['home_score'] + away_games['odds_home_spread'] - away_games['away_score']
        ats_wins = (home_margin_h > 0).sum() + (home_margin_a < 0).sum()
        ats_losses = (home_margin_h < 0).sum() + (home_margin_a > 0).sum()

        total_diff = pd.concat([
            home_games['home_score'] + home_games['away_score'] - home_games['odds_total'],
            away_games['home_score'] + away_games['away_score'] - away_games['odds_total'],
        ])
        over_wins = (total_diff > 0).sum()
        under_wins = (total_diff < 0).sum()

        m_ml = _team_model_record(sim_ml, team, False)
        m_sp = _team_model_record(sim_sp, team, False)
        m_tl = _team_model_record(sim_tl, team, True)
        wl = lambda r: f"{r[1]}-{r[2]}" + (f"-{r[3]}" if r[3] else '')

        rows.append({
            'Team': team,
            'GP': gp,
            'Actual SU': f"{su_wins}-{gp-su_wins}",
            'Actual ATS': f"{ats_wins}-{ats_losses}",
            'Actual O/U': f"{over_wins}-{under_wins}",
            'ML W-L': wl(m_ml),
            'ML ROI': m_ml[4],
            'Spread W-L': wl(m_sp),
            'Spread ROI': m_sp[4],
            'Totals W-L': wl(m_tl),
            'Totals ROI': m_tl[4],
        })

    return pd.DataFrame(rows)


def build_team_breakdown(df, team, bet_type, threshold, recorded=True, points_edge=None):
    """
    Build per-team breakdown tables. The model's bets are the recommended bets (recorded=True) or the what-if at `threshold`.
    Returns (actual_record_df, model_bets_df) — both pandas DataFrames ready for display.
    """
    completed = df[df['status'] == 'COMPLETED'].copy()
    if completed.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Ensure numeric
    for c in ['home_score', 'away_score', 'ml_prob', 'odds_home_ml', 'odds_away_ml',
              'odds_home_spread', 'odds_total', 'spread_pred', 'total_pred']:
        if c in completed.columns:
            completed[c] = pd.to_numeric(completed[c], errors='coerce')

    # Filter to games involving this team
    team_games = completed[
        (completed['home_team'] == team) | (completed['away_team'] == team)
    ].copy()
    if team_games.empty:
        return pd.DataFrame(), pd.DataFrame()

    mkt = bet_record.MARKET_OF_BET_TYPE[bet_type]
    if f'rec_{mkt}_side' not in team_games.columns:
        team_games = bet_record.attach_recommendations(team_games, None, None)
    dec = bet_record.decide(team_games, mkt, threshold, points_edge, recorded)
    res = bet_record.settle(team_games, mkt, dec)

    # --- Build per-game labels from team's perspective ---
    rows = []
    for gidx, g in team_games.iterrows():
        is_home = g['home_team'] == team
        ha = 'Home' if is_home else 'Away'

        team_score = g['home_score'] if is_home else g['away_score']
        opp_score  = g['away_score'] if is_home else g['home_score']
        if pd.isna(team_score) or pd.isna(opp_score):
            continue

        # SU (Straight Up / ML)
        su_win = team_score > opp_score

        # ATS
        spread = g.get('odds_home_spread')  # Home spread line
        if pd.notna(spread):
            # Home covers when home_score + spread > away_score
            home_cover_margin = g['home_score'] + spread - g['away_score']
            if home_cover_margin == 0:
                ats_win = None  # Push: neither a win nor a loss
            else:
                ats_win = (home_cover_margin > 0) if is_home else (home_cover_margin < 0)
            # Fav/Dog from team perspective
            team_spread = spread if is_home else -spread
            fav_dog = 'Favorite' if team_spread < 0 else 'Underdog'
        else:
            ats_win = None
            fav_dog = None

        # O/U
        total_line = g.get('odds_total')
        actual_total = g['home_score'] + g['away_score']
        if pd.notna(total_line) and actual_total != total_line:      # exactly on the line is a push: neither over nor under
            over_hit = actual_total > total_line
        else:
            over_hit = None

        # Intersection
        if fav_dog:
            fav_label = 'Fav' if fav_dog == 'Favorite' else 'Dog'
            intersection = f"{ha} {fav_label}"
        else:
            intersection = None

        # --- Model bet info: the recommended bet of this game (bet_record.decide), graded at its own line and price ---
        model_bet_on_team = None  # Did model bet on THIS team?
        model_bet_won = None
        model_bet_odds = None
        model_bet_push = False

        side = dec.at[gidx, 'side']
        if isinstance(side, str):
            if mkt == 'total':
                model_bet_on_team = True      # Totals are team-agnostic, counts for both
            else:
                model_bet_on_team = (side == 'home') == is_home
            if model_bet_on_team and res[gidx] is not None:
                model_bet_push = res[gidx] == 'P'
                model_bet_won = None if model_bet_push else res[gidx] == 'W'
                model_bet_odds = dec.at[gidx, 'price']

        rows.append({
            'home_away': ha,
            'fav_dog': fav_dog,
            'intersection': intersection,
            'su_win': su_win,
            'ats_win': ats_win,
            'over_hit': over_hit,
            'model_bet_on_team': model_bet_on_team,
            'model_bet_won': model_bet_won,
            'model_bet_odds': model_bet_odds,
            'model_bet_push': model_bet_push,
        })

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    tdf = pd.DataFrame(rows)

    # --- Table 1: Actual Team Record ---
    slices = [
        ('Overall', tdf),
        ('Home', tdf[tdf['home_away'] == 'Home']),
        ('Away', tdf[tdf['home_away'] == 'Away']),
        ('Favorite', tdf[tdf['fav_dog'] == 'Favorite']),
        ('Underdog', tdf[tdf['fav_dog'] == 'Underdog']),
        ('Home Fav', tdf[tdf['intersection'] == 'Home Fav']),
        ('Home Dog', tdf[tdf['intersection'] == 'Home Dog']),
        ('Away Fav', tdf[tdf['intersection'] == 'Away Fav']),
        ('Away Dog', tdf[tdf['intersection'] == 'Away Dog']),
    ]

    actual_rows = []
    for label, sdf in slices:
        n = len(sdf)
        if n == 0:
            actual_rows.append({'Split': label, 'GP': 0, 'SU (ML)': '-', 'ATS': '-', 'O/U (Over)': '-'})
            continue
        su_w = int(sdf['su_win'].sum())
        su_l = n - su_w
        ats_valid = sdf['ats_win'].dropna()
        if not ats_valid.empty:
            ats_w = int(ats_valid.sum())
            ats_l = len(ats_valid) - ats_w
            ats_pct = ats_w / len(ats_valid) * 100
            ats_str = f"{ats_w}-{ats_l} ({ats_pct:.0f}%)"
        else:
            ats_str = '-'
        ou_valid = sdf['over_hit'].dropna()
        if not ou_valid.empty:
            ou_w = int(ou_valid.sum())
            ou_l = len(ou_valid) - ou_w
            ou_pct = ou_w / len(ou_valid) * 100
            ou_str = f"{ou_w}-{ou_l} ({ou_pct:.0f}%)"
        else:
            ou_str = '-'
        actual_rows.append({
            'Split': label,
            'GP': n,
            'SU (ML)': f"{su_w}-{su_l} ({su_w/n*100:.0f}%)",
            'ATS': ats_str,
            'O/U (Over)': ou_str,
        })

    actual_df = pd.DataFrame(actual_rows)

    # --- Table 2: Model Bets on This Team ---
    model_rows = []
    bet_label = bet_type
    for label, sdf in slices:
        bet_on = sdf[sdf['model_bet_on_team'] == True]
        n = len(bet_on)
        if n == 0:
            model_rows.append({'Split': label, 'Bets': 0, 'Record': '-', 'ROI': '-'})
            continue
        pushes = int(bet_on['model_bet_push'].sum())
        valid = bet_on[bet_on['model_bet_won'].notna()]
        if valid.empty:
            model_rows.append({'Split': label, 'Bets': n, 'Record': f"0-0-{pushes}" if pushes else '-', 'ROI': '-'})
            continue
        w = int(valid['model_bet_won'].sum())
        l = len(valid) - w
        # Compute profit
        profit, staked = 0.0, 0
        for _, r in valid.iterrows():
            odds = r['model_bet_odds']
            if pd.isna(odds):
                if bet_type == 'Moneyline':
                    continue   # no moneyline price: not settled (and not staked) rather than assumed -110
                odds = STANDARD_PRICE
            profit += settle_bet(odds, bool(r['model_bet_won']))
            staked += 1
        roi = (profit / (staked * 100)) * 100 if staked else 0.0
        model_rows.append({
            'Split': label,
            'Bets': len(valid) + pushes,
            'Record': f"{w}-{l}" + (f"-{pushes}" if pushes else '') + f" ({w/len(valid)*100:.0f}%)",
            'ROI': f"{roi:+.1f}%",
        })

    model_df = pd.DataFrame(model_rows)

    return actual_df, model_df


def _clv_text(rec, unit):
    """'CLV +0.60 pts (24 bets with a close)' for a bet_record.summarize() dict; says so when there is no closing line in the data."""
    if not rec['n_clv']:
        return "CLV: no closing lines in this data"
    return f"Mean CLV {rec['mean_clv']:+.2f} {unit} ({rec['n_clv']} of {rec['bets']} bets with a close)"


def render_record_metrics(sims, columns):
    """W-L-P, units and ROI of the recommended bets of each market (sims: {bet type: simulate_strategies frame}), with the mean closing-line value under each."""
    for col, (bt, label, unit) in zip(columns, (("Moneyline", "Moneyline", "prob. pts"), ("Spread", "Spread", "pts"), ("Totals", "Totals", "pts"))):
        rec = record_of(sims[bt])
        col.metric(label, f"{bet_record.record_text(rec)}  ·  {rec['units']:+.2f}u", f"{rec['bets']} bets · ROI {rec['roi']:+.1f}%", delta_color='off')
        col.caption(_clv_text(rec, unit))


def render_record_panel(df, ctx, label):
    """The season record: the RECOMMENDED bets of each market (never the what-if sliders), with CLV, and how many of the bets are logged versus recomputed."""
    sims = {bt: simulate_strategies(df, DEFAULT_PROB_EDGE, bt) for bt in ("Moneyline", "Spread", "Totals")}
    st.markdown(f"**Record of the recommended bets** ({label})")
    render_record_metrics(sims, st.columns(3))
    by_src = {k: sum(record_of(s)['by_src'][k] for s in sims.values()) for k in bet_record.SOURCE_LABELS}
    logged, recomputed = bet_record.logged_share(by_src)
    st.caption(f"{logged + recomputed} bets: {bet_record.source_sentence(by_src)}. Each bet is graded at the line and the side's own price it was judged at "
               f"(* a spread or total price that was not stored is taken as -110); a push is a push (W-L-P, no profit or loss). "
               f"Recommendations logged: {ctx['log_source']}; sigmas for recomputed bets: {ctx['sigma_source']}. "
               "CLV = closing line value, positive when the line moved toward the bet after it was taken.")
    if recomputed:
        st.caption("A recomputed bet applies the live rule to the stored prediction and the stored (run-time) line, the deployed model set's sigma and, for spread and totals, "
                   "-110: the live run's own price and per-game total sigma are not stored for those games, so a few of them may differ from what the run actually recommended.")


def recommended_bets_table(sims):
    """One row per recommended settled bet of every market in `sims` ({bet type: simulate_strategies frame}), newest first."""
    rows = []
    for bt, sim in sims.items():
        m = bet_record.MARKET_OF_BET_TYPE[bt]
        for _, r in sim[sim['bet_outcome'] != 'No Bet'].iterrows():
            score = f"{r['away_score']:.0f}-{r['home_score']:.0f}" if pd.notna(r['home_score']) and pd.notna(r['away_score']) else ''
            rows.append({'Date': r['game_date'], 'Game': r['matchup'], 'Market': bt,
                         'Bet': bet_record.bet_label(m, r['bet_side'], r['bet_line'], r['bet_price'], r['bet_edge'], r['home_team'], r['away_team'], bool(r['bet_price_assumed'])),
                         'Final (away-home)': score, 'Result': r['bet_outcome'], 'Units': round(r['bet_result'] / 100, 2),
                         'CLV': None if pd.isna(r['bet_clv']) else round(r['bet_clv'], 2), 'Source': bet_record.SOURCE_LABELS.get(r['bet_src'], r['bet_src'])})
    table = pd.DataFrame(rows, columns=['Date', 'Game', 'Market', 'Bet', 'Final (away-home)', 'Result', 'Units', 'CLV', 'Source'])
    return table.sort_values('Date', ascending=False, kind='stable').reset_index(drop=True)


def main():
    st.title("🏀 NBA Model Dashboard")

    df, source = load_data()
    ctx = load_record_context()

    if df.empty:
        st.warning("No prediction data available.")
        st.stop() # Stop execution if no data

    # Sidebar Filters
    st.sidebar.header("Configuration")
    st.sidebar.caption(f"Data: {source} · {df['game_date'].min()} to {df['game_date'].max()}")
    if 'prediction_date' in df.columns and df['prediction_date'].notna().any():
        st.sidebar.caption(f"Latest prediction run: {df['prediction_date'].max():%Y-%m-%d %H:%M}")
    if st.sidebar.button("🔄 Refresh data", help=f"Reload the data now instead of waiting up to {DATA_TTL // 60} minutes. GitHub itself serves the published files "
                                                  "with a delay of up to about 5 minutes after a publish."):
        st.cache_data.clear()
        st.rerun()
    
    # Season + date filter: choosing a season resets the dates to that season's span (the current season runs from its
    # opening night to today); the dates can then be narrowed further.
    game_dates = [d for d in df['game_date'] if pd.notna(d)] if 'game_date' in df.columns else []
    today = nba_today()
    seasons = season_options(game_dates, today)
    season = st.sidebar.selectbox("Season", seasons, index=0, help="Defaults to the current season. Picking a season sets the start and end dates below; you can then move either date into another season.")
    start_default, end_default, _, _ = season_date_range(game_dates, season, today)
    min_date, max_date = date_bounds(game_dates, today)      # the pickers span ALL seasons, e.g. late 2025-26 into early 2026-27
    start_date = st.sidebar.date_input("Start Date", start_default, min_value=min_date, max_value=max_date, key=f"start_{season}")
    end_date = st.sidebar.date_input("End Date", end_default, min_value=min_date, max_value=max_date, key=f"end_{season}")

    # What-If Analysis
    st.sidebar.subheader("What-If Analysis")
    st.sidebar.caption("At the default settings the tabs show the RECOMMENDED bets (the live rule), which is the season record. Moving a slider switches to a what-if "
                       "with another minimum edge: it is not what the model recommended.")
    sigmas = ctx['sigmas']

    bet_type = st.sidebar.selectbox("Bet Type", ["Moneyline", "Spread", "Totals"])

    # One rule for every market (team/betting.py): bet when the model's win probability beats the price's break-even
    # probability by `threshold`. For the moneyline that is the natural control. For spread/totals a point edge over the line
    # is easier to reason about, so the slider is in points and converted: P(win) = Phi(edge / residual std) at -110, per game with its own line
    # (a whole-number line can push, so the same points are a slightly different probability) and the model set's sigma (not betting.py's defaults).
    if bet_type == "Moneyline":
        threshold = st.sidebar.slider("Minimum edge (probability points)", 0.0, 0.20, DEFAULT_PROB_EDGE, 0.005, format="%.3f",
                                      help="Bet only when the model's win probability exceeds the price's break-even probability by this much "
                                           "(0.05 = 5 percentage points, e.g. the model says 60% and the price implies 55%).")
        points_edge = None
        recorded = bool(np.isclose(threshold, DEFAULT_PROB_EDGE))
        if recorded:
            threshold = DEFAULT_PROB_EDGE
    else:
        std = sigmas['spread_std'] if bet_type == "Spread" else sigmas['total_std']
        line_word = "spread" if bet_type == "Spread" else "total"
        default_pts = round(points_threshold(std, DEFAULT_PROB_EDGE), 1)
        points_edge = st.sidebar.slider(f"Minimum edge (points vs the {line_word})", 0.0, 10.0, default_pts, 0.1,
                                        key=f"pts_{bet_type}",
                                        help=f"Bet only when the model's {line_word} differs from the market line by at least this many points. "
                                             f"{default_pts} points is the model's standard 5% edge rule. The probability shown below the slider is the model's own estimate, "
                                             f"not an observed win rate.")
        recorded = bool(np.isclose(points_edge, default_pts))
        threshold = DEFAULT_PROB_EDGE if recorded else implied_prob_edge(points_edge, std)
        st.sidebar.caption(f"{points_edge:.1f} pts: the model ESTIMATES {cover_probability(points_edge, std):.1%} of such bets win, vs 52.4% to break even at -110 "
                           f"(a {implied_prob_edge(points_edge, std):+.1%} claimed edge, half-point line; a whole-number line is converted per game). That estimate assumes its error is a bell curve "
                           f"with std {std:.1f} pts around its prediction (the deployed model set's: {ctx['sigma_source']}); "
                           f"backtested bets that clear the threshold have won less often than estimated, so read the results tabs for what actually happened.")
    if recorded:
        threshold_label = "the live rule (5% probability edge): the recommended bets"
    else:
        threshold_label = f"what-if, minimum edge {points_edge:.1f} pts ({threshold:+.1%} edge)" if points_edge is not None else f"what-if, minimum edge {threshold:.3f}"

    # Filter Data for Display
    mask = (df['game_date'] >= start_date) & (df['game_date'] <= end_date)
    display_df = df.loc[mask]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 Today's Games", "Recent Performance", "📊 Predictions Log & Analysis", "📋 Team Breakdown", "🏆 League Summary", "🔍 Model Drivers"])

    with tab1:
        st.subheader("Games Schedule")
        today = nba_today()
        todays_games = df[df['game_date'] == today].copy()
        
        display_game_df = pd.DataFrame()
        
        if todays_games.empty:
            # Find closest future games
            future_games = df[df['game_date'] > today].sort_values('game_date')
            if not future_games.empty:
                next_date = future_games['game_date'].iloc[0]
                st.info(f"No games found for Today ({today}). Showing Next Scheduled Games for **{next_date}**.")
                todays_games = future_games[future_games['game_date'] == next_date].copy()
            else:
                 todays_games = pd.DataFrame()
                 st.warning("No games found in database.")

        if not todays_games.empty:
            # Enhance Dataframe for Display
            
            # 1. Implied Odds (Home ML)
            todays_games['Implied Home Prob (%)'] = (100.0 * todays_games['odds_home_ml'].apply(calculate_implied_prob)).round(2)
            todays_games['Model Home Prob (%)'] = (100.0 * todays_games['ml_prob']).round(2)
            todays_games['Implied Away Prob (%)'] = (100.0 * todays_games['odds_away_ml'].apply(calculate_implied_prob)).round(2)
            todays_games['Model Away Prob (%)'] = (100.0 * (1 - todays_games['ml_prob'])).round(2)
            # 2. Bets (the recommended ones, bet_record.decide)
            def bet_column(m, no_line):
                dec = bet_record.decide(todays_games, m)
                labels = []
                for (idx, r), (_, d) in zip(todays_games.iterrows(), dec.iterrows()):
                    if no_line(r): labels.append(NO_ODDS)
                    else: labels.append(bet_record.bet_label(m, d['side'], d['line'], d['price'], d['edge'], r['home_team'], r['away_team'], d['price_assumed']) or "-")
                return labels

            todays_games['ML Bet'] = bet_column('ml', lambda r: pd.isna(r['odds_home_ml']) and pd.isna(r['odds_away_ml']))
            todays_games['Spread Bet'] = bet_column('spread', lambda r: pd.isna(r['odds_home_spread']))
            todays_games['Total Bet'] = bet_column('total', lambda r: pd.isna(r['odds_total']))

            # 3. Clean Columns: one row per game, the away-side and duplicate numbers folded into combined columns
            def pct(x):
                return f"{x:.0f}%" if pd.notna(x) else "-"
            def american(x):
                return f"{x:+.0f}" if pd.notna(x) else "-"
            def moneyline(a, h):
                return NO_ODDS if pd.isna(a) and pd.isna(h) else f"{american(a)} / {american(h)}"
            todays_games = todays_games.reset_index(drop=True)
            table = pd.DataFrame({
                'Why?': False,
                'Away Team': todays_games['away_team'],
                'Home Team': todays_games['home_team'],
                'Home win % (model / market)': [f"{pct(m)} / {pct(i)}" for m, i in zip(todays_games['Model Home Prob (%)'], todays_games['Implied Home Prob (%)'])],
                'Moneyline (away / home)': [moneyline(a, h) for a, h in zip(todays_games['odds_away_ml'], todays_games['odds_home_ml'])],
                'ML Bet': todays_games['ML Bet'],
                'Spread Bet': todays_games['Spread Bet'],
                'Total Bet': todays_games['Total Bet'],
            })
            render_rerun_status(_read_rerun_status(), todays_games['game_date'].iloc[0])
            if todays_games['game_date'].iloc[0] == today:
                render_live_scores(todays_games)
            st.caption(f"Games on {todays_games['game_date'].iloc[0]}. The **Bet** columns are the bets the live run recommended (5% probability edge over the price's break-even, "
                       "judged at the line and price shown; * = -110 assumed because the price is not stored). "
                       "Tick **Why?** on a game to see what moves the model's number away from the line.")
            edited = st.data_editor(table, hide_index=True, width='stretch', key="todays_games_table",
                                    disabled=[c for c in table.columns if c != 'Why?'],
                                    column_config={'Why?': st.column_config.CheckboxColumn('Why?', width='small')})
            st.metric("Total Games", len(todays_games))
            for i in edited.index[edited['Why?']]:
                render_explanation(load_explanation_data(), todays_games.iloc[i])

    with tab2:
        st.subheader("Recommended bets: last 14 days")
        st.caption("Only the bets the live run recommended (5% probability edge over the price's break-even), graded at the line and price they were judged at; a push is a push. "
                   "This is the betting record.")

        # Last 14 days, before today
        today = nba_today()
        previous_two_weeks = today - datetime.timedelta(days=14)
        window_df = display_df[(display_df['game_date'] >= previous_two_weeks) & (display_df['game_date'] < today)]
        sims = {bt: simulate_strategies(window_df, DEFAULT_PROB_EDGE, bt) for bt in ("Moneyline", "Spread", "Totals")}
        bets_by_type = {bt: sim[sim['bet_outcome'] != 'No Bet'] for bt, sim in sims.items()}

        if window_df.empty:
            st.info("No predictions yet this week.")
        else:
            render_record_metrics(sims, st.columns(3))
            if all(b.empty for b in bets_by_type.values()):
                st.info("No recommended bets settled in the last 14 days.")
            else:
                # --- Breakdown Metrics for the window ---
                st.divider()
                st.markdown("##### Performance Breakdown (Favorite vs Underdog / Over vs Under)")
                ml_df_b, sp_df_b, tl_df_b = bets_by_type["Moneyline"], bets_by_type["Spread"], bets_by_type["Totals"]

                # Row 1: Favorite / Underdog & Over / Under
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.caption("Moneyline — Fav vs Dog")
                    if not ml_df_b.empty:
                        display_subgroup_metrics(ml_df_b, 'bet_subgroup', 'bet_result', 'ML')
                    else:
                        st.write("-")
                with b2:
                    st.caption("Spread — Fav vs Dog")
                    if not sp_df_b.empty:
                        display_subgroup_metrics(sp_df_b, 'bet_subgroup', 'bet_result', 'Spread')
                    else:
                        st.write("-")
                with b3:
                    st.caption("Totals — Over vs Under")
                    if not tl_df_b.empty:
                        display_subgroup_metrics(tl_df_b, 'bet_subgroup', 'bet_result', 'Total')
                    else:
                        st.write("-")

                st.divider()

                # Row 2: Home vs Away (ML & Spread only)
                st.markdown("##### Home vs Away Performance")
                h1, h2 = st.columns(2)
                with h1:
                    st.caption("Moneyline — Home vs Away")
                    if not ml_df_b.empty:
                        display_subgroup_metrics(ml_df_b, 'bet_home_away', 'bet_result', 'ML', group_order=['Home', 'Away'])
                    else:
                        st.write("-")
                with h2:
                    st.caption("Spread — Home vs Away")
                    if not sp_df_b.empty:
                        display_subgroup_metrics(sp_df_b, 'bet_home_away', 'bet_result', 'Spread', group_order=['Home', 'Away'])
                    else:
                        st.write("-")

                st.divider()

                # Row 3: 4-way intersection (ML & Spread)
                st.markdown("##### Intersection: Home/Away × Favorite/Underdog")
                ix_order = ['Home Fav', 'Home Dog', 'Away Fav', 'Away Dog']
                i1, i2 = st.columns(2)
                with i1:
                    st.caption("Moneyline")
                    if not ml_df_b.empty:
                        display_subgroup_metrics(ml_df_b, 'bet_intersection', 'bet_result', 'ML', group_order=ix_order)
                    else:
                        st.write("-")
                with i2:
                    st.caption("Spread")
                    if not sp_df_b.empty:
                        display_subgroup_metrics(sp_df_b, 'bet_intersection', 'bet_result', 'Spread', group_order=ix_order)
                    else:
                        st.write("-")

                st.divider()
                st.markdown("##### The bets")
                st.dataframe(recommended_bets_table(sims), hide_index=True, width='stretch')

    with tab3:
        st.subheader(f"Simulation & Analysis: {bet_type}")
        render_record_panel(display_df, ctx, f"{start_date} to {end_date}")

        # Show all data first but collapsed
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(display_df.drop(columns=[c for c in display_df.columns if c.startswith('rec_')]))      # the rec_* columns are the recorded bets' inputs (dashboard/bet_record.py)
        
        st.caption(f"{bet_type} bets, {threshold_label}, from {start_date} to {end_date}. "
                   + ("These are the RECOMMENDED bets: the record above." if recorded else "WHAT-IF: the live rule with another minimum edge, NOT the record of the bets the model recommended."))

        # Reactive Simulation (Runs automatically)
        sim_df = simulate_strategies(display_df, threshold, bet_type, recorded, points_edge)
        
        # Check if we have final games
        final_games = display_df[display_df['status'] == 'COMPLETED']
        if final_games.empty:
            st.warning("No 'Final' games found in the selected range to run simulation. (All games might be 'PENDING')")
        
        # Metrics
        total_bets = len(sim_df[sim_df['bet_outcome'] != 'No Bet'])
        
        if total_bets == 0:
             st.info("No bets placed based on current threshold/data.")
        else:
            wins = len(sim_df[sim_df['bet_outcome'] == 'Win'])
            losses = len(sim_df[sim_df['bet_outcome'] == 'Loss'])
            pushes = len(sim_df[sim_df['bet_outcome'] == 'Push'])
            settled = wins + losses                       # a push returns the stake: it is in the record but not in the win rate or the ROI base
            win_rate = (wins / settled * 100) if settled > 0 else 0.0
            total_profit = sim_df['bet_result'].sum()
            roi = (total_profit / (settled * 100) * 100) if settled > 0 else 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Bets (W-L-P)", f"{total_bets} ({wins}-{losses}-{pushes})")
            c2.metric("Win Rate", f"{win_rate:.1f}%")
            c3.metric("Total Profit (Units)", f"{total_profit/100:.2f}u")
            c4.metric("ROI", f"{roi:.1f}%")
            
            # Breakdown Metrics
            st.markdown("---")
            bets_df = sim_df[sim_df['bet_outcome'] != 'No Bet'].copy()

            # Row 1: Fav/Dog or Over/Under
            st.markdown("##### Performance by Type")
            display_subgroup_metrics(bets_df, 'bet_subgroup', 'bet_result', bet_type)

            if bet_type in ('Moneyline', 'Spread'):
                st.divider()
                # Row 2: Home vs Away
                st.markdown("##### Home vs Away")
                display_subgroup_metrics(bets_df, 'bet_home_away', 'bet_result', bet_type,
                                         group_order=['Home', 'Away'])

                st.divider()
                # Row 3: 4-way intersection
                st.markdown("##### Home/Away × Favorite/Underdog")
                display_subgroup_metrics(bets_df, 'bet_intersection', 'bet_result', bet_type,
                                         group_order=['Home Fav', 'Home Dog', 'Away Fav', 'Away Dog'])

            st.markdown("---")
            
            # Charts
            # ROI over time
            fig_roi = px.line(sim_df[sim_df['bet_outcome'] != 'No Bet'], x='game_date', y='cumulative_profit', title=f'Cumulative Profit ({bet_type})')
            st.plotly_chart(fig_roi, width='stretch')
            
            # Monthly Breakdown
            sim_df['month'] = pd.to_datetime(sim_df['game_date']).dt.strftime('%Y-%m')
            monthly_results = sim_df[sim_df['bet_outcome'] != 'No Bet'].groupby('month')['bet_result'].sum().reset_index()
            
            fig_monthly = px.bar(monthly_results, x='month', y='bet_result', title='Profit by Month')
            st.plotly_chart(fig_monthly, width='stretch')
            
        st.markdown("### Detailed Log")
        st.caption("Tick **Why?** on a game to see what moved the model's numbers away from the line, and how the game ended.")
        log_cols = ['game_date', 'matchup', 'bet_outcome', 'bet_result', 'bet_side', 'bet_line', 'bet_price', 'bet_src', 'ml_prob', 'spread_pred', 'total_pred', 'odds_home_ml', 'odds_home_spread', 'odds_total', 'home_score', 'away_score']
        log_table = sim_df[log_cols].copy()
        log_table.insert(0, 'Why?', False)
        ticked = st.data_editor(log_table, key="detailed_log_table", width='stretch', hide_index=True, disabled=log_cols,
                                column_config={'Why?': st.column_config.CheckboxColumn('Why?', width='small')})
        picked = ticked.index[ticked['Why?']]
        if len(picked):
            history = load_explanation_history()
            for idx in picked[:5]:
                render_explanation(history, sim_df.loc[idx], settled=True)
            if len(picked) > 5:
                st.caption(f"Showing the first 5 of {len(picked)} ticked games.")

    with tab4:
        st.subheader("📋 Team Performance Breakdown")
        st.caption(f"Bet Type: **{bet_type}** · Bets: **{threshold_label}** · Date Range: {start_date} → {end_date}")

        # Build team list from filtered data
        all_teams = sorted(set(
            display_df['home_team'].dropna().unique().tolist() +
            display_df['away_team'].dropna().unique().tolist()
        ))

        if not all_teams:
            st.warning("No teams found in the selected date range.")
        else:
            selected_team = st.selectbox("Select Team", all_teams, key='team_breakdown_select')

            actual_df, model_df = build_team_breakdown(display_df, selected_team, bet_type, threshold, recorded, points_edge)

            if actual_df.empty:
                st.info(f"No completed games found for {selected_team} in the selected range.")
            else:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"#### {selected_team} — Actual Record")
                    st.dataframe(
                        actual_df,
                        hide_index=True,
                        width='stretch',
                        column_config={
                            'Split': st.column_config.TextColumn('Split', width='medium'),
                            'GP': st.column_config.NumberColumn('GP', width='small'),
                        }
                    )

                with col_b:
                    st.markdown(f"#### Model Bets on {selected_team} — {bet_type}")
                    st.dataframe(
                        model_df,
                        hide_index=True,
                        width='stretch',
                        column_config={
                            'Split': st.column_config.TextColumn('Split', width='medium'),
                            'Bets': st.column_config.NumberColumn('Bets', width='small'),
                        }
                    )

                # Show this team's recent games
                with st.expander(f"Recent Games for {selected_team}", expanded=False):
                    team_games = display_df[
                        ((display_df['home_team'] == selected_team) | (display_df['away_team'] == selected_team))
                        & (display_df['status'] == 'COMPLETED')
                    ].sort_values('game_date', ascending=False).head(20)
                    if not team_games.empty:
                        show_cols = ['game_date', 'matchup', 'home_score', 'away_score',
                                     'odds_home_ml', 'odds_home_spread', 'odds_total',
                                     'ml_prob', 'spread_pred', 'total_pred']
                        st.dataframe(team_games[[c for c in show_cols if c in team_games.columns]],
                                     hide_index=True, width='stretch')
                    else:
                        st.write("No completed games.")

    with tab6:
        render_drivers(df, start_date, end_date)

    with tab5:
        st.subheader("🏆 League-Wide Summary")
        st.caption((f"Model columns: the **recommended bets** (the live rule, 5% win-probability edge)" if recorded else
                    f"Model columns: WHAT-IF at a minimum edge of **{threshold:+.1%} win-probability edge** (applies to all three model bet columns; not the record)")
                   + f". Date Range: {start_date} → {end_date}")
        
        with st.spinner("Calculating league summary..."):
            league_df = build_league_summary(display_df, threshold, recorded)
            
            if league_df.empty:
                st.info("No data available.")
            else:
                st.dataframe(
                    league_df,
                    hide_index=True,
                    width='stretch',
                    height=1200,
                    column_config={
                        'Team': st.column_config.TextColumn("Team", width="medium"),
                        'GP': st.column_config.NumberColumn("GP", width="small"),
                        'ML ROI': st.column_config.NumberColumn("ML ROI", format="%.1f%%", width="small"),
                        'Spread ROI': st.column_config.NumberColumn("Spread ROI", format="%.1f%%", width="small"),
                        'Totals ROI': st.column_config.NumberColumn("Totals ROI", format="%.1f%%", width="small"),
                    }
                )

def render_drivers(df, start_date, end_date):
    """The Model Drivers tab: which groups of inputs move the model's numbers across the explained games, and whether they move them the right way."""
    st.subheader("What drives the model's numbers")
    st.caption("Across every explained game in the date range: how much each group of inputs pushes the model away from the line, and whether the game "
               "then landed on the pushed side of the line. A group that pushes a lot but whose results are indistinguishable from noise is a "
               "candidate for pruning. Descriptive only: with several groups tested, one 'signal' can be luck, and this is not a betting rule.")
    history = load_explanation_history()
    if history.empty:
        st.info("No explanations available yet.")
        return
    c1, c2 = st.columns(2)
    market_name = c1.selectbox("Market", ["Total", "Spread", "Moneyline"], key="drivers_market")
    level_name = c2.selectbox("Level", ["Groups of inputs", "Individual inputs"], key="drivers_level")
    market = {"Total": TOTAL, "Spread": SPREAD, "Moneyline": MONEYLINE}[market_name]
    in_range = df[(df['game_date'] >= start_date) & (df['game_date'] <= end_date)]
    results = in_range[['game_id', 'home_score', 'away_score']].assign(game_id=in_range['game_id'].astype(str).str.zfill(10))
    scoped = history[history['game_id'].isin(set(results['game_id']))]
    stats = driver_stats(scoped, results, market, level='group' if level_name.startswith('Groups') else 'feature')
    if stats.empty:
        st.info("No settled, explained games in the selected dates.")
        return
    n_games = scoped.loc[scoped['market'] == market, 'game_id'].nunique()
    st.caption(f"{n_games} explained games between {start_date} and {end_date}. "
               + ("Individual inputs: past games keep only their 15 biggest inputs." if not level_name.startswith('Groups') else ""))
    unit = "prob. points" if market == MONEYLINE else "pts"
    scale = 100.0 if market == MONEYLINE else 1.0
    table = pd.DataFrame({
        'Group' if level_name.startswith('Groups') else 'Input': stats['label'],
        f'Typical push ({unit})': (scale * stats['mean_abs']).round(2 if market != MONEYLINE or level_name.startswith('Groups') else 1),
        'Share of all pushes': (100 * stats['share']).round(1),
        'Games pushed': stats['games'],
        'Landed on the pushed side': (100 * stats['hit_rate']).round(1),
        f'Result beyond the line ({unit})': (scale * stats['edge']).round(2),
        'Verdict': stats['verdict']})
    if market == MONEYLINE:
        table = table.drop(columns=['Landed on the pushed side'])
    st.dataframe(table, hide_index=True, width='stretch', column_config={
        'Share of all pushes': st.column_config.NumberColumn(format="%.1f%%"),
        'Landed on the pushed side': st.column_config.NumberColumn(format="%.1f%%", help="Of the games where this pushed the number at least a little, how often the game landed on that side of the line. 50% = a coin flip."),
        f'Result beyond the line ({unit})': st.column_config.NumberColumn(help="Average result beyond the line, in the direction this group pushed. Positive = the push was right on average; the chart below shows the 95% interval.")})
    chart = stats[stats['games'] >= 30].copy()
    if not chart.empty:
        fig = go.Figure(go.Scatter(
            x=scale * chart['edge'], y=chart['label'], mode='markers', marker=dict(size=9, color='#2E86C1'),
            error_x=dict(type='data', symmetric=False, array=scale * (chart['ci_high'] - chart['edge']), arrayminus=scale * (chart['edge'] - chart['ci_low']))))
        fig.add_vline(x=0, line_dash='dash', line_color='#7F8C8D')
        fig.update_yaxes(autorange='reversed')
        fig.update_xaxes(title=f"Result beyond the line in the pushed direction ({unit}), 95% interval: right of 0 = the push was right")
        fig.update_layout(height=120 + 34 * len(chart), margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()
