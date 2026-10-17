import streamlit as st
import pandas as pd
import sqlite3
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add root to sys.path to allow importing models
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from models.betting import calculate_implied_prob, calculate_payout, calculate_bet_ev, get_ml_bet

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


@st.cache_data(ttl=3600)
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

        return df, source
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), "unavailable"

# Helper functions moved to models.betting
# calculate_implied_prob and calculate_payout are now imported


def compute_model_accuracy_and_roi(df):
    """Compute model's raw-pick correctness and ROI for each bet type (no threshold).
    Returns summary dict and df with correctness flags.
    """
    res = {}
    d = df.copy()
    # Ensure numeric
    d['home_score'] = pd.to_numeric(d['home_score'], errors='coerce')
    d['away_score'] = pd.to_numeric(d['away_score'], errors='coerce')

    # Moneyline pick (model pick based on Edge > 5%)
    def ml_pick_func(row):
        # Ensure market_prob is available or let get_ml_bet compute it from odds
        if 'odds_home_ml' not in row: return None
        pick, _ = get_ml_bet(row, threshold=0.05)
        return pick

    d['model_ml_pick'] = d.apply(ml_pick_func, axis=1)

    def ml_correct(row):
        if pd.isna(row['home_score']) or row['model_ml_pick'] is None: return None
        if row['model_ml_pick'] == 'home':
            return row['home_score'] > row['away_score']
        else:
            return row['away_score'] > row['home_score']

    d['ml_correct'] = d.apply(ml_correct, axis=1)

    # Spread pick (model predicts home margin = spread_pred; pick home if spread_pred + line > 0)
    def spread_pick(row):
        if pd.isna(row.get('spread_pred')) or pd.isna(row.get('odds_home_spread')): return None
        cover_margin = row['spread_pred'] + row['odds_home_spread']
        return 'home' if cover_margin > 0 else 'away'

    def spread_correct(row):
        if pd.isna(row['home_score']) or pd.isna(row.get('odds_home_spread')) or pd.isna(row.get('spread_pred')): return None
        line = row['odds_home_spread']
        # Home covers when home_score + line > away_score
        if row['model_spread_pick'] == 'home':
            return (row['home_score'] + line) > row['away_score']
        else:
            return (row['home_score'] + line) < row['away_score']

    d['model_spread_pick'] = d.apply(spread_pick, axis=1)
    d['spread_correct'] = d.apply(spread_correct, axis=1)

    # Totals pick (model picks over when total_pred > odds_total)
    def total_pick(row):
        if pd.isna(row.get('total_pred')) or pd.isna(row.get('odds_total')): return None
        return 'over' if row['total_pred'] > row['odds_total'] else 'under'

    def total_correct(row):
        if pd.isna(row['home_score']) or pd.isna(row.get('odds_total')) or pd.isna(row.get('total_pred')): return None
        total = row['home_score'] + row['away_score']
        if row['model_total_pick'] == 'over':
            return total > row['odds_total']
        else:
            return total < row['odds_total']

    d['model_total_pick'] = d.apply(total_pick, axis=1)
    d['total_correct'] = d.apply(total_correct, axis=1)

    # Compute ROI/win% for each bet type assuming $100 stake per bet.
    def compute_metrics_for(pick_col, correct_col, odds_cols):
        sub = d[d[pick_col].notnull() & d[correct_col].notnull()].copy()
        if sub.empty:
            return {'bets': 0, 'wins': 0, 'win_rate': 0.0, 'profit': 0.0, 'roi': 0.0}

        total_bets = len(sub)
        wins = int(sub[correct_col].sum())
        # profit computation
        profit = 0.0
        for _, r in sub.iterrows():
            won = r[correct_col]
            if pick_col == 'model_ml_pick':
                # determine which odds to use
                odds = r['odds_home_ml'] if r[pick_col] == 'home' else r['odds_away_ml']
                if pd.isna(odds):
                    # skip bet (treat as no bet)
                    total_bets -= 1
                    continue
                if won:
                    profit += calculate_payout(odds)
                else:
                    profit -= 100
            else:
                # use -110 market assumption for spread/totals
                if won:
                    profit += calculate_payout(-110)
                else:
                    profit -= 100

        if total_bets <= 0:
            return {'bets': 0, 'wins': 0, 'win_rate': 0.0, 'profit': 0.0, 'roi': 0.0}

        roi = (profit / (total_bets * 100)) * 100
        win_rate = (wins / total_bets) * 100
        return {'bets': total_bets, 'wins': wins, 'win_rate': win_rate, 'profit': profit, 'roi': roi}

    res['Moneyline'] = compute_metrics_for('model_ml_pick', 'ml_correct', ['odds_home_ml','odds_away_ml'])
    res['Spread'] = compute_metrics_for('model_spread_pick', 'spread_correct', [])
    res['Totals'] = compute_metrics_for('model_total_pick', 'total_correct', [])

    return res, d

def simulate_strategies(df, threshold, bet_type):
    """
    Simulates betting strategies based on a threshold.
    bet_type: 'Moneyline', 'Spread', 'Totals'
    """
    # Clone to avoid modifying original
    sim_df = df.copy()
    
    # Filter for completed games only
    sim_df = sim_df[sim_df['status'] == 'COMPLETED'].copy()
    
    sim_df['bet_result'] = 0.0
    sim_df['bet_outcome'] = 'No Bet'
    sim_df['bet_subgroup'] = None     # Favorite/Underdog/Over/Under
    sim_df['bet_home_away'] = None    # Home/Away (ML & Spread only)
    sim_df['bet_intersection'] = None # Home Fav / Home Dog / Away Fav / Away Dog
    
    for idx, row in sim_df.iterrows():
        bet_placed = False
        won = False
        odds = -110 # Default for Spread/Total
        
        if bet_type == 'Moneyline':
            home_prob = row.get('ml_prob')
            market_home_prob = calculate_implied_prob(row.get('odds_home_ml'))
            market_away_prob = calculate_implied_prob(row.get('odds_away_ml'))
            
            if pd.notna(home_prob) and pd.notna(market_home_prob) and pd.notna(market_away_prob):
                pick, edge = get_ml_bet(row, threshold=threshold)
                
                if pick == 'home':
                    bet_placed = True
                    odds = row.get('odds_home_ml')
                    is_fav = odds < 0
                    sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if is_fav else 'Underdog'
                    sim_df.at[idx, 'bet_home_away'] = 'Home'
                    sim_df.at[idx, 'bet_intersection'] = 'Home Fav' if is_fav else 'Home Dog'
                    if row['home_score'] > row['away_score']:
                        won = True
                        
                elif pick == 'away':
                    bet_placed = True
                    odds = row.get('odds_away_ml')
                    is_fav = odds < 0
                    sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if is_fav else 'Underdog'
                    sim_df.at[idx, 'bet_home_away'] = 'Away'
                    sim_df.at[idx, 'bet_intersection'] = 'Away Fav' if is_fav else 'Away Dog'
                    if row['away_score'] > row['home_score']:
                        won = True

        elif bet_type == 'Spread':
            spread_pred = row.get('spread_pred')
            line = row.get('odds_home_spread')
            
            if pd.notna(spread_pred) and pd.notna(line):
                cover_margin = spread_pred + line
                
                if cover_margin > threshold:
                    # Bet Home Spread
                    bet_placed = True
                    is_fav = line < 0
                    sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if is_fav else 'Underdog'
                    sim_df.at[idx, 'bet_home_away'] = 'Home'
                    sim_df.at[idx, 'bet_intersection'] = 'Home Fav' if is_fav else 'Home Dog'
                    if (row['home_score'] + line) > row['away_score']:
                        won = True
                elif cover_margin < -threshold:
                    # Bet Away Spread
                    bet_placed = True
                    is_fav = (-line) < 0
                    sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if is_fav else 'Underdog'
                    sim_df.at[idx, 'bet_home_away'] = 'Away'
                    sim_df.at[idx, 'bet_intersection'] = 'Away Fav' if is_fav else 'Away Dog'
                    if (row['home_score'] + line) < row['away_score']:
                        won = True

        elif bet_type == 'Totals':
            # Threshold is Points Edge
            # Compare total_pred vs odds_total
            total_pred = row.get('total_pred')
            line = row.get('odds_total')
            
            if pd.notna(total_pred) and pd.notna(line):
                diff = total_pred - line
                
                if diff > threshold:
                    # Bet Over
                    bet_placed = True
                    sim_df.at[idx, 'bet_subgroup'] = 'Over'
                    if (row['home_score'] + row['away_score']) > line:
                        won = True
                elif diff < -threshold:
                    # Bet Under
                    bet_placed = True
                    sim_df.at[idx, 'bet_subgroup'] = 'Under'
                    if (row['home_score'] + row['away_score']) < line:
                        won = True

        if bet_placed:
            if won:
                profit = calculate_payout(odds)
                sim_df.at[idx, 'bet_result'] = profit
                sim_df.at[idx, 'bet_outcome'] = 'Win'
            else:
                sim_df.at[idx, 'bet_result'] = -100
                sim_df.at[idx, 'bet_outcome'] = 'Loss'
                    
    # Cumulative ROI
    # Fix: Sort by date ASCENDING to calculate cumulative sum correctly (Past -> Future)
    sim_df = sim_df.sort_values(by='game_date', ascending=True)
    sim_df['cumulative_profit'] = sim_df['bet_result'].cumsum()
    
    # Sort back to DESCENDING for display (Latest -> Past)
    sim_df = sim_df.sort_values(by='game_date', ascending=False)
    
    return sim_df

def _team_model_record(sim_df, team_name, is_totals=False):
    """ (bets, wins, roi) for model bets involving team_name in a simulate_strategies() result. """
    if is_totals:
        team_bets = sim_df[((sim_df['home_team'] == team_name) | (sim_df['away_team'] == team_name)) & (sim_df['bet_outcome'] != 'No Bet')]
    else:
        team_bets_home = sim_df[(sim_df['home_team'] == team_name) & (sim_df['bet_home_away'] == 'Home')]
        team_bets_away = sim_df[(sim_df['away_team'] == team_name) & (sim_df['bet_home_away'] == 'Away')]
        team_bets = pd.concat([team_bets_home, team_bets_away])

    bets = len(team_bets)
    if bets == 0:
        return 0, 0, 0.0
    profit = team_bets['bet_result'].sum()
    roi = (profit / (bets * 100)) * 100
    wins = (team_bets['bet_result'] > 0).sum()
    return bets, wins, roi


@st.cache_data(ttl=600)
def build_league_summary(df, threshold):
    """ Builds a single league-wide summary table for all teams with Actual and Model records. """
    teams = sorted(set(df['home_team'].dropna().unique().tolist() + df['away_team'].dropna().unique().tolist()))
    
    completed = df[df['status'] == 'COMPLETED'].copy()
    for c in ['home_score', 'away_score', 'odds_home_spread', 'odds_total']:
        if c in completed.columns:
            completed[c] = pd.to_numeric(completed[c], errors='coerce')
            
    sim_ml = simulate_strategies(df, threshold, 'Moneyline')
    sim_sp = simulate_strategies(df, threshold, 'Spread')
    sim_tl = simulate_strategies(df, threshold, 'Totals')
    
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
        
        m_ml_bets, m_ml_wins, m_ml_roi = _team_model_record(sim_ml, team, False)
        m_sp_bets, m_sp_wins, m_sp_roi = _team_model_record(sim_sp, team, False)
        m_tl_bets, m_tl_wins, m_tl_roi = _team_model_record(sim_tl, team, True)
        
        rows.append({
            'Team': team,
            'GP': gp,
            'Actual SU': f"{su_wins}-{gp-su_wins}",
            'Actual ATS': f"{ats_wins}-{ats_losses}",
            'Actual O/U': f"{over_wins}-{under_wins}",
            'ML W-L': f"{m_ml_wins}-{m_ml_bets-m_ml_wins}",
            'ML ROI': m_ml_roi,
            'Spread W-L': f"{m_sp_wins}-{m_sp_bets-m_sp_wins}",
            'Spread ROI': m_sp_roi,
            'Totals W-L': f"{m_tl_wins}-{m_tl_bets-m_tl_wins}",
            'Totals ROI': m_tl_roi,
        })
        
    return pd.DataFrame(rows)


def build_team_breakdown(df, team, bet_type, threshold):
    """
    Build per-team breakdown tables.
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

    # --- Build per-game labels from team's perspective ---
    rows = []
    for _, g in team_games.iterrows():
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
        if pd.notna(total_line):
            over_hit = actual_total > total_line
        else:
            over_hit = None

        # Intersection
        if fav_dog:
            fav_label = 'Fav' if fav_dog == 'Favorite' else 'Dog'
            intersection = f"{ha} {fav_label}"
        else:
            intersection = None

        # --- Model bet info ---
        model_bet_on_team = None  # Did model bet on THIS team?
        model_bet_won = None
        model_bet_odds = None
        model_bet_type_label = None

        if bet_type == 'Moneyline':
            if pd.notna(g.get('ml_prob')) and pd.notna(g.get('odds_home_ml')) and pd.notna(g.get('odds_away_ml')):
                pick, edge = get_ml_bet(g, threshold=threshold)
                if pick == 'home' and is_home:
                    model_bet_on_team = True
                    model_bet_won = su_win
                    model_bet_odds = g['odds_home_ml']
                elif pick == 'away' and not is_home:
                    model_bet_on_team = True
                    model_bet_won = su_win
                    model_bet_odds = g['odds_away_ml']
                elif pick == 'home' and not is_home:
                    model_bet_on_team = False  # Bet against this team
                elif pick == 'away' and is_home:
                    model_bet_on_team = False

        elif bet_type == 'Spread':
            sp = g.get('spread_pred')
            line = g.get('odds_home_spread')
            if pd.notna(sp) and pd.notna(line):
                cover_margin = sp + line
                if cover_margin > threshold:
                    # Model bets home
                    if is_home:
                        model_bet_on_team = True
                        model_bet_won = ats_win
                        model_bet_odds = -110
                    else:
                        model_bet_on_team = False
                elif cover_margin < -threshold:
                    # Model bets away
                    if not is_home:
                        model_bet_on_team = True
                        model_bet_won = ats_win
                        model_bet_odds = -110
                    else:
                        model_bet_on_team = False

        elif bet_type == 'Totals':
            tp = g.get('total_pred')
            tl = g.get('odds_total')
            if pd.notna(tp) and pd.notna(tl):
                diff = tp - tl
                if abs(diff) > threshold:
                    model_bet_on_team = True  # Totals are team-agnostic, counts for both
                    if diff > 0:
                        model_bet_won = over_hit
                    else:
                        model_bet_won = (not over_hit) if over_hit is not None else None
                    model_bet_odds = -110

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
        valid = bet_on[bet_on['model_bet_won'].notna()]
        if valid.empty:
            model_rows.append({'Split': label, 'Bets': n, 'Record': '-', 'ROI': '-'})
            continue
        w = int(valid['model_bet_won'].sum())
        l = len(valid) - w
        # Compute profit
        profit = 0.0
        for _, r in valid.iterrows():
            odds = r['model_bet_odds']
            if pd.isna(odds):
                odds = -110
            if r['model_bet_won']:
                profit += calculate_payout(odds)
            else:
                profit -= 100
        roi = (profit / (len(valid) * 100)) * 100
        model_rows.append({
            'Split': label,
            'Bets': len(valid),
            'Record': f"{w}-{l} ({w/len(valid)*100:.0f}%)",
            'ROI': f"{roi:+.1f}%",
        })

    model_df = pd.DataFrame(model_rows)

    return actual_df, model_df


def main():
    st.title("🏀 NBA Model Dashboard")

    df, source = load_data()
    
    if df.empty:
        st.warning("No prediction data available.")
        st.stop() # Stop execution if no data

    # Sidebar Filters
    st.sidebar.header("Configuration")
    st.sidebar.caption(f"Data: {source} · {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Date Filter
    # Check if we have data
    if 'game_date' in df.columns and not df['game_date'].isnull().all():
        min_date = df['game_date'].min()
        max_date = df['game_date'].max()
    else:
        min_date = datetime.date.today()
        max_date = datetime.date.today()
    
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # What-If Analysis
    st.sidebar.subheader("What-If Analysis")
    
    bet_type = st.sidebar.selectbox("Bet Type", ["Moneyline", "Spread", "Totals"])
    
    if bet_type == "Moneyline":
        threshold = st.sidebar.slider("Edge Threshold", 0.0, 0.20, 0.05, 0.005, format="%.3f", help="Bet only when model prob exceeds market prob by this amount.")
    else:
        threshold = st.sidebar.slider("Points Edge Threshold", 0.0, 10.0, 1.0, 0.5, help="Bet only when predicted score differs from line by this much.")

    # Filter Data for Display
    mask = (df['game_date'] >= start_date) & (df['game_date'] <= end_date)
    display_df = df.loc[mask]

    # Compute model accuracy & ROI (no threshold) across available data (use filtered range)
    model_metrics, model_df = compute_model_accuracy_and_roi(display_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Today's Games", "Recent Performance", "📊 Predictions Log & Analysis", "📋 Team Breakdown", "🏆 League Summary"])

    with tab1:
        st.subheader("Games Schedule")
        today = datetime.date.today()
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
            # 2. Picks
            def get_ml_pick_display(row):
                # Using 5% threshold as default for "ML Pick" display recommendation.
                # get_ml_bet prices each side with its own moneyline (odds_home_ml / odds_away_ml).
                pick, edge = get_ml_bet(row, threshold=0.05)
                
                if pick == 'home':
                     return f"BET HOME (Edge {edge*100:.1f}%)"
                elif pick == 'away':
                     return f"BET AWAY (Edge {edge*100:.1f}%)"
                else:
                     # Optional: Show small edge or Pass
                     return ""
            
            def get_spread_pick(row):
                if pd.isna(row['spread_pred']) or pd.isna(row['odds_home_spread']): return ""
                # Pred Margin (Home - Away) vs Line
                # cover_margin = spread_pred + line
                # If > 0, Home Covers.
                margin = row['spread_pred'] + row['odds_home_spread']
                if margin > 0: return f"Home {row['odds_home_spread']} (Margin: {margin:.1f})"
                else: return f"Away {-1 * row['odds_home_spread']} (Margin: {abs(margin):.1f})"

            def get_total_pick(row):
                if pd.isna(row['total_pred']) or pd.isna(row['odds_total']): return ""
                diff = row['total_pred'] - row['odds_total']
                if diff > 0: return f"Over {row['odds_total']} (+{diff:.1f})"
                else: return f"Under {row['odds_total']} ({diff:.1f})"

            todays_games['ML Pick'] = todays_games.apply(get_ml_pick_display, axis=1)
            todays_games['Spread Pick'] = todays_games.apply(get_spread_pick, axis=1)
            todays_games['Total Pick'] = todays_games.apply(get_total_pick, axis=1)
            
            # 3. Clean Columns
            cols_to_show = [
                'matchup', 
                'game_date',
                'Model Home Prob (%)',
                'Implied Home Prob (%)',
                'Model Away Prob (%)',
                'Implied Away Prob (%)',
                'odds_away_ml',
                'odds_home_ml',
                'ML Pick',
                'Spread Pick',
                'Total Pick'
            ]
            
            todays_games['Implied Home Prob (%)'] = todays_games['Implied Home Prob (%)'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else x)
            todays_games['Model Home Prob (%)'] = todays_games['Model Home Prob (%)'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else x)
            todays_games['Implied Away Prob (%)'] = todays_games['Implied Away Prob (%)'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else x)
            todays_games['Model Away Prob (%)'] = todays_games['Model Away Prob (%)'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else x)

            # Filter existing cols
            final_cols = [c for c in cols_to_show if c in todays_games.columns]
            
            st.dataframe(todays_games[final_cols])
            st.metric("Total Games", len(todays_games))

    with tab2:
        st.subheader("Model Performance: Last 14 Days")
        
        # Filter to this week (Monday to today)
        today = datetime.date.today()
        previous_two_weeks = today - datetime.timedelta(days=14)
        week_df = model_df[(model_df['game_date'] >= previous_two_weeks) & (model_df['game_date'] < today)].copy()
        week_df = week_df.sort_values('game_date', ascending=False)
        
        if week_df.empty:
            st.info("No predictions yet this week.")
        else:
            # Compute records and ROI for each bet type
            def compute_record_and_roi(correct_series, pick_col, odds_col=None):
                """Returns (wins, losses, pushes, roi) tuple."""
                valid_mask = correct_series.notna()
                valid = correct_series[valid_mask]
                if valid.empty:
                    return 0, 0, 0, 0.0
                wins = int(valid.sum())
                losses = len(valid) - wins
                
                # Compute ROI
                profit = 0.0
                for idx in valid.index:
                    won = valid.loc[idx]
                    if pick_col and odds_col:
                        odds = week_df.loc[idx, odds_col]
                        if pd.notna(odds):
                            if won:
                                profit += calculate_payout(odds)
                            else:
                                profit -= 100
                    else:
                        # Use -110 for spreads/totals
                        if won:
                            profit += calculate_payout(-110)
                        else:
                            profit -= 100
                
                roi = (profit / (len(valid) * 100)) * 100 if len(valid) > 0 else 0.0
                return wins, losses, 0, roi
            
            ml_wins, ml_losses, ml_pushes, ml_roi = compute_record_and_roi(week_df['ml_correct'], 'model_ml_pick', None)
            sp_wins, sp_losses, sp_pushes, sp_roi = compute_record_and_roi(week_df['spread_correct'], None, None)
            tl_wins, tl_losses, tl_pushes, tl_roi = compute_record_and_roi(week_df['total_correct'], None, None)
            
            # Display records with ROI as top metrics
            cols = st.columns(3)
            cols[0].metric("Moneyline", f"{ml_wins}-{ml_losses}-{ml_pushes}", f"{ml_roi:.1f}%")
            cols[1].metric("Spread", f"{sp_wins}-{sp_losses}-{sp_pushes}", f"{sp_roi:.1f}%")
            cols[2].metric("Totals", f"{tl_wins}-{tl_losses}-{tl_pushes}", f"{tl_roi:.1f}%")

            # --- Breakdown Metrics for Week ---
            st.divider()
            st.markdown("##### Performance Breakdown (Favorite vs Underdog / Over vs Under)")
            
            # Helper to calculate profit + subgroup labels for a row in week_df
            def get_row_profit(row, pick_col, correct_col, odds_col=None, spread_total=False):
                """Returns (profit, fav_dog_subgroup, home_away, intersection) or (0.0, None, None, None) if skip."""
                if pd.isna(row.get(correct_col)) or pd.isna(row.get(pick_col)):
                    return 0.0, None, None, None
                
                subgroup = None
                home_away = None
                is_fav = False
                
                if spread_total:
                    if 'total' in pick_col:  # Totals: no home/away intersection
                        subgroup = 'Over' if row[pick_col] == 'over' else 'Under'
                        home_away = None
                    else:  # Spread
                        home_away = 'Home' if row[pick_col] == 'home' else 'Away'
                        line = row.get('odds_home_spread', 0)
                        if row[pick_col] == 'home':
                            is_fav = line < 0
                        else:
                            is_fav = (-line) < 0
                        subgroup = 'Favorite' if is_fav else 'Underdog'
                    odds = -110
                else:  # Moneyline
                    home_away = 'Home' if row[pick_col] == 'home' else 'Away'
                    odds = row.get('odds_home_ml') if row[pick_col] == 'home' else row.get('odds_away_ml')
                    if pd.isna(odds): return 0.0, None, None, None
                    is_fav = odds < 0
                    subgroup = 'Favorite' if is_fav else 'Underdog'

                # Build intersection label (only for ML & Spread)
                intersection = None
                if home_away and subgroup in ('Favorite', 'Underdog'):
                    fav_label = 'Fav' if subgroup == 'Favorite' else 'Dog'
                    intersection = f"{home_away} {fav_label}"

                won = row[correct_col]
                profit = calculate_payout(odds) if won else -100
                    
                return profit, subgroup, home_away, intersection

            # Compute Breakdowns — collect profit + all label dimensions
            ml_data, sp_data, tl_data = [], [], []
            for _, row in week_df.iterrows():
                p, sg, ha, ix = get_row_profit(row, 'model_ml_pick', 'ml_correct', spread_total=False)
                if sg is not None:
                    ml_data.append({'profit': p, 'subgroup': sg, 'home_away': ha, 'intersection': ix})
                p, sg, ha, ix = get_row_profit(row, 'model_spread_pick', 'spread_correct', spread_total=True)
                if sg is not None:
                    sp_data.append({'profit': p, 'subgroup': sg, 'home_away': ha, 'intersection': ix})
                p, sg, ha, ix = get_row_profit(row, 'model_total_pick', 'total_correct', spread_total=True)
                if sg is not None:
                    tl_data.append({'profit': p, 'subgroup': sg, 'home_away': ha, 'intersection': ix})

            ml_df_b = pd.DataFrame(ml_data) if ml_data else pd.DataFrame()
            sp_df_b = pd.DataFrame(sp_data) if sp_data else pd.DataFrame()
            tl_df_b = pd.DataFrame(tl_data) if tl_data else pd.DataFrame()

            # Row 1: Favorite / Underdog & Over / Under
            b1, b2, b3 = st.columns(3)
            with b1:
                st.caption("Moneyline — Fav vs Dog")
                if not ml_df_b.empty:
                    display_subgroup_metrics(ml_df_b, 'subgroup', 'profit', 'ML')
                else:
                    st.write("-")
            with b2:
                st.caption("Spread — Fav vs Dog")
                if not sp_df_b.empty:
                    display_subgroup_metrics(sp_df_b, 'subgroup', 'profit', 'Spread')
                else:
                    st.write("-")
            with b3:
                st.caption("Totals — Over vs Under")
                if not tl_df_b.empty:
                    display_subgroup_metrics(tl_df_b, 'subgroup', 'profit', 'Total')
                else:
                    st.write("-")

            st.divider()

            # Row 2: Home vs Away (ML & Spread only)
            st.markdown("##### Home vs Away Performance")
            h1, h2 = st.columns(2)
            with h1:
                st.caption("Moneyline — Home vs Away")
                if not ml_df_b.empty:
                    display_subgroup_metrics(ml_df_b, 'home_away', 'profit', 'ML', group_order=['Home', 'Away'])
                else:
                    st.write("-")
            with h2:
                st.caption("Spread — Home vs Away")
                if not sp_df_b.empty:
                    display_subgroup_metrics(sp_df_b, 'home_away', 'profit', 'Spread', group_order=['Home', 'Away'])
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
                    display_subgroup_metrics(ml_df_b, 'intersection', 'profit', 'ML', group_order=ix_order)
                else:
                    st.write("-")
            with i2:
                st.caption("Spread")
                if not sp_df_b.empty:
                    display_subgroup_metrics(sp_df_b, 'intersection', 'profit', 'Spread', group_order=ix_order)
                else:
                    st.write("-")

            st.divider()
            # ----------------------------------
            
            # Format predictions table (similar to tab1)
            week_df = week_df.copy()
            
            # Helper functions for formatting picks
            def format_ml_pick(row):
                if pd.isna(row['model_ml_pick']):
                    return ""
                team = row['home_team'] if row['model_ml_pick'] == 'home' else row['away_team']
                prob = row['ml_prob'] if row['model_ml_pick'] == 'home' else (1 - row['ml_prob'])
                return f"{team} ({prob:.1%})"
            
            def format_spread_pick(row):
                if pd.isna(row.get('model_spread_pick')) or pd.isna(row.get('odds_home_spread')):
                    return ""
                line = row['odds_home_spread']
                if row['model_spread_pick'] == 'home':
                    margin = row.get('spread_pred', 0) + line
                    return f"Home {line} (Margin: {margin:.1f})"
                else:
                    margin = -(row.get('spread_pred', 0) + line)
                    return f"Away {-line} (Margin: {margin:.1f})"
            
            def format_total_pick(row):
                if pd.isna(row.get('model_total_pick')) or pd.isna(row.get('odds_total')):
                    return ""
                line = row['odds_total']
                diff = row.get('total_pred', 0) - line
                if row['model_total_pick'] == 'over':
                    return f"Over {line} (+{diff:.1f})"
                else:
                    return f"Under {line} ({diff:.1f})"
            
            week_df['ML Pick'] = week_df.apply(format_ml_pick, axis=1)
            week_df['Spread Pick'] = week_df.apply(format_spread_pick, axis=1)
            week_df['Total Pick'] = week_df.apply(format_total_pick, axis=1)
            week_df['Model Home Prob (%)'] = (100.0 * week_df['ml_prob'])
            week_df['Implied Home Prob (%)'] = (100.0 * week_df['odds_home_ml'].apply(calculate_implied_prob)).round(2)
            
            # Build display table
            cols_to_show = [
                'home_team',
                'away_team',
                'game_date',
                'home_score',
                'away_score',
                'Model Home Prob (%)',
                'Implied Home Prob (%)',
                'odds_home_ml',
                'ML Pick',
                'Spread Pick',
                'Total Pick'
            ]


            # Format columns on display_df BEFORE selecting into show_df
            if 'Model Home Prob (%)' in week_df.columns:
                week_df['Model Home Prob (%)'] = week_df['Model Home Prob (%)'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else x)
            if 'Implied Home Prob (%)' in week_df.columns:
                week_df['Implied Home Prob (%)'] = week_df['Implied Home Prob (%)'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else x)
            if 'odds_home_ml' in week_df.columns:
                week_df['odds_home_ml'] = week_df['odds_home_ml'].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x < 0 else f"+{x}")
            if 'home_score' in week_df.columns:
                week_df['home_score'] = week_df['home_score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else x)
            if 'away_score' in week_df.columns:
                week_df['away_score'] = week_df['away_score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else x)
            
            final_cols = [c for c in cols_to_show if c in week_df.columns]
            # Include correctness columns for styling (not displayed, but used by highlight_wins)
            show_df = week_df[final_cols + ['ml_correct', 'spread_correct', 'total_correct']].copy()
            
            # Function to highlight winning bet pick columns
            def highlight_wins(row):
                """Apply green background to pick columns when predictions are correct."""
                result = [''] * len(row)
                for i, col in enumerate(show_df.columns):
                    if col == 'ML Pick' and pd.notna(row.get('ml_correct')) and row['ml_correct'] is True:
                        result[i] = 'background-color: #90EE90'
                    elif col == 'Spread Pick' and pd.notna(row.get('spread_correct')) and row['spread_correct'] is True:
                        result[i] = 'background-color: #90EE90'
                    elif col == 'Total Pick' and pd.notna(row.get('total_correct')) and row['total_correct'] is True:
                        result[i] = 'background-color: #90EE90'
                return result
                        
            # Apply row-by-row styling
            styled_df = show_df.style.apply(highlight_wins, axis=1)

            st.dataframe(styled_df,
                         column_config={'ml_correct':None,
                                        'spread_correct':None,
                                        'total_correct':None},
                                        hide_index=True)

    with tab3:
        st.subheader(f"Simulation & Analysis: {bet_type}")
        
        # Show all data first but collapsed
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(display_df)
        
        st.caption(f"Simulating {bet_type} bets with threshold {threshold} from {start_date} to {end_date}.")
        
        # Reactive Simulation (Runs automatically)
        sim_df = simulate_strategies(display_df, threshold, bet_type)
        
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
            win_rate = (wins / total_bets * 100) if total_bets > 0 else 0.0
            total_profit = sim_df['bet_result'].sum()
            roi = (total_profit / (total_bets * 100) * 100) if total_bets > 0 else 0.0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Bets", total_bets)
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
            st.plotly_chart(fig_roi, use_container_width=True)
            
            # Monthly Breakdown
            sim_df['month'] = pd.to_datetime(sim_df['game_date']).dt.strftime('%Y-%m')
            monthly_results = sim_df[sim_df['bet_outcome'] != 'No Bet'].groupby('month')['bet_result'].sum().reset_index()
            
            fig_monthly = px.bar(monthly_results, x='month', y='bet_result', title='Profit by Month')
            st.plotly_chart(fig_monthly, use_container_width=True)
            
        st.markdown("### Detailed Log")
        st.dataframe(sim_df[['game_date', 'matchup', 'bet_outcome', 'bet_result', 'ml_prob', 'spread_pred', 'total_pred', 'odds_home_ml', 'odds_home_spread', 'odds_total', 'home_score', 'away_score']])

    with tab4:
        st.subheader("📋 Team Performance Breakdown")
        st.caption(f"Bet Type: **{bet_type}** · Threshold: **{threshold}** · Date Range: {start_date} → {end_date}")

        # Build team list from filtered data
        all_teams = sorted(set(
            display_df['home_team'].dropna().unique().tolist() +
            display_df['away_team'].dropna().unique().tolist()
        ))

        if not all_teams:
            st.warning("No teams found in the selected date range.")
        else:
            selected_team = st.selectbox("Select Team", all_teams, key='team_breakdown_select')

            actual_df, model_df = build_team_breakdown(display_df, selected_team, bet_type, threshold)

            if actual_df.empty:
                st.info(f"No completed games found for {selected_team} in the selected range.")
            else:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"#### {selected_team} — Actual Record")
                    st.dataframe(
                        actual_df,
                        hide_index=True,
                        use_container_width=True,
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
                        use_container_width=True,
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
                                     hide_index=True, use_container_width=True)
                    else:
                        st.write("No completed games.")

    with tab5:
        st.subheader("🏆 League-Wide Summary")
        st.caption(f"Model Threshold: **{threshold}** (Applies to all model bet columns). Date Range: {start_date} → {end_date}")
        
        with st.spinner("Calculating league summary..."):
            league_df = build_league_summary(display_df, threshold)
            
            if league_df.empty:
                st.info("No data available.")
            else:
                st.dataframe(
                    league_df,
                    hide_index=True,
                    use_container_width=True,
                    height=1200,
                    column_config={
                        'Team': st.column_config.TextColumn("Team", width="medium"),
                        'GP': st.column_config.NumberColumn("GP", width="small"),
                        'ML ROI': st.column_config.NumberColumn("ML ROI", format="%.1f%%", width="small"),
                        'Spread ROI': st.column_config.NumberColumn("Spread ROI", format="%.1f%%", width="small"),
                        'Totals ROI': st.column_config.NumberColumn("Totals ROI", format="%.1f%%", width="small"),
                    }
                )

if __name__ == "__main__":
    main()
