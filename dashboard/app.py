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
def display_subgroup_metrics(df, subgroup_col, result_col, title):
    """
    Displays metrics for unique values in subgroup_col.
    df: DataFrame with bet results
    subgroup_col: Column name defining the group (e.g. 'bet_subgroup')
    result_col: Column containing bet profit/loss (e.g. 'bet_result' or calculated profit)
    """
    if subgroup_col not in df.columns or df.empty:
        return

    groups = df[subgroup_col].unique()
    groups = [g for g in groups if pd.notna(g) and g != '']
    groups.sort() # Ensure consistent order
    
    # Custom sort for specific terms
    if 'Favorite' in groups and 'Underdog' in groups:
        groups = ['Favorite', 'Underdog']
    if 'Over' in groups and 'Under' in groups:
        groups = ['Over', 'Under']

    cols = st.columns(len(groups))
    
    for i, group in enumerate(groups):
        sub_df = df[df[subgroup_col] == group]
        if sub_df.empty:
            continue
            
        bets = len(sub_df)
        # Check if result_col is a boolean (Win/Loss) or float (Profit)
        # Ideally we standardized on Profit.
        # If boolean (correct/incorrect), we assume standard -110 or track profit externally.
        # But here we want to flexible. 
        
        # Case A: Simulation DF (has 'bet_result' as float profit)
        if df[result_col].dtype.kind in 'fc': # float or complex
             profit = sub_df[result_col].sum()
             # We assume units are stake=100
             # ROI = Profit / (Bets * 100) * 100
             roi = (profit / (bets * 100)) * 100
             wins = len(sub_df[sub_df[result_col] > 0])
             losses = len(sub_df[sub_df[result_col] < 0])
             # Pushes are 0 profit?
             pushes = len(sub_df[sub_df[result_col] == 0]) # rough approx
        
        # Case B: Tab 2 DF (has boolean Correct/Incorrect and we need to calculate profit)
        # This helper is best suited for Tab 3 style data or pre-processed data.
        # For Tab 2, we should probably stick to custom logic or pre-process it to look like Tab 3.
        else:
             # Fallback or strict requirement
             st.error("Invalid column type for metrics")
             return

        cols[i].metric(f"{title}: {group}", f"{wins}-{losses}-{pushes}", f"{roi:.1f}%")


# Constants
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nba_stats.db"
print(DB_PATH)
@st.cache_data
def load_data():
    """Loads prediction data from the sqlite database."""
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM predictions_log_v2"
        df = pd.read_sql_query(query, conn)
        df = df.sort_values(by=['game_date', 'game_id'], ascending=[False, True])
        conn.close()
        
        # Convert date columns to datetime
        df['game_date'] = pd.to_datetime(df['game_date']).dt.date
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        
        # Ensure numeric columns are actually numeric
        numeric_cols = ['home_score', 'away_score', 'ml_prob', 'odds_home_ml', 'odds_away_ml', 'spread_pred', 'total_pred']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

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
    sim_df['bet_subgroup'] = None # Favorite/Underdog/Over/Under
    
    for idx, row in sim_df.iterrows():
        bet_placed = False
        won = False
        odds = -110 # Default for Spread/Total
        
        if bet_type == 'Moneyline':
            # Threshold is now Edge Threshold (e.g. 0.05 for 5%)
            # Calculate Edge: Model Prob - Market Prob
            
            home_prob = row.get('ml_prob')
            market_home_prob = calculate_implied_prob(row.get('odds_home_ml'))
            
            if pd.notna(home_prob) and pd.notna(market_home_prob):
                # Construct a row-like object or just use the values
                # We can't pass 'row' directly if it lacks 'market_prob', but get_ml_bet handles odds_home_ml looking up
                # However, simulate_strategies Loop `row` is a Series.
                
                # Let's use get_ml_bet logic inline or call it.
                # Calling it is cleaner but we need to ensure it finds odds.
                
                # Check for bet
                # Note: 'row' in iterrows contains all columns.
                pick, edge = get_ml_bet(row, threshold=threshold)
                
                if pick == 'home':
                    bet_placed = True
                    odds = row.get('odds_home_ml')
                    sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if odds < 0 else 'Underdog'
                    if row['home_score'] > row['away_score']:
                        won = True
                        
                elif pick == 'away':
                    bet_placed = True
                    odds = row.get('odds_away_ml')
                    sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if odds < 0 else 'Underdog'
                    if row['away_score'] > row['home_score']:
                        won = True

        elif bet_type == 'Spread':
            # Threshold is Points Edge
            # Assumption: spread_pred is Predicted Home Margin (Home - Away)
            # Assumption: odds_home_spread is Home Line (e.g. +2.5)
            # Win Condition: Home Score + Line > Away Score
            
            spread_pred = row.get('spread_pred')
            line = row.get('odds_home_spread')
            
            if pd.notna(spread_pred) and pd.notna(line):
                # Predicted Cover Margin = Predicted Margin + Line
                # Example: Pred -1, Line +2.5 -> Cover Margin +1.5 (Home covers by 1.5)
                cover_margin = spread_pred + line
                
                if cover_margin > threshold:
                    # Bet Home Spread
                    bet_placed = True
                    # Favorite if Home Spread is negative (giving points)
                    sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if line < 0 else 'Underdog'
                    if (row['home_score'] + line) > row['away_score']:
                        won = True
                elif cover_margin < -threshold:
                    # Bet Away Spread (Opposite)
                    # Away Spread is -1 * Home Spread
                    # If Home line is +5 (Underdog), Away is -5 (Favorite)
                    sim_df.at[idx, 'bet_subgroup'] = 'Favorite' if (-line) < 0 else 'Underdog'
                    bet_placed = True
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

def main():
    st.title("🏀 NBA Model Dashboard")

    df = load_data()
    
    if df.empty:
        st.warning("No data returned from database.")
        st.stop() # Stop execution if no data

    # Sidebar Filters
    st.sidebar.header("Configuration")
    
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

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📅 Today's Games", "Recent Performance", "📊 Predictions Log & Analysis"])

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
            
            # 2. Picks
            # 2. Picks
            def get_ml_pick_display(row):
                # Using 5% threshold as default for "ML Pick" display recommendation
                # Ensure we have market_prob or compute it
                if 'market_prob' not in row:
                    row['market_prob'] = calculate_implied_prob(row['odds_home_ml'])
                
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
                'odds_home_ml',
                'ML Pick',
                'Spread Pick',
                'Total Pick'
            ]
            
            todays_games['Implied Home Prob (%)'] = todays_games['Implied Home Prob (%)'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else x)
            todays_games['Model Home Prob (%)'] = todays_games['Model Home Prob (%)'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else x)

            # Filter existing cols
            final_cols = [c for c in cols_to_show if c in todays_games.columns]
            
            st.dataframe(todays_games[final_cols])
            st.metric("Total Games", len(todays_games))

    with tab2:
        st.subheader("Model Performance: This Week's Predictions")
        
        # Filter to this week (Monday to today)
        today = datetime.date.today()
        start_of_week = today - datetime.timedelta(days=today.weekday())
        week_df = model_df[(model_df['game_date'] >= start_of_week) & (model_df['game_date'] < today)].copy()
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
            
            # Helper to calculate profit for a row in week_df
            def get_row_profit(row, pick_col, correct_col, odds_col=None, spread_total=False):
                if pd.isna(row.get(correct_col)) or pd.isna(row.get(pick_col)): return 0.0, None
                
                # Determine subgroup
                subgroup = None
                is_fav = False
                
                if spread_total:
                    # Spread or Total
                    if 'total' in pick_col: # Total
                         subgroup = 'Over' if row[pick_col] == 'over' else 'Under'
                    else: # Spread
                         # Check if pick was Home or Away
                         # If pick Home, check odds_home_spread. If < 0 Fav
                         line = row.get('odds_home_spread', 0)
                         if row[pick_col] == 'home':
                             is_fav = line < 0
                         else:
                             is_fav = (-line) < 0
                         subgroup = 'Favorite' if is_fav else 'Underdog'
                    odds = -110
                else:
                    # Moneyline
                    # Pick Home -> check odds_home_ml
                    odds = row.get('odds_home_ml') if row[pick_col] == 'home' else row.get('odds_away_ml')
                    if pd.isna(odds): return 0.0, None
                    subgroup = 'Favorite' if odds < 0 else 'Underdog'

                # Calculate Result
                won = row[correct_col]
                profit = 0.0
                if won:
                    profit = calculate_payout(odds)
                else:
                    profit = -100
                    
                return profit, subgroup

            # Compute Breakdowns
            # We need to construct mini DFS
            
            # ML Breakdown
            ml_data = []
            for _, row in week_df.iterrows():
                p, g = get_row_profit(row, 'model_ml_pick', 'ml_correct', spread_total=False)
                if g: ml_data.append({'profit': p, 'subgroup': g})
            
            # Spread Breakdown
            sp_data = []
            for _, row in week_df.iterrows():
                p, g = get_row_profit(row, 'model_spread_pick', 'spread_correct', spread_total=True)
                if g: sp_data.append({'profit': p, 'subgroup': g})

            # Total Breakdown
            tl_data = []
            for _, row in week_df.iterrows():
                 p, g = get_row_profit(row, 'model_total_pick', 'total_correct', spread_total=True)
                 if g: tl_data.append({'profit': p, 'subgroup': g})

            b1, b2, b3 = st.columns(3)
            
            with b1:
                st.caption("Moneyline Splits")
                if ml_data:
                    display_subgroup_metrics(pd.DataFrame(ml_data), 'subgroup', 'profit', 'ML')
                else: st.write("-")

            with b2:
                 st.caption("Spread Splits")
                 if sp_data:
                     display_subgroup_metrics(pd.DataFrame(sp_data), 'subgroup', 'profit', 'Spread')
                 else: st.write("-")

            with b3:
                 st.caption("Totals Splits")
                 if tl_data:
                      display_subgroup_metrics(pd.DataFrame(tl_data), 'subgroup', 'profit', 'Total')
                 else: st.write("-")
            
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
            st.markdown("##### Performance by Type")
            display_subgroup_metrics(sim_df[sim_df['bet_outcome'] != 'No Bet'], 'bet_subgroup', 'bet_result', bet_type)
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

if __name__ == "__main__":
    main()
