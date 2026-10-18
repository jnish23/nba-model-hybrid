import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure we can import our betting logic
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from team.betting import (calculate_implied_prob, calculate_payout, get_spread_bet, get_total_bet,
                            DEFAULT_SPREAD_STD, DEFAULT_TOTAL_STD, STANDARD_PRICE)

def load_data(csv_path="./models/saved/backtest_predictions_calibrated.csv"):
    if csv_path:
        df = pd.read_csv(csv_path)
        # Compute scores from spread_actual (home-away) and total_actual (home+away)
        df['home_score'] = (df['total_actual'] + df['spread_actual']) / 2
        df['away_score'] = df['total_actual'] - df['home_score']
        # Map columns to match db schema
        rename_map = {
            'gameDate': 'game_date',
            'gameId': 'game_id',
            'teamId_home': 'home_team', # ids fine for grouping
            'teamId_away': 'away_team'
        }
        df.rename(columns=rename_map, inplace=True)
    else:
        db_path = Path(__file__).resolve().parent.parent / "data" / "nba_stats.db"
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM predictions_log_v2 WHERE status = 'COMPLETED'", conn)
        conn.close()
        
        # Deduplicate and clean
        df = df.sort_values(by=['game_date', 'game_id'], ascending=[False, True])
        df = df.drop_duplicates(subset=['game_id'], keep='first')
        
    for c in ['home_score', 'away_score', 'ml_prob', 'odds_home_ml', 'odds_away_ml', 
              'odds_home_spread', 'odds_total', 'spread_pred', 'total_pred']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    return df

def analyze_calibration(df):
    print("=== MODEL CALIBRATION (MONEYLINE) ===")
    df = df[df['ml_prob'].notna() & df['home_score'].notna()].copy()
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    # Bin by 10%
    bins = np.arange(0, 1.1, 0.1)
    df['prob_bin'] = pd.cut(df['ml_prob'], bins=bins)
    
    cal_df = df.groupby('prob_bin', observed=False)['home_win'].agg(['count', 'mean']).reset_index()
    cal_df.rename(columns={'count': 'Games', 'mean': 'Actual Win Rate'}, inplace=True)
    cal_df['Predicted Win Rate'] = cal_df['prob_bin'].apply(lambda x: float(x.mid) if pd.notna(x) else np.nan).astype(float)
    cal_df['Diff'] = cal_df['Actual Win Rate'] - cal_df['Predicted Win Rate']
    
    print(cal_df[['prob_bin', 'Games', 'Predicted Win Rate', 'Actual Win Rate', 'Diff']].to_string(index=False))
    print()

def get_ml_bets(df, threshold):
    bets = []
    for _, g in df.iterrows():
        if pd.isna(g['ml_prob']) or pd.isna(g['odds_home_ml']) or pd.isna(g['odds_away_ml']):
            continue
            
        home_implied = calculate_implied_prob(g['odds_home_ml'])
        away_prob = 1.0 - g['ml_prob']
        away_implied = calculate_implied_prob(g['odds_away_ml'])
        
        home_edge = g['ml_prob'] - home_implied
        away_edge = away_prob - away_implied
        
        # Determine subset (Home/Away -> Fav/Dog)
        is_home_fav = g['odds_home_ml'] < 0
        
        bet_side = None
        odds = None
        edge = None
        won = None
        subset = None
        
        if home_edge > threshold and home_edge > away_edge:
            bet_side = 'Home'
            odds = g['odds_home_ml']
            edge = home_edge
            won = g['home_score'] > g['away_score']
            subset = 'Home Fav' if is_home_fav else 'Home Dog'
        elif away_edge > threshold:
            bet_side = 'Away'
            odds = g['odds_away_ml']
            edge = away_edge
            won = g['away_score'] > g['home_score']
            subset = 'Away Dog' if is_home_fav else 'Away Fav'
            
        if bet_side:
            payout = calculate_payout(odds) if won else -100
            bets.append({
                'season': g.get('seasonYear', 'Unknown'),
                'side': bet_side, 'subset': subset, 'edge': edge, 'won': won, 'profit': payout
            })
    return pd.DataFrame(bets)

def _std(g, col, default):
    v = g.get(col)
    return float(v) if v is not None and pd.notna(v) else default


def get_spread_bets(df, threshold):
    """Spread bets under the shared rule (models/betting.get_spread_bet): bet the side whose
    P(cover) = Phi(point edge / residual std) beats the -110 break-even by more than `threshold`
    (a probability edge). The residual std is the fold's own calibration-holdout std."""
    bets = []
    for _, g in df.iterrows():
        if pd.isna(g['spread_pred']) or pd.isna(g['odds_home_spread']): continue

        sp = g['spread_pred']
        line = g['odds_home_spread']
        side, edge = get_spread_bet(sp, line, _std(g, 'spread_std', DEFAULT_SPREAD_STD), threshold)
        if side is None:
            continue
        is_home_fav = line < 0
        margin = g['home_score'] - g['away_score'] + line          # >0: home covers, 0: push
        push = margin == 0
        won = (margin > 0) if side == 'home' else (margin < 0)
        subset = (('Home Fav' if is_home_fav else 'Home Dog') if side == 'home'
                  else ('Away Dog' if is_home_fav else 'Away Fav'))
        profit = 0.0 if push else (calculate_payout(STANDARD_PRICE) if won else -100)
        bets.append({'season': g.get('seasonYear', 'Unknown'), 'side': side.title(), 'subset': subset,
                     'edge': edge, 'won': bool(won), 'push': bool(push), 'profit': profit})
    return pd.DataFrame(bets)

def get_totals_bets(df, threshold):
    """Totals bets under the same probability-edge rule."""
    bets = []
    for _, g in df.iterrows():
        if pd.isna(g['total_pred']) or pd.isna(g['odds_total']) or pd.isna(g['odds_home_spread']): continue

        line = g['odds_total']
        side, edge = get_total_bet(g['total_pred'], line, _std(g, 'total_std', DEFAULT_TOTAL_STD), threshold)
        if side is None:
            continue
        is_home_fav = g['odds_home_spread'] < 0
        actual_total = g['home_score'] + g['away_score']
        push = actual_total == line
        won = (actual_total > line) if side == 'over' else (actual_total < line)
        subset = f"{'Home Fav Game' if is_home_fav else 'Away Fav Game'} - {side.title()}"
        profit = 0.0 if push else (calculate_payout(STANDARD_PRICE) if won else -100)
        bets.append({'season': g.get('seasonYear', 'Unknown'), 'side': side.title(), 'subset': subset,
                     'edge': edge, 'won': bool(won), 'push': bool(push), 'profit': profit})
    return pd.DataFrame(bets)

def analyze_strategy(df_bets, name):
    print(f"=== {name.upper()} STRATEGY ANALYSIS ===")
    if df_bets.empty:
        print("No bets matched criteria.\n")
        return
        
    df_bets = pd.DataFrame(df_bets)
    
    # Overall
    if 'push' in df_bets.columns:
        pushes = int(df_bets['push'].sum())
        df_bets = df_bets[~df_bets['push']]          # a push returns the stake: not a bet
    else:
        pushes = 0
    bets = len(df_bets)
    if bets == 0:
        print("No bets matched criteria.\n")
        return
    wins = df_bets['won'].sum()
    profit = df_bets['profit'].sum()
    roi = (profit / (bets * 100)) * 100
    print(f"OVERALL: {wins}-{bets-wins} ({(wins/bets*100):.1f}%) | ROI: {roi:+.1f}% | Profit: {profit/100:+.2f}u"
          + (f" | pushes: {pushes}" if pushes else ""))
    
    # By subset
    summary = df_bets.groupby('subset').agg(
        Bets=('won', 'count'),
        Wins=('won', 'sum'),
        Profit=('profit', 'sum')
    )
    summary['Win%'] = (summary['Wins'] / summary['Bets']) * 100
    summary['ROI%'] = (summary['Profit'] / (summary['Bets'] * 100)) * 100
    summary['Profit(u)'] = summary['Profit'] / 100
    
    print("\nBreakdown by Scenario:")
    print(summary[['Bets', 'Wins', 'Win%', 'ROI%', 'Profit(u)']].sort_values('ROI%', ascending=False).to_string())
    
    # By Season
    if 'season' in df_bets.columns:
        season_summary = df_bets.groupby('season').agg(
            Bets=('won', 'count'),
            Wins=('won', 'sum'),
            Profit=('profit', 'sum')
        )
        season_summary['Win%'] = (season_summary['Wins'] / season_summary['Bets']) * 100
        season_summary['ROI%'] = (season_summary['Profit'] / (season_summary['Bets'] * 100)) * 100
        season_summary['Profit(u)'] = season_summary['Profit'] / 100
        print("\nBreakdown by Season:")
        print(season_summary[['Bets', 'Wins', 'Win%', 'ROI%', 'Profit(u)']].to_string())

    print()

def main():
    import sys
    if len(sys.argv) > 1:
        df = load_data(sys.argv[1])
        print(f"Loaded {len(df)} games from CSV: {sys.argv[1]}\n")
    else:
        df = load_data()
        print(f"Loaded {len(df)} completed, deduplicated games from DB.\n")
    
    analyze_calibration(df)
    
    # One threshold definition for every market: probability edge over the price's break-even
    for t in [0.00, 0.05, 0.10]:
        print(f"--- ML Edge > {t*100:.1f}% ---")
        analyze_strategy(get_ml_bets(df, t), f"Moneyline (Edge > {t*100}%)")

    for t in [0.0, 0.05, 0.10]:
        from team.betting import points_threshold
        print(f"--- Spread Edge > {t*100:.1f}% (= {points_threshold(DEFAULT_SPREAD_STD, t):.1f} pts at std {DEFAULT_SPREAD_STD}) ---")
        analyze_strategy(get_spread_bets(df, t), f"Spread (Edge > {t*100}%)")

    for t in [0.0, 0.05, 0.10]:
        print(f"--- Totals Edge > {t*100:.1f}% (= {points_threshold(DEFAULT_TOTAL_STD, t):.1f} pts at std {DEFAULT_TOTAL_STD}) ---")
        analyze_strategy(get_totals_bets(df, t), f"Totals (Edge > {t*100}%)")

if __name__ == '__main__':
    main()
