
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import lightgbm as lgb
from models.game_features import GameFeatureBuilder
from models.backtest import Backtester

# Config
logging.basicConfig(level=logging.INFO)

def main():
    # 1. Generate Training Set with Archive Odds AND Player Features
    db_path = r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db'
    archive_path = r'c:\Users\jorda\nba-model-hybrid\data\nba_odds_archive.csv'
    
    logging.info("Building Training Set (Archive + Player Features)...")
    builder = GameFeatureBuilder(db_path)
    # Filter for requested seasons
    seasons = ['2019-20', '2020-21', '2021-22', '2022-23']
    df = builder.build_training_set_with_archive(archive_path, seasons=seasons)
    
    # Save for inspection
    output_path = Path('data/processed/training_set_player.csv')
    df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df)} games to {output_path}")
    
    # 2. Walk-Forward Validation (Train model, get preds)
    logging.info("Running Walk-Forward Model...")
    
    # Features (same as baseline, but roster_bpm_home/away are implicitly INCLUDED because they are not excluded)
    cols_to_exclude = ['gameId', 'gameDate', 'seasonYear', 'teamId_home', 'teamId_away', 
                       'matchup_home', 'wl_home', 'wl_away', 'points_home', 'points_away',
                       'is_home_home', 'is_home_away', 'matchup_away',
                       'target_home_win', 'target_point_diff', 'target_total_pts',
                       'odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_away_spread',
                       'odds_total', 'odds_home_ml_open', 'odds_away_ml_open',
                       'odds_home_spread_open', 'odds_away_spread_open', 'odds_total_open']
    
    features = [c for c in df.columns if c not in cols_to_exclude and 'meta' not in c]
    logging.info(f"Training with features: {features}")
    
    target = 'target_home_win'
    
    preds_dfs = []
    
    # Sort by date
    df = df.sort_values('gameDate')
    
    for season in seasons:
        logging.info(f"Processing Season {season}...")
        test_df = df[df['seasonYear'] == season].copy()
        
        train_df = df[df['gameDate'] < test_df['gameDate'].min()]
        
        if train_df.empty:
            logging.warning(f"No training data for {season} (First season in set?). Skipping.")
            continue
            
        X_train = train_df[features]
        X_test = test_df[features]
        
        if X_test.empty: continue
        
        # 1. Moneyline Model (Classifier)
        y_train_ml = train_df['target_home_win']
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=-1)
        clf.fit(X_train, y_train_ml)
        test_df['prob_home_win'] = clf.predict_proba(X_test)[:, 1]
        
        # 2. Spread Model (Regressor for Point Diff)
        # Target: Home - Away
        y_train_spread = train_df['target_point_diff']
        reg_spread = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=-1)
        reg_spread.fit(X_train, y_train_spread)
        test_df['pred_point_diff'] = reg_spread.predict(X_test)
        
        # 3. Total Model (Regressor for Total Pts)
        # Target: Home + Away
        y_train_total = train_df['target_total_pts']
        reg_total = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=-1)
        reg_total.fit(X_train, y_train_total)
        test_df['pred_total'] = reg_total.predict(X_test)
        
        preds_dfs.append(test_df)
        
    if not preds_dfs:
        logging.error("No predictions generated. Check season alignment.")
        return

    full_preds = pd.concat(preds_dfs)
    full_preds.to_csv('data/processed/backtest_predictions_player.csv', index=False)
    
    # 3. Backtest All Markets
    backtester = Backtester(full_preds)
    
    logging.info("--- Moneyline Backtest ---")
    pnl_ml, roi_ml, bets_ml = backtester.run_backtest(edge_threshold=0.05)
    
    logging.info("--- Spread Backtest ---")
    pnl_spread, roi_spread, bets_spread = backtester.run_spread_backtest(edge_threshold=1.0)
    
    logging.info("--- Total Backtest ---")
    pnl_total, roi_total, bets_total = backtester.run_total_backtest(edge_threshold=1.0)
    
    OUTPUT_FILE = "backtest_travel_metrics.txt"
    PLOT_FILE = "bankroll_travel_combined.png"
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("Travel-Adjusted Backtest (2020-21 to 2022-23)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Moneyline (Edge > 5%):\n")
        f.write(f"  ROI: {roi_ml:.4f}\n")
        f.write(f"  PnL: {pnl_ml:.2f}\n")
        f.write(f"  Bets: {bets_ml}\n")
        f.write("-"*30 + "\n")
        f.write(f"Spread (Edge > 1.0 pts):\n")
        f.write(f"  ROI: {roi_spread:.4f}\n")
        f.write(f"  PnL: {pnl_spread:.2f}\n")
        f.write(f"  Bets: {bets_spread}\n")
        f.write("-"*30 + "\n")
        f.write(f"Total (Edge > 1.0 pts):\n")
        f.write(f"  ROI: {roi_total:.4f}\n")
        f.write(f"  PnL: {pnl_total:.2f}\n")
        f.write(f"  Bets: {bets_total}\n")

    # Plot Combined Bankroll
    def get_daily(df_res, name):
        if 'gameDate' not in df_res.columns: return pd.Series()
        return df_res.groupby('gameDate')['pnl_total'].sum().cumsum().rename(name)

    daily_ml = get_daily(backtester.df, "Moneyline")
    daily_spread = get_daily(backtester.df_spread, "Spread")
    daily_total = get_daily(backtester.df_total, "Total")
    
    plt.figure(figsize=(12, 6))
    if not daily_ml.empty: plt.plot(daily_ml.index, daily_ml, label=f'Moneyline (ROI {roi_ml:.1%})')
    if not daily_spread.empty: plt.plot(daily_spread.index, daily_spread, label=f'Spread (ROI {roi_spread:.1%})')
    if not daily_total.empty: plt.plot(daily_total.index, daily_total, label=f'Total (ROI {roi_total:.1%})')
    
    plt.title(f"Player-Adjusted: Multi-Market Bankroll")
    plt.xlabel("Date")
    plt.ylabel("Profit ($100 unit)")
    plt.legend()
    plt.grid(True)
    plt.savefig('bankroll_player_combined.png')
    logging.info("Analysis Complete. Results in backtest_player_metrics.txt")

if __name__ == "__main__":
    main()
