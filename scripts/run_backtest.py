
import pandas as pd
import logging
from pathlib import Path
from team.train_baseline import BaselineModel
from team.backtest import Backtester
import matplotlib.pyplot as plt

# Config
logging.basicConfig(level=logging.INFO)

def main():
    data_path = Path('data/processed/training_set_v1.csv')
    
    # 1. Load Data (using BaselineModel's loader for consistency)
    logging.info("Loading Data...")
    if not data_path.exists():
        logging.error(f"Data not found: {data_path}")
        return

    # We can use BaselineModel just to load/prep, or load manually.
    # Manual load to keep all cols
    df = pd.read_csv(data_path)
    # Ensure date
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    
    # 2. Get Probs (Need a trained model or train on fly)
    # For backtest, we need OUT OF SAMPLE probs.
    # So we should run the Walk-Forward Validation and SAVE probs.
    
    # Re-instantiate Baseline to run WF and get preds
    logging.info("Running Walk-Forward Model to get predictions...")
    model = BaselineModel(data_path)
    
    # We need to modify BaselineModel to return the dataframe WITH predictions
    # Or we can do it here manually similar to backtest_walk_forward
    # But BaselineModel logic is good.
    # Let's import the class and use it. 
    # Actually BaselineModel.backtest_walk_forward merely prints metrics.
    # We need predictions stored.
    
    # Quick fix: Run predictions generation here
    preds_dfs = []
    
    model.load_data() # This sets model.df
    
    seasons = sorted(model.df['seasonYear'].unique())
    start_season = '2019-20'
    try:
        start_idx = seasons.index(start_season)
    except:
        start_idx = 0
        
    import lightgbm as lgb
    
    for i in range(start_idx, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]
        
        train_df = model.df[model.df['seasonYear'].isin(train_seasons)]
        test_df = model.df[model.df['seasonYear'] == test_season].copy()
        
        if train_df.empty or test_df.empty: continue
        
        X_train = train_df[model.features]
        y_train = train_df[model.target]
        X_test = test_df[model.features]
        
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=-1)
        clf.fit(X_train, y_train)
        
        # Predict Probs
        probs = clf.predict_proba(X_test)[:, 1]
        test_df['prob_home_win'] = probs
        preds_dfs.append(test_df)
        
    full_preds = pd.concat(preds_dfs)
    full_preds.to_csv('data/processed/backtest_predictions.csv', index=False)
    logging.info(f"Generated predictions for {len(full_preds)} games (Saved to CSV).")
    
    # 3. Run Backtest
    backtester = Backtester(full_preds)
    total_pnl, roi, total_bets = backtester.run_backtest(edge_threshold=0.05)
    
    logging.info("--- Verification Steps ---")
    
    with open('backtest_metrics.txt', 'w') as f:
        f.write(f"ROI: {roi:.4f}\n")
        f.write(f"PnL: {total_pnl:.2f}\n")
        f.write(f"Bets: {total_bets}\n")
    
    # A. Bankroll Graph
    backtester.df['cumulative_pnl'] = backtester.df['pnl_total'].cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(backtester.df['gameDate'], backtester.df['cumulative_pnl'])
    plt.title(f"Bankroll Over Time (ROI: {roi:.2%})")
    plt.xlabel("Date")
    plt.ylabel("Profit ($100 unit)")
    plt.grid(True)
    plt.savefig('bankroll_over_time.png')
    logging.info("Saved bankroll_over_time.png")
    
    # B. Inspect Specific Games
    logging.info("\n--- Significant Bets Inspection ---")
    # Filter for bets placed
    bets = backtester.df[backtester.df['bet_count'] > 0].copy()
    if not bets.empty:
        # Sort by PnL to see biggest wins/losses
        bets = bets.sort_values('pnl_total', ascending=False)
        
        cols = ['gameDate', 'matchup_home', 'prob_home_win', 'implied_home', 'odds_home_ml', 'target_home_win', 'pnl_total']
        
        logging.info("Top 3 Wins:")
        print(bets[cols].head(3).to_string(index=False))
        
        logging.info("\nTop 3 Losses:")
        print(bets[cols].tail(3).to_string(index=False))
        
        logging.info("\n--- Worst Days Analysis ---")
        daily_pnl = backtester.df.groupby('gameDate')['pnl_total'].sum().sort_values()
        print(daily_pnl.head(5))
        
        worst_date = daily_pnl.index[0]
        logging.info(f"\nInspecting Worst Day: {worst_date.date()}")
        bad_day_bets = backtester.df[backtester.df['gameDate'] == worst_date]
        print(bad_day_bets[['matchup_home', 'prob_home_win', 'odds_home_ml', 'target_home_win', 'bet_home', 'bet_away', 'pnl_total']].to_string(index=False))

        logging.info("\nRandom Sample (5 bets):")
        print(bets[cols].sample(min(5, len(bets))).to_string(index=False))
        
    # C. Leakage Sanity Check (Manual Verification)
    logging.info("\n--- Leakage Check (Manual) ---")
    # Pick a random row from training set
    sample = backtester.df.sample(1).iloc[0]
    gdate = pd.to_datetime(sample['gameDate'])
    tid = sample['teamId_home']
    
    logging.info(f"Checking Game: {sample['matchup_home']} on {gdate.date()}")
    logging.info(f"Feature 'days_rest_home': {sample.get('days_rest_home', 'N/A')}")
    
    # Query DB to confirm
    import sqlite3
    with sqlite3.connect('data/nba_stats.db') as conn:
        # Find previous game for this team
        q = f"""
        SELECT gameDate 
        FROM team_game_full 
        WHERE teamId = {tid} AND gameDate < '{gdate.strftime('%Y-%m-%d')}'
        ORDER BY gameDate DESC LIMIT 1
        """
        prev_game = pd.read_sql(q, conn)
        
    if not prev_game.empty:
        prev_date = pd.to_datetime(prev_game.iloc[0]['gameDate'])
        calc_rest = (gdate - prev_date).days
        logging.info(f"DB Previous Game: {prev_date.date()}")
        logging.info(f"Calculated Rest: {calc_rest}")
        
        # Determine if match (logic in code clamps at 7)
        feat_rest = sample.get('days_rest_home')
        if feat_rest is not None:
            # Code clamps to 7? Wait, I added that manually in generate script?
            # Yes: np.where(feats['days_rest'] > 7, 7, feats['days_rest'])
            expected = min(calc_rest, 7)
            if float(feat_rest) == float(expected):
                 logging.info("SUCCESS: Feature matches History (No Leakage detected for Rest metrics)")
            else:
                 logging.warning(f"MISMATCH: Feature {feat_rest} != Calculated {expected}")
    else:
        logging.info("No previous game found (Season Opener?).")

if __name__ == "__main__":
    main()
