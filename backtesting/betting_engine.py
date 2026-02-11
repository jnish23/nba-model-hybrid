import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class BettingEngine:
    def __init__(self):
        pass

    def prepare_betting_data(self, predictions, odds, id_map):
        """
        Joins Model Predictions with Historical Odds.
        """
        # A. Aggregate Predictions using ID map
        home_preds = predictions[predictions['isHome'] == 1].copy()
        away_preds = predictions[predictions['isHome'] == 0].copy()
        
        home_preds['home_abbr'] = home_preds['teamId'].map(id_map)
        away_preds['away_abbr'] = away_preds['teamId'].map(id_map)
        
        game_preds = pd.merge(
            home_preds[['gameId', 'gameDate', 'teamId', 'home_abbr', 'points', 'pred_points', 'season_end']],
            away_preds[['gameId', 'teamId', 'away_abbr', 'points', 'pred_points']],
            on='gameId', suffixes=('_home', '_away')
        )
        
        # B. Prepare Odds
        odds_clean = odds.copy()
        # Convert to datetime if string
        if odds_clean['GAME_DATE'].dtype == 'object':
            odds_clean['GAME_DATE'] = pd.to_datetime(odds_clean['GAME_DATE'])
        
        odds_clean = odds_clean.rename(columns={
            'GAME_DATE': 'gameDate',
            'home_team_abbr': 'home_abbr',
            'away_team_abbr': 'away_abbr'
        })
        
        # C. Join predictions + odds
        full_df = pd.merge(
            game_preds,
            odds_clean,
            left_on=['gameDate', 'home_abbr'],
            right_on=['gameDate', 'home_abbr'],
            how='inner' 
        )
        
        # D. Calculate Edges
        full_df['pred_margin'] = full_df['pred_points_home'] - full_df['pred_points_away']
        full_df['actual_margin'] = full_df['points_home'] - full_df['points_away']
        
        # Market Spread (DraftKings)
        full_df['market_margin'] = -1 * full_df['home_spread_draftkings']
        full_df = full_df.sort_values(['gameDate'])
        
        return full_df

    def run_backtest(self, df, threshold=1.5):
        """
        Places bets where |Pred_Margin - Market_Margin| > threshold.
        """
        df = df.copy()
        
        df['bet_home'] = df['pred_margin'] > (df['market_margin'] + threshold)
        df['bet_away'] = df['pred_margin'] < (df['market_margin'] - threshold)
        
        df['home_cover_result'] = np.sign(df['actual_margin'] - df['market_margin'])
        df['away_cover_result'] = -1 * df['home_cover_result']
        
        price = 0.909 # -110 odds
        df['pnl'] = 0.0
        
        df.loc[df['bet_home'] & (df['home_cover_result'] == 1), 'pnl'] += price
        df.loc[df['bet_home'] & (df['home_cover_result'] == -1), 'pnl'] -= 1.0
        
        df.loc[df['bet_away'] & (df['away_cover_result'] == 1), 'pnl'] += price
        df.loc[df['bet_away'] & (df['away_cover_result'] == -1), 'pnl'] -= 1.0
        
        return df

    def calculate_pnl(self, df):
        """
        Calculates PnL for a dataframe that already has 'is_bet_home' and 'is_bet_away' columns.
        """
        df = df.copy()
        
        # Ensure cover results exist
        if 'home_cover_result' not in df.columns:
            df['home_cover_result'] = np.sign(df['actual_margin'] - df['market_margin'])
            df['away_cover_result'] = -1 * df['home_cover_result']
            
        price = 0.909 # -110 odds
        df['pnl'] = 0.0
        
        # Rename for compatibility if needed (validator uses is_bet_home, run_backtest uses bet_home)
        if 'bet_home' not in df.columns and 'is_bet_home' in df.columns:
            df['bet_home'] = df['is_bet_home']
        if 'bet_away' not in df.columns and 'is_bet_away' in df.columns:
            df['bet_away'] = df['is_bet_away']
            
        df.loc[df['bet_home'] & (df['home_cover_result'] == 1), 'pnl'] += price
        df.loc[df['bet_home'] & (df['home_cover_result'] == -1), 'pnl'] -= 1.0
        
        df.loc[df['bet_away'] & (df['away_cover_result'] == 1), 'pnl'] += price
        df.loc[df['bet_away'] & (df['away_cover_result'] == -1), 'pnl'] -= 1.0
        
        return df

    def generate_report(self, results, output_dir="reports", config=None):
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Calculation Logic
        bets_made = results['bet_home'].sum() + results['bet_away'].sum()
        total_pnl = results['pnl'].sum()
        roi = total_pnl / bets_made if bets_made > 0 else 0
        
        wins = results.loc[(results['bet_home'] & (results['home_cover_result'] == 1)) | 
                        (results['bet_away'] & (results['away_cover_result'] == 1))].shape[0]
        win_rate = wins / bets_made if bets_made > 0 else 0

        # 1. Save results to CSV
        csv_path = os.path.join(output_dir, f"backtest_results_{config['features']}_{config['model_type']}.csv")
        results.to_csv(csv_path, index=False)

        # 2. Print Summary
        print("\n--- Betting Backtest Report ---")
        print(f"Total Bets: {bets_made}")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Total Units Won: {total_pnl:.2f}")
        print(f"ROI: {roi*100:.2f}%")
        print(f"Full results saved to: {csv_path}")

        # 3. Handle Plotting
        results['date_str'] = results['gameDate'].dt.date
        daily_pnl = results.groupby('gameDate')['pnl'].sum().sort_index().cumsum()
        
        plt.figure(figsize=(12, 6))
        daily_pnl.plot(title='Cumulative Profit (Units) - Modular Backtest')
        plt.xlabel('Date')
        plt.ylabel('Units Won')
        plt.grid(True)
        
        # 4. Market Benchmark
        # Market prediction for margin is exactly 'market_margin' (which is -Spread)
        # Model prediction is 'pred_margin'
        
        # Filter to where we have valid market lines
        valid_bench = results.dropna(subset=['market_margin', 'pred_margin', 'actual_margin'])
        
        if not valid_bench.empty:
            market_mae = (valid_bench['actual_margin'] - valid_bench['market_margin']).abs().mean()
            model_mae = (valid_bench['actual_margin'] - valid_bench['pred_margin']).abs().mean()
            
            print("\n--- Market Benchmark ---")
            print(f"Market MAE (Margin): {market_mae:.2f}")
            print(f"Model MAE (Margin):  {model_mae:.2f}")
            print(f"Difference:          {model_mae - market_mae:.2f} (Negative is better)")
            
            # Save benchmark to text file
            with open(os.path.join(output_dir, f"benchmark_{config['features']}_{config['model_type']}.txt"), 'w') as f:
                f.write(f"Market MAE: {market_mae:.2f}\n")
                f.write(f"Model MAE: {model_mae:.2f}\n")
                f.write(f"Diff: {model_mae - market_mae:.2f}\n")

        # Save the plot
        plot_path = os.path.join(output_dir, f"cumulative_pnl_{config['features']}_{config['model_type']}.png")
        plt.savefig(plot_path)
        plt.close()  # Important: closes the figure to free up memory
        
        print(f"Plot saved to: {plot_path}")

