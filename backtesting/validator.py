import pandas as pd
import numpy as np

class WalkForwardValidator:
    def __init__(self):
        pass

    def optimize_thresholds(self, betting_df: pd.DataFrame, start_test_season=2019):
        """
        DYNAMIC Walk-Forward Optimization to prevent data leakage.
        For each season, we pick the best threshold based ONLY on previous seasons.
        """
        print("\n--- Running Dynamic Walk-Forward Optimization (No Leakage) ---")
        
        # Ensure season info is present (we might need to merge it back if missing)
        # betting_df should have 'gameDate'
        betting_df['season'] = betting_df['gameDate'].dt.year # Approx, since season spans years. 
        # Better: Use the season column if it exists or derive it properly
        # In run_pipeline, we had season_end. Let's assume it's passed or available.
        if 'season_end' not in betting_df.columns:
            # Re-derive standard season end (e.g. June 2023 -> 2023)
             betting_df['season_end'] = np.where(betting_df['gameDate'].dt.month > 9, 
                                                 betting_df['gameDate'].dt.year + 1, 
                                                 betting_df['gameDate'].dt.year)

        seasons = sorted(betting_df['season_end'].unique())
        thresholds = np.arange(1.0, 10.0, 0.5)
        
        # We need at least one historic season to optimize on
        test_seasons = [s for s in seasons if s >= start_test_season]
        
        wf_bets = []
        
        for season in test_seasons:
            # 1. OPTIMIZATION WINDOW: All history BEFORE this season
            history = betting_df[betting_df['season_end'] < season].copy()
            
            if len(history) < 500: # Need minimum sample
                print(f"Skipping Season {season}: Insufficient history ({len(history)} games)")
                continue
                
            # Find Best Threshold on History
            best_thresh = 0
            best_roi = -999.0
            
            # Add cover results for history
            history['home_cover_result'] = np.sign(history['actual_margin'] - history['market_margin'])
            history['away_cover_result'] = -1 * history['home_cover_result']
            price = 0.909
            
            for t in thresholds:
                bet_home = history['pred_margin'] > (history['market_margin'] + t)
                bet_away = history['pred_margin'] < (history['market_margin'] - t)
                
                # Fast PnL Calc
                pnl = 0.0
                pnl += (bet_home & (history['home_cover_result'] == 1)).sum() * price
                pnl -= (bet_home & (history['home_cover_result'] == -1)).sum()
                pnl += (bet_away & (history['away_cover_result'] == 1)).sum() * price
                pnl -= (bet_away & (history['away_cover_result'] == -1)).sum()
                
                bets = bet_home.sum() + bet_away.sum()
                roi = pnl / bets if bets > 0 else -999.0
                
                if roi > best_roi and bets > 100: # Min bet filter for stability
                    best_roi = roi
                    best_thresh = t
            
            if best_roi == -999.0:
                best_thresh = 3.0  # Sensible default
                print(f"Season {season}: Using default threshold {best_thresh} (No valid history)")
            
            else:   
                print(f"Season {season}: Selected Threshold {best_thresh} (Hist ROI: {best_roi*100:.2f}%)")
            
            # 2. TEST WINDOW: Apply selected threshold to CURRENT season
            current = betting_df[betting_df['season_end'] == season].copy()
            
            # Apply Best Threshold
            current['is_bet_home'] = current['pred_margin'] > (current['market_margin'] + best_thresh)
            current['is_bet_away'] = current['pred_margin'] < (current['market_margin'] - best_thresh)
            current['selected_threshold'] = best_thresh
            
            # Filter to only actual bets
            active_bets = current[current['is_bet_home'] | current['is_bet_away']].copy()
            active_bets['bet_team_id'] = np.where(active_bets['is_bet_home'], active_bets['teamId_home'], active_bets['teamId_away'])
            active_bets['bet_type'] = np.where(active_bets['is_bet_home'], 'HOME', 'AWAY')
            
            wf_bets.append(active_bets)
            
        if not wf_bets:
             return pd.DataFrame()
             
        return pd.concat(wf_bets)

    def print_optimization_report(self, results_df):
        print("\nOptimization Results (Sorted by ROI):")
        # Filter for meaningful sample size (e.g., at least 100 bets per year approx? say 500 total)
        valid_results = results_df[results_df['bets'] > 500].sort_values('roi', ascending=False)
        
        if valid_results.empty:
            print("No strategies met the minimum bet count (500). Showing all:")
            print(results_df)
        else:
            print(valid_results.to_string(index=False))
            
        best_strat = results_df.sort_values('units_won', ascending=False).iloc[0]
        print(f"\nBest Strategy by Units Won: Threshold {best_strat['threshold']} -> {best_strat['units_won']:.2f} Units ({best_strat['roi']:.2f}% ROI)")
