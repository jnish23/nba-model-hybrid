
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

class Backtester:
    def __init__(self, df):
        self.df = df.copy()
        
    def run_backtest(self, prob_col='prob_home_win', edge_threshold=0.05):
        """
        Run betting simulation.
        Bet when Model Prob > Implied Prob + Threshold.
        """
        logging.info("Running Backtest...")
        
        # 1. Calculate Implied Probs
        # American Odds to Decimal: >0 -> (Odds/100) + 1; <0 -> (100/|Odds|) + 1
        def american_to_decimal(series):
            return np.where(series > 0, (series/100)+1, np.where(series < 0, (100/np.abs(series))+1, np.nan))

        # Uses Closing Odds for now (can switch to Opening)
        # A game without a price cannot be bet: the price is NaN, so no edge clears the threshold. (It used to be filled with
        # -110, which invented a price for every game that had no moneyline and let the model "bet" it.)
        self.df['decimal_home'] = american_to_decimal(self.df['odds_home_ml'])
        self.df['decimal_away'] = american_to_decimal(self.df['odds_away_ml'])
        
        self.df['implied_home'] = 1 / self.df['decimal_home']
        self.df['implied_away'] = 1 / self.df['decimal_away']
        
        # 2. Identify Bets
        # Bet Home
        self.df['bet_home'] = (self.df[prob_col] - self.df['implied_home']) > edge_threshold
        # Bet Away
        self.df['bet_away'] = ((1 - self.df[prob_col]) - self.df['implied_away']) > edge_threshold
        
        # 3. Calculate Results
        # If bet home and home wins -> Profit = Stake * (Decimal - 1)
        # If bet home and lose -> Loss = Stake
        stake = 100
        
        self.df['pnl_home'] = np.where(
            self.df['bet_home'],
            np.where(self.df['target_home_win'] == 1, stake * (self.df['decimal_home'] - 1), -stake),
            0
        )
        
        self.df['pnl_away'] = np.where(
            self.df['bet_away'],
            np.where(self.df['target_home_win'] == 0, stake * (self.df['decimal_away'] - 1), -stake),
            0
        )
        
        self.df['pnl_total'] = self.df['pnl_home'] + self.df['pnl_away']
        self.df['bet_count'] = self.df['bet_home'].astype(int) + self.df['bet_away'].astype(int)
        
        # 4. Calculate EV (Expected Value per unit bet)
        # EV = P(Win) * (Decimal - 1) - P(Loss) * 1
        #    = P * (Decimal - 1) - (1 - P)
        #    = P*Decimal - P - 1 + P
        #    = P * Decimal - 1
        self.df['ev_home'] = self.df[prob_col] * self.df['decimal_home'] - 1
        self.df['ev_away'] = (1 - self.df[prob_col]) * self.df['decimal_away'] - 1
        
        total_pnl = self.df['pnl_total'].sum()
        total_bets = self.df['bet_count'].sum()
        roi = (total_pnl / (total_bets * stake)) if total_bets > 0 else 0
        
        logging.info(f"Backtest Results: Bets={total_bets}, PnL=${total_pnl:.2f}, ROI={roi:.2%}")
        return total_pnl, roi, total_bets

    def run_spread_backtest(self, pred_col='pred_point_diff', edge_threshold=1.0):
        """
        Backtest against the Spread.
        Condition: ABS(ModelDiff - SpreadLine) > Edge
        Bet strategy:
        - If Model says Home wins by 10, Spread is -5: Model Edge is 5 points -> Bet Home Cover.
        - If Model says Home wins by 2, Spread is -5: Model Edge is -3 points (Model says Away Cover) -> Bet Away Cover.
        
        Note: Spread is usually "Home -5" meaning Home must win by >5.
        Point Diff is defined as Home - Away.
        So Spread -5 corresponds to Diff +5?
        Standard notation: Home -5.5.
        If Home scores 100, Away 90. Diff = +10.
        Cover: 10 > 5.5 -> Home Cover.
        
        Our `target_point_diff` is Home - Away.
        Our `odds_home_spread` is like -5.5.
        So we compare `pred_point_diff` vs `(-1 * odds_home_spread)`?
        Wait, usually spread is stored as "-5.5".
        To cover, Home Pts - Away Pts + Spread > 0? No.
        Home Pts - Away Pts > -Spread.
        Example: Spread -5.5. Home needs to win by 6.
        Diff > 5.5.
        
        Let's assume `odds_home_spread` is the number we need to beat.
        If spread is negative (-5.5), favorite.
        Winning condition: Diff + Spread > 0.
        Example: Diff +10. Spread -5.5. 10 + (-5.5) = 4.5 > 0. Win.
        
        Model Logic:
        Predicted Diff = 10.
        Spread = -5.5.
        Predicted Cover Margin = 10 + (-5.5) = 4.5.
        If Margin > Edge (e.g. 1.0), Bet Home Cover.
        If Margin < -Edge (e.g. -1.0), Bet Away Cover.
        """
        logging.info("Running Spread Backtest...")
        
        # Filter for valid spread lines
        df = self.df[self.df['odds_home_spread'].notna()].copy()
        
        # Calculate predicted cover margin
        # Spread is typically negative for favorite, positive for underdog.
        # Home Cover Condition: Actual Diff > -Spread (if spread is "points to give")
        # Standard DB representation: spread is the value added to home score?
        # Let's verify with actual data or assume standard:
        # Home -5.5 -> odds_home_spread = -5.5.
        # Home 100, Away 90. Diff = 10. Result = 10 + (-5.5) = 4.5 > 0. Home Covers.
        
        df['pred_margin'] = df[pred_col] + df['odds_home_spread']
        
        # Bet Logic
        df['bet_home'] = df['pred_margin'] > edge_threshold
        df['bet_away'] = df['pred_margin'] < -edge_threshold
        
        # PnL (Standard -110 odds = 1.909 decimal)
        decimal_odds = 1.909
        stake = 100
        
        # Determine actual outcome
        df['actual_margin'] = df['target_point_diff'] + df['odds_home_spread']
        df['home_cover'] = df['actual_margin'] > 0
        df['push'] = df['actual_margin'] == 0
        
        # Calculate PnL
        df['pnl'] = 0.0
        
        # Home Bets
        # Win
        df.loc[df['bet_home'] & df['home_cover'], 'pnl'] += stake * (decimal_odds - 1)
        # Loss (and not push)
        df.loc[df['bet_home'] & (~df['home_cover']) & (~df['push']), 'pnl'] -= stake
        
        # Away Bets
        # Win (Home did NOT cover, and no push)
        df.loc[df['bet_away'] & (~df['home_cover']) & (~df['push']), 'pnl'] += stake * (decimal_odds - 1)
        # Loss (Home Covered)
        df.loc[df['bet_away'] & df['home_cover'], 'pnl'] -= stake
        
        # Stats
        df['bet_count'] = df['bet_home'].astype(int) + df['bet_away'].astype(int)
        df['pnl_total'] = df['pnl']
        total_pnl = df['pnl_total'].sum()
        total_bets = df['bet_count'].sum()
        roi = (total_pnl / (total_bets * stake)) if total_bets > 0 else 0
        
        logging.info(f"Spread Backtest: Bets={total_bets}, PnL=${total_pnl:.2f}, ROI={roi:.2%}")
        
        # Store for plotting
        self.df_spread = df
        return total_pnl, roi, total_bets

    def run_total_backtest(self, pred_col='pred_total', edge_threshold=1.0):
        """
        Backtest Totals (Over/Under).
        Condition: ABS(ModelTotal - Line) > Edge
        """
        logging.info("Running Total Backtest...")
        df = self.df[self.df['odds_total'].notna()].copy()
        
        df['diff_vs_line'] = df[pred_col] - df['odds_total']
        
        # Bet Over: Model says more points than line
        df['bet_over'] = df['diff_vs_line'] > edge_threshold
        # Bet Under: Model says fewer points
        df['bet_under'] = df['diff_vs_line'] < -edge_threshold
        
        # Standard -110
        decimal_odds = 1.909
        stake = 100
        
        df['actual_total'] = df['target_total_pts']
        df['over_hit'] = df['actual_total'] > df['odds_total']
        df['push'] = df['actual_total'] == df['odds_total']
        
        df['pnl'] = 0.0
        
        # Over Bets
        df.loc[df['bet_over'] & df['over_hit'], 'pnl'] += stake * (decimal_odds - 1)
        df.loc[df['bet_over'] & (~df['over_hit']) & (~df['push']), 'pnl'] -= stake
        
        # Under Bets
        df.loc[df['bet_under'] & (~df['over_hit']) & (~df['push']), 'pnl'] += stake * (decimal_odds - 1)
        df.loc[df['bet_under'] & df['over_hit'], 'pnl'] -= stake
        
        # Stats
        df['bet_count'] = df['bet_over'].astype(int) + df['bet_under'].astype(int)
        df['pnl_total'] = df['pnl']
        total_pnl = df['pnl_total'].sum()
        total_bets = df['bet_count'].sum()
        roi = (total_pnl / (total_bets * stake)) if total_bets > 0 else 0
        
        logging.info(f"Total Backtest: Bets={total_bets}, PnL=${total_pnl:.2f}, ROI={roi:.2%}")
        self.df_total = df
        return total_pnl, roi, total_bets

    def plot_calibration(self, prob_col='prob_home_win', n_bins=10):
        """
        Plot Reliability Diagram.
        """
        prob_true, prob_pred = calibration_curve(self.df['target_home_win'], self.df[prob_col], n_bins=n_bins)
        
        plt.figure(figsize=(8, 8))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid()
        
        # Save plot? or return
        # plt.savefig('calibration.png')
        # plt.close()
        return prob_true, prob_pred

    def analyze_clv(self):
        """
        Compare Bet Odds vs Closing Odds (if we bet Opening).
        Requires Opening Odds presence.
        """
        pass
