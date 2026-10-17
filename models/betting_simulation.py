# betting_simulation.py
import pandas as pd
import numpy as np
from pathlib import Path

from models.betting import points_threshold, DEFAULT_SPREAD_STD, DEFAULT_TOTAL_STD, DEFAULT_PROB_EDGE

class BettingSimulator:
    def __init__(self, predictions_path):
        self.df = pd.read_csv(predictions_path)
        self.df['gameDate'] = pd.to_datetime(self.df['gameDate'])
        
    def implied_prob_from_american(self, odds):
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)
    
    def simulate_moneyline(self, edge_threshold=0.05, flat_stake=100):
        """
        Bet when model probability exceeds implied probability by threshold.
        """
        results = []
        
        for _, game in self.df.iterrows():
            model_prob = game['ml_prob']
            
            # Check home bet
            if 'odds_home_ml' in game and pd.notna(game['odds_home_ml']):
                implied = self.implied_prob_from_american(game['odds_home_ml'])
                edge = model_prob - implied
                
                if edge > edge_threshold:
                    won = game['ml_actual'] == 1
                    payout = self._calc_payout(game['odds_home_ml'], flat_stake, won)
                    results.append({
                        'gameId': game['gameId'],
                        'date': game['gameDate'],
                        'bet_type': 'home_ml',
                        'model_prob': model_prob,
                        'implied_prob': implied,
                        'edge': edge,
                        'won': won,
                        'profit': payout
                    })
            
            # Check away bet
            if 'odds_away_ml' in game and pd.notna(game['odds_away_ml']):
                implied = self.implied_prob_from_american(game['odds_away_ml'])
                model_away = 1 - model_prob
                edge = model_away - implied
                
                if edge > edge_threshold:
                    won = game['ml_actual'] == 0
                    payout = self._calc_payout(game['odds_away_ml'], flat_stake, won)
                    results.append({
                        'gameId': game['gameId'],
                        'date': game['gameDate'],
                        'bet_type': 'away_ml',
                        'model_prob': model_away,
                        'implied_prob': implied,
                        'edge': edge,
                        'won': won,
                        'profit': payout
                    })
        
        return pd.DataFrame(results)

    def simulate_spread(self, threshold=None, flat_stake=100):
        """
        Bet on the spread if the predicted margin differs from the line by > threshold (points).
        threshold=None uses the shared rule's equivalent: the point edge at which P(cover) beats the
        -110 break-even by DEFAULT_PROB_EDGE (models/betting.py). Assumes standard -110 odds.
        """
        if threshold is None:
            threshold = points_threshold(DEFAULT_SPREAD_STD, DEFAULT_PROB_EDGE)
        results = []
        # Standard -110 price payout multiplier (Profit / Stake)
        standard_payout_mult = 100 / 110 

        for _, game in self.df.iterrows():
            if 'spread_pred' not in game or pd.isna(game['spread_pred']):
                continue

            pred_margin = game['spread_pred'] # Home - Away
            actual_margin = game['spread_actual']

            # Home Spread
            if 'odds_home_spread' in game and pd.notna(game['odds_home_spread']):
                line_home = game['odds_home_spread']
                # Condition: (Pred Margin + Line) > Threshold
                # Ex: Pred +5 (Home wins by 5), Line -3. Edge = 2.
                # Ex: Pred -5 (Away wins by 5), Line +7. (-5 + 7) = +2. Edge = 2.
                # Basically: Does our prediction cover the line with edge?
                
                # Check coverage: (Pred Margin + Line)
                # If > 0, we think Home covers. 
                # How much > 0? That's the edge.
                edge = (pred_margin + line_home)
                
                if edge > threshold:
                    # Bet Home Spread
                    # Win if Actual + Line > 0
                    if (actual_margin + line_home) > 0:
                        profit = flat_stake * standard_payout_mult
                        won = 1
                    elif (actual_margin + line_home) == 0:
                        profit = 0 # Push
                        won = 0.5 # Treat push as neutral or specific val? Let's use 0 profit.
                    else:
                        profit = -flat_stake
                        won = 0
                        
                    results.append({
                        'gameId': game['gameId'],
                        'date': game['gameDate'],
                        'bet_type': 'home_spread',
                        'pred_margin': pred_margin,
                        'line': line_home,
                        'edge': edge,
                        'won': won,
                        'profit': profit
                    })

            # Away Spread
            if 'odds_away_spread' in game and pd.notna(game['odds_away_spread']):
                line_away = game['odds_away_spread']
                # Away Margin = -(Home Margin)
                pred_margin_away = -pred_margin
                actual_margin_away = -actual_margin
                
                edge = (pred_margin_away + line_away)
                
                if edge > threshold:
                    # Bet Away Spread
                    if (actual_margin_away + line_away) > 0:
                        profit = flat_stake * standard_payout_mult
                        won = 1
                    elif (actual_margin_away + line_away) == 0:
                        profit = 0
                        won = 0.5
                    else:
                        profit = -flat_stake
                        won = 0
                        
                    results.append({
                        'gameId': game['gameId'],
                        'date': game['gameDate'],
                        'bet_type': 'away_spread',
                        'pred_margin': pred_margin, # Keep original context
                        'line': line_away,
                        'edge': edge,
                        'won': won,
                        'profit': profit
                    })
                    
        return pd.DataFrame(results)

    def simulate_total(self, threshold=None, flat_stake=100):
        """
        Bet Over/Under if predicted total differs from line by > threshold (points).
        threshold=None uses the shared rule's equivalent (see simulate_spread). Assumes -110 odds.
        """
        if threshold is None:
            threshold = points_threshold(DEFAULT_TOTAL_STD, DEFAULT_PROB_EDGE)
        results = []
        standard_payout_mult = 100 / 110
        
        for _, game in self.df.iterrows():
            if 'total_pred' not in game or pd.isna(game['total_pred']):
                continue
            if 'odds_total' not in game or pd.isna(game['odds_total']):
                continue
                
            pred_total = game['total_pred']
            actual_total = game['total_actual']
            line = game['odds_total']
            
            # Check Over
            if pred_total > (line + threshold):
                # Bet Over
                if actual_total > line:
                    profit = flat_stake * standard_payout_mult
                    won = 1
                elif actual_total == line:
                    profit = 0
                    won = 0.5
                else:
                    profit = -flat_stake
                    won = 0
                    
                results.append({
                    'gameId': game['gameId'],
                    'date': game['gameDate'],
                    'bet_type': 'over',
                    'pred_total': pred_total,
                    'line': line,
                    'edge': pred_total - line,
                    'won': won,
                    'profit': profit
                })
                
            # Check Under
            elif pred_total < (line - threshold):
                # Bet Under
                if actual_total < line:
                    profit = flat_stake * standard_payout_mult
                    won = 1
                elif actual_total == line:
                    profit = 0
                    won = 0.5
                else:
                    profit = -flat_stake
                    won = 0
                
                results.append({
                    'gameId': game['gameId'],
                    'date': game['gameDate'],
                    'bet_type': 'under',
                    'pred_total': pred_total,
                    'line': line,
                    'edge': line - pred_total, # Positive edge
                    'won': won,
                    'profit': profit
                })
                
        return pd.DataFrame(results)

    def _calc_payout(self, odds, stake, won):
        if not won:
            return -stake
        if odds > 0:
            return stake * (odds / 100)
        else:
            return stake * (100 / -odds)
    
    def summarize(self, bets_df, label="Bets"):
        print(f"\n--- {label} Summary ---")
        if bets_df.empty:
            print("No bets placed.")
            return
            
        print(f"Total bets: {len(bets_df)}")
        # Check if 'won' has 0.5 (pushes). Exclude from win rate? Or count as 0.5?
        # Typically Win% = Wins / (Wins + Losses). Pushes exclude.
        # But 'won' column is 1, 0, 0.5.
        wins = bets_df[bets_df['won'] == 1]
        losses = bets_df[bets_df['won'] == 0]
        pushes = bets_df[bets_df['won'] == 0.5]
        
        decided_bets = len(wins) + len(losses)
        if decided_bets > 0:
            win_rate = len(wins) / decided_bets
            print(f"Win rate (excl. push): {win_rate:.1%} ({len(wins)}-{len(losses)}-{len(pushes)})")
        else:
            print(f"Win rate: N/A (0-{len(losses)}-{len(pushes)})")

        total_profit = bets_df['profit'].sum()
        roi = total_profit / (len(bets_df) * 100) # Assuming flat 100 stake
        
        print(f"Total profit: ${total_profit:.2f}")
        print(f"ROI: {roi:.1%}")
        if 'edge' in bets_df.columns:
            print(f"Avg edge on bets: {bets_df['edge'].mean():.2f}")
        
        # By season
        bets_df['season'] = pd.to_datetime(bets_df['date']).dt.year
        # Or better separate by 'seasonYear' if available
        # But date year is rough proxy standard
        
        print("\nBy year:")
        annual = bets_df.groupby('season').agg({
            'won': lambda x: (x==1).sum(),
            'profit': ['count', 'sum']
        })
        # Clean up columns
        # Structure: won lambda, profit count, profit sum
        # annual.columns = ['wins', 'bets', 'profit']
        print(annual)

    def diagnose_moneyline_bets(self, edge_threshold=0.05):
        """Figure out why moneyline bets are losing."""
        df = self.df.copy()
        
        def implied_prob(odds):
            if pd.isna(odds) or odds == 0:
                return np.nan
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return -odds / (-odds + 100)
        
        df['market_prob'] = df['odds_home_ml'].apply(implied_prob)
        df['market_prob_away'] = df['odds_away_ml'].apply(implied_prob)
        df = df.dropna(subset=['market_prob', 'market_prob_away', 'ml_prob', 'ml_actual'])
        
        # Each side against its OWN price (1 - home implied would credit the away bet with the vig)
        df['model_edge_home'] = df['ml_prob'] - df['market_prob']
        df['model_edge_away'] = (1 - df['ml_prob']) - df['market_prob_away']
        
        print("="*60)
        print("MONEYLINE DIAGNOSTIC")
        print("="*60)
        
        # Home bets (model more bullish than market)
        home_bets = df[df['model_edge_home'] > edge_threshold].copy()
        if len(home_bets) > 0:
            home_wr = home_bets['ml_actual'].mean()
            home_model_avg = home_bets['ml_prob'].mean()
            home_market_avg = home_bets['market_prob'].mean()
            print(f"\nHOME BETS ({len(home_bets)} bets):")
            print(f"  Win rate: {home_wr:.1%}")
            print(f"  Model predicted: {home_model_avg:.1%}")
            print(f"  Market implied: {home_market_avg:.1%}")
            print(f"  Model overconfident by: {home_model_avg - home_wr:.1%}")
            
            # Break down by market probability (favorite vs underdog)
            home_bets['is_favorite'] = home_bets['market_prob'] > 0.5
            print("\n  By favorite status:")
            for fav, group in home_bets.groupby('is_favorite'):
                label = "Favorites" if fav else "Underdogs"
                print(f"    {label}: {len(group)} bets, {group['ml_actual'].mean():.1%} win rate")
        
        # Away bets (model more bearish on home than market)
        away_bets = df[df['model_edge_away'] > edge_threshold].copy()
        if len(away_bets) > 0:
            away_wr = 1 - away_bets['ml_actual'].mean()  # Away win rate
            away_model_avg = 1 - away_bets['ml_prob'].mean()
            away_market_avg = away_bets['market_prob_away'].mean()
            print(f"\nAWAY BETS ({len(away_bets)} bets):")
            print(f"  Win rate: {away_wr:.1%}")
            print(f"  Model predicted: {away_model_avg:.1%}")
            print(f"  Market implied: {away_market_avg:.1%}")
            print(f"  Model overconfident by: {away_model_avg - away_wr:.1%}")
            
            # Break down by market probability
            away_bets['away_is_favorite'] = away_bets['market_prob_away'] > 0.5
            print("\n  By favorite status:")
            for fav, group in away_bets.groupby('away_is_favorite'):
                label = "Favorites" if fav else "Underdogs"
                print(f"    {label}: {len(group)} bets, {1 - group['ml_actual'].mean():.1%} win rate")
        
        # Calibration by confidence bucket
        print("\n" + "="*60)
        print("CALIBRATION BY MODEL CONFIDENCE")
        print("="*60)
        
        df['prob_bucket'] = pd.cut(df['ml_prob'], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
        calibration = df.groupby('prob_bucket').agg({
            'ml_actual': ['mean', 'count'],
            'ml_prob': 'mean',
            'market_prob': 'mean'
        }).round(3)
        print(calibration)
        
        # Are we betting mostly on underdogs?
        print("\n" + "="*60)
        print("BET DISTRIBUTION")
        print("="*60)
        
        all_bets = pd.concat([
            home_bets.assign(bet_type='home'),
            away_bets.assign(bet_type='away')
        ])
        
        if len(all_bets) > 0:
            all_bets['bet_on_favorite'] = (
                ((all_bets['bet_type'] == 'home') & (all_bets['market_prob'] > 0.5)) |
                ((all_bets['bet_type'] == 'away') & (all_bets['market_prob'] < 0.5))
            )
            
            fav_bets = all_bets[all_bets['bet_on_favorite']]
            dog_bets = all_bets[~all_bets['bet_on_favorite']]
            
            print(f"Bets on favorites: {len(fav_bets)} ({len(fav_bets)/len(all_bets):.1%})")
            print(f"Bets on underdogs: {len(dog_bets)} ({len(dog_bets)/len(all_bets):.1%})")
            
            if len(fav_bets) > 0:
                fav_won = ((fav_bets['bet_type'] == 'home') & (fav_bets['ml_actual'] == 1)) | \
                        ((fav_bets['bet_type'] == 'away') & (fav_bets['ml_actual'] == 0))
                print(f"Favorite bet win rate: {fav_won.mean():.1%}")
            
            if len(dog_bets) > 0:
                dog_won = ((dog_bets['bet_type'] == 'home') & (dog_bets['ml_actual'] == 1)) | \
                        ((dog_bets['bet_type'] == 'away') & (dog_bets['ml_actual'] == 0))
                print(f"Underdog bet win rate: {dog_won.mean():.1%}")


    def simulate_moneyline_v2(self, edge_threshold=0.075, underdog_edge_threshold=0.1, flat_stake=100):
        """
        Bet when model probability exceeds implied probability by threshold.
        Use higher threshold for underdog bets.
        """
        results = []
        
        for _, game in self.df.iterrows():
            model_prob = game['ml_prob']
            
            # Check home bet
            if 'odds_home_ml' in game and pd.notna(game['odds_home_ml']):
                implied = self.implied_prob_from_american(game['odds_home_ml'])
                edge = model_prob - implied
                
                # Is home team the favorite?
                home_is_favorite = implied > 0.5
                
                # Use different thresholds
                required_edge = edge_threshold if home_is_favorite else underdog_edge_threshold
                
                if edge > required_edge:
                    won = game['ml_actual'] == 1
                    payout = self._calc_payout(game['odds_home_ml'], flat_stake, won)
                    results.append({
                        'gameId': game['gameId'],
                        'date': game['gameDate'],
                        'bet_type': 'home_ml',
                        'model_prob': model_prob,
                        'implied_prob': implied,
                        'edge': edge,
                        'is_favorite': home_is_favorite,
                        'won': won,
                        'profit': payout
                    })
            
            # Check away bet
            if 'odds_away_ml' in game and pd.notna(game['odds_away_ml']):
                implied = self.implied_prob_from_american(game['odds_away_ml'])
                model_away = 1 - model_prob
                edge = model_away - implied
                
                # Is away team the favorite?
                away_is_favorite = implied > 0.5
                
                # Use different thresholds
                required_edge = edge_threshold if away_is_favorite else underdog_edge_threshold
                
                if edge > required_edge:
                    won = game['ml_actual'] == 0
                    payout = self._calc_payout(game['odds_away_ml'], flat_stake, won)
                    results.append({
                        'gameId': game['gameId'],
                        'date': game['gameDate'],
                        'bet_type': 'away_ml',
                        'model_prob': model_away,
                        'implied_prob': implied,
                        'edge': edge,
                        'is_favorite': away_is_favorite,
                        'won': won,
                        'profit': payout
                    })
        
        return pd.DataFrame(results)


    def simulate_moneyline_favorites_only(self, edge_threshold=0.075, flat_stake=100):
        """
        Only bet on favorites where model sees edge.
        """
        results = []
        
        for _, game in self.df.iterrows():
            model_prob = game['ml_prob']
            
            # Check home bet - only if home is favorite
            if 'odds_home_ml' in game and pd.notna(game['odds_home_ml']):
                implied = self.implied_prob_from_american(game['odds_home_ml'])
                
                # Only bet if market says home is favorite AND model agrees AND sees edge
                if implied > 0.5 and model_prob > 0.5:
                    edge = model_prob - implied
                    if edge > edge_threshold:
                        won = game['ml_actual'] == 1
                        payout = self._calc_payout(game['odds_home_ml'], flat_stake, won)
                        results.append({
                            'gameId': game['gameId'],
                            'date': game['gameDate'],
                            'bet_type': 'home_ml',
                            'model_prob': model_prob,
                            'implied_prob': implied,
                            'edge': edge,
                            'won': won,
                            'profit': payout
                        })
            
            # Check away bet - only if away is favorite
            if 'odds_away_ml' in game and pd.notna(game['odds_away_ml']):
                implied = self.implied_prob_from_american(game['odds_away_ml'])
                model_away = 1 - model_prob
                
                # Only bet if market says away is favorite AND model agrees AND sees edge
                if implied > 0.5 and model_away > 0.5:
                    edge = model_away - implied
                    if edge > edge_threshold:
                        won = game['ml_actual'] == 0
                        payout = self._calc_payout(game['odds_away_ml'], flat_stake, won)
                        results.append({
                            'gameId': game['gameId'],
                            'date': game['gameDate'],
                            'bet_type': 'away_ml',
                            'model_prob': model_away,
                            'implied_prob': implied,
                            'edge': edge,
                            'won': won,
                            'profit': payout
                        })
        
        return pd.DataFrame(results)


    def analyze_spread_by_year_and_threshold(self, thresholds=[3.0, 4.0, 5.0]):
        """Check if spread edge is consistent across years."""
        
        print("="*70)
        print("SPREAD PERFORMANCE BY YEAR AND THRESHOLD")
        print("="*70)
        
        for thresh in thresholds:
            bets = self.simulate_spread(threshold=thresh)
            if bets.empty:
                continue
                
            bets['year'] = pd.to_datetime(bets['date']).dt.year
            
            print(f"\n--- Threshold: {thresh} pts ---")
            
            yearly = bets.groupby('year').agg({
                'won': lambda x: ((x == 1).sum(), (x == 0).sum(), len(x)),
                'profit': 'sum'
            })
            
            for year in sorted(bets['year'].unique()):
                year_bets = bets[bets['year'] == year]
                wins = (year_bets['won'] == 1).sum()
                losses = (year_bets['won'] == 0).sum()
                total = wins + losses
                wr = wins / total if total > 0 else 0
                profit = year_bets['profit'].sum()
                roi = profit / (len(year_bets) * 100)
                print(f"  {year}: {len(year_bets):3d} bets, {wr:.1%} WR, ${profit:+8.0f} ({roi:+.1%} ROI)")


    def analyze_spread_direction(self, threshold=5.0):
        """Check if spread bets are biased toward home or away."""
        bets = self.simulate_spread(threshold=threshold)
        
        print(f"\n--- Spread Direction Analysis (thresh={threshold}) ---")
        
        home_bets = bets[bets['bet_type'] == 'home_spread']
        away_bets = bets[bets['bet_type'] == 'away_spread']
        
        print(f"Home spread bets: {len(home_bets)}")
        if len(home_bets) > 0:
            home_wr = (home_bets['won'] == 1).sum() / ((home_bets['won'] == 1).sum() + (home_bets['won'] == 0).sum())
            home_roi = home_bets['profit'].sum() / (len(home_bets) * 100)
            print(f"  Win rate: {home_wr:.1%}, ROI: {home_roi:+.1%}")
        
        print(f"Away spread bets: {len(away_bets)}")
        if len(away_bets) > 0:
            away_wr = (away_bets['won'] == 1).sum() / ((away_bets['won'] == 1).sum() + (away_bets['won'] == 0).sum())
            away_roi = away_bets['profit'].sum() / (len(away_bets) * 100)
            print(f"  Win rate: {away_wr:.1%}, ROI: {away_roi:+.1%}")
        
        # Are we mostly betting favorites or underdogs to cover?
        print("\n--- Favorite/Underdog Coverage ---")
        
        # Home spread negative = home favored
        bets['betting_favorite'] = (
            ((bets['bet_type'] == 'home_spread') & (bets['line'] < 0)) |
            ((bets['bet_type'] == 'away_spread') & (bets['line'] > 0))
        )
        
        fav_bets = bets[bets['betting_favorite']]
        dog_bets = bets[~bets['betting_favorite']]
        
        print(f"Betting favorite to cover: {len(fav_bets)} bets")
        if len(fav_bets) > 0:
            fav_wr = (fav_bets['won'] == 1).sum() / ((fav_bets['won'] == 1).sum() + (fav_bets['won'] == 0).sum())
            fav_roi = fav_bets['profit'].sum() / (len(fav_bets) * 100)
            print(f"  Win rate: {fav_wr:.1%}, ROI: {fav_roi:+.1%}")
        
        print(f"Betting underdog to cover: {len(dog_bets)} bets")
        if len(dog_bets) > 0:
            dog_wr = (dog_bets['won'] == 1).sum() / ((dog_bets['won'] == 1).sum() + (dog_bets['won'] == 0).sum())
            dog_roi = dog_bets['profit'].sum() / (len(dog_bets) * 100)
            print(f"  Win rate: {dog_wr:.1%}, ROI: {dog_roi:+.1%}")

    def investigate_2023_collapse(self):
        """Figure out why 2023 broke."""
        df = self.df.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df['year'] = df['gameDate'].dt.year
        
        print("="*60)
        print("2023 COLLAPSE INVESTIGATION")
        print("="*60)
        
        # Compare feature distributions across years
        for year in [2021, 2022, 2023, 2024, 2025]:
            year_df = df[df['year'] == year]
            if len(year_df) == 0:
                continue
                
            print(f"\n--- {year} ---")
            print(f"Games: {len(year_df)}")
            
            # Spread prediction stats
            print(f"Spread pred mean: {year_df['spread_pred'].mean():.2f}")
            print(f"Spread pred std: {year_df['spread_pred'].std():.2f}")
            print(f"Spread actual mean: {year_df['spread_actual'].mean():.2f}")
            print(f"Spread actual std: {year_df['spread_actual'].std():.2f}")
            
            # Model error
            error = year_df['spread_pred'] - year_df['spread_actual']
            print(f"Mean error: {error.mean():.2f}")
            print(f"MAE: {error.abs().mean():.2f}")
            
            # Correlation between prediction and actual
            corr = year_df['spread_pred'].corr(year_df['spread_actual'])
            print(f"Pred-Actual correlation: {corr:.3f}")    

    def analyze_losses(self, year, threshold=5.0):
        """Deep dive on spread bets for a specific year."""
        bets = self.simulate_spread(threshold=threshold)
        bets['year'] = pd.to_datetime(bets['date']).dt.year
        
        bets_year = bets[bets['year'] == year]
        bets_other = bets[bets['year'] != year]
        
        print("="*60)
        print(f"{year} vs OTHER YEARS COMPARISON")
        print("="*60)
        
        print(f"\n{year}: {len(bets_year)} bets")
        print(f"Other years: {len(bets_other)} bets")
        
        if len(bets_year) == 0:
            print(f"No bets found for {year}")
            return

        # Edge distribution
        print(f"\n--- Edge Distribution ---")
        print(f"{year} avg edge: {bets_year['edge'].mean():.1f} pts")
        if len(bets_other) > 0:
            print(f"Other years avg edge: {bets_other['edge'].mean():.1f} pts")
        
        # By bet type
        print(f"\n--- By Bet Type ---")
        for bt in ['home_spread', 'away_spread']:
            byear = bets_year[bets_year['bet_type'] == bt]
            bother = bets_other[bets_other['bet_type'] == bt]
            
            if len(byear) > 5:
                wryear = (byear['won'] == 1).sum() / ((byear['won'] == 1).sum() + (byear['won'] == 0).sum())
                if len(bother) > 5:
                    wrother = (bother['won'] == 1).sum() / ((bother['won'] == 1).sum() + (bother['won'] == 0).sum())
                    print(f"{bt}: {year} WR={wryear:.1%}, Other WR={wrother:.1%}")
                else:
                    print(f"{bt}: {year} WR={wryear:.1%}")
        
        # By edge bucket
        print(f"\n--- By Edge Size ---")
        for edge_min in [3, 5, 7, 10]:
            byear = bets_year[bets_year['edge'] >= edge_min]
            bother = bets_other[bets_other['edge'] >= edge_min]
            
            if len(byear) > 5:
                wryear = (byear['won'] == 1).sum() / ((byear['won'] == 1).sum() + (byear['won'] == 0).sum())
                if len(bother) > 5:
                    wrother = (bother['won'] == 1).sum() / ((bother['won'] == 1).sum() + (bother['won'] == 0).sum())
                    print(f"Edge >= {edge_min}: {year} WR={wryear:.1%} ({len(byear)}), Other WR={wrother:.1%} ({len(bother)})")
                else:
                     print(f"Edge >= {edge_min}: {year} WR={wryear:.1%} ({len(byear)})")


    def investigate_odds_data(self):
        """Check for odds data quality issues."""
        df = self.df.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df['year'] = df['gameDate'].dt.year
        
        print("="*60)
        print("ODDS DATA INVESTIGATION")
        print("="*60)
        
        for year in [2021, 2022, 2023, 2024, 2025]:
            year_df = df[df['year'] == year]
            if len(year_df) == 0:
                continue
            
            print(f"\n--- {year} ({len(year_df)} games) ---")
            
            # Home spread line distribution
            if 'odds_home_spread' in year_df.columns:
                hs = year_df['odds_home_spread'].dropna()
                print(f"Home spread line: mean={hs.mean():.1f}, std={hs.std():.1f}, min={hs.min():.1f}, max={hs.max():.1f}")
                
                # Count suspicious values
                suspicious = (hs.abs() > 20).sum()
                print(f"  Lines > ±20: {suspicious} ({suspicious/len(hs)*100:.1f}%)")
            
            # Check if we have both home and away spreads
            if 'odds_home_spread' in year_df.columns and 'odds_away_spread' in year_df.columns:
                both = year_df.dropna(subset=['odds_home_spread', 'odds_away_spread'])
                if len(both) > 0:
                    # Should sum to ~0 (home spread + away spread ≈ 0)
                    spread_sum = both['odds_home_spread'] + both['odds_away_spread']
                    print(f"Home + Away spread: mean={spread_sum.mean():.2f} (should be ~0)")
            
            # Coverage
            odds_cols = ['odds_home_spread', 'odds_away_spread', 'odds_home_ml', 'odds_away_ml', 'odds_total']
            for col in odds_cols:
                if col in year_df.columns:
                    coverage = year_df[col].notna().mean()
                    print(f"  {col} coverage: {coverage:.1%}")


    def debug_spread_bets(self, threshold=5.0, sample_year=2022):
        """Print sample bets to verify edge calculation."""
        bets = self.simulate_spread(threshold=threshold)
        bets['year'] = pd.to_datetime(bets['date']).dt.year
        
        sample = bets[bets['year'] == sample_year].head(20)
        
        print("="*60)
        print(f"SAMPLE SPREAD BETS ({sample_year})")
        print("="*60)
        
        for _, row in sample.iterrows():
            print(f"\nGame: {row['gameId']}")
            print(f"  Bet type: {row['bet_type']}")
            print(f"  Line: {row['line']}")
            print(f"  Pred margin: {row['pred_margin']}")
            print(f"  Edge: {row['edge']}")
            print(f"  Won: {row['won']}")


    def investigate_2021(self):
        """Deep dive on 2021 data issues."""
        df = self.df.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df['year'] = df['gameDate'].dt.year
        
        df_2021 = df[df['year'] == 2021].copy()
        df_other = df[df['year'] != 2021].copy()
        
        print("="*60)
        print("2021 DATA INVESTIGATION")
        print("="*60)
        
        print(f"\n2021 games: {len(df_2021)}")
        print(f"Other years: {len(df_other)}")
        
        # Check odds values
        print("\n--- Odds Sanity Check ---")
        for col in ['odds_home_ml', 'odds_away_ml', 'odds_home_spread']:
            if col in df_2021.columns:
                v21 = df_2021[col].dropna()
                vother = df_other[col].dropna()
                print(f"\n{col}:")
                print(f"  2021:  mean={v21.mean():.1f}, std={v21.std():.1f}, min={v21.min():.1f}, max={v21.max():.1f}")
                print(f"  Other: mean={vother.mean():.1f}, std={vother.std():.1f}, min={vother.min():.1f}, max={vother.max():.1f}")
        
        # Check if model and market agree on favorite
        def implied_prob(odds):
            if pd.isna(odds) or odds == 0:
                return np.nan
            return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)
        
        df_2021['market_prob'] = df_2021['odds_home_ml'].apply(implied_prob)
        df_other['market_prob'] = df_other['odds_home_ml'].apply(implied_prob)
        
        # Does model agree with market on who's favored?
        df_2021['model_fav_home'] = df_2021['ml_prob'] > 0.5
        df_2021['market_fav_home'] = df_2021['market_prob'] > 0.5
        df_2021['agree_on_favorite'] = df_2021['model_fav_home'] == df_2021['market_fav_home']
        
        df_other['model_fav_home'] = df_other['ml_prob'] > 0.5
        df_other['market_fav_home'] = df_other['market_prob'] > 0.5
        df_other['agree_on_favorite'] = df_other['model_fav_home'] == df_other['market_fav_home']
        
        print("\n--- Model vs Market Agreement ---")
        print(f"2021 agree on favorite: {df_2021['agree_on_favorite'].mean():.1%}")
        print(f"Other agree on favorite: {df_other['agree_on_favorite'].mean():.1%}")
        
        # Check actual outcomes
        print("\n--- Actual Outcomes ---")
        print(f"2021 home win rate: {df_2021['ml_actual'].mean():.1%}")
        print(f"Other home win rate: {df_other['ml_actual'].mean():.1%}")
        
        # Sample some 2021 games
        print("\n--- Sample 2021 Games ---")
        sample = df_2021.dropna(subset=['market_prob', 'ml_prob']).head(10)
        for _, row in sample.iterrows():
            print(f"\nDate: {row['gameDate'].date()}")
            print(f"  Market prob (home): {row['market_prob']:.1%}")
            print(f"  Model prob (home):  {row['ml_prob']:.1%}")
            print(f"  Home ML odds: {row['odds_home_ml']}")
            print(f"  Actual: {'Home won' if row['ml_actual'] == 1 else 'Away won'}")
        
        # Check spread predictions vs actual
        print("\n--- Spread Sanity ---")
        print(f"2021 spread_pred mean: {df_2021['spread_pred'].mean():.2f}")
        print(f"2021 spread_actual mean: {df_2021['spread_actual'].mean():.2f}")
        print(f"Other spread_pred mean: {df_other['spread_pred'].mean():.2f}")
        print(f"Other spread_actual mean: {df_other['spread_actual'].mean():.2f}")


    def simulate_moneyline_exclude_2021(self, edge_threshold=0.05, flat_stake=100):
        """Same as simulate_moneyline but excludes 2021."""
        df_clean = self.df[pd.to_datetime(self.df['gameDate']).dt.year != 2021].copy()
        
        original_df = self.df
        self.df = df_clean
        results = self.simulate_moneyline(edge_threshold=edge_threshold, flat_stake=flat_stake)
        self.df = original_df
        
        return results

    def simulate_spread_exclude_2021(self, threshold=2, flat_stake=100):
            """Same as simulate_spread but excludes 2021."""
            df_clean = self.df[pd.to_datetime(self.df['gameDate']).dt.year != 2021].copy()
            
            original_df = self.df
            self.df = df_clean
            results = self.simulate_spread(threshold=threshold, flat_stake=flat_stake)
            self.df = original_df
            
            return results

    def simulate_total_exclude_2021(self, threshold=2, flat_stake=100):
            """Same as simulate_total but excludes 2021."""
            df_clean = self.df[pd.to_datetime(self.df['gameDate']).dt.year != 2021].copy()
            
            original_df = self.df
            self.df = df_clean
            results = self.simulate_total(threshold=threshold, flat_stake=flat_stake)
            self.df = original_df
            
            return results

    def analyze_totals_edge(self, threshold=5.0):
        """Deep dive on totals to verify signal."""
        bets = self.simulate_total(threshold=threshold)
        bets = bets[pd.to_datetime(bets['date']).dt.year != 2021]  # Exclude 2021
        bets['year'] = pd.to_datetime(bets['date']).dt.year
        
        print("="*60)
        print(f"TOTALS ANALYSIS (Threshold: {threshold})")
        print("="*60)
        
        # Year-by-year consistency
        print("\n--- By Year ---")
        for year in sorted(bets['year'].unique()):
            yb = bets[bets['year'] == year]
            wins = (yb['won'] == 1).sum()
            losses = (yb['won'] == 0).sum()
            wr = wins / (wins + losses) if (wins + losses) > 0 else 0
            roi = yb['profit'].sum() / (len(yb) * 100)
            print(f"{year}: {len(yb)} bets, {wr:.1%} WR, {roi:+.1%} ROI")
        
        # Over vs Under breakdown
        print("\n--- Over vs Under ---")
        overs = bets[bets['bet_type'] == 'over']
        unders = bets[bets['bet_type'] == 'under']
        
        if len(overs) > 0:
            over_wr = (overs['won'] == 1).sum() / ((overs['won'] == 1).sum() + (overs['won'] == 0).sum())
            over_roi = overs['profit'].sum() / (len(overs) * 100)
            print(f"Overs:  {len(overs)} bets, {over_wr:.1%} WR, {over_roi:+.1%} ROI")
        
        if len(unders) > 0:
            under_wr = (unders['won'] == 1).sum() / ((unders['won'] == 1).sum() + (unders['won'] == 0).sum())
            under_roi = unders['profit'].sum() / (len(unders) * 100)
            print(f"Unders: {len(unders)} bets, {under_wr:.1%} WR, {under_roi:+.1%} ROI")
        
        # Edge size distribution
        print("\n--- By Edge Size ---")
        for edge_min in [5, 7, 10, 15]:
            edge_bets = bets[bets['edge'] >= edge_min]
            if len(edge_bets) > 50:
                wr = (edge_bets['won'] == 1).sum() / ((edge_bets['won'] == 1).sum() + (edge_bets['won'] == 0).sum())
                roi = edge_bets['profit'].sum() / (len(edge_bets) * 100)
                print(f"Edge >= {edge_min}: {len(edge_bets)} bets, {wr:.1%} WR, {roi:+.1%} ROI")
        
        # Monthly distribution (check for seasonality)
        print("\n--- By Month ---")
        bets['month'] = pd.to_datetime(bets['date']).dt.month
        for month in sorted(bets['month'].unique()):
            mb = bets[bets['month'] == month]
            if len(mb) > 30:
                wr = (mb['won'] == 1).sum() / ((mb['won'] == 1).sum() + (mb['won'] == 0).sum())
                roi = mb['profit'].sum() / (len(mb) * 100)
                month_name = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month]
                print(f"{month_name}: {len(mb)} bets, {wr:.1%} WR, {roi:+.1%} ROI")


    def simulate_moneyline_inverse(self, edge_threshold=0.05, flat_stake=100):
        """
        Bet AGAINST the model's edge.
        When model says "home is undervalued," bet away.
        When model says "away is undervalued," bet home.
        """
        results = []
        
        for _, game in self.df.iterrows():
            # Skip 2021
            if pd.to_datetime(game['gameDate']).year == 2021:
                continue
                
            model_prob = game['ml_prob']
            
            if 'odds_home_ml' not in game or pd.isna(game['odds_home_ml']):
                continue
            if 'odds_away_ml' not in game or pd.isna(game['odds_away_ml']):
                continue
                
            implied_home = self.implied_prob_from_american(game['odds_home_ml'])
            implied_away = self.implied_prob_from_american(game['odds_away_ml'])
            
            model_edge_home = model_prob - implied_home
            model_edge_away = (1 - model_prob) - implied_away
            
            # INVERSE: When model likes home, bet away
            if model_edge_home > edge_threshold:
                won = game['ml_actual'] == 0  # Away wins
                payout = self._calc_payout(game['odds_away_ml'], flat_stake, won)
                results.append({
                    'gameId': game['gameId'],
                    'date': game['gameDate'],
                    'bet_type': 'away_ml (inverse)',
                    'model_prob_home': model_prob,
                    'implied_prob_home': implied_home,
                    'model_edge': model_edge_home,
                    'odds': game['odds_away_ml'],
                    'won': won,
                    'profit': payout
                })
            
            # INVERSE: When model likes away, bet home
            if model_edge_away > edge_threshold:
                won = game['ml_actual'] == 1  # Home wins
                payout = self._calc_payout(game['odds_home_ml'], flat_stake, won)
                results.append({
                    'gameId': game['gameId'],
                    'date': game['gameDate'],
                    'bet_type': 'home_ml (inverse)',
                    'model_prob_away': 1 - model_prob,
                    'implied_prob_away': implied_away,
                    'model_edge': model_edge_away,
                    'odds': game['odds_home_ml'],
                    'won': won,
                    'profit': payout
                })
        
        return pd.DataFrame(results)


    def analyze_inverse_strategy(self):
        """Compare original vs inverse moneyline strategies."""
        
        print("="*60)
        print("ORIGINAL VS INVERSE MONEYLINE STRATEGY")
        print("="*60)
        
        for thresh in [0.05, 0.075, 0.10, 0.15]:
            # Original
            orig = self.simulate_moneyline(edge_threshold=thresh)
            orig = orig[pd.to_datetime(orig['date']).dt.year != 2021]
            
            # Inverse
            inv = self.simulate_moneyline_inverse(edge_threshold=thresh)
            
            if len(orig) > 100 and len(inv) > 100:
                orig_wr = (orig['won'] == True).sum() / len(orig)
                orig_roi = orig['profit'].sum() / (len(orig) * 100)
                
                inv_wr = (inv['won'] == True).sum() / len(inv)
                inv_roi = inv['profit'].sum() / (len(inv) * 100)
                
                print(f"\nThreshold: {thresh}")
                print(f"  Original: {len(orig)} bets, {orig_wr:.1%} WR, {orig_roi:+.1%} ROI")
                print(f"  Inverse:  {len(inv)} bets, {inv_wr:.1%} WR, {inv_roi:+.1%} ROI")


    def analyze_inverse_high_threshold(self, edge_threshold=0.15):
        """Deep dive on high-threshold inverse strategy."""
        inv = self.simulate_moneyline_inverse(edge_threshold=edge_threshold)
        inv['year'] = pd.to_datetime(inv['date']).dt.year
        
        print("="*60)
        print(f"INVERSE STRATEGY DEEP DIVE (Threshold: {edge_threshold})")
        print("="*60)
        
        print(f"\nTotal bets: {len(inv)}")
        print(f"Win rate: {(inv['won'] == True).mean():.1%}")
        print(f"ROI: {inv['profit'].sum() / (len(inv) * 100):+.1%}")
        
        # Year-by-year
        print("\n--- By Year ---")
        for year in sorted(inv['year'].unique()):
            yb = inv[inv['year'] == year]
            if len(yb) > 5:
                wr = (yb['won'] == True).sum() / len(yb)
                roi = yb['profit'].sum() / (len(yb) * 100)
                print(f"{year}: {len(yb)} bets, {wr:.1%} WR, {roi:+.1%} ROI")
        
        # By bet type (betting home vs away)
        print("\n--- By Bet Type ---")
        for bt in inv['bet_type'].unique():
            bt_bets = inv[inv['bet_type'] == bt]
            if len(bt_bets) > 10:
                wr = (bt_bets['won'] == True).sum() / len(bt_bets)
                roi = bt_bets['profit'].sum() / (len(bt_bets) * 100)
                avg_odds = bt_bets['odds'].mean()
                print(f"{bt}: {len(bt_bets)} bets, {wr:.1%} WR, {roi:+.1%} ROI, avg odds: {avg_odds:.0f}")
        
        # By odds range
        print("\n--- By Odds ---")
        inv['is_favorite'] = inv['odds'] < 0
        
        fav = inv[inv['is_favorite']]
        dog = inv[~inv['is_favorite']]
        
        if len(fav) > 10:
            wr = (fav['won'] == True).sum() / len(fav)
            roi = fav['profit'].sum() / (len(fav) * 100)
            print(f"Favorites: {len(fav)} bets, {wr:.1%} WR, {roi:+.1%} ROI")
        
        if len(dog) > 10:
            wr = (dog['won'] == True).sum() / len(dog)
            roi = dog['profit'].sum() / (len(dog) * 100)
            print(f"Underdogs: {len(dog)} bets, {wr:.1%} WR, {roi:+.1%} ROI")
        
        # What was the original model predicting?
        print("\n--- Model Confidence on These Games ---")
        print(f"Avg model edge that triggered bet: {inv['model_edge'].mean():.1%}")
        print(f"Max model edge: {inv['model_edge'].max():.1%}")


    def simulate_total_refined(self, threshold=7.0, exclude_months=[10, 11], overs_only=False, flat_stake=100):
        """
        Refined totals strategy based on analysis.
        - Higher threshold (7+ pts)
        - Exclude October/November
        - Optionally focus on overs only
        """
        results = []
        standard_payout_mult = 100 / 110
        
        for _, game in self.df.iterrows():
            # Skip 2021 (bad data)
            game_date = pd.to_datetime(game['gameDate'])
            if game_date.year == 2021:
                continue
                
            # Skip early season
            if game_date.month in exclude_months:
                continue
                
            if 'total_pred' not in game or pd.isna(game['total_pred']):
                continue
            if 'odds_total' not in game or pd.isna(game['odds_total']):
                continue
                
            pred_total = game['total_pred']
            actual_total = game['total_actual']
            line = game['odds_total']
            
            # Check Over
            if pred_total > (line + threshold):
                if actual_total > line:
                    profit = flat_stake * standard_payout_mult
                    won = 1
                elif actual_total == line:
                    profit = 0
                    won = 0.5
                else:
                    profit = -flat_stake
                    won = 0
                    
                results.append({
                    'gameId': game['gameId'],
                    'date': game['gameDate'],
                    'bet_type': 'over',
                    'pred_total': pred_total,
                    'line': line,
                    'edge': pred_total - line,
                    'won': won,
                    'profit': profit
                })
                
            # Check Under (skip if overs_only)
            elif not overs_only and pred_total < (line - threshold):
                if actual_total < line:
                    profit = flat_stake * standard_payout_mult
                    won = 1
                elif actual_total == line:
                    profit = 0
                    won = 0.5
                else:
                    profit = -flat_stake
                    won = 0
                
                results.append({
                    'gameId': game['gameId'],
                    'date': game['gameDate'],
                    'bet_type': 'under',
                    'pred_total': pred_total,
                    'line': line,
                    'edge': line - pred_total,
                    'won': won,
                    'profit': profit
                })
                    
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Load predictions
    predictions_path = Path(__file__).parent / 'saved' / 'backtest_predictions_calibrated.csv'
    if not predictions_path.exists():
        print(f"Predictions not found at {predictions_path}")
    else:
        simulator = BettingSimulator(predictions_path)
        
        # simulator.investigate_2021()
        
            # 1. Moneyline
        # print("MONEYLINE EXCLUDING 2021:")
        # for thresh in [0.01, 0.025, 0.05, 0.075, 0.10]:
        #     bets = simulator.simulate_moneyline_exclude_2021(edge_threshold=thresh)
        #     if len(bets) > 100:
        #         roi = bets['profit'].sum() / (len(bets) * 100)
        #         wr = (bets['won'] == 1).sum() / ((bets['won'] == 1).sum() + (bets['won'] == 0).sum())
        #         print(f"Thresh {thresh}: {len(bets)} bets, {wr:.1%} WR, {roi:+.1%} ROI")

        # simulator.analyze_inverse_strategy()
        # simulator.analyze_inverse_high_threshold()

        # # Also test intermediate thresholds
        # print("\nFINDING OPTIMAL INVERSE THRESHOLD:")
        # for thresh in [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20]:
        #     inv = simulator.simulate_moneyline_inverse(edge_threshold=thresh)
        #     if len(inv) >= 30:
        #         wr = (inv['won'] == True).sum() / len(inv)
        #         roi = inv['profit'].sum() / (len(inv) * 100)
        #         print(f"Thresh {thresh}: {len(inv)} bets, {wr:.1%} WR, {roi:+.1%} ROI")


        print("REFINED TOTALS STRATEGIES:")
        print("\n--- All bets, exclude Oct/Nov, threshold 7 ---")

        for t in [0, 1, 2, 3, 4, 5, 6, 7]:
            bets = simulator.simulate_total_refined(threshold=t, exclude_months=[10, 11], overs_only=False)
            simulator.summarize(bets, f"Totals Refined (threshold {t})")


        ml_bets = simulator.simulate_moneyline_exclude_2021(edge_threshold=0.05)
        simulator.summarize(ml_bets, "Moneyline")   

        # print("\n--- Overs only, exclude Oct/Nov, threshold 7 ---")
        # bets_overs = simulator.simulate_total_refined(threshold=7.0, exclude_months=[10, 11], overs_only=True)
        # simulator.summarize(bets_overs, "Overs Only")

        # print("\n--- Overs only, exclude Oct/Nov, threshold 5 ---")
        # bets_overs_5 = simulator.simulate_total_refined(threshold=5.0, exclude_months=[10, 11], overs_only=True)
        # simulator.summarize(bets_overs_5, "Overs Only (thresh 5)")

        # # 2. Spread (Threshold: Model must differ from line by > 1.5 points)
        # # Test spread with varying thresholds
        # print("SPREAD EXCLUDING 2021:")
        # for thresh in [1.0, 2.0, 3.0, 4.0, 5.0]:
        #     bets = simulator.simulate_spread_exclude_2021(threshold=thresh)
        #     if len(bets) > 100:
        #         roi = bets['profit'].sum() / (len(bets) * 100)
        #         wr = (bets['won'] == 1).sum() / ((bets['won'] == 1).sum() + (bets['won'] == 0).sum())
        #         print(f"Thresh {thresh}: {len(bets)} bets, {wr:.1%} WR, {roi:+.1%} ROI")

        # print("TOTAL EXCLUDING 2021:")
        # for thresh in [1.0, 2.0, 3.0, 4.0, 5.0]:
        #     bets = simulator.simulate_total_exclude_2021(threshold=thresh)
        #     if len(bets) > 100:
        #         roi = bets['profit'].sum() / (len(bets) * 100)
        #         wr = (bets['won'] == 1).sum() / ((bets['won'] == 1).sum() + (bets['won'] == 0).sum())
        #         print(f"Thresh {thresh}: {len(bets)} bets, {wr:.1%} WR, {roi:+.1%} ROI")


        # simulator.analyze_totals_edge(threshold=5.0)

        # simulator.analyze_spread_by_year_and_threshold()
        # simulator.analyze_spread_direction()
        
        # # Analyze losses for each year
        # for yr in [2021, 2022, 2023, 2024, 2025]:
        #     simulator.analyze_losses(yr, threshold=5.0)
            
        # # simulator.investigate_odds_data()
        # # simulator.debug_spread_bets()


        # bets_spread = simulator.simulate_spread(threshold=1.5)
        # simulator.summarize(bets_spread, "Spread (Thresh=1.5)")
        
        # # 3. Total (Threshold: Model must differ from line by > 2.0 points)
        # bets_total = simulator.simulate_total(threshold=2.0)
        # simulator.summarize(bets_total, "Total (Thresh=2.0)")

        # simulator.diagnose_moneyline_bets()


# if __name__ == "__main__":
#     predictions_path = Path(__file__).parent / 'saved' / 'backtest_predictions.csv'
#     simulator = BettingSimulator(predictions_path)
    
#     # Original (broken)
#     print("\n" + "="*60)
#     print("ORIGINAL STRATEGY (all bets)")
#     print("="*60)
#     bets_ml = simulator.simulate_moneyline(edge_threshold=0.05)
#     simulator.summarize(bets_ml, "Moneyline")
    
#     # Favorites only
#     print("\n" + "="*60)
#     print("FAVORITES ONLY STRATEGY")
#     print("="*60)
#     bets_fav = simulator.simulate_moneyline_favorites_only(edge_threshold=0.05)
#     simulator.summarize(bets_fav, "Moneyline (Favorites Only)")
    
#     # Tiered thresholds
#     print("\n" + "="*60)
#     print("TIERED THRESHOLDS (5% favorites, 15% underdogs)")
#     print("="*60)
#     bets_tiered = simulator.simulate_moneyline_v2(edge_threshold=0.05, underdog_edge_threshold=0.15)
#     simulator.summarize(bets_tiered, "Moneyline (Tiered)")