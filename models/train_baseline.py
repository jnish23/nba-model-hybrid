import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, classification_report, mean_squared_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from scipy import stats
import matplotlib.pyplot as plt
import logging
import joblib
import json
from pathlib import Path
from flaml import AutoML

class BaselineModel:
    def __init__(self, data_path, model_dir='models/saved'):
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.features = []
        self.target = 'target_home_win'

        # Target-specific feature sets (populated after selection)
        self.features_ml = None
        self.features_spread = None
        self.features_total = None


    def load_data(self, use_odds=False):
        logging.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['gameDate'] = pd.to_datetime(self.df['gameDate'])
        self.df = self.df.sort_values('gameDate').reset_index(drop=True)
        
        # Explicitly sanitize column names to avoid Spaces vs Underscores confusion
        self.df.columns = [c.replace(' ', '_') for c in self.df.columns]
        self.odds_cols = [c for c in self.df.columns if c.startswith('odds_')]
        
        self._validate_odds_data()

        # Identify features (numeric columns excluding targets/metadata)
        exclude_cols = [
            'gameDate', 'seasonYear', 'gameId', 'teamId_home', 'teamId_away', 
            'matchup', 'wl_home', 'wl_away', 'target_home_win', 'target_point_diff', 
            'target_total_pts', 'points_home', 'points_away', 'is_home_home', 'is_home_away'
        ]
        # Also exclude text columns if any leaked
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if use_odds:
            # Include odds but exclude specific raw ones if needed? 
            # We keep them all as they contain valuable info (spread, ml, total)
            self.features = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('target_')]
            logging.info(f"Training features: {len(self.features)} (INCLUDING ODDS)")
        else:
            self.features = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('target_') and not c.startswith('odds_')]
            logging.info(f"Training features: {len(self.features)} (odds excluded)")
            
        logging.info(f"Odds columns preserved for evaluation: {len(self.odds_cols)}")

    def _validate_odds_data(self):
        """Clean obviously bad odds values."""
        odds_issues = []
        
        # Spread should be between -25 and +25
        for col in ['odds_home_spread', 'odds_away_spread']:
            if col in self.df.columns:
                bad_mask = self.df[col].abs() > 25
                if bad_mask.sum() > 0:
                    odds_issues.append(f"{col}: {bad_mask.sum()} values > ±25")
                    self.df.loc[bad_mask, col] = np.nan
    
        # ML odds should be reasonable (-5000 to +5000)
        for col in ['odds_home_ml', 'odds_away_ml']:
            if col in self.df.columns:
                bad_mask = self.df[col].abs() > 5000
                if bad_mask.sum() > 0:
                    odds_issues.append(f"{col}: {bad_mask.sum()} extreme values")
                    self.df.loc[bad_mask, col] = np.nan
    
        if odds_issues:
            logging.warning(f"Odds validation issues fixed: {odds_issues}")


    def implied_prob_from_american(self, odds):
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)

    def evaluate_vs_market(self, predictions_df):
        """Compare model predictions to market implied probabilities."""
        df = predictions_df.copy()
        
        # Calculate implied prob from odds
        df['market_prob'] = df['odds_home_ml'].apply(self.implied_prob_from_american)
    

        # Debug: Check for remaining NaNs
        print(f"Rows before dropna: {len(df)}")
        print(f"NaN in market_prob: {df['market_prob'].isna().sum()}")
        print(f"NaN in ml_prob: {df['ml_prob'].isna().sum()}")
        print(f"NaN in ml_actual: {df['ml_actual'].isna().sum()}")

        # Drop rows with NaN in any of the key columns
        df = df.dropna(subset=['ml_actual', 'ml_prob', 'market_prob'])
        print(f"Rows after dropna: {len(df)}")
        
        if df.empty:
            logging.warning("No valid rows after dropping NaNs")
            return

        # Convert to numpy arrays explicitly
        y_true = df['ml_actual'].values.astype(float)
        market_probs = df['market_prob'].values.astype(float)
        model_probs = df['ml_prob'].values.astype(float)
    
        # Final NaN check
        valid_mask = ~(np.isnan(y_true) | np.isnan(market_probs) | np.isnan(model_probs))
        y_true = y_true[valid_mask]
        market_probs = market_probs[valid_mask]
        model_probs = model_probs[valid_mask]
    
        print(f"Final valid samples: {len(y_true)}")
    
        if len(y_true) == 0:
            logging.warning("No valid samples for comparison")
            return


        # Model calibration
        model_brier = brier_score_loss(y_true, model_probs)
        market_brier = brier_score_loss(y_true, market_probs)
    
        print("\n" + "="*60)
        print("MODEL VS MARKET COMPARISON")
        print("="*60)
        print(f"Brier Score - Model: {model_brier:.4f}, Market: {market_brier:.4f}")
        print(f"Model beats market: {model_brier < market_brier}")
        
        # Log loss comparison
        model_ll = log_loss(y_true, model_probs)
        market_ll = log_loss(y_true, market_probs)
        print(f"Log Loss  - Model: {model_ll:.4f}, Market: {market_ll:.4f}")
        
        # Accuracy comparison
        model_acc = (y_true == (model_probs > 0.5).astype(int)).mean()
        market_acc = (y_true == (market_probs > 0.5).astype(int)).mean()
        print(f"Accuracy  - Model: {model_acc:.4f}, Market: {market_acc:.4f}")
        
        # Edge analysis using the cleaned dataframe
        df_clean = df.iloc[np.where(valid_mask)[0]] if len(valid_mask) == len(df) else df
        df_clean['model_edge'] = df_clean['ml_prob'] - df_clean['market_prob']
        
        print("\n--- Edge Analysis ---")
        for thresh in [0.05, 0.10, 0.15]:
            home_bets = df_clean[df_clean['model_edge'] > thresh]
            if len(home_bets) > 0:
                actual_wr = home_bets['ml_actual'].mean()
                model_avg = home_bets['ml_prob'].mean()
                market_avg = home_bets['market_prob'].mean()
                print(f"\nHome edge > {thresh:.0%}: {len(home_bets)} bets")
                print(f"  Actual win rate: {actual_wr:.1%} | Model: {model_avg:.1%} | Market: {market_avg:.1%}")
            
            away_bets = df_clean[df_clean['model_edge'] < -thresh]
            if len(away_bets) > 0:
                actual_wr = 1 - away_bets['ml_actual'].mean()
                model_avg = 1 - away_bets['ml_prob'].mean()
                market_avg = 1 - away_bets['market_prob'].mean()
                print(f"\nAway edge > {thresh:.0%}: {len(away_bets)} bets")
                print(f"  Actual win rate: {actual_wr:.1%} | Model: {model_avg:.1%} | Market: {market_avg:.1%}")

    def sanity_check_odds(self):
        """Verify odds data makes sense."""
        df = self.df.copy()
        
        if 'odds_home_ml' not in df.columns:
            print("No odds_home_ml column")
            return
            
        def implied_prob(odds):
            if pd.isna(odds) or odds == 0:
                return np.nan
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return -odds / (-odds + 100)
        
        df['market_prob'] = df['odds_home_ml'].apply(implied_prob)
        df = df.dropna(subset=['market_prob', 'target_home_win'])
        
        # Bin by market probability and check calibration
        df['prob_bin'] = pd.cut(df['market_prob'], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
        
        calibration = df.groupby('prob_bin').agg({
            'target_home_win': ['mean', 'count'],
            'market_prob': 'mean'
        }).round(3)
        
        print("\nMarket Calibration Check:")
        print("(Market implied prob should ≈ actual win rate)")
        print(calibration)
        
        # Overall market accuracy
        market_pred = (df['market_prob'] > 0.5).astype(int)
        market_acc = (market_pred == df['target_home_win']).mean()
        print(f"\nMarket accuracy: {market_acc:.1%}")



    def find_lgbm_blockers(self):
        """Identify what's breaking LightGBM"""
        X = self.df[self.features]
        
        # 1. Check for inf (the likely killer)
        inf_counts = np.isinf(X.select_dtypes(include=[np.number])).sum()
        inf_cols = inf_counts[inf_counts > 0]
        print(f"Columns with inf values: {len(inf_cols)}")
        if len(inf_cols) > 0:
            print(inf_cols.sort_values(ascending=False).head(10))
        
        # 2. Check variance extremes
        variances = X.var()
        print(f"\nVariance range: {variances.min():.2e} to {variances.max():.2e}")
        print(f"Zero/near-zero variance (<1e-10): {(variances < 1e-10).sum()}")
        print(f"Extreme variance (>1e10): {(variances > 1e10).sum()}")
        
        # 3. Check value ranges
        print(f"\nValue range: {X.min().min():.2e} to {X.max().max():.2e}")
        
        # 4. Quick fix attempt - clean data
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.loc[:, X_clean.var() > 1e-10]  # drop zero-variance
        
        print(f"\nAfter cleaning: {X_clean.shape[1]} features (dropped {X.shape[1] - X_clean.shape[1]})")
        
        return X_clean

    def clean_features(self, X):
        """Replace inf and drop zero-variance columns"""
        X = X.replace([np.inf, -np.inf], np.nan)
        # Drop columns that are constant or have zero variance
        valid_cols = X.columns[X.var() > 1e-10]
        return X[valid_cols]
    
    def deeper_debug(self):
        """Run after load_data()"""
        X = self.df[self.features]
        y = self.df['target_home_win']
        
        # 1. Basic target correlation
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        print("Top 15 features by |correlation| with target:")
        print(correlations.head(15))
        print(f"\nMax correlation: {correlations.max():.4f}")
        print(f"Features with |corr| > 0.05: {(correlations > 0.05).sum()}")
        
        # 2. Feature-feature correlation (sample)
        sample_feats = correlations.head(20).index.tolist()
        corr_matrix = X[sample_feats].corr()
        high_corr = (corr_matrix.abs() > 0.9) & (corr_matrix != 1.0)
        print(f"\nHighly correlated pairs (>0.9) among top 20: {high_corr.sum().sum() // 2}")
        
        # 3. Sanity check - can logistic regression learn anything?
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Use top correlated features only
        X_top = X[correlations.head(20).index]
        X_scaled = StandardScaler().fit_transform(X_top.fillna(0))
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_scaled, y)
        lr_acc = (lr.predict(X_scaled) == y).mean()
        print(f"\nLogistic Regression (top 20 features, in-sample): {lr_acc:.4f}")
        print(f"Baseline (always predict majority): {y.mean():.4f}")


    def verify_model_health(self):
        """Check if the model is actually working despite warnings"""
        X = self.df[self.features].replace([np.inf, -np.inf], np.nan)
        y = self.df['target_home_win']
        
        # Fit with verbosity to see what happens
        model = lgb.LGBMClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=6,
            verbose=-1,
            importance_type='gain'
        )
        model.fit(X, y)
        
        # Check if model learned anything
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        
        print(f"Prediction distribution: {pd.Series(preds).value_counts(normalize=True).to_dict()}")
        print(f"Probability range: {probs.min():.3f} to {probs.max():.3f}")
        print(f"In-sample accuracy: {(preds == y).mean():.4f}")
        print(f"Baseline: {y.mean():.4f}")
        print(f"Non-zero importances: {(model.feature_importances_ > 0).sum()} / {len(self.features)}")
        print(f"Top 5 importances: {sorted(model.feature_importances_, reverse=True)[:5]}")

    def select_all_feature_sets(self, n_top=100):
        """Select features for all three targets."""
        if self.df is None:
            self.load_data()
            
        logging.info("Selecting target-specific feature sets...")
        
        self.features_ml = self.select_features('target_home_win', n_top=n_top)
        self.features_spread = self.select_features('target_point_diff', n_top=n_top)
        self.features_total = self.select_features('target_total_pts', n_top=n_top)
        
        # Log overlap analysis
        ml_set = set(self.features_ml)
        spread_set = set(self.features_spread)
        total_set = set(self.features_total)
        
        logging.info(f"ML features: {len(self.features_ml)}")
        logging.info(f"Spread features: {len(self.features_spread)}")
        logging.info(f"Total features: {len(self.features_total)}")
        logging.info(f"ML ∩ Spread overlap: {len(ml_set & spread_set)}")
        logging.info(f"ML ∩ Total overlap: {len(ml_set & total_set)}")
        logging.info(f"Spread ∩ Total overlap: {len(spread_set & total_set)}")
        logging.info(f"All three overlap: {len(ml_set & spread_set & total_set)}")
        
    def get_features_for_target(self, target: str) -> list:
        """Return the appropriate feature set for a target."""
        if target == 'target_home_win':
            if self.features_ml is None:
                self.features_ml = self.select_features(target, n_top=100)
            return self.features_ml
        elif target == 'target_point_diff':
            if self.features_spread is None:
                self.features_spread = self.select_features(target, n_top=100)
            return self.features_spread
        elif target == 'target_total_pts':
            if self.features_total is None:
                self.features_total = self.select_features(target, n_top=100)
            return self.features_total
        else:
            raise ValueError(f"Unknown target: {target}")


    def _select_features_from_subset(self, df_subset, target, n_top=100):
        """
        Feature selection using ONLY the provided subset (no leakage).
        Used during walk-forward backtesting.
        """
        # Use only the features that exist and are valid
        available_features = [f for f in self.features if f in df_subset.columns]
        
        X = df_subset[available_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Drop zero-variance columns within this subset
        variances = X.var()
        valid_cols = variances[variances > 1e-10].index.tolist()
        X = X[valid_cols]
        
        y = df_subset[target]
        
        if 'win' in target:
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, 
                verbose=-1, random_state=42
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, 
                verbose=-1, random_state=42
            )
        
        model.fit(X, y)
        
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected = imp_df.head(n_top)['feature'].tolist()
        
        return selected

    # def backtest_walk_forward(self, start_season='2019-20', use_tuned_params=True, use_odds=False, save_suffix='_no_odds'):
    #     """Walk-forward validation for all three models."""

    #     if use_tuned_params and (self.model_dir / 'best_config_ml.json').exists():
    #         with open(self.model_dir / 'best_config_ml.json') as f:
    #             ml_params = json.load(f)
    #         logging.info(f"Using tuned ML params: {ml_params}")
    #     else:
    #         ml_params = dict(n_estimators=1000, learning_rate=0.01, max_depth=6,
    #                         num_leaves=31, min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5)

    #     if use_tuned_params and (self.model_dir / 'best_config_spread.json').exists():
    #         with open(self.model_dir / 'best_config_spread.json') as f:
    #             spread_params = json.load(f)
    #         logging.info(f"Using tuned Spread params: {spread_params}")
    #     else:
    #         spread_params = dict(n_estimators=1000, learning_rate=0.01, max_depth=6,
    #                             num_leaves=31, min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5)

    #     if use_tuned_params and (self.model_dir / 'best_config_total.json').exists():
    #         with open(self.model_dir / 'best_config_total.json') as f:
    #             total_params = json.load(f)
    #         logging.info(f"Using tuned Total params: {total_params}")
    #     else:
    #         total_params = dict(n_estimators=1000, learning_rate=0.01, max_depth=6,
    #                             num_leaves=31, min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5)

    #     # Reload data if needed or if odds flag changed (force reload to be safe or rely on caller)
    #     # Simplest: Force reload if we want to ensure feature set matches flag
    #     self.load_data(use_odds=use_odds)
        
    #     # VERIFY: Check for odds leakage
    #     if not use_odds:
    #         all_features = set(self.features_ml + self.features_spread + self.features_total) if self.features_ml else []
    #         odds_in_features = [f for f in self.features if 'odds' in f.lower()]
    #         if odds_in_features:
    #             print(f"\n⚠️  LEAKAGE CHECK: Odds features in training: {odds_in_features}")
    #             raise ValueError("Odds features detected in training set (but use_odds=False)")
        
    #     seasons = sorted(self.df['seasonYear'].unique())
    #     start_idx = seasons.index(start_season)
        
    #     all_predictions = []
    #     metrics = []
        
    #     for i in range(start_idx, len(seasons)):
    #         test_season = seasons[i]
    #         train_seasons = seasons[:i]
            
    #         train_df = self.df[self.df['seasonYear'].isin(train_seasons)]
    #         test_df = self.df[self.df['seasonYear'] == test_season]
            
    #         if train_df.empty or test_df.empty:
    #             continue
            
    #         logging.info(f"--- Season: {test_season} (train: {len(train_df)}, test: {len(test_df)}) ---")
            
    #         # Select features using ONLY training data
    #         features_ml = self._select_features_from_subset(train_df, 'target_home_win', n_top=100)
    #         features_spread = self._select_features_from_subset(train_df, 'target_point_diff', n_top=100)
    #         features_total = self._select_features_from_subset(train_df, 'target_total_pts', n_top=100)
        
    #         # ===== MONEYLINE =====
    #         X_train_ml = train_df[features_ml]
    #         X_test_ml = test_df[features_ml]
    #         y_train_ml = train_df['target_home_win']
    #         y_test_ml = test_df['target_home_win']
            
    #         clf_ml = lgb.LGBMClassifier(**ml_params, verbosity=-1, random_state=42, importance_type='gain')
    #         clf_ml.fit(X_train_ml, y_train_ml)
            
    #         ml_probs = clf_ml.predict_proba(X_test_ml)[:, 1]
    #         ml_preds = (ml_probs > 0.5).astype(int)
            
    #         # ===== SPREAD =====
    #         X_train_spread = train_df[features_spread]
    #         X_test_spread = test_df[features_spread]
    #         y_train_spread = train_df['target_point_diff']
    #         y_test_spread = test_df['target_point_diff']
            
    #         reg_spread = lgb.LGBMRegressor(**spread_params, verbosity=-1, random_state=42, importance_type='gain')
    #         reg_spread.fit(X_train_spread, y_train_spread)
    #         spread_pred = reg_spread.predict(X_test_spread)
            
    #         # ===== TOTAL =====
    #         X_train_total = train_df[features_total]
    #         X_test_total = test_df[features_total]
    #         y_train_total = train_df['target_total_pts']
    #         y_test_total = test_df['target_total_pts']
            
    #         reg_total = lgb.LGBMRegressor(**total_params, verbosity=-1, random_state=42, importance_type='gain')
    #         reg_total.fit(X_train_total, y_train_total)
    #         total_pred = reg_total.predict(X_test_total)
            
    #         # ===== METRICS =====
    #         season_metrics = {
    #             'season': test_season,
    #             'n_games': len(test_df),
    #             # Moneyline
    #             'ml_accuracy': accuracy_score(y_test_ml, ml_preds),
    #             'ml_log_loss': log_loss(y_test_ml, ml_probs),
    #             'ml_brier': brier_score_loss(y_test_ml, ml_probs),
    #             # Spread
    #             'spread_mae': np.abs(y_test_spread - spread_pred).mean(),
    #             'spread_rmse': np.sqrt(mean_squared_error(y_test_spread, spread_pred)),
    #             # Total
    #             'total_mae': np.abs(y_test_total - total_pred).mean(),
    #             'total_rmse': np.sqrt(mean_squared_error(y_test_total, total_pred)),
    #         }
    #         metrics.append(season_metrics)
            
    #         # ===== STORE PREDICTIONS FOR BETTING SIM =====
    #         fold_preds = test_df[['gameId', 'gameDate', 'seasonYear', 
    #                             'teamId_home', 'teamId_away']].copy()
    #         fold_preds['ml_prob'] = ml_probs
    #         fold_preds['ml_actual'] = y_test_ml.values
    #         fold_preds['spread_pred'] = spread_pred
    #         fold_preds['spread_actual'] = y_test_spread.values
    #         fold_preds['total_pred'] = total_pred
    #         fold_preds['total_actual'] = y_test_total.values
            
    #         # Include odds if available for betting sim
    #         odds_cols = [c for c in test_df.columns if c.startswith('odds_')]
    #         for col in odds_cols:
    #             fold_preds[col] = test_df[col].values
                
    #         all_predictions.append(fold_preds)
        
    #     # ===== SUMMARY =====
    #     metrics_df = pd.DataFrame(metrics)
        
    #     print("\n" + "="*60)
    #     print(f"WALK-FORWARD RESULTS BY SEASON ({save_suffix})")
    #     print("="*60)
    #     print(metrics_df.to_string(index=False))
        
    #     print("\n" + "="*60)
    #     print("AGGREGATED PERFORMANCE")
    #     print("="*60)
    #     if not metrics_df.empty:
    #         print(f"Moneyline Accuracy:  {metrics_df['ml_accuracy'].mean():.4f} (±{metrics_df['ml_accuracy'].std():.4f})")
    #         print(f"Moneyline Log Loss:  {metrics_df['ml_log_loss'].mean():.4f}")
    #         print(f"Moneyline Brier:     {metrics_df['ml_brier'].mean():.4f}")
    #         print(f"Spread MAE:          {metrics_df['spread_mae'].mean():.2f} pts")
    #         print(f"Total MAE:           {metrics_df['total_mae'].mean():.2f} pts")
        
    #     # Save predictions for betting simulation
    #     if all_predictions:
    #         predictions_df = pd.concat(all_predictions, ignore_index=True)
    #         predictions_df.to_csv(self.model_dir / f'backtest_predictions{save_suffix}.csv', index=False)
    #         metrics_df.to_csv(self.model_dir / f'backtest_metrics{save_suffix}.csv', index=False)
        
    #     return metrics_df, pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()


    def backtest_walk_forward(self, start_season='2019-20', use_tuned_params=True, 
                            use_odds=False, save_suffix='_no_odds', calibrate=True,
                            calib_holdout_frac=0.2):
        """
        Walk-forward validation with optional isotonic calibration.
        
        If calibrate=True, each fold's training data is split:
        - (1 - calib_holdout_frac) for model training
        - calib_holdout_frac for fitting calibrators
        """
        
        # Load params (same as before)
        if use_tuned_params and (self.model_dir / 'best_config_ml.json').exists():
            with open(self.model_dir / 'best_config_ml.json') as f:
                ml_params = json.load(f)
        else:
            ml_params = dict(n_estimators=1000, learning_rate=0.01, max_depth=6,
                            num_leaves=31, min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5)

        if use_tuned_params and (self.model_dir / 'best_config_spread.json').exists():
            with open(self.model_dir / 'best_config_spread.json') as f:
                spread_params = json.load(f)
        else:
            spread_params = dict(n_estimators=1000, learning_rate=0.01, max_depth=6,
                                num_leaves=31, min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5)

        if use_tuned_params and (self.model_dir / 'best_config_total.json').exists():
            with open(self.model_dir / 'best_config_total.json') as f:
                total_params = json.load(f)
        else:
            total_params = dict(n_estimators=1000, learning_rate=0.01, max_depth=6,
                                num_leaves=31, min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5)

        self.load_data(use_odds=use_odds)
        
        if not use_odds:
            odds_in_features = [f for f in self.features if 'odds' in f.lower()]
            if odds_in_features:
                raise ValueError(f"Odds features detected in training set: {odds_in_features}")
        
        seasons = sorted(self.df['seasonYear'].unique())
        start_idx = seasons.index(start_season)
        
        all_predictions = []
        metrics = []
        calibration_stats = []  # Track calibration effectiveness
        
        for i in range(start_idx, len(seasons)):
            test_season = seasons[i]
            train_seasons = seasons[:i]
            
            full_train_df = self.df[self.df['seasonYear'].isin(train_seasons)].copy()
            test_df = self.df[self.df['seasonYear'] == test_season].copy()
            
            if full_train_df.empty or test_df.empty:
                continue
            
            # Sort by date for proper temporal split
            full_train_df = full_train_df.sort_values('gameDate').reset_index(drop=True)
            
            if calibrate:
                # Split: model training | calibration holdout
                n_model_train = int(len(full_train_df) * (1 - calib_holdout_frac))
                model_train_df = full_train_df.iloc[:n_model_train]
                calib_df = full_train_df.iloc[n_model_train:]
                logging.info(f"--- Season: {test_season} | model_train: {len(model_train_df)}, "
                            f"calib: {len(calib_df)}, test: {len(test_df)} ---")
            else:
                model_train_df = full_train_df
                calib_df = None
                logging.info(f"--- Season: {test_season} | train: {len(model_train_df)}, test: {len(test_df)} ---")
            
            # Feature selection on model_train only (no leakage)
            features_ml = self._select_features_from_subset(model_train_df, 'target_home_win', n_top=100)
            features_spread = self._select_features_from_subset(model_train_df, 'target_point_diff', n_top=100)
            features_total = self._select_features_from_subset(model_train_df, 'target_total_pts', n_top=100)
            
            # ===== MONEYLINE =====
            X_model_train_ml = model_train_df[features_ml]
            y_model_train_ml = model_train_df['target_home_win']
            X_test_ml = test_df[features_ml]
            y_test_ml = test_df['target_home_win']
            
            clf_ml = lgb.LGBMClassifier(**ml_params, verbosity=-1, random_state=42, importance_type='gain')
            clf_ml.fit(X_model_train_ml, y_model_train_ml)
            
            # Raw predictions
            ml_probs_raw = clf_ml.predict_proba(X_test_ml)[:, 1]
            
            if calibrate and calib_df is not None and len(calib_df) > 150:
                # Fit isotonic calibration on holdout
                X_calib_ml = calib_df[features_ml]
                y_calib_ml = calib_df['target_home_win']
                calib_probs_ml = clf_ml.predict_proba(X_calib_ml)[:, 1]
                
                iso_ml = IsotonicRegression(out_of_bounds='clip')
                iso_ml.fit(calib_probs_ml, y_calib_ml)
                
                # Apply calibration
                ml_probs = np.clip(iso_ml.predict(ml_probs_raw), 1e-15, 1 - 1e-15)
            else:
                ml_probs = ml_probs_raw
                iso_ml = None
            
            ml_preds = (ml_probs > 0.5).astype(int)
            
            # ===== SPREAD =====
            X_model_train_spread = model_train_df[features_spread]
            y_model_train_spread = model_train_df['target_point_diff']
            X_test_spread = test_df[features_spread]
            y_test_spread = test_df['target_point_diff']
            
            reg_spread = lgb.LGBMRegressor(**spread_params, verbosity=-1, random_state=42, importance_type='gain')
            reg_spread.fit(X_model_train_spread, y_model_train_spread)
            
            spread_pred = reg_spread.predict(X_test_spread)
            
            # Estimate prediction uncertainty from calibration set
            if calibrate and calib_df is not None and len(calib_df) > 50:
                X_calib_spread = calib_df[features_spread]
                y_calib_spread = calib_df['target_point_diff']
                calib_spread_pred = reg_spread.predict(X_calib_spread)
                spread_errors = y_calib_spread.values - calib_spread_pred
                spread_std = spread_errors.std()
                spread_bias = spread_errors.mean()
            else:
                spread_std = None
                spread_bias = 0
            
            # ===== TOTAL =====
            X_model_train_total = model_train_df[features_total]
            y_model_train_total = model_train_df['target_total_pts']
            X_test_total = test_df[features_total]
            y_test_total = test_df['target_total_pts']
            
            reg_total = lgb.LGBMRegressor(**total_params, verbosity=-1, random_state=42, importance_type='gain')
            reg_total.fit(X_model_train_total, y_model_train_total)
            
            total_pred = reg_total.predict(X_test_total)
            
            if calibrate and calib_df is not None and len(calib_df) > 50:
                X_calib_total = calib_df[features_total]
                y_calib_total = calib_df['target_total_pts']
                calib_total_pred = reg_total.predict(X_calib_total)
                total_errors = y_calib_total.values - calib_total_pred
                total_std = total_errors.std()
                total_bias = total_errors.mean()
            else:
                total_std = None
                total_bias = 0
            
            # ===== METRICS =====
            season_metrics = {
                'season': test_season,
                'n_games': len(test_df),
                'n_calib': len(calib_df) if calib_df is not None else 0,
                # Moneyline
                'ml_accuracy': accuracy_score(y_test_ml, ml_preds),
                'ml_log_loss': log_loss(y_test_ml, ml_probs),
                'ml_brier': brier_score_loss(y_test_ml, ml_probs),
                'ml_brier_raw': brier_score_loss(y_test_ml, ml_probs_raw) if calibrate else None,
                # Spread
                'spread_mae': np.abs(y_test_spread - spread_pred).mean(),
                'spread_rmse': np.sqrt(mean_squared_error(y_test_spread, spread_pred)),
                'spread_std_est': spread_std,
                'spread_bias_est': spread_bias,
                # Total
                'total_mae': np.abs(y_test_total - total_pred).mean(),
                'total_rmse': np.sqrt(mean_squared_error(y_test_total, total_pred)),
                'total_std_est': total_std,
                'total_bias_est': total_bias,
            }
            metrics.append(season_metrics)
            
            # ===== STORE PREDICTIONS =====
            fold_preds = test_df[['gameId', 'gameDate', 'seasonYear', 
                                'teamId_home', 'teamId_away']].copy()
            fold_preds['ml_prob'] = ml_probs
            fold_preds['ml_prob_raw'] = ml_probs_raw  # Keep raw for comparison
            fold_preds['ml_actual'] = y_test_ml.values
            fold_preds['spread_pred'] = spread_pred
            fold_preds['spread_actual'] = y_test_spread.values
            fold_preds['spread_std'] = spread_std  # For probabilistic betting
            fold_preds['total_pred'] = total_pred
            fold_preds['total_actual'] = y_test_total.values
            fold_preds['total_std'] = total_std
            
            # Include odds for betting sim
            odds_cols = [c for c in test_df.columns if c.startswith('odds_')]
            for col in odds_cols:
                fold_preds[col] = test_df[col].values
                
            all_predictions.append(fold_preds)
        
        # ===== SUMMARY =====
        metrics_df = pd.DataFrame(metrics)
        
        print("\n" + "="*60)
        print(f"WALK-FORWARD RESULTS BY SEASON ({save_suffix})")
        print("="*60)
        print(metrics_df.to_string(index=False))
        
        print("\n" + "="*60)
        print("AGGREGATED PERFORMANCE")
        print("="*60)
        if not metrics_df.empty:
            print(f"Moneyline Accuracy:  {metrics_df['ml_accuracy'].mean():.4f} (±{metrics_df['ml_accuracy'].std():.4f})")
            print(f"Moneyline Log Loss:  {metrics_df['ml_log_loss'].mean():.4f}")
            print(f"Moneyline Brier:     {metrics_df['ml_brier'].mean():.4f}")
            
            if calibrate and 'ml_brier_raw' in metrics_df.columns:
                raw_brier = metrics_df['ml_brier_raw'].dropna().mean()
                cal_brier = metrics_df['ml_brier'].mean()
                print(f"  (Raw Brier: {raw_brier:.4f}, Calibration improvement: {raw_brier - cal_brier:.4f})")
            
            print(f"Spread MAE:          {metrics_df['spread_mae'].mean():.2f} pts")
            print(f"Spread Std (avg):    {metrics_df['spread_std_est'].dropna().mean():.2f} pts")
            print(f"Total MAE:           {metrics_df['total_mae'].mean():.2f} pts")
            print(f"Total Std (avg):     {metrics_df['total_std_est'].dropna().mean():.2f} pts")
        
        # Save
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            predictions_df.to_csv(self.model_dir / f'backtest_predictions{save_suffix}.csv', index=False)
            metrics_df.to_csv(self.model_dir / f'backtest_metrics{save_suffix}.csv', index=False)
        
        return metrics_df, pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()


    def save_averaged_feature_importance(self, all_importances: list):
        """Average feature importances across all backtest folds."""
        
        # Stack into array: (n_folds, n_features)
        importance_array = np.array(all_importances)
        
        # Calculate mean and std across folds
        avg_importance = importance_array.mean(axis=0)
        std_importance = importance_array.std(axis=0)
        
        imp_df = pd.DataFrame({
            'feature': self.features,
            'importance_mean': avg_importance,
            'importance_std': std_importance,
            'importance_cv': std_importance / (avg_importance + 1e-10)  # Coefficient of variation
        }).sort_values('importance_mean', ascending=False)
        
        path = self.model_dir / 'feature_importance_averaged.csv'
        imp_df.to_csv(path, index=False)
        
        print("\nTop 15 Features (Averaged Across Folds):")
        print(imp_df.head(15).to_string(index=False))
        
        # Also save per-fold importances if you want to analyze stability
        fold_imp_df = pd.DataFrame(
            np.array(all_importances).T,  # Transpose: (n_features, n_folds)
            index=self.features
        )
        fold_imp_df.columns = [f'fold_{i}' for i in range(len(all_importances))]
        fold_imp_df.to_csv(self.model_dir / 'feature_importance_by_fold.csv')


    def select_features(self, target, n_top=100):
        """Train a proxy model to select the top N features for a specific target."""
        if self.df is None: self.load_data()
        
        logging.info(f"Selecting top {n_top} features for target: {target}...")
        
        # Use full dataset for selection (or could sample)
        X = self.clean_features(self.df[self.features])

        self.features = X.columns.tolist()

        y = self.df[target]
        
        # Proxy Model (Fast)
        if 'win' in target:
             model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
        else:
             model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
             
        model.fit(X, y)
        
        imp_df = pd.DataFrame({
            'feature': self.features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected = imp_df.head(n_top)['feature'].tolist()
        logging.info(f"Top 5 Selection: {selected[:5]}")
        return selected

    def save_model(self, model, filename):
        path = self.model_dir / filename
        joblib.dump(model, path)
        logging.info(f"Model saved to {path}")
        
    def save_feature_importance(self, model):
        imp_df = pd.DataFrame({
            'feature': self.features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        path = self.model_dir / 'feature_importance.csv'
        imp_df.to_csv(path, index=False)
        print("\nTop 10 Features:")
        print(imp_df.head(10))

    def evaluate_calibration(self, test_season='2024-25'):
        """
        Evaluate calibration using a proper temporal split:
        Train (Historic) -> Calibration (Recent Historic) -> Test (Target Season)
        """
        if self.df is None:
            self.load_data()
        
        logging.info(f"Evaluating Calibration (Test Season: {test_season})...")
        
        test_df = self.df[self.df['seasonYear'] == test_season].copy()
        date_thresh = test_df['gameDate'].min()
        
        # All history before test season
        full_train_df = self.df[self.df['gameDate'] < date_thresh].copy()
        full_train_df = full_train_df.sort_values('gameDate')
        
        if full_train_df.empty or test_df.empty:
            logging.warning("Not enough data for calibration evaluation.")
            return

        # Temporal Split: 80% Train, 20% Calibration
        n_train = int(len(full_train_df) * 0.8)
        train_df = full_train_df.iloc[:n_train]
        cal_df = full_train_df.iloc[n_train:]
        
        logging.info(f"Split Sizes - Train: {len(train_df)}, Calib: {len(cal_df)}, Test: {len(test_df)}")

        # SELECT FEATURES FROM TRAIN ONLY (no leakage)
        features = self._select_features_from_subset(train_df, 'target_home_win', n_top=100)
        
        X_train = train_df[features]
        y_train = train_df['target_home_win']
        
        X_cal = cal_df[features]
        y_cal = cal_df['target_home_win']
        
        X_test = test_df[features]
        y_test = test_df['target_home_win']

        # debug_data(X_train, y_train, "Training Data")
        # debug_data(X_cal, y_cal, "Calibration Data")
        # debug_data(X_test, y_test, "Test Data")

        # 1. Train Base Model (Uncalibrated)
        logging.info("Training Base Model (for Eval)...")
        # Use simpler params for eval speed, or match production
        base_clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01, max_depth=-1, 
                                    num_leaves=31, min_child_samples=20, subsample=0.8, 
                                    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                                     random_state=42, verbosity=-1, importance_type='gain')
        base_clf.fit(X_train, y_train)
        
        # Predict Uncalibrated on Test
        prob_uncal = base_clf.predict_proba(X_test)[:, 1]
        
        # 2. Calibrate (Isotonic on Calibration Set)
        logging.info("Fitting Isotonic Calibration on held-out Calibration Set...")
        # Get raw probs on calibration set
        prob_val_raw = base_clf.predict_proba(X_cal)[:, 1]
        
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(prob_val_raw, y_cal)
        
        # Predict Calibrated on Test
        # Raw test probs -> Isotonic Transform
        prob_cal = iso_reg.predict(prob_uncal)
        
        # 3. Plot & Compare
        plt.figure(figsize=(10, 10))
        
        # Uncalibrated Curve
        frac_pos_uncal, mean_pred_uncal = calibration_curve(y_test, prob_uncal, n_bins=10)
        plt.plot(mean_pred_uncal, frac_pos_uncal, "s-", label="Uncalibrated")
        
        # Calibrated Curve
        frac_pos_cal, mean_pred_cal = calibration_curve(y_test, prob_cal, n_bins=10)
        plt.plot(mean_pred_cal, frac_pos_cal, "d-", label="Isotonic Calibrated")
        
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        plt.ylabel("Fraction of Positives")
        plt.xlabel("Mean Predicted Value")
        plt.title(f"Reliability Diagram ({test_season}) - Temporal Split")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig('calibration_plot.png')
        logging.info("Calibration plot saved to calibration_plot.png")
        
        # Metrics
        brier_uncal = brier_score_loss(y_test, prob_uncal)
        brier_cal = brier_score_loss(y_test, prob_cal)
        print(f"Brier Score - Uncalibrated: {brier_uncal:.4f}, Calibrated: {brier_cal:.4f}")

    def analyze_regression_errors(self, test_season='2024-25'):
        """Analyze prediction errors for Spread and Total models to enable probabilistic outputs."""
        if self.df is None: self.load_data()
        
        logging.info(f"Analyzing Regression Errors (Test Season: {test_season})...")
        
        test_df = self.df[self.df['seasonYear'] == test_season]
        train_df = self.df[self.df['gameDate'] < test_df['gameDate'].min()]
        
        if train_df.empty or test_df.empty:
            logging.warning("Not enough data for error analysis.")
            return

        X_train = train_df[self.features]
        X_test = test_df[self.features]
        
        calib_meta = {}
        
        # 1. Spread (Point Diff)
        
        # SELECT FEATURES FROM TRAIN ONLY
        features_spread = self._select_features_from_subset(train_df, 'target_point_diff', n_top=100)
        features_total = self._select_features_from_subset(train_df, 'target_total_pts', n_top=100)
        
        X_train_spread = train_df[features_spread]
        X_test_spread = test_df[features_spread]
        y_train_spread = train_df['target_point_diff']
        y_test_spread = test_df['target_point_diff']
        
        reg_spread = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=-1, importance_type='gain')
        reg_spread.fit(X_train_spread, y_train_spread)
        pred_spread = reg_spread.predict(X_test_spread)
        
        errors_spread = y_test_spread - pred_spread
        mean_s = errors_spread.mean()
        std_s = errors_spread.std()
        
        logging.info(f"Spread Error: Mean={mean_s:.4f}, Std={std_s:.4f}")
        calib_meta['spread_std'] = float(std_s)
        calib_meta['spread_mean'] = float(mean_s)
        
        # 2. Total (Points)
        features_total = self.get_features_for_target('target_total_pts')
        
        X_train_total = train_df[features_total]
        X_test_total = test_df[features_total]    
        y_train_total = train_df['target_total_pts']
        y_test_total = test_df['target_total_pts']
        
        reg_total = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=-1, importance_type='gain')
        reg_total.fit(X_train_total, y_train_total)
        pred_total = reg_total.predict(X_test_total)
        
        errors_total = y_test_total - pred_total
        mean_t = errors_total.mean()
        std_t = errors_total.std()
        
        logging.info(f"Total Error: Mean={mean_t:.4f}, Std={std_t:.4f}")
        calib_meta['total_std'] = float(std_t)
        calib_meta['total_mean'] = float(mean_t)
        
        # Save Metadata
        with open('calibration_metadata.json', 'w') as f:
            json.dump(calib_meta, f)
        logging.info("Saved calibration_metadata.json")
        
        # Plot Distributions
        plt.figure(figsize=(12, 5))
        
        # Spread Plot
        plt.subplot(1, 2, 1)
        plt.hist(errors_spread, bins=30, density=True, alpha=0.6, color='g', label='Errors')
        # Normal Fit
        x = np.linspace(min(errors_spread), max(errors_spread), 100)
        plt.plot(x, stats.norm.pdf(x, mean_s, std_s), 'r-', lw=2, label='Normal Dist')
        plt.title(f"Spread Error Dist (Std={std_s:.2f})")
        plt.legend()
        
        # Total Plot
        plt.subplot(1, 2, 2)
        plt.hist(errors_total, bins=30, density=True, alpha=0.6, color='b', label='Errors')
        x2 = np.linspace(min(errors_total), max(errors_total), 100)
        plt.plot(x2, stats.norm.pdf(x2, mean_t, std_t), 'r-', lw=2, label='Normal Dist')
        plt.title(f"Total Error Dist (Std={std_t:.2f})")
        plt.legend()
        
        plt.savefig('error_distribution.png')
        logging.info("Saved error_distribution.png")


    # def train_and_save_all(self, use_odds=False, save_suffix=''):
    #     """Train final models on ALL available data and save them."""
    #     # Force reload to ensure correct feature set
    #     self.load_data(use_odds=use_odds)
        
    #     # Use temporal split for feature selection too
    #     n_feature_select = int(len(self.df) * 0.7)  # Use first 70% for feature selection
    #     feature_select_df = self.df.iloc[:n_feature_select]
        
    #     # Select features from historical portion only
    #     self.features_ml = self._select_features_from_subset(feature_select_df, 'target_home_win', n_top=100)
    #     self.features_spread = self._select_features_from_subset(feature_select_df, 'target_point_diff', n_top=100)
    #     self.features_total = self._select_features_from_subset(feature_select_df, 'target_total_pts', n_top=100)
            
    #     # Save feature config
    #     feature_config = {
    #         'ml': self.features_ml,
    #         'spread': self.features_spread,
    #         'total': self.features_total
    #     }
    #     with open(self.model_dir / f'feature_config{save_suffix}.json', 'w') as f:
    #         json.dump(feature_config, f)

    #     # ===== MONEYLINE =====
    #     X_ml = self.df[self.features_ml]
    #     y_ml = self.df['target_home_win']
        
    #     # Temporal split for FLAML validation
    #     n_train = int(len(X_ml) * 0.8)
        
    #     logging.info("Training FLAML AutoML for Moneyline...")
    #     automl_ml = AutoML()
    #     automl_ml.fit(
    #         X_ml.iloc[:n_train], y_ml.iloc[:n_train],
    #         task="classification",
    #         time_budget=300,
    #         metric='log_loss',
    #         estimator_list=['lgbm'],
    #         X_val=X_ml.iloc[n_train:],
    #         y_val=y_ml.iloc[n_train:],
    #         verbose=0
    #     )
    #     logging.info(f"Best ML Config: {automl_ml.best_config}")
        
    #     # Refit on full data with best params
    #     best_ml_params = automl_ml.best_config.copy()
    #     final_ml = lgb.LGBMClassifier(**best_ml_params, verbosity=-1, random_state=42)
    #     final_ml.fit(X_ml, y_ml)
    #     self.save_model(final_ml, f'latest_baseline{save_suffix}.pkl')
        
    #     # Save best config for backtest consistency
    #     with open(self.model_dir / f'best_config_ml{save_suffix}.json', 'w') as f:
    #         json.dump(best_ml_params, f)

    #         # ===== SPREAD =====
    #     X_spread = self.df[self.features_spread]
    #     y_spread = self.df['target_point_diff']
        
    #     # Temporal split for FLAML validation
    #     n_train = int(len(X_spread) * 0.8)
        
    #     logging.info("Training FLAML AutoML for Spread...")
    #     automl_spread = AutoML()
    #     automl_spread.fit(
    #         X_spread.iloc[:n_train], y_spread.iloc[:n_train],
    #         task="regression",
    #         time_budget=300,
    #         metric='mse',
    #         estimator_list=['lgbm'],
    #         X_val=X_spread.iloc[n_train:],
    #         y_val=y_spread.iloc[n_train:],
    #         verbose=0
    #     )
    #     logging.info(f"Best Spread Config: {automl_spread.best_config}")
        
    #     # Refit on full data with best params
    #     best_spread_params = automl_spread.best_config.copy()
    #     final_spread = lgb.LGBMRegressor(**best_spread_params, verbosity=-1, random_state=42)
    #     final_spread.fit(X_spread, y_spread)
    #     self.save_model(final_spread, f'latest_spread{save_suffix}.pkl')
        
    #     # Save best config for backtest consistency
    #     with open(self.model_dir / f'best_config_spread{save_suffix}.json', 'w') as f:
    #         json.dump(best_spread_params, f)

    #         # ===== TOTAL =====
    #     X_total = self.df[self.features_total]
    #     y_total = self.df['target_total_pts']
        
    #     # Temporal split for FLAML validation
    #     n_train = int(len(X_total) * 0.8)
        
    #     logging.info("Training FLAML AutoML for Total...")
    #     automl_total = AutoML()
    #     automl_total.fit(
    #         X_total.iloc[:n_train], y_total.iloc[:n_train],
    #         task="regression",
    #         time_budget=300,
    #         metric='mse',
    #         estimator_list=['lgbm'],
    #         X_val=X_total.iloc[n_train:],
    #         y_val=y_total.iloc[n_train:],
    #         verbose=0
    #     )
    #     logging.info(f"Best Total Config: {automl_total.best_config}")
        
    #     # Refit on full data with best params
    #     best_total_params = automl_total.best_config.copy()
    #     final_total = lgb.LGBMRegressor(**best_total_params, verbosity=-1, random_state=42)
    #     final_total.fit(X_total, y_total)
    #     self.save_model(final_total, f'latest_total{save_suffix}.pkl')
        
    #     # Save best config for backtest consistency
    #     with open(self.model_dir / f'best_config_total{save_suffix}.json', 'w') as f:
    #         json.dump(best_total_params, f)


    def train_and_save_all(self, use_odds=False, save_suffix='', calib_holdout_frac=0.2):
        """
        Train final models on ALL available data and save them, including calibrators.
        
        Split: [model_train (80%)] [calibration (20%)]
        - Models trained on model_train
        - Calibrators fitted on calibration set
        - Both saved for production use
        """
        self.load_data(use_odds=use_odds)
        
        # Sort by date for proper temporal split
        self.df = self.df.sort_values('gameDate').reset_index(drop=True)
        
        n_model_train = int(len(self.df) * (1 - calib_holdout_frac))
        model_train_df = self.df.iloc[:n_model_train]
        calib_df = self.df.iloc[n_model_train:]
        
        logging.info(f"Final training split: model_train={len(model_train_df)}, calib={len(calib_df)}")
        
        # Select features from model_train only (no leakage)
        self.features_ml = self._select_features_from_subset(model_train_df, 'target_home_win', n_top=100)
        self.features_spread = self._select_features_from_subset(model_train_df, 'target_point_diff', n_top=100)
        self.features_total = self._select_features_from_subset(model_train_df, 'target_total_pts', n_top=100)
        
        # Save feature config
        feature_config = {
            'ml': self.features_ml,
            'spread': self.features_spread,
            'total': self.features_total
        }
        with open(self.model_dir / f'feature_config{save_suffix}.json', 'w') as f:
            json.dump(feature_config, f)
        logging.info(f"Saved feature config with {len(self.features_ml)}/{len(self.features_spread)}/{len(self.features_total)} features")

        calibration_meta = {}

        # ===== MONEYLINE =====
        X_model_train_ml = model_train_df[self.features_ml]
        y_model_train_ml = model_train_df['target_home_win']
        X_calib_ml = calib_df[self.features_ml]
        y_calib_ml = calib_df['target_home_win']
        
        # Temporal split for FLAML validation (within model_train)
        n_flaml_train = int(len(X_model_train_ml) * 0.8)
        
        logging.info("Training FLAML AutoML for Moneyline...")
        automl_ml = AutoML()
        automl_ml.fit(
            X_model_train_ml.iloc[:n_flaml_train], y_model_train_ml.iloc[:n_flaml_train],
            task="classification",
            time_budget=300,
            metric='log_loss',
            estimator_list=['lgbm'],
            X_val=X_model_train_ml.iloc[n_flaml_train:],
            y_val=y_model_train_ml.iloc[n_flaml_train:],
            verbose=0
        )
        logging.info(f"Best ML Config: {automl_ml.best_config}")
        
        # Refit on full model_train with best params
        best_ml_params = automl_ml.best_config.copy()
        final_ml = lgb.LGBMClassifier(**best_ml_params, verbosity=-1, random_state=42)
        final_ml.fit(X_model_train_ml, y_model_train_ml)
        
        # Fit isotonic calibrator on calibration set
        calib_probs_ml = final_ml.predict_proba(X_calib_ml)[:, 1]
        iso_ml = IsotonicRegression(out_of_bounds='clip')
        iso_ml.fit(calib_probs_ml, y_calib_ml)
        
        # Evaluate calibration improvement
        brier_raw = brier_score_loss(y_calib_ml, calib_probs_ml)
        calib_probs_clipped = np.clip(iso_ml.predict(calib_probs_ml), 1e-15, 1 - 1e-15)
        brier_cal = brier_score_loss(y_calib_ml, calib_probs_clipped)
        logging.info(f"ML Calibration - Brier raw: {brier_raw:.4f}, calibrated: {brier_cal:.4f}")
        
        calibration_meta['ml_brier_raw'] = float(brier_raw)
        calibration_meta['ml_brier_calibrated'] = float(brier_cal)
        
        # Save model and calibrator
        self.save_model(final_ml, f'model_ml{save_suffix}.pkl')
        self.save_model(iso_ml, f'calibrator_ml{save_suffix}.pkl')
        
        with open(self.model_dir / f'best_config_ml{save_suffix}.json', 'w') as f:
            json.dump(best_ml_params, f)

        # ===== SPREAD =====
        X_model_train_spread = model_train_df[self.features_spread]
        y_model_train_spread = model_train_df['target_point_diff']
        X_calib_spread = calib_df[self.features_spread]
        y_calib_spread = calib_df['target_point_diff']
        
        n_flaml_train = int(len(X_model_train_spread) * 0.8)
        
        logging.info("Training FLAML AutoML for Spread...")
        automl_spread = AutoML()
        automl_spread.fit(
            X_model_train_spread.iloc[:n_flaml_train], y_model_train_spread.iloc[:n_flaml_train],
            task="regression",
            time_budget=300,
            metric='mse',
            estimator_list=['lgbm'],
            X_val=X_model_train_spread.iloc[n_flaml_train:],
            y_val=y_model_train_spread.iloc[n_flaml_train:],
            verbose=0
        )
        logging.info(f"Best Spread Config: {automl_spread.best_config}")
        
        best_spread_params = automl_spread.best_config.copy()
        final_spread = lgb.LGBMRegressor(**best_spread_params, verbosity=-1, random_state=42)
        final_spread.fit(X_model_train_spread, y_model_train_spread)
        
        # Compute error distribution on calibration set
        calib_spread_pred = final_spread.predict(X_calib_spread)
        spread_errors = y_calib_spread.values - calib_spread_pred
        spread_std = float(spread_errors.std())
        spread_bias = float(spread_errors.mean())
        spread_mae = float(np.abs(spread_errors).mean())
        
        logging.info(f"Spread Calibration - MAE: {spread_mae:.2f}, Std: {spread_std:.2f}, Bias: {spread_bias:.2f}")
        
        calibration_meta['spread_std'] = spread_std
        calibration_meta['spread_bias'] = spread_bias
        calibration_meta['spread_mae'] = spread_mae
        
        self.save_model(final_spread, f'model_spread{save_suffix}.pkl')
        
        with open(self.model_dir / f'best_config_spread{save_suffix}.json', 'w') as f:
            json.dump(best_spread_params, f)

        # ===== TOTAL =====
        X_model_train_total = model_train_df[self.features_total]
        y_model_train_total = model_train_df['target_total_pts']
        X_calib_total = calib_df[self.features_total]
        y_calib_total = calib_df['target_total_pts']
        
        n_flaml_train = int(len(X_model_train_total) * 0.8)
        
        logging.info("Training FLAML AutoML for Total...")
        automl_total = AutoML()
        automl_total.fit(
            X_model_train_total.iloc[:n_flaml_train], y_model_train_total.iloc[:n_flaml_train],
            task="regression",
            time_budget=300,
            metric='mse',
            estimator_list=['lgbm'],
            X_val=X_model_train_total.iloc[n_flaml_train:],
            y_val=y_model_train_total.iloc[n_flaml_train:],
            verbose=0
        )
        logging.info(f"Best Total Config: {automl_total.best_config}")
        
        best_total_params = automl_total.best_config.copy()
        final_total = lgb.LGBMRegressor(**best_total_params, verbosity=-1, random_state=42)
        final_total.fit(X_model_train_total, y_model_train_total)
        
        # Compute error distribution on calibration set
        calib_total_pred = final_total.predict(X_calib_total)
        total_errors = y_calib_total.values - calib_total_pred
        total_std = float(total_errors.std())
        total_bias = float(total_errors.mean())
        total_mae = float(np.abs(total_errors).mean())
        
        logging.info(f"Total Calibration - MAE: {total_mae:.2f}, Std: {total_std:.2f}, Bias: {total_bias:.2f}")
        
        calibration_meta['total_std'] = total_std
        calibration_meta['total_bias'] = total_bias
        calibration_meta['total_mae'] = total_mae
        
        self.save_model(final_total, f'model_total{save_suffix}.pkl')
        
        with open(self.model_dir / f'best_config_total{save_suffix}.json', 'w') as f:
            json.dump(best_total_params, f)

        # ===== SAVE CALIBRATION METADATA =====
        calibration_meta['calib_set_size'] = len(calib_df)
        calibration_meta['calib_date_range'] = {
            'start': str(calib_df['gameDate'].min()),
            'end': str(calib_df['gameDate'].max())
        }
        
        with open(self.model_dir / f'calibration_meta{save_suffix}.json', 'w') as f:
            json.dump(calibration_meta, f, indent=2)
        
        logging.info(f"Saved calibration metadata to calibration_meta{save_suffix}.json")
        
        # Print summary
        print("\n" + "="*60)
        print(f"FINAL MODELS SAVED ({save_suffix})")
        print("="*60)
        print(f"ML Model:      model_ml{save_suffix}.pkl")
        print(f"ML Calibrator: calibrator_ml{save_suffix}.pkl")
        print(f"Spread Model:  model_spread{save_suffix}.pkl")
        print(f"Total Model:   model_total{save_suffix}.pkl")
        print(f"Feature Config: feature_config{save_suffix}.json")
        print(f"Calibration Meta: calibration_meta{save_suffix}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Assuming script run from project root
    data_path = Path('data/processed/training_set_v4.csv')
    
    model = BaselineModel(data_path)
    

    metrics, preds = model.backtest_walk_forward(
        start_season='2019-20',
        use_odds=True,
        calibrate=True,
        save_suffix = '_calibrated'
        )
    
    model.evaluate_vs_market(preds)

    print("\n" + "="*80)
    print("TRAINING FINAL MODELS (WITH ODDS)")
    print("="*80)
    model.train_and_save_all(use_odds=True, save_suffix='_with_odds_v3')
    
