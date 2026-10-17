import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, classification_report, mean_squared_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from scipy import stats
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import logging
import joblib
import json
from pathlib import Path
from flaml import AutoML, tune
from datetime import date

from models.model_registry import DEPLOYED_SUFFIX

class BaselineModel:
    def __init__(self, data_path, model_dir='models/saved'):
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.features = []        # legacy: union of all features
        self.features_diff = []   # diff_* + matchup + odds (for ML/Spread)
        self.features_sep = []    # *_home + *_away + odds, NO diff_* (for Totals)
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
            all_features = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('target_')]
        else:
            all_features = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('target_') and not c.startswith('odds_')]

        # --- Build two feature pools ---
        # 1. Diff pool (for ML / Spread): diff_*, matchup_*, odds_*, plus context cols
        #    that are NOT per-team (e.g. season_game_number-style context). Per-team pairs such as
        #    roster_bpm_home/away enter the diff pool only through diff_roster_bpm*.
        self.features_diff = [
            c for c in all_features
            if c.startswith('diff_') or c.startswith('matchup_') or c.startswith('odds_')
            or not (c.endswith('_home') or c.endswith('_away'))
        ]

        # 2. Sep pool (for Totals): *_home, *_away, odds_*, context cols — NO diff_*
        self.features_sep = [
            c for c in all_features
            if not c.startswith('diff_') and not c.startswith('matchup_')
        ]

        # Legacy: keep union for backward compat with older methods
        self.features = all_features

        logging.info(f"Feature pools — diff (ML/Spread): {len(self.features_diff)}, "
                     f"sep (Totals): {len(self.features_sep)}, "
                     f"total union: {len(self.features)} "
                     f"({'WITH' if use_odds else 'WITHOUT'} odds)")
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
        
        # Edge analysis using the cleaned dataframe. Each side is judged against its OWN price:
        # "1 - home implied" is not the away price (the two implied probabilities sum to > 1),
        # and using it credits every away bet with the vig.
        df_clean = df.iloc[np.where(valid_mask)[0]].copy() if len(valid_mask) == len(df) else df.copy()
        df_clean['market_prob_away'] = df_clean['odds_away_ml'].apply(self.implied_prob_from_american)
        df_clean = df_clean.dropna(subset=['market_prob_away'])
        df_clean['edge_home'] = df_clean['ml_prob'] - df_clean['market_prob']
        df_clean['edge_away'] = (1 - df_clean['ml_prob']) - df_clean['market_prob_away']
        
        print("\n--- Edge Analysis (each side vs its own price) ---")
        for thresh in [0.05, 0.10, 0.15]:
            home_bets = df_clean[(df_clean['edge_home'] > thresh) & (df_clean['edge_home'] >= df_clean['edge_away'])]
            if len(home_bets) > 0:
                actual_wr = home_bets['ml_actual'].mean()
                model_avg = home_bets['ml_prob'].mean()
                market_avg = home_bets['market_prob'].mean()
                print(f"\nHome edge > {thresh:.0%}: {len(home_bets)} bets")
                print(f"  Actual win rate: {actual_wr:.1%} | Model: {model_avg:.1%} | Market: {market_avg:.1%}")
                print(f"  ROI at the actual moneylines: {self._ml_roi(home_bets, 'home'):+.1%}")
            
            away_bets = df_clean[(df_clean['edge_away'] > thresh) & (df_clean['edge_away'] > df_clean['edge_home'])]
            if len(away_bets) > 0:
                actual_wr = 1 - away_bets['ml_actual'].mean()
                model_avg = 1 - away_bets['ml_prob'].mean()
                market_avg = away_bets['market_prob_away'].mean()
                print(f"\nAway edge > {thresh:.0%}: {len(away_bets)} bets")
                print(f"  Actual win rate: {actual_wr:.1%} | Model: {model_avg:.1%} | Market: {market_avg:.1%}")
                print(f"  ROI at the actual moneylines: {self._ml_roi(away_bets, 'away'):+.1%}")

    @staticmethod
    def _ml_roi(bets, side):
        """ROI of flat $100 moneyline bets on `side`, each settled at ITS OWN price (not a flat -110)."""
        from models.betting import settle_bet
        odds = bets['odds_home_ml'] if side == 'home' else bets['odds_away_ml']
        won = bets['ml_actual'] == (1 if side == 'home' else 0)
        profit = sum(settle_bet(o, w) for o, w in zip(odds, won))
        return profit / (100 * len(bets))

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

    def _get_pool_for_target(self, target: str) -> list:
        """Return the correct feature pool for a given target."""
        if target in ('target_home_win', 'target_point_diff'):
            return self.features_diff   # Differences for ML / Spread
        elif target == 'target_total_pts':
            return self.features_sep    # Separate home/away for Totals
        else:
            return self.features        # Fallback: full union

    def get_features_for_target(self, target: str) -> list:
        """Return the appropriate feature set for a target."""
        if self.df is None:
            self.load_data()
        if target == 'target_home_win':
            if self.features_ml is None:
                self.features_ml = self._select_features_from_subset(self.df, target)
            return self.features_ml
        elif target == 'target_point_diff':
            if self.features_spread is None:
                self.features_spread = self._select_features_from_subset(self.df, target)
            return self.features_spread
        elif target == 'target_total_pts':
            if self.features_total is None:
                self.features_total = self._select_features_from_subset(self.df, target)
            return self.features_total
        else:
            raise ValueError(f"Unknown target: {target}")


    def _select_features_from_subset(self, df_subset, target, n_top=150, n_min=20):
        """
        Two-Stage Feature Selection (No Leakage):
        1. Hierarchical Clustering (Spearman) to remove highly correlated redundant features.
        2. Null Importance Selection to filter out noise.
        
        Uses the target-appropriate feature pool (diff for ML/Spread, sep for Totals).
        """
        # 1. Filter available/valid features from the correct pool
        feature_pool = self._get_pool_for_target(target)
        available_features = [f for f in feature_pool if f in df_subset.columns]
        X = df_subset[available_features].copy().replace([np.inf, -np.inf], np.nan)
        
        # Drop zero-variance columns
        variances = X.var()
        valid_cols = variances[variances > 1e-10].index.tolist()
        X = X[valid_cols]
        y = df_subset[target]
        
        # Initial proxy model to gauge baseline importance
        if 'win' in target:
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, verbose=-1, random_state=42, n_jobs=-1)
        else:
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1, random_state=42, n_jobs=-1)
            
        model.fit(X, y)
        initial_imp = dict(zip(X.columns, model.feature_importances_))
        
        # ==========================================
        # STAGE 1: Redundancy Filter (Clustering)
        # ==========================================
        # Calculate Spearman rank correlation matrix
        corr_matrix = X.corr(method='spearman').abs()
        
        # Fill NaNs with 0 (no correlation) just in case a column became effectively constant
        corr_matrix = corr_matrix.fillna(0)
        
        # Convert to distance matrix (clip to avoid precision issues < 0)
        dist_matrix = np.clip(1 - corr_matrix, 0, 1)
        dist_array = squareform(dist_matrix, checks=False)
        
        # Hierarchical clustering (average linkage)
        linked = sch.linkage(dist_array, method='average')
        
        # Group features with < 0.15 distance (> 0.85 correlation)
        labels = sch.fcluster(linked, t=0.15, criterion='distance')
        
        cluster_to_feature = {}
        for feat, label in zip(X.columns, labels):
            if label not in cluster_to_feature:
                cluster_to_feature[label] = feat
            else:
                # Keep the one with the highest initial LightGBM importance
                current_best = cluster_to_feature[label]
                if initial_imp[feat] > initial_imp[current_best]:
                    cluster_to_feature[label] = feat
                    
        stage1_features = list(cluster_to_feature.values())
        logging.info(f"[{target}] Stage 1 (Clustering): Reduced from {len(X.columns)} to {len(stage1_features)} independent features.")
        
        # If we have very few features, just return what we have
        if len(stage1_features) <= n_min:
            return sorted(stage1_features, key=lambda f: initial_imp[f], reverse=True)
            
        # ==========================================
        # STAGE 2: Noise Filter (Null Importance)
        # ==========================================
        X_stage1 = X[stage1_features]
        n_permutations = 10
        rng = np.random.default_rng(42)
        null_importances = np.zeros((n_permutations, len(stage1_features)))
        
        for i in range(n_permutations):
            y_shuffled = rng.permutation(y)
            model.fit(X_stage1, y_shuffled)
            null_importances[i, :] = model.feature_importances_
            
        # 90th percentile of null importances bounds the noise for each feature
        null_p90 = np.percentile(null_importances, 90, axis=0)
        
        # Get actual importance using only the stage1 features
        model.fit(X_stage1, y)
        actual_imp = model.feature_importances_
        
        # Filter: Actual must be > Null P90 AND > 0
        final_candidates = []
        for idx, feat in enumerate(stage1_features):
            if actual_imp[idx] > null_p90[idx] and actual_imp[idx] > 0:
                final_candidates.append((feat, actual_imp[idx]))
                
        # Sort by actual importance
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in final_candidates]
        
        logging.info(f"[{target}] Stage 2 (Null Imp): Reduced from {len(stage1_features)} to {len(selected)} high-signal features.")
        
        # Enforce constraints (n_min and n_top)
        if len(selected) > n_top:
            selected = selected[:n_top]
        elif len(selected) < n_min:
            # Fallback to the top stage1 features if too many were stripped
            logging.warning(f"[{target}] Too few features passed Null Imp. Padding to {n_min}.")
            fallback = sorted(stage1_features, key=lambda f: initial_imp[f], reverse=True)
            for f in fallback:
                if f not in selected:
                    selected.append(f)
                if len(selected) == n_min:
                    break
                    
        return selected



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
            odds_in_diff = [f for f in self.features_diff if 'odds' in f.lower()]
            odds_in_sep = [f for f in self.features_sep if 'odds' in f.lower()]
            if odds_in_diff or odds_in_sep:
                raise ValueError(f"Odds features detected in training set: diff={odds_in_diff}, sep={odds_in_sep}")
        
        seasons = sorted(self.df['seasonYear'].unique())
        start_idx = seasons.index(start_season)
        
        all_predictions = []
        metrics = []
        calibration_stats = []  # Track calibration effectiveness

        # ===== DIAGNOSTIC: Track previous fold features for stability check =====
        prev_features_ml = None
        prev_features_spread = None
        prev_features_total = None
        
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

            # =====================================================================
            # DIAGNOSTIC 1: Temporal Leakage Assertion
            # =====================================================================
            assert model_train_df['gameDate'].max() < test_df['gameDate'].min(), \
                f"Temporal leak: train max {model_train_df['gameDate'].max()} >= test min {test_df['gameDate'].min()}"
            logging.info(f"[DIAG] Temporal check PASSED: train ends {model_train_df['gameDate'].max().date()}, "
                        f"test starts {test_df['gameDate'].min().date()}")

            # =====================================================================
            # DIAGNOSTIC (bonus): NaN Audit on training features
            # =====================================================================
            all_pool_features = list(set(self.features_diff + self.features_sep))
            avail_pool = [f for f in all_pool_features if f in model_train_df.columns]
            nan_rates = model_train_df[avail_pool].isnull().mean()
            high_nan = nan_rates[nan_rates > 0.10]
            if len(high_nan) > 0:
                logging.warning(f"[DIAG] {len(high_nan)} features with >10% NaN in train: "
                               f"{dict(high_nan.nlargest(5).round(3))}")
            else:
                logging.info(f"[DIAG] NaN audit PASSED: no features with >10% NaN in model_train")

            # =====================================================================
            # DIAGNOSTIC (bonus): Target Distribution Check
            # =====================================================================
            train_home_win_rate = model_train_df['target_home_win'].mean()
            test_home_win_rate = test_df['target_home_win'].mean()
            logging.info(f"[DIAG] Home win rate — train: {train_home_win_rate:.3f}, test: {test_home_win_rate:.3f}")
            logging.info(f"[DIAG] Total pts — train mean: {model_train_df['target_total_pts'].mean():.1f}, "
                        f"test mean: {test_df['target_total_pts'].mean():.1f}, "
                        f"train std: {model_train_df['target_total_pts'].std():.1f}")

            # Feature selection on model_train only (no leakage)
            features_ml = self._select_features_from_subset(model_train_df, 'target_home_win')
            features_spread = self._select_features_from_subset(model_train_df, 'target_point_diff')
            features_total = self._select_features_from_subset(model_train_df, 'target_total_pts')
            
            logging.info(f"Fold {test_season} features - ML: {len(features_ml)}, Spread: {len(features_spread)}, Total: {len(features_total)}")

            # =====================================================================
            # DIAGNOSTIC 2: Feature Stability (Jaccard similarity vs previous fold)
            # =====================================================================
            if prev_features_ml is not None:
                for label, curr, prev in [
                    ('ML', features_ml, prev_features_ml),
                    ('Spread', features_spread, prev_features_spread),
                    ('Total', features_total, prev_features_total),
                ]:
                    overlap = len(set(curr) & set(prev))
                    union = len(set(curr) | set(prev))
                    jaccard = overlap / union if union > 0 else 0
                    logging.info(f"[DIAG] {label} feature stability vs prev fold: "
                                f"Jaccard={jaccard:.2f}, overlap={overlap}/{len(curr)}")
                    if jaccard < 0.50:
                        logging.warning(f"[DIAG] LOW {label} feature stability (Jaccard={jaccard:.2f})! "
                                       f"Features may be unstable across folds.")

            prev_features_ml = features_ml
            prev_features_spread = features_spread
            prev_features_total = features_total

            # =====================================================================
            # DIAGNOSTIC 7: Explicit Odds Feature Check
            # =====================================================================
            for label, feats in [('ML', features_ml), ('Spread', features_spread), ('Total', features_total)]:
                odds_selected = [f for f in feats if 'odds' in f.lower()]
                if odds_selected:
                    logging.info(f"[DIAG] Odds features in {label} selection: {odds_selected}")
                else:
                    logging.info(f"[DIAG] No odds features in {label} selection")

                # Also verify feature pool correctness
                if label in ('ML', 'Spread'):
                    home_away_leaked = [f for f in feats if (f.endswith('_home') or f.endswith('_away'))
                                       and not f.startswith('odds_')]
                    if home_away_leaked:
                        logging.warning(f"[DIAG] {label} has raw home/away features (expected diff only): "
                                       f"{home_away_leaked[:5]}...")
                elif label == 'Total':
                    diff_leaked = [f for f in feats if f.startswith('diff_')]
                    if diff_leaked:
                        logging.warning(f"[DIAG] Total has diff features (expected sep only): "
                                       f"{diff_leaked[:5]}...")

            # ===== MONEYLINE =====
            X_model_train_ml = model_train_df[features_ml]
            y_model_train_ml = model_train_df['target_home_win']
            X_test_ml = test_df[features_ml]
            y_test_ml = test_df['target_home_win']
            
            clf_ml = lgb.LGBMClassifier(**ml_params, verbosity=-1, random_state=42, importance_type='gain')
            clf_ml.fit(X_model_train_ml, y_model_train_ml)
            
            # Raw predictions
            ml_probs_raw = clf_ml.predict_proba(X_test_ml)[:, 1]

            # =====================================================================
            # DIAGNOSTIC 4 (ML): Train-vs-Test Performance Gap
            # =====================================================================
            ml_train_probs = clf_ml.predict_proba(X_model_train_ml)[:, 1]
            ml_train_acc = accuracy_score(y_model_train_ml, (ml_train_probs > 0.5).astype(int))
            ml_train_logloss = log_loss(y_model_train_ml, ml_train_probs)
            ml_test_acc_raw = accuracy_score(y_test_ml, (ml_probs_raw > 0.5).astype(int))
            ml_test_logloss_raw = log_loss(y_test_ml, ml_probs_raw)
            logging.info(f"[DIAG] ML train acc: {ml_train_acc:.4f}, test acc: {ml_test_acc_raw:.4f} "
                        f"(gap: {ml_train_acc - ml_test_acc_raw:+.4f})")
            logging.info(f"[DIAG] ML train logloss: {ml_train_logloss:.4f}, test logloss: {ml_test_logloss_raw:.4f} "
                        f"(gap: {ml_test_logloss_raw - ml_train_logloss:+.4f})")
            if (ml_train_acc - ml_test_acc_raw) > 0.10:
                logging.warning(f"[DIAG] ML train-test accuracy gap > 10% — possible overfitting!")
            
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

            # =====================================================================
            # DIAGNOSTIC 3 (ML): Prediction Distribution Sanity Check
            # =====================================================================
            logging.info(f"[DIAG] ML prob range: [{ml_probs.min():.3f}, {ml_probs.max():.3f}], "
                        f"std={ml_probs.std():.3f}, mean={ml_probs.mean():.3f}")
            if ml_probs.std() < 0.01:
                logging.warning(f"[DIAG] ML predictions are near-constant (std={ml_probs.std():.4f})! "
                               f"Model may not be learning meaningful patterns.")

            # =====================================================================
            # DIAGNOSTIC 6 (ML): Feature Importance Consistency — Top 5
            # =====================================================================
            ml_imp = pd.Series(clf_ml.feature_importances_, index=features_ml).sort_values(ascending=False)
            top5_ml = ml_imp.head(5)
            logging.info(f"[DIAG] ML Top 5 features by importance:")
            for feat, imp in top5_ml.items():
                feat_vals = X_model_train_ml[feat]
                logging.info(f"  {feat}: importance={imp:.1f}, "
                            f"mean={feat_vals.mean():.4f}, std={feat_vals.std():.4f}, "
                            f"NaN%={feat_vals.isnull().mean():.1%}, "
                            f"range=[{feat_vals.min():.3f}, {feat_vals.max():.3f}]")
                if feat_vals.std() < 1e-6:
                    logging.warning(f"[DIAG] Top ML feature '{feat}' has near-zero variance! "
                                   f"Data pipeline issue suspected.")

            # ===== SPREAD =====
            X_model_train_spread = model_train_df[features_spread]
            y_model_train_spread = model_train_df['target_point_diff']
            X_test_spread = test_df[features_spread]
            y_test_spread = test_df['target_point_diff']
            
            reg_spread = lgb.LGBMRegressor(**spread_params, verbosity=-1, random_state=42, importance_type='gain')
            reg_spread.fit(X_model_train_spread, y_model_train_spread)
            
            spread_pred = reg_spread.predict(X_test_spread)

            # =====================================================================
            # DIAGNOSTIC 4 (Spread): Train-vs-Test Performance Gap
            # =====================================================================
            spread_train_pred = reg_spread.predict(X_model_train_spread)
            spread_train_mae = np.abs(y_model_train_spread - spread_train_pred).mean()
            spread_test_mae = np.abs(y_test_spread - spread_pred).mean()
            logging.info(f"[DIAG] Spread train MAE: {spread_train_mae:.2f}, test MAE: {spread_test_mae:.2f} "
                        f"(gap: {spread_test_mae - spread_train_mae:+.2f})")
            if (spread_test_mae - spread_train_mae) > 3.0:
                logging.warning(f"[DIAG] Spread train-test MAE gap > 3 pts — possible overfitting!")

            # =====================================================================
            # DIAGNOSTIC 3 (Spread): Prediction Distribution Sanity Check
            # =====================================================================
            logging.info(f"[DIAG] Spread pred range: [{spread_pred.min():.1f}, {spread_pred.max():.1f}], "
                        f"std={spread_pred.std():.1f}, mean={spread_pred.mean():.1f}")

            # =====================================================================
            # DIAGNOSTIC 6 (Spread): Feature Importance — Top 5
            # =====================================================================
            spread_imp = pd.Series(reg_spread.feature_importances_, index=features_spread).sort_values(ascending=False)
            top5_spread = spread_imp.head(5)
            logging.info(f"[DIAG] Spread Top 5 features by importance:")
            for feat, imp in top5_spread.items():
                feat_vals = X_model_train_spread[feat]
                logging.info(f"  {feat}: importance={imp:.1f}, "
                            f"mean={feat_vals.mean():.4f}, std={feat_vals.std():.4f}, "
                            f"NaN%={feat_vals.isnull().mean():.1%}")
                if feat_vals.std() < 1e-6:
                    logging.warning(f"[DIAG] Top Spread feature '{feat}' has near-zero variance!")

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

            # =====================================================================
            # DIAGNOSTIC 4 (Total): Train-vs-Test Performance Gap
            # =====================================================================
            total_train_pred = reg_total.predict(X_model_train_total)
            total_train_mae = np.abs(y_model_train_total - total_train_pred).mean()
            total_test_mae = np.abs(y_test_total - total_pred).mean()
            logging.info(f"[DIAG] Total train MAE: {total_train_mae:.2f}, test MAE: {total_test_mae:.2f} "
                        f"(gap: {total_test_mae - total_train_mae:+.2f})")
            if (total_test_mae - total_train_mae) > 5.0:
                logging.warning(f"[DIAG] Total train-test MAE gap > 5 pts — possible overfitting!")

            # =====================================================================
            # DIAGNOSTIC 3 (Total): Prediction Distribution Sanity Check
            # =====================================================================
            logging.info(f"[DIAG] Total pred range: [{total_pred.min():.1f}, {total_pred.max():.1f}], "
                        f"std={total_pred.std():.1f}, mean={total_pred.mean():.1f}")
            if total_pred.std() < 1.0:
                logging.warning(f"[DIAG] Total predictions have very low variance (std={total_pred.std():.2f})! "
                               f"Model may be predicting near-constant values.")

            # =====================================================================
            # DIAGNOSTIC 6 (Total): Feature Importance — Top 5
            # =====================================================================
            total_imp = pd.Series(reg_total.feature_importances_, index=features_total).sort_values(ascending=False)
            top5_total = total_imp.head(5)
            logging.info(f"[DIAG] Total Top 5 features by importance:")
            for feat, imp in top5_total.items():
                feat_vals = X_model_train_total[feat]
                logging.info(f"  {feat}: importance={imp:.1f}, "
                            f"mean={feat_vals.mean():.4f}, std={feat_vals.std():.4f}, "
                            f"NaN%={feat_vals.isnull().mean():.1%}")
                if feat_vals.std() < 1e-6:
                    logging.warning(f"[DIAG] Top Total feature '{feat}' has near-zero variance!")
            
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

            # =====================================================================
            # DIAGNOSTIC 5: Per-Fold Calibration Plot
            # =====================================================================
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f'Fold {test_season} — Diagnostics', fontsize=14)

                # ML Calibration
                frac_pos, mean_pred = calibration_curve(y_test_ml, ml_probs, n_bins=8, strategy='uniform')
                axes[0].plot(mean_pred, frac_pos, 's-', label='Calibrated' if iso_ml else 'Raw')
                if iso_ml:
                    frac_pos_raw, mean_pred_raw = calibration_curve(y_test_ml, ml_probs_raw, n_bins=8, strategy='uniform')
                    axes[0].plot(mean_pred_raw, frac_pos_raw, 'd--', alpha=0.5, label='Raw')
                axes[0].plot([0, 1], [0, 1], 'k:', alpha=0.5)
                axes[0].set_title(f'ML Reliability (Brier={brier_score_loss(y_test_ml, ml_probs):.4f})')
                axes[0].set_xlabel('Predicted Probability')
                axes[0].set_ylabel('Observed Frequency')
                axes[0].legend(fontsize=8)

                # Spread Error Distribution
                spread_test_errors = y_test_spread.values - spread_pred
                axes[1].hist(spread_test_errors, bins=25, alpha=0.7, color='steelblue', density=True)
                axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)
                axes[1].set_title(f'Spread Errors (MAE={np.abs(spread_test_errors).mean():.1f})')
                axes[1].set_xlabel('Actual - Predicted')

                # Total Error Distribution
                total_test_errors = y_test_total.values - total_pred
                axes[2].hist(total_test_errors, bins=25, alpha=0.7, color='coral', density=True)
                axes[2].axvline(0, color='red', linestyle='--', alpha=0.7)
                axes[2].set_title(f'Total Errors (MAE={np.abs(total_test_errors).mean():.1f})')
                axes[2].set_xlabel('Actual - Predicted')

                plt.tight_layout()
                plot_path = self.model_dir / f'diagnostics_fold_{test_season}{save_suffix}.png'
                plt.savefig(plot_path, dpi=100)
                plt.close(fig)
                logging.info(f"[DIAG] Saved fold diagnostics plot: {plot_path.name}")
            except Exception as e:
                logging.warning(f"[DIAG] Could not generate fold plot: {e}")
            
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
                # Diagnostic: train-test gaps
                'ml_train_acc': ml_train_acc,
                'ml_train_test_gap': ml_train_acc - ml_test_acc_raw,
                'spread_train_mae': spread_train_mae,
                'spread_train_test_gap': spread_test_mae - spread_train_mae,
                'total_train_mae': total_train_mae,
                'total_train_test_gap': total_test_mae - total_train_mae,
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

            logging.info(f"[DIAG] === Fold {test_season} complete ===\n")
        
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

            # =====================================================================
            # DIAGNOSTIC SUMMARY: Overfitting Report
            # =====================================================================
            print("\n" + "="*60)
            print("OVERFITTING REPORT (Train-Test Gaps)")
            print("="*60)
            print(f"ML Accuracy Gap (avg):   {metrics_df['ml_train_test_gap'].mean():+.4f} "
                  f"(per-fold: {metrics_df['ml_train_test_gap'].tolist()})")
            print(f"Spread MAE Gap (avg):    {metrics_df['spread_train_test_gap'].mean():+.2f} pts "
                  f"(per-fold: {[f'{v:.1f}' for v in metrics_df['spread_train_test_gap']]})")
            print(f"Total MAE Gap (avg):     {metrics_df['total_train_test_gap'].mean():+.2f} pts "
                  f"(per-fold: {[f'{v:.1f}' for v in metrics_df['total_train_test_gap']]})")
        
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
        features = self._select_features_from_subset(train_df, 'target_home_win')
        
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
        features_spread = self._select_features_from_subset(train_df, 'target_point_diff')
        
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
        features_total = self._select_features_from_subset(train_df, 'target_total_pts')
        
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


    def train_and_save_all(self, use_odds=False, save_suffix='', calib_holdout_frac=0.2, time_budget=600):
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
        self.features_ml = self._select_features_from_subset(model_train_df, 'target_home_win')
        self.features_spread = self._select_features_from_subset(model_train_df, 'target_point_diff')
        self.features_total = self._select_features_from_subset(model_train_df, 'target_total_pts')
        
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

        custom_hp = {
            "lgbm": {
                "reg_alpha": {"domain": tune.loguniform(lower=0.01, upper=50.0)},
                "reg_lambda": {"domain": tune.loguniform(lower=0.01, upper=50.0)},
                "colsample_bytree": {"domain": tune.uniform(lower=0.4, upper=0.9)},
                "min_child_samples": {"domain": tune.randint(lower=20, upper=100)},
                "n_estimators": {"domain": 1000},
            }
        }

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
            time_budget=time_budget,
            metric='log_loss',
            estimator_list=['lgbm'],
            custom_hp=custom_hp,
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
            time_budget=time_budget,
            metric='mse',
            estimator_list=['lgbm'],
            custom_hp=custom_hp,
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
            time_budget=time_budget,
            metric='mse',
            estimator_list=['lgbm'],
            custom_hp=custom_hp,
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-backtest', action='store_true', help='skip the walk-forward backtest')
    parser.add_argument('--deploy', action='store_true',
                        help=f'train the production model set (opening odds as inputs) and save it as '
                             f'"{DEPLOYED_SUFFIX}", the set scripts/run_inference_v2.py loads')
    parser.add_argument('--no-research-model', action='store_true', help='skip the dated no-odds research model')
    parser.add_argument('--time-budget', type=int, default=600, help='FLAML seconds per model when training final models')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Latest dated training set written by scripts/generate_training_set.py (ISO dates sort lexically)
    from models.data_paths import latest_training_set
    data_path = latest_training_set()
    
    model = BaselineModel(data_path)
    today = date.today()

    if not args.skip_backtest:
        metrics, preds = model.backtest_walk_forward(
            start_season='2019-20',
            use_odds=False,
            calibrate=True,
            save_suffix='_calibrated'
            )
        model.evaluate_vs_market(preds)

    if not args.no_research_model:
        print("\n" + "="*80)
        print("TRAINING FINAL MODELS (NO ODDS)")
        print("="*80)
        model.train_and_save_all(use_odds=False, save_suffix=f'_no_odds_{today}', time_budget=args.time_budget)

    if args.deploy:
        print("\n" + "="*80)
        print(f"TRAINING DEPLOYED MODEL SET ({DEPLOYED_SUFFIX}, WITH opening odds)")
        print("="*80)
        BaselineModel(data_path).train_and_save_all(use_odds=True, save_suffix=DEPLOYED_SUFFIX,
                                                    time_budget=args.time_budget)


if __name__ == "__main__":
    main()
