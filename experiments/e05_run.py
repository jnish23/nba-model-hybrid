"""E5 runner: python -m experiments.e05_run"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import features_e05                                  # noqa: E402
from experiments.feature_exp import run_feature_experiment           # noqa: E402
from models.data_paths import latest_training_set                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VARIANT = ROOT / 'data' / 'processed' / 'experiments' / 'e05_training_set.csv'

if __name__ == '__main__':
    VARIANT.parent.mkdir(parents=True, exist_ok=True)
    features_e05.build_variant(latest_training_set(), VARIANT, ROOT / 'data' / 'nba_stats.db')
    run_feature_experiment('E05_pace_adjusted', VARIANT, 'pace-normalised, opponent-adjusted team stats (added candidates)')
