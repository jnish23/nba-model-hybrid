"""Check 9d: my close-priced bet table equals the project's experiments.metrics.make_bets fed the close lines."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix()); sys.path.insert(0, 'C:/Users/jorda/nba-pregame-audit')
from common import *
from mymetrics import *
from experiments import metrics as pm
import warnings; warnings.filterwarnings('ignore')
p = load_e25()
q = p.dropna(subset=['odds_home_spread', 'close_home_spread', 'odds_total', 'close_total']).copy()
pb = pm.make_bets(q.assign(odds_home_spread=q.close_home_spread, odds_total=q.close_total), markets=('spread', 'total'))
mine, _ = make_bets(q, markets=('spread', 'total'), line_sp='close_home_spread', line_tot='close_total')
print('games with all four lines:', len(q), ' project bets:', pb.groupby('market').size().to_dict(), ' mine:', mine.groupby('market').size().to_dict())
for mk in ('spread', 'total'):
    a = mine[mine.market == mk].set_index('gameId'); b = pb[pb.market == mk].copy(); b['gameId'] = b.gameId.astype(str).str.zfill(10); b = b.set_index('gameId')
    print(mk, 'same game set:', set(a.index) == set(b.index), ' max |profit diff|:', (a.profit - b.profit.reindex(a.index)).abs().max())
