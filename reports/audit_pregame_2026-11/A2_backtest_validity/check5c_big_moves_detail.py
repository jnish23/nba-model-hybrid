"""Check 5c follow-up: for games with |close-open|>8 (spread), which line is wrong? Compare each line's sign with the same snapshot's
de-vigged moneyline favourite, and the training-set (single-book) opening spread. Also: how often does an opening/closing spread sign
disagree with its own moneyline favourite across all games?"""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix()); sys.path.insert(0,'C:/Users/jorda/nba-pregame-audit')
from common import *
from core.odds_clean import load_clean_odds
import warnings; warnings.filterwarnings('ignore')
o=load_clean_odds(DB); o['gameId']=o.gameId.astype(str).str.zfill(10); o['gameDate']=pd.to_datetime(o.gameDate)
o['margin']=o.home_score-o.away_score
t=load_train()[['gameId','odds_home_spread','odds_total','odds_home_ml','odds_away_ml']]
o=o.merge(t,on='gameId',how='left')
for pre in ('open_','close_'):
    d=o.dropna(subset=[pre+'home_spread',pre+'home_prob']); d=d[(d[pre+'home_spread'].abs()>=1.0)]
    dis=((d[pre+'home_spread']<0)!=(d[pre+'home_prob']>0.5))
    print(pre,'spread favourite disagrees with ML favourite (|spread|>=1): %d of %d'%(dis.sum(),len(d)))
d=o.dropna(subset=['odds_home_spread','odds_home_ml','odds_away_ml']); d=d[d.odds_home_spread.abs()>=1.0]
ph=np.where(d.odds_home_ml<0,-d.odds_home_ml/(-d.odds_home_ml+100),100/(d.odds_home_ml+100)); pa=np.where(d.odds_away_ml<0,-d.odds_away_ml/(-d.odds_away_ml+100),100/(d.odds_away_ml+100))
print('training-set odds_*: spread favourite disagrees with ML favourite (|spread|>=1): %d of %d'%(((d.odds_home_spread<0)!=(ph/(ph+pa)>0.5)).sum(),len(d)))
b=o[((o.close_home_spread-o.open_home_spread).abs()>8)].copy()
b['open_sign_ok_vs_ml']=((b.open_home_spread<0)==(b.open_home_prob>0.5)); b['close_sign_ok_vs_ml']=((b.close_home_spread<0)==(b.close_home_prob>0.5))
b['exact_flip']=((b.open_home_spread+b.close_home_spread).abs()<=2.5)&(np.sign(b.open_home_spread)!=np.sign(b.close_home_spread))
print('44 big-move games: open sign consistent with open ML:',int(b.open_sign_ok_vs_ml.sum()),' close sign consistent with close ML:',int(b.close_sign_ok_vs_ml.sum()),' sign flips with |open+close|<=2.5:',int(b.exact_flip.sum()), 'of', len(b))
print(b[['gameDate','open_home_spread','odds_home_spread','close_home_spread','open_home_prob','close_home_prob','margin','open_sign_ok_vs_ml','close_sign_ok_vs_ml']].round(3).to_string())
print('Which is closer to the realised margin (|margin+line|): open closer in',int(((b.margin+b.open_home_spread).abs()<(b.margin+b.close_home_spread).abs()).sum()),'of',len(b))
