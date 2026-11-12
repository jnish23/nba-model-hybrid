"""Check 5a (part 2): is nba_odds ('closing') the line AT tip (frozen), the last pre-game snapshot (30 min earlier), or an in-game value?
For 2026-27 games: per book/field equality of nba_odds with (i) last in_game=0 snapshot, (ii) first in_game=1 snapshot (line shown at tip),
(iii) last snapshot of the game; drift of in-game rows from the last pre-game row by minutes after tip; and accuracy against results
(MAE of margin+spread, total-total) for opening / last pre-game snapshot / nba_odds."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from common import *
BOOKS=['betmgm','fanduel','caesars','bet365','draftkings','bet_rivers_ny']
with conn() as c:
    s=pd.read_sql('select * from odds_snapshots',c)
    cl=pd.read_sql("select * from nba_odds where gameDate>='2026-10-01'",c)
    op=pd.read_sql("select * from nba_odds_opening where gameDate>='2026-10-01'",c)
    res=pd.read_sql("select gameId,gameDate,teamId,MATCHUP,points from team_game_full where seasonYear='2026-27'",c)
for d in (s,cl,op):
    for col in ('home_team','away_team'): d[col]=d[col].replace({'LA Clippers':'Los Angeles Clippers'})
s['fetched']=pd.to_datetime(s.fetched_at_et)
cl['game_date']=pd.to_datetime(cl.gameDate).dt.strftime('%Y-%m-%d'); op['game_date']=pd.to_datetime(op.gameDate).dt.strftime('%Y-%m-%d')
cur=s[s.line_type=='current'].sort_values('fetched')
key=['game_date','home_team','away_team','book']
last_pre=cur[cur.in_game==0].drop_duplicates(key,keep='last'); first_ig=cur[cur.in_game==1].drop_duplicates(key,keep='first'); last_any=cur.drop_duplicates(key,keep='last')
rows=[]
for b in BOOKS:
    for f in ('home_spread','total','home_ml','away_ml'):
        col=f'{f}_{b}'
        m=cl[['game_date','home_team','away_team',col]].copy(); m[col]=pd.to_numeric(m[col],errors='coerce'); m['book']=b
        for nm,src in (('last_pre',last_pre),('first_ingame',first_ig),('last_any',last_any)):
            m=m.merge(src[key+[f]].rename(columns={f:nm}),on=key,how='left')
        ok=m[col].notna()&m['last_pre'].notna()&m['first_ingame'].notna()
        rows.append({'book':b,'field':f,'n':int(ok.sum()),'eq_last_pregame':(m.loc[ok,col]==m.loc[ok,'last_pre']).mean(),
                     'eq_first_ingame_row':(m.loc[ok,col]==m.loc[ok,'first_ingame']).mean(),'eq_last_any':(m.loc[ok,col]==m.loc[ok,'last_any']).mean(),
                     'pre_vs_firstIG_equal':(m.loc[ok,'last_pre']==m.loc[ok,'first_ingame']).mean()})
r=pd.DataFrame(rows); print(r.groupby('field')[['n','eq_last_pregame','eq_first_ingame_row','eq_last_any','pre_vs_firstIG_equal']].agg({'n':'sum','eq_last_pregame':'mean','eq_first_ingame_row':'mean','eq_last_any':'mean','pre_vs_firstIG_equal':'mean'}).round(3))
r.to_csv(OUT/'check5a2_close_vs_tip_rows.csv',index=False)
# accuracy vs results, consensus per game
def cons(df,fld): return df.groupby(['game_date','home_team','away_team'])[fld].median()
tgm=res.copy(); tgm['home']=tgm.MATCHUP.str.contains(' vs. ',regex=False)
h=tgm[tgm.home][['gameId','gameDate','points']].rename(columns={'points':'hp'}); a=tgm[~tgm.home][['gameId','points']].rename(columns={'points':'ap'})
g=h.merge(a,on='gameId'); g['margin']=g.hp-g.ap; g['tot']=g.hp+g.ap
nm=pd.read_sql("select distinct gameId, home_team, away_team from nba_odds",conn()) if False else None
# map game -> teams through nba_odds rows (game_date + home_team) -> need names: use cl rows with scores
cl2=cl.copy(); cl2['hs']=pd.to_numeric(cl2.home_score,errors='coerce'); cl2['as_']=pd.to_numeric(cl2.away_score,errors='coerce')
base=pd.DataFrame({'spread_open':cons(op.assign(**{c:pd.to_numeric(op[c],errors='coerce') for c in [f'home_spread_{b}' for b in BOOKS]}).melt(id_vars=['game_date','home_team','away_team'],value_vars=[f'home_spread_{b}' for b in BOOKS]).rename(columns={'value':'v'}),'v')})
def long(df,f): 
    return df.melt(id_vars=['game_date','home_team','away_team'],value_vars=[f'{f}_{b}' for b in BOOKS]).assign(value=lambda d:pd.to_numeric(d['value'],errors='coerce'))
out=pd.DataFrame({'open_sp':long(op,'home_spread').groupby(['game_date','home_team','away_team']).value.median(),
                  'open_tot':long(op,'total').groupby(['game_date','home_team','away_team']).value.median(),
                  'tab_sp':long(cl,'home_spread').groupby(['game_date','home_team','away_team']).value.median(),
                  'tab_tot':long(cl,'total').groupby(['game_date','home_team','away_team']).value.median(),
                  'pre_sp':cons(last_pre,'home_spread'),'pre_tot':cons(last_pre,'total')})
cl2s=cl2.set_index(['game_date','home_team','away_team'])[['hs','as_']]
out=out.join(cl2s)
out['margin']=out.hs-out.as_; out['tot']=out.hs+out.as_
out=out[(out.hs>0)&(out.as_>0)].dropna(subset=['open_sp','tab_sp','pre_sp'])
print('games with all of open/table-close/pre-game snapshot and a final score:',len(out))
for nm_,spc,totc in (('opening','open_sp','open_tot'),('last pre-game snapshot (-30min)','pre_sp','pre_tot'),('nba_odds "close"','tab_sp','tab_tot')):
    print(f'{nm_:34s} spread MAE {(out.margin+out[spc]).abs().mean():.3f}  total MAE {(out.tot-out[totc]).abs().mean():.3f}  spread RMSE {np.sqrt(((out.margin+out[spc])**2).mean()):.3f}')
print('|tab - pre| spread mean %.3f, total mean %.3f; share different: spread %.2f total %.2f'%((out.tab_sp-out.pre_sp).abs().mean(),(out.tab_tot-out.pre_tot).abs().mean(),(out.tab_sp!=out.pre_sp).mean(),(out.tab_tot!=out.pre_tot).mean()))
