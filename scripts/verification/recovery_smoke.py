"""Exercise real restored artifacts offline and save reproducible comparison fixtures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import traceback

ROOT=Path.cwd()
REPORT=Path(os.environ['MILESTONE_REPORT'])
OWNER='models'
OUT=REPORT/'fixtures'/OWNER
OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
RESULTS=[]
ONLY=''
RESTART_ONLY=ONLY=='restart'
if ONLY and (REPORT/(OWNER+'_smoke.json')).exists():
    RESULTS=json.loads((REPORT/(OWNER+'_smoke.json')).read_text())

def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2,default=str,sort_keys=True))

def check(name,fn):
    RESULTS[:]=[r for r in RESULTS if r['check']!=name]
    try:
        detail=fn()
        RESULTS.append({'check':name,'status':'pass','detail':detail})
    except Exception as e:
        RESULTS.append({'check':name,'status':'fail','error':str(e),'traceback':traceback.format_exc()})
    (REPORT/(OWNER+'_smoke.json')).write_text(json.dumps(RESULTS,indent=2,default=str))
    print(name,RESULTS[-1]['status'],flush=True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def model_loads():
    import joblib
    from team import model_registry as mr
    mr.alias_legacy_pickle_modules()
    result=[]
    for p in sorted((ROOT/'models/saved').glob('*.pkl')):
        try:
            obj=joblib.load(p)
            result.append({'file':p.name,'status':'loaded','type':type(obj).__module__+'.'+type(obj).__name__})
        except Exception as e:result.append({'file':p.name,'status':'failed','error':str(e)})
    save('model_loads',result)
    deployed=mr.load_bundle(model_dir=ROOT/'models/saved')
    assert all(r['status']=='loaded' for r in result), 'Some historical artifacts failed; see fixture'
    return {'artifacts':len(result),'deployed_suffix':deployed.suffix,'model_set_hash':deployed.model_set_hash}

def team_predictions():
    import pandas as pd
    from team.live_features import build_live_game_frame
    from team.model_registry import load_bundle
    from team.market_models import predict_markets
    with sqlite3.connect(ROOT/'data/nba_stats.db') as c:
        date=c.execute('SELECT MAX(substr(gameDate,1,10)) FROM team_game_full').fetchone()[0]
        day=pd.read_sql('SELECT gameId,gameDate,teamId,MATCHUP FROM team_game_full WHERE substr(gameDate,1,10)=?',c,params=[date])
    games=[]
    for gid,g in day.groupby('gameId',sort=True):
        if len(g)!=2:continue
        home=g[g.MATCHUP.str.contains('vs.',regex=False)]
        if len(home)!=1:continue
        tid=int(home.iloc[0].teamId)
        games.append({'gameId':str(gid).zfill(10),'gameDate':date,'teamId_home':tid,
          'teamId_away':int(g[g.teamId!=tid].iloc[0].teamId),'odds_home_ml':-110.,'odds_away_ml':-110.,
          'odds_home_spread':-2.5,'odds_away_spread':2.5,'odds_total':225.5})
        if len(games)==2:break
    assert games,'No paired games'
    inputs=pd.DataFrame(games)
    frame=build_live_game_frame(str(ROOT/'data/nba_stats.db'),inputs,injury_features=None)
    assert len(frame)==len(games)
    bundle=load_bundle(model_dir=ROOT/'models/saved')
    a=predict_markets(bundle,frame);b=predict_markets(bundle,frame)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    inputs.to_csv(OUT/'team_games.csv',index=False)
    frame.to_csv(OUT/'team_features.csv',index=False)
    a.to_csv(OUT/'team_predictions.csv',index=False)
    return {'games':len(games),'feature_columns':len(frame.columns),'repeat_exact':True,
      'fixture_policy':'fixed illustrative opening odds; injury unavailable encoded by existing builder; not new production predictions'}

def props_predictions():
    import joblib
    import numpy as np
    from props.models import data
    from props.live.predict import predict_rows
    from props.live import config
    cfg=config.load();config.verify(cfg,check_features_db=True)
    bundle=joblib.load(config.bundle_path(cfg))
    with sqlite3.connect(ROOT/'data/props_features_v3.db') as c:
        ids=[x[0] for x in c.execute('SELECT DISTINCT game_id FROM player_game_base ORDER BY game_id DESC LIMIT 1')]
    rows=data.load_frame(ROOT/'data/props_features_v3.db',ids).head(6)
    assert len(rows)==6,'Prediction fixture must exercise six real feature rows'
    a,p=predict_rows(bundle,rows,n_sims=20000,seed=21)
    b,q=predict_rows(bundle,rows,n_sims=20000,seed=21)
    np.testing.assert_array_equal(a,b)
    for k in p:
        np.testing.assert_array_equal(p[k],q[k])
        np.testing.assert_allclose(p[k].sum(axis=1),1.,atol=2e-6,rtol=0)
        assert np.isfinite(p[k]).all()
    rows.to_csv(OUT/'props_features.csv',index=False)
    np.savez(OUT/'props_probabilities.npz',minutes=a,**p)
    return {'model':cfg['model_version'],'rows':len(rows),'seed':21,'n_sims':20000,'repeat_exact':True}

def exports():
    import pandas as pd
    import export_dashboard_data as team_export
    from props.live import export,publish,config
    df=team_export.load_predictions(ROOT/'data/nba_stats.db')
    assert len(df)>0,'Prediction export must not be empty'
    expl=team_export.load_explanations_table(df,ROOT/'data/nba_stats.db')
    team_export.write_files(OUT/'team_export',df,expl)
    export.export_live(ROOT/'data/props_live.db',OUT/'props_export',config.load(),n_boot=100)
    bundle=OUT/'props_deployment';bundle.mkdir(exist_ok=True)
    written=publish.build_bundle(bundle,src_root=ROOT,live_dir=OUT/'props_export',backtest_dir=ROOT/'data/props_dashboard',
        now=pd.Timestamp('2026-11-07T05:00:00Z'))
    return {'team_rows':len(df),'props_rows':len(pd.read_csv(OUT/'props_export/bets.csv')),'deployment_files':len(written),
            'publish_called':False}

def dashboard_apps():
    # A subprocess imports only the generated bundle, with source imports removed.
    bundle=OUT/'props_deployment'
    script=OUT/'dashboard_app_check.py'
    script.write_text('''import json,os,sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
import importlib.abc
class NoSource(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'props','team','experiments','scripts'}:raise AssertionError(fullname)
sys.meta_path.insert(0,NoSource())
bundle=Path(sys.argv[1]);sys.path.insert(0,str(bundle))
at=AppTest.from_file(str(bundle/'dashboard_props/app.py'),default_timeout=120).run()
errors=[str(x.value) for x in at.exception]
print(json.dumps({'exceptions':errors,'titles':[x.value for x in at.title],'dataframes':len(at.dataframe),'metrics':len(at.metric)}))
assert not errors,errors
assert len(at.dataframe)>0,'Generated bundle did not render any data table'
''')
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).parent)
    r=subprocess.run([sys.executable,str(script),str(bundle)],cwd=bundle,env=env,capture_output=True,text=True)
    (OUT/'props_deployment_check.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,'Standalone props bundle failed; see deployment log'
    from streamlit.testing.v1 import AppTest
    os.environ['NBA_DASHBOARD_DB']=str(ROOT/'data/nba_stats.db')
    os.environ['NBA_DASHBOARD_DATA_BASE']=str(OUT/'team_export')+os.sep
    at=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=120).run()
    errors=[str(x.value) for x in at.exception]
    assert not errors,errors
    assert len(at.dataframe)+len(at.metric)>0,'Team app did not render data'
    return {'props_standalone':True,'team_restored_database':True,'external_network':'denied'}


for name,fn in [('model_bundle_loads',model_loads),('team_features_predictions',team_predictions),('props_features_probabilities',props_predictions),('exports',exports),('dashboard_apps',dashboard_apps)]:
    check(name,fn)
raise SystemExit(any(x['status']=='fail' for x in RESULTS))
