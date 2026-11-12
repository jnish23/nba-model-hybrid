"""Check 2: ordering / granularity facts and segment results (regular season, play-in, playoffs, NBA Cup, 2020 bubble) for E25_with_odds."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from common import *
from mymetrics import *
import warnings; warnings.filterwarnings('ignore')
p = load_e25(); t = load_train()
with conn() as c:
    tg = pd.read_sql('select gameId,teamId,gameDate,seasonYear,MATCHUP from team_game_full', c)
tg['gameDate'] = pd.to_datetime(tg.gameDate); tg['gameId'] = tg.gameId.astype(str).str.zfill(10)
print('gameDate string length distribution:', pd.read_sql('select length(gameDate) l, count(*) n from team_game_full group by l', conn()).to_dict('records'))
print('team-games sharing (teamId, gameDate):', int(tg.duplicated(['teamId', 'gameDate']).sum()), ' games per (teamId,date) max:', int(tg.groupby(['teamId', 'gameDate']).size().max()))
tg = tg.sort_values(['teamId', 'gameDate', 'gameId'])
tg['prev'] = tg.groupby('teamId').gameDate.shift(1)
tg['days'] = (tg.gameDate - tg.prev).dt.days
print('previous game on the SAME date:', int((tg.days == 0).sum()), '; days_rest==1 (back-to-backs):', int((tg.days == 1).sum()))
# same-day other-team results: features of a team never read other teams' rows, by construction verified in check1 (strictly-before recompute matched to 1e-13)
# game ordering across the league inside a date does not matter for any feature (all per-team or strictly-before-date)
# segments
p['type'] = np.select([p.gameId.str[:3] == '002', p.gameId.str[:3] == '004', p.gameId.str[:3] == '005'], ['regular', 'playoff', 'play-in'], 'other')
p['segment'] = p['type']
p.loc[(p.seasonYear == '2019-20') & (p.gameDate >= '2020-07-30'), 'segment'] = 'bubble(' + p['type'] + ')'
print(p.groupby('segment').size())
rows = []
for seg, g in p.groupby('segment'):
    r = {'segment': seg, 'n': len(g),
         'spread_mae_model': (g.target_point_diff - g.spread_pred).abs().mean(),
         'spread_mae_open': (g.target_point_diff + g.odds_home_spread).abs().mean(),
         'total_mae_model': (g.target_total_pts - g.total_pred).abs().mean(),
         'total_mae_open': (g.target_total_pts - g.odds_total).abs().mean(),
         'home_win_rate': g.target_home_win.mean(), 'mean_margin': g.target_point_diff.mean(),
         'mean_open_margin_implied': (-g.odds_home_spread).mean()}
    rows.append(r)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(3).to_string())
b, push = make_bets(p)
b = b.merge(p[['gameId', 'segment']], on='gameId')
out = []
for mk in ('ml', 'spread', 'total'):
    for seg in ('regular', 'playoff', 'play-in'):
        x = b[(b.market == mk) & (b.segment == seg)]
        if len(x) > 20:
            st = day_boot_stat(x)
            out.append({'market': mk, 'segment': seg, 'bets': st['bets'], 'hit': round(st['hit'], 1), 'roi': round(st['roi'], 2), 'lo': round(st['roi_ci'][0], 1), 'hi': round(st['roi_ci'][1], 1)})
print(pd.DataFrame(out).to_string())
print('share of E25 test rows that are playoffs/play-in:', round(float((p.type != 'regular').mean()), 3))
rs = p[p.type == 'regular']
print('Regular season only: ML logloss(clip 1e-6) %.4f, spread MAE model %.3f open %.3f, total MAE model %.3f open %.3f' % (
    logloss(rs.target_home_win, rs.ml_prob.clip(1e-6, 1 - 1e-6)), (rs.target_point_diff - rs.spread_pred).abs().mean(), (rs.target_point_diff + rs.odds_home_spread).abs().mean(),
    (rs.target_total_pts - rs.total_pred).abs().mean(), (rs.target_total_pts - rs.odds_total).abs().mean()))
# NBA Cup knockout games (gameId prefix 006) in DB?
print('team_game_full gameId prefixes:', tg.gameId.str[:3].value_counts().to_dict(), '(006 = NBA Cup knockout/final is absent from the database)')
