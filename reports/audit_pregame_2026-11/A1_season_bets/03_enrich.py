"""Step 3: add own-view / line-move decomposition, closing-line value and the split variables to every bet. Writes bets.csv."""
from common import *

B = pd.read_pickle(OUT / '_bets_stage2.pkl')
G = pd.read_csv(OUT / 'games.csv', dtype={'game_id': str})
c = con()

# ------------------------------------------------------------------ split variables from the database
f = pd.read_sql("select game_id, prediction_date, market, feature, value from prediction_explanations where kind='feature' and "
                "feature in ('inj_lost_new_home','diff_inj_lost_new') and prediction_date>='2026-10-20'", c)
f['game_id'] = f.game_id.str.zfill(10)
f = f.sort_values('prediction_date').drop_duplicates(['game_id', 'feature'], keep='last')
W = f.pivot(index='game_id', columns='feature', values='value')
W['inj_new_home'] = W.inj_lost_new_home.clip(lower=0)
W['inj_new_away'] = (W.inj_lost_new_home - W.diff_inj_lost_new).clip(lower=0)
W = W[['inj_new_home', 'inj_new_away']].reset_index()

T = pd.read_sql("select teamId, gameId, gameDate, MATCHUP from team_game_full", c)
T['gameDate'] = pd.to_datetime(T.gameDate)
T = T.sort_values(['teamId', 'gameDate'])
T['prev'] = T.groupby('teamId').gameDate.shift(1)
T['rest'] = (T.gameDate - T.prev).dt.days
T['gameId'] = T.gameId.astype(str).str.zfill(10)
T['home'] = T.MATCHUP.str.contains(' vs. ')
rest = T[T.home].set_index('gameId').rest.rename('rest_home').to_frame().join(T[~T.home].set_index('gameId').rest.rename('rest_away')).reset_index().rename(columns={'gameId': 'game_id'})

B = B.merge(W, on='game_id', how='left').merge(rest, on='game_id', how='left')
B['new_inj'] = np.where(B.inj_new_home.notna(), ((B.inj_new_home > 0.5) | (B.inj_new_away > 0.5)).astype(float), np.nan)
B['b2b_home'] = (B.rest_home == 1).astype(float)
B['b2b_away'] = (B.rest_away == 1).astype(float)

# ------------------------------------------------------------------ own view vs line move
s = np.where(B.side.isin(['home', 'over']), 1, -1)
B['s'] = s
B['open_source'] = 'none'
open_sp = B.ex_open_home_spread.copy()
open_tt = B.ex_open_total.copy()
src_sp = np.where(open_sp.notna(), 'explanations', 'none')
src_tt = np.where(open_tt.notna(), 'explanations', 'none')
open_sp = open_sp.fillna(B.snap_open_spread)
open_tt = open_tt.fillna(B.snap_open_total)
src_sp = np.where((src_sp == 'none') & open_sp.notna(), 'snapshot proxy', src_sp)
src_tt = np.where((src_tt == 'none') & open_tt.notna(), 'snapshot proxy', src_tt)

B['open_line'] = np.nan
B['own_view'] = np.nan
B['move'] = np.nan
B['own_view_snapproxy'] = np.nan
B['move_snapproxy'] = np.nan
B['edge_points'] = np.nan
for i, b in B.iterrows():
    if b.market == 'Spread':
        o, cur, m = open_sp[i], b.odds_home_spread, b.spread_pred
        B.at[i, 'open_source'] = src_sp[i]
        B.at[i, 'edge_points'] = b.s * (m + cur)
        if pd.notna(o):
            B.at[i, 'open_line'] = o if b.s == 1 else -o                              # open line of the side bet
            B.at[i, 'own_view'] = b.s * (m + o)
            B.at[i, 'move'] = b.s * (cur - o)
        o2 = b.snap_open_spread
        if pd.notna(o2):
            B.at[i, 'own_view_snapproxy'], B.at[i, 'move_snapproxy'] = b.s * (m + o2), b.s * (cur - o2)
    elif b.market == 'Total':
        o, cur, m = open_tt[i], b.odds_total, b.total_pred
        B.at[i, 'open_source'] = src_tt[i]
        B.at[i, 'edge_points'] = b.s * (m - cur)
        if pd.notna(o):
            B.at[i, 'open_line'] = o
            B.at[i, 'own_view'] = b.s * (m - o)
            B.at[i, 'move'] = b.s * (o - cur)
        o2 = b.snap_open_total
        if pd.notna(o2):
            B.at[i, 'own_view_snapproxy'], B.at[i, 'move_snapproxy'] = b.s * (m - o2), b.s * (o2 - cur)
    else:                                                                              # moneyline, probability points
        pm = b.ml_prob if b.side == 'home' else 1 - b.ml_prob
        pc_home = float(implied(b.odds_home_ml))
        pc_away = float(implied(b.odds_away_ml))
        be_cur = pc_home if b.side == 'home' else pc_away
        B.at[i, 'edge_points'] = pm - be_cur
        if pd.notna(b.ex_p_open_home_ml):
            po_home = b.ex_p_open_home_ml
            B.at[i, 'open_source'] = 'explanations'
        elif pd.notna(b.snap_open_p_home):                                              # fair opening prob -> add the current overround split evenly
            fair_cur = float(devig2(b.odds_home_ml, b.odds_away_ml))
            po_home = b.snap_open_p_home + (pc_home - fair_cur)
            B.at[i, 'open_source'] = 'snapshot proxy'
        else:
            continue
        be_open = po_home if b.side == 'home' else pc_away + (pc_home - po_home)        # constant-vig assumption for the away price at the open
        B.at[i, 'open_line'] = be_open
        B.at[i, 'own_view'] = pm - be_open
        B.at[i, 'move'] = be_open - be_cur
B['own_share'] = B.own_view / B.edge_points
B['fade'] = np.where(B.move.isna(), np.nan, np.where(B.move > 0, 1.0, np.where(B.move < 0, -1.0, 0.0)))   # +1 fades the open->run move, -1 follows it
B['own_dom'] = np.where(B.own_view.isna(), np.nan, (B.own_view >= B.move).astype(float))                  # own view at least half of the edge

# ------------------------------------------------------------------ closing line value
valid_close = G.set_index('game_id').snap_close_mins_to_tip.le(90)
B['close_valid'] = B.game_id.map(valid_close).fillna(False)
B['close_mins_to_tip'] = B.snap_close_mins_to_tip
for col in ['close_line', 'clv', 'close_line_book', 'clv_book', 'clv_prob_price', 'clv_fair_to_fair']:
    B[col] = np.nan
for i, b in B.iterrows():
    if not b.close_valid:
        continue
    if b.market == 'Spread':
        cl = b.snap_close_spread
        B.at[i, 'close_line'] = cl if b.s == 1 else -cl
        B.at[i, 'clv'] = b.s * (b.odds_home_spread - cl)
        if pd.notna(b.bk_close_spread) and b.bet_book_match >= 3:
            B.at[i, 'close_line_book'] = b.bk_close_spread if b.s == 1 else -b.bk_close_spread
            B.at[i, 'clv_book'] = b.s * (b.odds_home_spread - b.bk_close_spread)
    elif b.market == 'Total':
        cl = b.snap_close_total
        B.at[i, 'close_line'] = cl
        B.at[i, 'clv'] = b.s * (cl - b.odds_total)
        if pd.notna(b.bk_close_total) and b.bet_book_match >= 3:
            B.at[i, 'close_line_book'] = b.bk_close_total
            B.at[i, 'clv_book'] = b.s * (b.bk_close_total - b.odds_total)
    else:
        pc = b.snap_close_p_home
        if pd.isna(pc):
            continue
        fair_bet = float(devig2(b.odds_home_ml, b.odds_away_ml))
        side_close = pc if b.side == 'home' else 1 - pc
        side_bet_fair = fair_bet if b.side == 'home' else 1 - fair_bet
        B.at[i, 'close_line'] = side_close
        B.at[i, 'clv'] = side_close - side_bet_fair                                       # fair-to-fair: pure movement
        B.at[i, 'clv_fair_to_fair'] = side_close - side_bet_fair
        B.at[i, 'clv_prob_price'] = side_close - float(implied(b.price))                  # vs the vigged price actually taken
        if pd.notna(b.bk_close_p_home) and b.bet_book_match >= 3:
            sc = b.bk_close_p_home if b.side == 'home' else 1 - b.bk_close_p_home
            B.at[i, 'clv_book'] = sc - side_bet_fair
            B.at[i, 'close_line_book'] = sc

# ------------------------------------------------------------------ favourite / underdog / group flags
def favdog(b):
    if b.market == 'Total':
        return None
    if b.market == 'ML':
        return 'fav' if b.price < 0 else 'dog'
    return 'fav' if b.line < 0 else ('dog' if b.line > 0 else 'pk')


B['fav_dog'] = B.apply(favdog, axis=1)
B['special_run'] = np.where(B.game_id.isin(G[(G.run_et.astype(str).str[11:16] > '17:00')].game_id), 'late_api_run_1859ET',
                   np.where(G.set_index('game_id').odds_source.reindex(B.game_id).values == 'theoddsapi', 'odds_api_open_equals_current', ''))
B['edge_bucket'] = pd.cut(B.edge, [0.05, 0.07, 0.10, 1.0], labels=['5-7%', '7-10%', '10%+'], right=False)
B['first_two_weeks'] = (pd.to_datetime(B.game_date) <= pd.Timestamp('2026-11-02')).astype(int)   # opening night Oct 20 + 14 days -> Nov 2
B['day_index'] = (pd.to_datetime(B.game_date) - pd.Timestamp('2026-10-20')).dt.days

keep = ['game_id', 'game_date', 'matchup', 'market', 'side', 'source', 'status', 'edge', 'edge_points', 'line', 'price', 'price_source',
        'open_source', 'open_line', 'own_view', 'move', 'own_share', 'fade', 'own_dom', 'own_view_snapproxy', 'move_snapproxy',
        'close_valid', 'close_mins_to_tip', 'close_line', 'clv', 'close_line_book', 'clv_book', 'clv_prob_price', 'clv_fair_to_fair',
        'result', 'profit', 'fav_dog', 'edge_bucket', 'new_inj', 'inj_new_home', 'inj_new_away', 'rest_home', 'rest_away', 'b2b_home', 'b2b_away',
        'first_two_weeks', 'day_index', 'special_run', 'bet_book', 'bet_book_match', 'ml_prob', 'spread_pred', 'total_pred', 'odds_home_ml',
        'odds_away_ml', 'odds_home_spread', 'odds_total', 'home_score', 'away_score', 'run_et', 'odds_source']
B[keep].sort_values(['game_date', 'game_id', 'market']).to_csv(OUT / 'bets.csv', index=False)
print(B[keep].groupby(['market', 'status']).size())
print(B.groupby('market')[['clv', 'own_view', 'move', 'edge_points']].agg(['count', 'mean']))
