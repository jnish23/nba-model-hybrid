"""
NBA player props dashboard (paper-trading view). Separate from the team dashboard (dashboard/app.py).

    streamlit run dashboard_props/app.py

Two data sources, chosen in the sidebar: the LIVE paper record (scripts/export_props_dashboard_data.py --live -> data/props_dashboard_live/) and the BACKTEST replay of the frozen rule on
2023-24 to 2025-26 (scripts/export_props_dashboard_data.py -> data/props_dashboard/). Plans: reports/props/dashboard_plan.md, reports/props/live_pipeline_plan.md.
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                         # core/ sits at the repository root (here and in the published bundle); logic.py imports it, so this comes first
    sys.path.insert(0, str(ROOT))
try:
    from dashboard_props import logic as L
except ImportError:                                   # `streamlit run dashboard_props/app.py` puts this folder on sys.path
    import logic as L

try:
    from core import live_feed as F
except ImportError:                                   # an older bundle without core/: the board works, just without the live panel
    F = None
BACKTEST_DIR = Path(os.environ.get('PROPS_DASH_DATA', ROOT / 'data' / 'props_dashboard'))
LIVE_DIR = Path(os.environ.get('PROPS_DASH_LIVE', ROOT / 'data' / 'props_dashboard_live'))
STAT_CHOICES = ['rebounds', 'assists', 'threes', 'ra']
T_LABEL = {240: 'T-4h', 60: 'T-1h', 5: 'T-5m'}
LIVE_REFRESH_S = 30
if F is not None:
    F.log.setLevel(logging.INFO)                      # one 'live_feed fetch' line per request in the server log
    if not F.log.handlers:
        F.log.addHandler(logging.StreamHandler())

st.set_page_config(page_title='NBA Props', page_icon='🏀', layout='wide')


@st.cache_data(show_spinner=False, ttl=300)
def load(data_dir, stamp=None):
    """`stamp` (the export's modification time) is part of the cache key, so a hosted app picks up new data as soon as the files change; the 5-minute expiry is a backstop."""
    d = Path(data_dir)
    bets = pd.read_csv(d / 'bets.csv', dtype={'game_id': str})
    bets['played'] = bets['played'].fillna(0).astype(bool)
    bets['tip_et'] = pd.to_datetime(bets['tip_ref_utc'], utc=True).dt.tz_convert('America/New_York')
    out = {'bets': bets, 'summary': pd.read_csv(d / 'season_summary.csv'), 'reliability': pd.read_csv(d / 'reliability.csv'), 'bias': pd.read_csv(d / 'bias_monthly.csv'),
           'meta': json.loads((d / 'meta.json').read_text(encoding='utf-8')), 'pmf': json.loads((d / 'pmf.json').read_text(encoding='utf-8')),
           'books': json.loads((d / 'books.json').read_text(encoding='utf-8'))}
    for name in ('pipeline_health', 'slot_log', 'checks', 'current', 'results_now'):
        out[name] = pd.read_csv(d / f'{name}.csv') if (d / f'{name}.csv').exists() else pd.DataFrame()
    return out


def pct(x, nd=1):
    return '' if x is None or not np.isfinite(x) else f'{100 * x:+.{nd}f}%'


def bet_label(r):
    return f"{r['player_name']} · {r['side'].upper()} {L.STAT_LABEL.get(r['stat'], r['stat'])} {r['line']:g}"


def board_table(d, mode, treatment, reveal, cur=None, results_now=None, now=None, live=False):
    ev_c, dec_c = L.ev_col(mode), L.dec_col(mode)
    side_model = L.side_prob(d, 'p_over_star')
    side_mkt = L.side_prob(d, 'p_over_market')
    t = pd.DataFrame({
        'Tip (ET)': d['tip_et'].dt.strftime('%H:%M').to_numpy(),
        'Game': d['matchup'].to_numpy(),
        'Player': d['player_name'].to_numpy(),
        'Bet': (d['side'].str.upper() + ' ' + d['stat'].map(lambda s: L.STAT_LABEL.get(s, s)) + ' ' + d['line'].map(lambda v: f'{v:g}')).to_numpy(),
        'Price': L.american(d[dec_c]),
        'Best': [f'{a} {b}' for a, b in zip(L.american(d['dec_best']), d['book'])],
        'Shrunk P': side_model, 'Market': side_mkt, 'EV': d[ev_c].to_numpy(),
        'Flagged': d['minutes_before'].map(T_LABEL).to_numpy(), 'Books': d['n_at_line'].to_numpy()})
    if live:
        t = L.with_odds_now(t, d, cur, mode, now if now is not None else pd.Timestamp.now(tz='UTC'))
    if reveal:
        r = L.results_for_display(d, results_now, mode, treatment)
        t['Actual'] = r['Actual'].to_numpy()
        t['Result'] = r['Result'].to_numpy()
        t['Units'] = r['Units'].to_numpy()
    return t


def colour_results(t):
    """Row shading by result: green win, red loss, grey push/void; a provisional result (finished game, not yet officially settled) is shaded the same."""
    if 'Result' not in t.columns:
        return t
    shade = {'win': 'background-color: rgba(46, 160, 67, 0.30)', 'loss': 'background-color: rgba(218, 54, 51, 0.30)', 'other': 'background-color: rgba(128, 128, 128, 0.18)', '': ''}
    kinds = [L.result_kind(r) for r in t['Result']]
    return t.style.apply(lambda row: [shade[kinds[row.name]]] * len(row), axis=1)


@st.cache_data(ttl=LIVE_REFRESH_S, show_spinner=False)
def live_scoreboard(game_date):
    """ESPN scoreboard of one ET date, shared by every viewer: at most one request per date per 30 s."""
    return F.scoreboard(game_date)


@st.cache_data(ttl=LIVE_REFRESH_S, show_spinner=False)
def live_box(nba_game_id, espn_event_id, roster):
    """Box lines of one live game for the bet players (`roster`: ((player_id, player_name), ...)), shared by every viewer: at most one request per game per 30 s."""
    board = pd.DataFrame([{**dict.fromkeys(F.SCOREBOARD_COLS), 'nba_game_id': nba_game_id, 'espn_event_id': espn_event_id, 'status': 'live'}])
    return F.box(nba_game_id, board, pd.DataFrame(list(roster), columns=['player_id', 'player_name'])).assign(nba_game_id=nba_game_id)


def colour_live(t):
    shade = {'hit': 'background-color: rgba(46, 160, 67, 0.30)', 'busted': 'background-color: rgba(218, 54, 51, 0.30)'}
    kinds = t['kind'].tolist()
    return t.drop(columns='kind').style.apply(lambda row: [shade.get(kinds[row.name], '')] * len(row), axis=1)


@st.fragment(run_every=LIVE_REFRESH_S)
def live_panel(bets):
    """Open bets in games being played right now, against the player's current stat. Shown only while such a game is live; display only."""
    cand = L.live_candidates(bets, pd.Timestamp.now(tz='UTC'))
    if F is None or cand is None or cand.empty:
        return
    board = pd.concat([live_scoreboard(d) for d in sorted(cand['game_date'].unique())], ignore_index=True)
    board = F.attach_nba_ids(board, cand[['game_id', 'game_date', 'home_team', 'away_team']].drop_duplicates())
    if board.empty:
        st.caption('Live stats unavailable right now.')
        return
    games = board[(board['status'] == 'live') & board['nba_game_id'].isin(cand['game_id'])]
    if games.empty:
        return
    parts = []
    for _, g in games.iterrows():
        roster = cand.loc[cand['game_id'] == g['nba_game_id'], ['player_id', 'player_name']].drop_duplicates('player_id')
        parts.append(live_box(g['nba_game_id'], g['espn_event_id'], tuple(map(tuple, roster.itertuples(index=False)))))
    lines = pd.concat(parts, ignore_index=True)
    view = L.live_view(cand, board, lines)
    st.subheader('Live')
    if lines.empty:
        st.caption('Live stats unavailable right now.')
        return
    st.caption(f"Live, not final: updates every {LIVE_REFRESH_S} s (last {pd.Timestamp.now(tz='America/New_York'):%H:%M:%S} ET). Box scores from ESPN; minutes are whole minutes. "
               'Green = over already hit, red = under already busted. "On pace" scales the current stat to the expected minutes. The recorded bets and their settlement do not change.')
    cfg = {'Now': st.column_config.NumberColumn('Now', format='%d'), 'Min': st.column_config.NumberColumn('Min', format='%d'),
           'On pace': st.column_config.NumberColumn('On pace', format='%.1f')}
    st.dataframe(colour_live(view), width='stretch', hide_index=True, column_config=cfg)


def record_text(r):
    return f"{r['won']}-{r['lost']}" + (f"-{r['other']}" if r['other'] else '')


def render_day_record(rec, day):
    """Record of the bets in the table below (after the filters): overall with units, and the overs and unders; counts what the table colours, provisional grades included."""
    today = pd.Timestamp.now(tz='America/New_York').date()
    st.markdown("**Today's record**" if str(day) == str(today) else f'**Record for {day}**')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Bets (W-L-push/void)', record_text(rec['all']), f"{rec['all']['open']} open", delta_color='off', help='Graded bets only; "open" bets have no result yet. Provisional grades from a finished box score count.')
    c2.metric('Units', f"{rec['units']:+.2f}", help='Profit of the graded bets at the chosen price, 1 unit per bet.')
    c3.metric('Overs', record_text(rec['over']), f"{rec['over']['open']} open", delta_color='off')
    c4.metric('Unders', record_text(rec['under']), f"{rec['under']['open']} open", delta_color='off')


def page_board(D, mode, treatment):
    bets = D['bets']
    live = D['meta']['mode'] == 'live'
    if live:
        live_panel(bets)
    c1, c2, c3 = st.columns([1, 1, 2])
    season = c1.selectbox('Season', sorted(bets['season'].unique(), reverse=True))
    in_season = bets[bets['season'] == season]
    dates = sorted(in_season['game_date'].unique(), reverse=True)
    games_per_day = in_season.groupby('game_date')['game_id'].nunique()
    default_idx = 0 if live else next((i for i, dd in enumerate(dates) if games_per_day[dd] >= 6), 0)       # live: the latest slate; replay: the latest typical slate, not a one-game Finals night
    day = c2.selectbox('Date' if live else 'Replay date', dates, index=default_idx)
    reveal = c3.checkbox('Show results (settled bets)' if live else 'Reveal results (replay)', value=live, help='Off: see the slate as it looked before the games. On: add actual results where settled.')
    f1, f2, f3, f4 = st.columns(4)
    stats = f1.multiselect('Stat', STAT_CHOICES, default=STAT_CHOICES, format_func=lambda s: L.STAT_LABEL[s])
    sides = f2.multiselect('Side', ['over', 'under'], default=['over', 'under'])
    min_ev = f3.slider('Min EV at the chosen price (%)', -20.0, 20.0, -20.0, 0.5, help='Raise it to see only the highest-value plays at the price you chose. At the far left every bet of the slate is shown.') / 100.0
    day_df = bets[(bets['season'] == season) & (bets['game_date'] == day)]
    sort_options = list(L.SORTS) if live else ['ev', 'tip']
    sort = f4.selectbox('Sort by', sort_options, format_func=lambda k: L.SORTS[k])
    now = pd.Timestamp.now(tz='UTC')
    upcoming_only = st.checkbox('Only games that have not started', value=False, help='Hides games that have already tipped, so every row shown can still be bet.') if live else False
    cur = D.get('current')
    results_now = D.get('results_now')
    n_all = len(day_df)
    top_n = max(n_all, 1)
    if n_all > 5:
        top_n = st.slider('Rows shown', 5, n_all, n_all, 1, help='Every bet of the slate is shown by default.')
    shown, hidden = L.board(day_df, mode, min_ev, stats, sides, top_n, sort=sort, upcoming_only=upcoming_only, now=now, cur=cur)
    below = int((day_df[L.ev_col(mode)] < min_ev).sum())
    st.caption(f'{n_all} rule-qualified bets on {day} across {day_df["game_id"].nunique()} games; {len(shown)} shown'
               + (f', {hidden} more pass the filters (raise "Rows shown")' if hidden else '') + (f'; {below} hidden by the minimum-EV filter' if below else '')
               + f'. EV and price use: {L.PRICE_MODES[mode]}. Sorted by: {L.SORTS[sort].lower()}.')
    if shown.empty:
        st.info('No bets pass the filters on this slate.')
        return
    if live:
        n_up = int((~L.started(shown, now)).sum())
        st.caption(f'{n_up} of the {len(shown)} rows are for games that have not started (these can still be bet; "Price now" is their current price). The others have tipped: '
                   f'"Price now" reads "game started" and the result fills in once the box score is final.')
        if cur is not None and len(cur):
            asof = pd.to_datetime(cur['as_of_utc'], utc=True).max().tz_convert('America/New_York')
            st.caption(f"Odds now: latest snapshot {asof:%a %H:%M} ET. 'Price now', 'Line now' and 'EV now' re-value the recorded bet at today's prices and line; the recorded bet itself never changes. "
                       f"The odds collector snapshots about every 30 minutes, so a later refresh can show newer prices.")
        if reveal:
            st.caption('Green = win, red = loss, grey = push or void. "(prov.)" is a provisional grade from the final box score; the official settlement follows the next morning sync.')
        settled = bets[bets[f'status_{treatment}'].notna()]
        if len(settled):
            sel_ = settled[settled[L.ev_col(mode)] >= min_ev]
            s_ = L.roi_summary(sel_, 'median', treatment, n_boot=300)
            if s_['bets']:
                sb_ = L.roi_summary(sel_, 'best', treatment, n_boot=300)
                st.caption(f"Live record so far for settled bets with EV of at least {100 * min_ev:.1f}% at the chosen price: {s_['bets']} bets, ROI at the median price {pct(s_['roi'])} "
                           f"(95% date-clustered interval {ci_text(s_)}); at the best quoted price (not an assumed fill) {pct(sb_['roi'])} "
                           f"({ci_text(sb_)}). {L.noise_note(s_['bets'], (s_['lo'], s_['hi']), s_['clusters'])}")
                segs_ = L.clean_segments(D['meta'].get('segments'))
                cur_ = L.current_segment(segs_)
                if cur_ is not None:
                    sc_ = L.roi_summary(sel_[(L.assign_segments(sel_, segs_) == cur_['key']).to_numpy()], 'median', treatment, n_boot=300)
                    st.caption(f"Current segment ({cur_['label']}, {L.segment_period_text(cur_)}): " + (f"{sc_['bets']} settled bets, ROI at the median price {pct(sc_['roi'])} "
                               f"(95% date-clustered interval {ci_text(sc_)}). Only the segment is forward-test evidence; the record above mixes configurations." if sc_['bets'] else 'no settled bets yet.'))
    if reveal:
        render_day_record(L.day_record(shown, L.results_for_display(shown, results_now, mode, treatment)), day)
    t = board_table(shown, mode, treatment, reveal, cur, results_now, now, live)
    cfg = {'Shrunk P': st.column_config.ProgressColumn('Shrunk P', min_value=0.0, max_value=1.0, format='%.2f', help='Probability of the bet side after shrinking the raw model toward the market (p*, the probability the EV is computed from). The raw model probability is in the bet detail.'),
           'Market': st.column_config.NumberColumn('Market P', format='%.2f'),
           'EV': st.column_config.NumberColumn('EV', format='%.3f'), 'EV now': st.column_config.NumberColumn('EV now', format='%.3f'),
           'Units': st.column_config.NumberColumn('Units', format='%+.2f')}
    st.dataframe(colour_results(t), width='stretch', hide_index=True, column_config=cfg)
    with st.expander('All shown bets grouped by game'):
        for game, g in shown.groupby('matchup', sort=False):
            st.markdown(f'**{game}**')
            tg = board_table(g, mode, treatment, reveal, cur, results_now, now, live).drop(columns=['Game'])
            st.dataframe(colour_results(tg), width='stretch', hide_index=True, column_config=cfg)
    st.download_button('Download this slate (CSV)', t.to_csv(index=False).encode('utf-8'), file_name=f'props_board_{day}.csv')
    st.subheader('Inspect a bet')
    labels = {r['bet_id']: bet_label(r) for _, r in shown.iterrows()}
    bid = st.selectbox('Bet', list(labels), format_func=lambda b: labels[b])
    detail(D, shown[shown['bet_id'] == bid].iloc[0], mode, treatment, reveal)


def detail(D, r, mode, treatment, reveal):
    pmf = np.array(D['pmf'].get(r['bet_id'], []))
    book_px = D['books'].get(r['bet_id'], {})
    has_raw = 'p_over_model' in r.index and pd.notna(r['p_over_model'])
    m1, m1b, m2, m3, m4 = st.columns(5)
    m1.metric('Shrunk P(side)', f"{L.side_prob(pd.DataFrame([r]), 'p_over_star')[0]:.3f}", help='The raw model probability shrunk toward the market (p*). The EV is computed from this.')
    if has_raw:
        m1b.metric('Raw model P(side) (before shrinking)', f"{L.side_prob(pd.DataFrame([r]), 'p_over_model')[0]:.3f}", help='The model output before the shrink toward the market. Shown for reference only: the rule does not bet on it.')
    else:
        m1b.metric('Raw model P(side)', 'not in this export')
    m2.metric('Market P(side)', f"{L.side_prob(pd.DataFrame([r]), 'p_over_market')[0]:.3f}")
    m3.metric(f'EV ({L.PRICE_MODES[mode].split(":")[0]})', f"{r[L.ev_col(mode)]:+.3f}")
    m4.metric('Flagged', f"{T_LABEL.get(r['minutes_before'], r['minutes_before'])} before tip")
    left, right = st.columns(2)
    if len(pmf):
        ys = np.arange(len(pmf))
        colors = ['#d62728' if (y > r['line'] if r['side'] == 'over' else y < r['line']) else '#7f7f7f' for y in ys]
        fig = go.Figure(go.Bar(x=ys, y=pmf, marker_color=colors))
        fig.add_vline(x=r['line'], line_dash='dash', annotation_text=f"line {r['line']:g}")
        if reveal and np.isfinite(r['y']):
            fig.add_vline(x=r['y'], line_color='green', annotation_text=f"actual {r['y']:g}", annotation_position='bottom right')
        fig.update_layout(title=f"Model distribution ({L.STAT_LABEL.get(r['stat'], r['stat'])}; coloured = the bet wins)", height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title='value', yaxis_title='probability')
        left.plotly_chart(fig, width='stretch')
    if book_px:
        bk = pd.Series(book_px).sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=bk.index, y=bk.values, marker_color=['#2ca02c' if v == bk.max() else '#aec7e8' for v in bk.values], text=L.american(bk.values), textposition='outside'))
        fig.add_hline(y=r['dec_median'], line_dash='dot', annotation_text='median')
        fig.update_layout(title='Price of this bet by book at the flag time (decimal)', height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[min(1.5, bk.min() - 0.1), bk.max() + 0.15])
        right.plotly_chart(fig, width='stretch')
    hist = pd.DataFrame({'Decision time': ['T-4h', 'T-1h', 'T-5m'], 'Consensus line': [r['line_240'], r['line_60'], r['line_5']], 'Market P(over)': [r['mkt_240'], r['mkt_60'], r['mkt_5']]})
    st.dataframe(hist, hide_index=True, width='content')
    if reveal:
        w = L.with_profit(pd.DataFrame([r]), mode, treatment).iloc[0]
        status = r[f'status_{treatment}']
        if not isinstance(status, str):
            st.info('Pending: the game has not been settled yet.')
        elif w['stake']:
            st.success(f"Result: {status.upper()} · actual {r['y']:g} · {w['profit']:+.2f} units at {L.PRICE_MODES[mode].lower()}")
        else:
            st.info(f"Result: {status} · actual {r['y']} (no stake)")


def ci_text(r):
    """'-2.3% to +9.0%' for a summary dict with lo and hi; a plain note when the interval does not exist yet (one game date)."""
    return f"{pct(r['lo'])} to {pct(r['hi'])}" if np.isfinite(r.get('lo', np.nan)) and np.isfinite(r.get('hi', np.nan)) else 'not available yet (it needs at least two game dates)'


def pct_ci(r, lo, hi):
    """'+3.3% [-2.3%, +9.0%]' (blank when the ROI is undefined)."""
    if r is None or not np.isfinite(r):
        return ''
    return f'{pct(r)} [{pct(lo)}, {pct(hi)}]' if np.isfinite(lo) and np.isfinite(hi) else pct(r)


DIAG_LABELS = {'segment': 'Segment', 'side': 'Side', 'slot': 'Slot', 'stat': 'Stat', 'model_version': 'Model version', 'ev_bucket': 'EV claimed (best price)'}


def diag_display(t, keys):
    """Human-readable view of a `L.diagnostic_table` frame: intervals folded into one text cell; returns (display frame, too_small mask)."""
    out = pd.DataFrame({DIAG_LABELS.get(k, k): t[k].to_numpy() for k in keys})
    out['Stakes'] = t['stakes'].to_numpy()
    out['Win %'] = (100 * t['win_rate']).round(1).to_numpy()
    out['Mkt P(side)'] = t['p_market'].round(3).to_numpy()
    out['Shrunk P(side)'] = t['p_shrunk'].round(3).to_numpy()
    out['ROI median price [95% CI]'] = [pct_ci(*x) for x in zip(t['roi_median'], t['roi_median_lo'], t['roi_median_hi'])]
    out['ROI best quoted price [95% CI]'] = [pct_ci(*x) for x in zip(t['roi_best'], t['roi_best_lo'], t['roi_best_hi'])]
    out['CLV pts (n)'] = [f'{c:+.2f} ({n})' if np.isfinite(c) else f'n/a ({n})' for c, n in zip(t['clv_pts'], t['clv_n'])]
    small = t['too_small'].to_numpy(dtype=bool)
    out['Readable?'] = np.where(small, f'too small (<{L.MIN_STAKES} stakes)', '')
    return out, small


def grey_small(t, small):
    """Styler greying the rows flagged too small to read (they stay in the table)."""
    return t.style.apply(lambda row: ['color: #8a8a8a; font-style: italic' if small[row.name] else ''] * len(row), axis=1)


def show_diag(t, keys):
    disp, small = diag_display(t, keys)
    if disp.empty:
        st.caption('No settled bets in this selection.')
        return
    st.dataframe(grey_small(disp, small), hide_index=True, width='stretch')


def render_diagnostics(d, treatment):
    """Tables for spotting a broken model early: by side and slot, by stat, claimed versus realised EV, line moves. Computed by core/props_board_logic from bets.csv."""
    st.subheader('Diagnostics: is the model breaking?')
    st.caption(f'Settled stakes only (pushes and voids are not stakes). "Mkt P" and "Shrunk P" are the mean probabilities of the bet side: compare them with the win rate. ROI intervals are 95%, resampling whole '
               f'dates. CLV is in points of consensus probability toward the bet side by T-5 and exists only where the consensus line did not move. Grey italic rows have fewer than {L.MIN_STAKES} stakes: '
               f'too small to read, shown anyway.')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**By side**')
        show_diag(L.diagnostic_table(d, 'side', treatment), ['side'])
    with c2:
        st.markdown('**By slot**')
        show_diag(L.diagnostic_table(d, 'slot', treatment), ['slot'])
    st.markdown('**Side x slot**')
    show_diag(L.diagnostic_table(d, ['side', 'slot'], treatment), ['side', 'slot'])
    st.markdown('**By stat**')
    show_diag(L.diagnostic_table(d, 'stat', treatment), ['stat'])
    st.markdown('**Claimed versus realised, by EV claimed at the best price when flagged**')
    ev = L.ev_claim_table(d, treatment)
    if ev.empty:
        st.caption('No settled bets in this selection.')
    else:
        view = pd.DataFrame({'EV claimed (best price)': ev['ev_bucket'], 'Stakes': ev['stakes'], 'Mean claimed EV': [pct(x) for x in ev['claimed_ev']],
                             'Realised ROI, best quoted price [95% CI]': [pct_ci(*x) for x in zip(ev['roi_best'], ev['roi_best_lo'], ev['roi_best_hi'])],
                             'Realised ROI, median price [95% CI]': [pct_ci(*x) for x in zip(ev['roi_median'], ev['roi_median_lo'], ev['roi_median_hi'])],
                             'Readable?': np.where(ev['too_small'], f'too small (<{L.MIN_STAKES} stakes)', '')})
        st.dataframe(grey_small(view, ev['too_small'].to_numpy(dtype=bool)), hide_index=True, width='stretch')
        st.caption('If the edge were real and the claims right, realised ROI at the best price would rise with the claimed EV. Flat or falling realised ROI with a rising claim means the EV figure overstates the edge.')
    st.markdown('**Line movement from the flag to the T-5 close**')
    lm = L.line_move_summary(d)
    if lm.empty:
        st.caption('This export has no closing lines.')
    else:
        view = pd.DataFrame({'Group': lm['group'], 'Bets with a close': lm['bets_with_close'], 'Line moved': lm['moved'], 'Share moved': [f'{100 * x:.1f}%' if np.isfinite(x) else '' for x in lm['share_moved']],
                             'Moved in the bettor favour': [f'{100 * x:.1f}%' if np.isfinite(x) else '' for x in lm['share_in_favour']],
                             'Mean line move, points (+ = favour)': lm['mean_line_move_favour'].round(2)})
        st.dataframe(view, hide_index=True, width='stretch')
        st.caption('Bets flagged at T-5 have no later close and are not counted. A line that moved against the bet before the close is a sign the market disagrees; CLV (above) skips moved lines.')


def roi_text(s):
    """'+3.3% [-2.3%, +9.0%] on 412 settled bets, 9 dates' (or a plain 'no settled bets yet') from a `roi_summary` dict."""
    if not s['bets']:
        return 'no settled bets yet'
    return f"{pct_ci(s['roi'], s['lo'], s['hi'])} on {s['bets']:,} settled bets, {s['clusters']} dates"


def render_segments(D, d_all, treatment):
    """Record by forward-test segment (definitions from meta.json `segments`) plus an overall row. Nothing is drawn for an older export without segments."""
    segs = L.clean_segments(D['meta'].get('segments'))
    if not segs:
        return
    ss = L.segment_summary(d_all, segs, treatment, n_boot=1000)          # the same resamples as the headline, so the intervals agree
    disp, small = diag_display(ss.assign(segment=ss['label']), ['segment'])
    disp.insert(1, 'First date', ss['first_game_date'].to_numpy())
    disp.insert(2, 'Last date', ss['last_game_date'].to_numpy())
    disp.insert(3, 'Bets recorded', ss['bets'].to_numpy())
    disp.insert(4, 'Game dates', ss['game_dates'].to_numpy())
    disp['ROI flat -110 [95% CI]'] = [pct_ci(*x) for x in zip(ss['roi_flat'], ss['roi_flat_lo'], ss['roi_flat_hi'])]
    disp['Model version'] = [mv or 'mixed' for mv in ss['model_version']]
    disp['Mismatched'] = ss['mismatched'].to_numpy()
    disp = disp[[c for c in disp.columns if c != 'Readable?'] + ['Readable?']]
    st.markdown('**By forward-test segment**')
    st.caption('A segment is a stretch of game dates played under one fixed model, rule and roster rule (see the Rules & about page); only the segment rows are forward-test evidence, '
               'while the overall row mixes configurations and the earlier record.')
    st.dataframe(grey_small(disp, small), hide_index=True, width='stretch')
    n_bad = int(ss.loc[ss['segment'] == L.OVERALL_KEY, 'mismatched'].sum())
    if n_bad:
        st.warning(f"{n_bad} bets carry a model version, rule or roster rule that differs from their segment's definition. They are counted in their segment, not dropped: check the configuration switch.", icon='⚠️')


def render_headline(D, d, treatment):
    """First thing on the Tracker: the pooled ROI next to the current (open-ended) segment's ROI, both at the median price, whatever the filters below are set to."""
    segs = L.clean_segments(D['meta'].get('segments'))
    cur = L.current_segment(segs)
    if cur is None or D['meta'].get('mode') != 'live':
        return
    settled = d[d[f'status_{treatment}'].notna()]
    overall = L.roi_summary(settled, 'median', treatment)
    seg = L.roi_summary(settled[(L.assign_segments(settled, segs) == cur['key']).to_numpy()], 'median', treatment)
    c1, c2 = st.columns(2)
    c1.metric('Overall ROI, median price (all segments)', pct(overall['roi']) or 'n/a', help=f"Every settled bet in the selected seasons: {roi_text(overall)}. Mixes configurations and the earlier record.")
    c2.metric(f"{cur['label']} ROI, median price (current segment)", pct(seg['roi']) or 'no settled bets yet', help=f"Settled bets {L.segment_period_text(cur)}: {roi_text(seg)}.")
    st.caption(f"Overall: {roi_text(overall)}. {cur['label']} ({L.segment_period_text(cur)}, model {cur.get('model_version') or 'n/a'}): {roi_text(seg)}. "
               f"Only the segment is forward-test evidence; the overall figure mixes configurations and the earlier record. The numbers below follow the filters you pick.")


def render_versions(D, d_all, treatment):
    """Per-model-version summary: provenance from meta.json (or 'unknown' for an older export) next to the version's own settled record."""
    vt = L.version_meta_table(D['meta'], D['bets'])
    vs = L.version_summary(d_all, treatment)
    prov = dict(zip(vt['version'], vt['provenance']))
    disp, small = diag_display(vs, ['model_version'])
    disp.insert(1, 'First date', vs['first_game_date'].to_numpy())
    disp.insert(2, 'Last date', vs['last_game_date'].to_numpy())
    disp.insert(3, 'Bets recorded', vs['bets'].to_numpy())
    disp['Provenance'] = ['all versions combined' if v == 'all' else prov.get(v, L.UNKNOWN_PROVENANCE) for v in vs['model_version']]
    st.markdown('**By model version**')
    st.dataframe(grey_small(disp, small), hide_index=True, width='stretch')


def page_tracker(D, mode, treatment):
    bets = D['bets']
    seasons = st.multiselect('Seasons', sorted(bets['season'].unique()), default=sorted(bets['season'].unique()))
    d = bets[bets['season'].isin(seasons)]
    if d.empty:
        st.info('Select at least one season.')
        return
    vt = L.version_meta_table(D['meta'], bets)
    versions = list(vt['version'])
    d_all_versions = d
    segs = L.clean_segments(D['meta'].get('segments'))
    render_headline(D, d, treatment)
    has_versions = len(versions) > 1 or (len(versions) == 1 and versions[0] != 'unknown')
    seg_col, ver_col = st.columns(2) if segs and has_versions else (st.container(), st.container())
    if segs:
        seg_of = L.assign_segments(d, segs)
        seg_names = {s_['key']: s_['label'] for s_ in segs} | {L.UNASSIGNED_KEY: L.UNASSIGNED_LABEL}
        present = [k for k in [s_['key'] for s_ in segs] + [L.UNASSIGNED_KEY] if (seg_of == k).any()]
        seg_pick = seg_col.selectbox('Segment', ['All'] + [seg_names[k] for k in present], index=0,
                                     help='Filters every chart and table below, like the model version. Default: all segments. The segment table is always shown in full.')
        if seg_pick != 'All':
            d = d[(seg_of == next(k for k in present if seg_names[k] == seg_pick)).to_numpy()]
    if has_versions:
        pick = ver_col.selectbox('Model version', ['All'] + versions, index=0, help='The record mixes model versions (see "By model version" below). Default: all of them.')
        if pick != 'All':
            d = d[L.version_series(d) == pick]
        shown_versions = versions if pick == 'All' else [pick]
        restored = [v for v in L.restored_versions(vt) if v in shown_versions and (L.version_series(d) == v).any()]
        if restored:
            n_r = int(L.version_series(d).isin(restored).sum())
            st.info(f'{n_r:,} of the {len(d):,} bets in this selection ({", ".join(restored)}) were {L.RESTORED_PROVENANCE}: snapshot time, quote age and code commit were not recorded for them, '
                    f'and their results came from the export. Pick another model version above to leave them out.', icon='ℹ️')
    if d.empty:
        st.info('No bets for this selection.')
        return
    if D['meta']['mode'] == 'live':
        pending = int(d[f'status_{treatment}'].isna().sum())
        d = d[d[f'status_{treatment}'].notna()]
        st.caption(f'{pending} bets are placed and not yet settled; only settled bets count below.')
        if d.empty:
            st.info('No settled bets yet.')
            return
    s_med = L.roi_summary(d, 'median', treatment)
    s_best = L.roi_summary(d, 'best', treatment)
    s_flat = L.roi_summary(d, 'flat', treatment, n_boot=300)
    units = float(L.with_profit(d, mode, treatment)['profit'].sum())
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric('Settled bets', f"{s_med['bets']:,}", help=f"{s_med['player_games']:,} distinct player-games on {s_med['clusters']} dates")
    t2.metric('ROI, median price (primary)', pct(s_med['roi']), help=f"95% date-clustered interval {ci_text(s_med)}. The median bettable price is what an ordinary bettor gets.")
    t3.metric('ROI, best quoted price (not an assumed fill)', pct(s_best['roi']), help=f"95% date-clustered interval {ci_text(s_best)}. The best quote of the bettable books: a target "
              f"for limit orders, not a price you can assume you get.")
    t4.metric('ROI, flat -110', pct(s_flat['roi']), help=f"95% date-clustered interval {ci_text(s_flat)}")
    t5.metric('Win rate', f"{100 * s_med['win_rate']:.1f}%")
    st.caption(f"95% date-clustered intervals: median price {ci_text(s_med)}; best quoted price (not an assumed fill) {ci_text(s_best)}. "
               f"Units at the chosen price ({L.PRICE_MODES[mode].split(':')[0]}): {units:+.1f}. " + L.noise_note(s_med['bets'], (s_med['lo'], s_med['hi']), s_med['clusters']))
    fig = go.Figure()
    deff = 1.0
    for m in L.PRICE_MODES:
        c = L.cumulative(d, m, treatment)
        fig.add_trace(go.Scatter(x=c['game_date'], y=c['cum_units'], name=L.PRICE_MODES[m], line=dict(width=3 if m == mode else 1.2), opacity=1 if m == mode else 0.6))
        if m == mode:
            deff = c.attrs.get('design_effect', 1.0)
            fig.add_trace(go.Scatter(x=list(c['game_date']) + list(c['game_date'])[::-1], y=list(c['hi']) + list(c['lo'])[::-1], fill='toself', fillcolor='rgba(31,119,180,0.12)',
                                     line=dict(width=0), name=f'approx. 95% band (x{np.sqrt(deff):.2f} for same-date clustering)', hoverinfo='skip'))
    fig.update_layout(title='Cumulative units (flat 1-unit bets)', height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title='units')
    st.plotly_chart(fig, width='stretch')
    st.caption(f'The shaded band is approximate, not a date-block bootstrap: it is the band for independent bets, widened by {np.sqrt(deff):.2f}x (square root of a design effect of {deff:.2f}, estimated from how '
               f'much more the daily totals vary than independent bets would) because bets on one slate move together. Read it as a rough guide to the size of the noise.')
    b1, b2 = st.columns(2)
    fmt = {'roi': '{:+.1%}', 'units': '{:+.0f}', 'bets': '{:,.0f}'}
    b1.markdown('**By stat**'); b1.dataframe(L.breakdown(d, 'stat', mode, treatment).style.format(fmt), hide_index=True, width='stretch')
    b2.markdown('**By side**'); b2.dataframe(L.breakdown(d, 'side', mode, treatment).style.format(fmt), hide_index=True, width='stretch')
    b3, b4 = st.columns(2)
    de = d.assign(ev_band=L.ev_band(d[L.ev_col(mode)]))
    st.markdown('**By EV when flagged (at the chosen price): do the higher-value plays earn more?**')
    st.dataframe(L.breakdown(de, 'ev_band', mode, treatment).set_index('ev_band').reindex([n for *_, n in L.EV_BANDS]).dropna(how='all').reset_index().style.format(fmt), hide_index=True, width='stretch')
    st.caption('Bands hold few bets early in the live record, so the ROI per band is noise until there are several hundred settled bets in each.')
    b3.markdown('**By book (best-price quote)**'); b3.dataframe(L.breakdown(d, 'book', mode, treatment).style.format(fmt), hide_index=True, width='stretch')
    dd = d.assign(decision=d['minutes_before'].map(T_LABEL))
    b4.markdown('**By decision time**'); b4.dataframe(L.breakdown(dd, 'decision', mode, treatment).style.format(fmt), hide_index=True, width='stretch')
    render_diagnostics(d, treatment)
    if D['meta']['mode'] == 'live':
        render_segments(D, d_all_versions, treatment)
        render_versions(D, d_all_versions, treatment)
    st.subheader('Season summary' if D['meta']['mode'] != 'live' else 'Live record summary (placebo CLV and the information slope need a longer record)')
    sm = D['summary'].copy()
    wanted = ['season', 'kind', 'model_version', 'bets', 'player_games', 'roi_median_stands', 'roi_best_stands', 'roi_flat_stands', 'clv_real_pts', 'clv_placebo_mean_pts', 'clv_excess_pts',
              'clv_excess_lower_pts', 'placebo_roi', 'slope4', 'slope4_lo', 'slope4_hi']
    show = sm[[c for c in wanted if c in sm.columns]]
    formats = {'roi_best_stands': '{:+.1%}', 'roi_median_stands': '{:+.1%}', 'roi_flat_stands': '{:+.1%}', 'placebo_roi': '{:+.1%}', 'clv_real_pts': '{:+.2f}',
               'clv_placebo_mean_pts': '{:+.2f}', 'clv_excess_pts': '{:+.2f}', 'clv_excess_lower_pts': '{:+.2f}', 'slope4': '{:.2f}', 'slope4_lo': '{:.2f}', 'slope4_hi': '{:.2f}'}
    st.dataframe(show.style.format({k: v for k, v in formats.items() if k in show.columns}), hide_index=True, width='stretch')
    st.caption('The median-price ROI is the primary figure; the best-price ROI is the best quoted price, a target and not an assumed fill. CLV is in points of consensus probability toward the bet side by T-5; '
               '"excess" subtracts the no-information placebo (same k, rule, prices). Slope = realized-minus-market regressed on model-minus-market over the four traded stats.')


def page_health(D):
    if D['meta']['mode'] == 'live':
        st.warning(D['meta'].get('disclosure', ''), icon='⚠️')
        if len(D['checks']):
            st.markdown('**Checks at export time**')
            st.dataframe(D['checks'], hide_index=True, width='stretch')
        st.markdown('**Pipeline runs (latest first)**')
        st.dataframe(D['pipeline_health'].head(40), hide_index=True, width='stretch')
        st.markdown('**Decision slots, last 7 days (done, missed, no quotes)**')
        ss = L.slot_summary(D['slot_log'], pd.Timestamp.now(tz='UTC'), days=7)
        st.dataframe(pd.DataFrame({'Slot': ss['slot'], 'Done': ss['done'], 'Missed': ss['missed'], 'No quotes': ss['no_quotes'], 'Total': ss['total'],
                                   'Done share': [f'{100 * x:.0f}%' if np.isfinite(x) else '' for x in ss['done_share']]}), hide_index=True, width='stretch')
        st.caption('A missed slot is a game whose decision time passed without a recorded run: those bets were never made, which also biases the record toward the slots that did run. '
                   'The T-5m slot uses the snapshot taken about 10 minutes before tip.')
        st.markdown('**Decision slots (latest first)**')
        st.dataframe(D['slot_log'].head(60), hide_index=True, width='stretch')
        st.caption('Forecast-bias and reliability charts need a few hundred settled forecasts; they appear in the backtest view meanwhile.')
        return
    st.caption('Is the forecast still centred, and is it calibrated at the book line? Backtest values here; the live page will use rolling windows.')
    bias = D['bias']
    stat = st.selectbox('Stat', sorted(bias['stat'].unique()), index=0, format_func=lambda s: L.STAT_LABEL.get(s, s))
    b = bias[bias['stat'] == stat]
    fig = go.Figure()
    for season, g in b.groupby('season'):
        fig.add_trace(go.Scatter(x=g['month'], y=g['ratio'], name=season, mode='lines+markers'))
    fig.add_hline(y=1.0, line_dash='dash')
    fig.update_layout(title='Observed / predicted mean by month (1.0 = centred)', height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0.85, 1.15])
    st.plotly_chart(fig, width='stretch')
    rel = D['reliability']
    fig = go.Figure()
    for season, g in rel.groupby('season'):
        fig.add_trace(go.Scatter(x=g['mean_p_model'], y=g['hit_rate'], mode='lines+markers', name=f'{season} model'))
    fig.add_trace(go.Scatter(x=[0.3, 0.7], y=[0.3, 0.7], mode='lines', name='perfect', line=dict(dash='dash', color='grey')))
    fig.update_layout(title='Reliability at the book line, traded stats (model P(over) deciles vs hit rate)', height=340, margin=dict(l=10, r=10, t=40, b=10), xaxis_title='model P(over)', yaxis_title='hit rate')
    st.plotly_chart(fig, width='stretch')


def page_about(D):
    meta, rule = D['meta'], D['meta']['rule']
    seg_md = L.segments_markdown(meta.get('segments'))
    seg_note = ('\n\nThe record is also split into forward-test segments, each run under one fixed configuration. The Tracker shows them in a table and has a segment filter:\n' + seg_md) if seg_md else ''
    st.markdown(f"""
### What this is
A **paper-trading** view of one fixed rule (see the segments below for which rule applied when). No real money, no bet placement. Mode: **{meta['mode']}** ({'the live paper record, one row per rule-qualified bet at first qualification, flat 1 unit' if meta['mode'] == 'live' else 'historical replay'}), built {meta.get('built_at_utc', 'unknown')}.

### The rule
- Stats: {', '.join(L.STAT_LABEL[s] for s in rule['stats'])}; edge threshold {rule['primary']['e']:.0%} expected value at the best price; earliest decision time (T-4h, T-1h, T-5m) at which a market qualifies;
  at least {rule['primary']['min_books']} books agreeing on the line; one bet per player-game-stat; flat 1 unit.
- {L.pstar_text(rule)}
- No rolling recalibration (declared sensitivity only). Rule file hash `{L.short_hash(meta.get('rule_sha256'), 12)}`, dataset hash `{L.short_hash(meta.get('features_db_sha256'), 12)}`, code `{L.short_hash(meta.get('code_commit'), 8)}`.
- Decision slots: T-4h, T-1h and T-5m. The **T-5m slot uses the snapshot taken about 10 minutes before tip** (it started working on 2026-11-12), not a snapshot at exactly 5 minutes.

### Pricing views (choose in the sidebar)
- **Generic (default, primary)**: the median of the bettable books' quotes at the consensus line: what an ordinary bettor gets. Every single ROI headline on this dashboard is the median-price ROI.
- **Best available**: the best quote among the bettable books at the flag time. This is what the rule was selected on. It is shown next to the median as the **best quoted price (not an assumed fill)**: a
  target for working limit orders at the best sportsbook price on an exchange, not a price the record assumes you get.
- **Flat -110**: the textbook generic bookie.

### How to read the noise
Bets are not independent: many share a player-game with another bet and whole slates move together. The ROI intervals are therefore 95% intervals from a **date-clustered bootstrap** (whole game dates are
resampled), which is wider than the usual 1/sqrt(n) figure. The shaded band on the cumulative chart is the independent-bets band widened by a design effect estimated from the same-date clustering; it is an
approximation, not a bootstrap. Table cells with fewer than {L.MIN_STAKES} settled stakes are shown greyed as too small to read.

### Model versions and CLV
The live record mixes model versions. The first version (`v3_live_2026-27`, October 22-31) was **restored from a dashboard export in November 2026 after a data loss**: its snapshot time, quote age and code
commit are not recorded and it cannot be re-derived; its results came from the export. The Tracker has a model-version filter and a note whenever restored rows are included. CLV for bets before
2026-11-12 was **backfilled on 2026-11-12 from snapshots about 9-10 minutes before tip**, not recorded at the time.{seg_note}

### What the history says (reports/props/phase5_final_report.md)
The model's signal replicated across three seasons, but the profit depends on price: best-price ROI was about +4.9%, +3.3% and +4.6%; at the median price +0.8%, -1.7% and -0.5%. The approved gate
(median-price ROI non-negative) was **not passed** on the final season (-0.5%). 2023-24 and 2024-25 are in-sample for k and the rule. 2025-26 was scored once with the frozen rule and failed the
median-price gate, so it is **no longer a clean held-out season for this rule**.

### Limits
Quotes can move or disappear before placement; prop limits are small; winning accounts get restricted; early-exit settlement differs by book (see the settlement selector). Unibet no longer quotes in 2025-26.
""")


def render_freshness(D, treatment):
    """Top-of-page banner: a warning when the export is older than 45 minutes while an unsettled bet tips within the next 8 hours or the last 4 hours; otherwise a small caption."""
    meta = D['meta']
    built = meta.get('built_at_utc')
    if meta.get('mode') != 'live':
        st.caption(f'Backtest data built {built or "at an unknown time"}.')
        return
    fr = L.freshness_status(built, D['bets'], pd.Timestamp.now(tz='UTC'), treatment)
    if fr['warn']:
        st.warning(f"Data may be out of date: this export was built {L.age_text(fr['age_min'])} ago ({built}), and {fr['n_open_near']} unsettled bet{'s' if fr['n_open_near'] != 1 else ''} tip within the next "
                   f"{L.FRESH_AHEAD_H} hours or in the last {L.FRESH_BEHIND_H} hours. Prices, 'odds now', slots and results below may not show the latest state.", icon='⏱️')
    elif fr['age_min'] is None:
        st.caption(f'Data built {built or "at an unknown time"}.')
    else:
        st.caption(f"Data built {built} ({L.age_text(fr['age_min'])} ago).")


def main():
    st.title('🏀 NBA Player Props · paper-trading dashboard')
    sources = {name: p for name, p in (('Live paper record', LIVE_DIR), ('Backtest replay', BACKTEST_DIR)) if (p / 'bets.csv').exists()}
    if not sources:
        st.error(f'No data. Build it with: python scripts/export_props_dashboard_data.py --live   (or without --live for the backtest). Looked in {LIVE_DIR} and {BACKTEST_DIR}.')
        return
    st.sidebar.header('Data')
    source = st.sidebar.radio('Source', list(sources), index=0)
    D = load(str(sources[source]), (sources[source] / 'meta.json').stat().st_mtime)
    if D['meta']['mode'] == 'live':
        st.warning('LIVE PAPER RECORD: no real money. ' + D['meta'].get('disclosure', ''), icon='⚠️')
        st.sidebar.caption(f"Record started {D['meta'].get('record_started_utc') or 'not yet'} · model {D['meta'].get('model_version', 'n/a')} · built {D['meta'].get('built_at_utc', 'unknown')}")
    else:
        st.warning('BACKTEST MODE: historical replays of the frozen rule, not live bets. 2023-24 and 2024-25 are in-sample for k and the rule; 2025-26 was scored once, failed the median-price gate (-0.5%) and is no longer a clean held-out season for this rule.', icon='⚠️')
    st.sidebar.header('Pricing')
    mode = st.sidebar.radio('Price used for EV and profit', list(L.PRICE_MODES), format_func=lambda m: L.PRICE_MODES[m], index=0)
    treatment = st.sidebar.selectbox('Early-exit settlement', list(L.TREATMENTS), format_func=lambda t: L.TREATMENTS[t], index=0)
    st.sidebar.caption('Team dashboard: `streamlit run dashboard/app.py`')
    render_freshness(D, treatment)
    tab1, tab2, tab3, tab4 = st.tabs(['Board', 'Tracker', 'Model health', 'Rules & about'])
    with tab1:
        page_board(D, mode, treatment)
    with tab2:
        page_tracker(D, mode, treatment)
    with tab3:
        page_health(D)
    with tab4:
        page_about(D)


main()
