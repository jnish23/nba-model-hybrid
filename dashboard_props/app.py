"""
NBA player props dashboard (paper-trading view). Separate from the team dashboard (dashboard/app.py).

    streamlit run dashboard_props/app.py

Two data sources, chosen in the sidebar: the LIVE paper record (scripts/export_props_dashboard_data.py --live -> data/props_dashboard_live/) and the BACKTEST replay of the frozen rule on
2023-24 to 2025-26 (scripts/export_props_dashboard_data.py -> data/props_dashboard/). Plans: reports/props/dashboard_plan.md, reports/props/live_pipeline_plan.md.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from dashboard_props import logic as L
except ImportError:                                   # `streamlit run dashboard_props/app.py` puts this folder on sys.path
    import logic as L

ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DIR = Path(os.environ.get('PROPS_DASH_DATA', ROOT / 'data' / 'props_dashboard'))
LIVE_DIR = Path(os.environ.get('PROPS_DASH_LIVE', ROOT / 'data' / 'props_dashboard_live'))
STAT_CHOICES = ['rebounds', 'assists', 'threes', 'ra']
T_LABEL = {240: 'T-4h', 60: 'T-1h', 5: 'T-5m'}

st.set_page_config(page_title='NBA Props', page_icon='🏀', layout='wide')


@st.cache_data(show_spinner=False)
def load(data_dir):
    d = Path(data_dir)
    bets = pd.read_csv(d / 'bets.csv', dtype={'game_id': str})
    bets['played'] = bets['played'].fillna(0).astype(bool)
    bets['tip_et'] = pd.to_datetime(bets['tip_ref_utc'], utc=True).dt.tz_convert('America/New_York')
    out = {'bets': bets, 'summary': pd.read_csv(d / 'season_summary.csv'), 'reliability': pd.read_csv(d / 'reliability.csv'), 'bias': pd.read_csv(d / 'bias_monthly.csv'),
           'meta': json.loads((d / 'meta.json').read_text(encoding='utf-8')), 'pmf': json.loads((d / 'pmf.json').read_text(encoding='utf-8')),
           'books': json.loads((d / 'books.json').read_text(encoding='utf-8'))}
    for name in ('pipeline_health', 'slot_log', 'checks'):
        out[name] = pd.read_csv(d / f'{name}.csv') if (d / f'{name}.csv').exists() else pd.DataFrame()
    return out


def pct(x, nd=1):
    return '' if x is None or not np.isfinite(x) else f'{100 * x:+.{nd}f}%'


def bet_label(r):
    return f"{r['player_name']} · {r['side'].upper()} {L.STAT_LABEL.get(r['stat'], r['stat'])} {r['line']:g}"


def board_table(d, mode, treatment, reveal):
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
        'Model': side_model, 'Market': side_mkt, 'EV': d[ev_c].to_numpy(),
        'Flagged': d['minutes_before'].map(T_LABEL).to_numpy(), 'Books': d['n_at_line'].to_numpy()})
    if reveal:
        w = L.with_profit(d, mode, treatment)
        t['Actual'] = d['y'].to_numpy()
        t['Result'] = d[f'status_{treatment}'].to_numpy()
        t['Units'] = w['profit'].to_numpy()
    return t


def page_board(D, mode, treatment):
    bets = D['bets']
    c1, c2, c3 = st.columns([1, 1, 2])
    live = D['meta']['mode'] == 'live'
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
    min_ev = f3.slider('Min EV at the chosen price (%)', -5.0, 10.0, 0.0, 0.5) / 100.0
    top_n = f4.slider('Rows shown', 5, 60, 15, 5)
    day_df = bets[(bets['season'] == season) & (bets['game_date'] == day)]
    shown, hidden = L.board(day_df, mode, min_ev, stats, sides, top_n)
    st.caption(f'{len(day_df)} rule-qualified bets on {day} across {day_df["game_id"].nunique()} games; {len(shown)} shown'
               + (f', {hidden} more pass the filters (raise "Rows shown")' if hidden else '') + f'. EV and price use: {L.PRICE_MODES[mode]}.')
    if shown.empty:
        st.info('No bets pass the filters on this slate.')
        return
    t = board_table(shown, mode, treatment, reveal)
    cfg = {'Model': st.column_config.ProgressColumn('Model P', min_value=0.0, max_value=1.0, format='%.2f'), 'Market': st.column_config.NumberColumn('Market P', format='%.2f'),
           'EV': st.column_config.NumberColumn('EV', format='%.3f')}
    st.dataframe(t, use_container_width=True, hide_index=True, column_config=cfg)
    with st.expander('All shown bets grouped by game'):
        for game, g in shown.groupby('matchup', sort=False):
            st.markdown(f'**{game}**')
            st.dataframe(board_table(g, mode, treatment, reveal).drop(columns=['Game']), use_container_width=True, hide_index=True, column_config=cfg)
    st.download_button('Download this slate (CSV)', t.to_csv(index=False).encode('utf-8'), file_name=f'props_board_{day}.csv')
    st.subheader('Inspect a bet')
    labels = {r['bet_id']: bet_label(r) for _, r in shown.iterrows()}
    bid = st.selectbox('Bet', list(labels), format_func=lambda b: labels[b])
    detail(D, shown[shown['bet_id'] == bid].iloc[0], mode, treatment, reveal)


def detail(D, r, mode, treatment, reveal):
    pmf = np.array(D['pmf'].get(r['bet_id'], []))
    book_px = D['books'].get(r['bet_id'], {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Model P(side)', f"{L.side_prob(pd.DataFrame([r]), 'p_over_star')[0]:.3f}")
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
        left.plotly_chart(fig, use_container_width=True)
    if book_px:
        bk = pd.Series(book_px).sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=bk.index, y=bk.values, marker_color=['#2ca02c' if v == bk.max() else '#aec7e8' for v in bk.values], text=L.american(bk.values), textposition='outside'))
        fig.add_hline(y=r['dec_median'], line_dash='dot', annotation_text='median')
        fig.update_layout(title='Price of this bet by book at the flag time (decimal)', height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[min(1.5, bk.min() - 0.1), bk.max() + 0.15])
        right.plotly_chart(fig, use_container_width=True)
    hist = pd.DataFrame({'Decision time': ['T-4h', 'T-1h', 'T-5m'], 'Consensus line': [r['line_240'], r['line_60'], r['line_5']], 'Market P(over)': [r['mkt_240'], r['mkt_60'], r['mkt_5']]})
    st.dataframe(hist, hide_index=True, use_container_width=False)
    if reveal:
        w = L.with_profit(pd.DataFrame([r]), mode, treatment).iloc[0]
        status = r[f'status_{treatment}']
        if not isinstance(status, str):
            st.info('Pending: the game has not been settled yet.')
        elif w['stake']:
            st.success(f"Result: {status.upper()} · actual {r['y']:g} · {w['profit']:+.2f} units at {L.PRICE_MODES[mode].lower()}")
        else:
            st.info(f"Result: {status} · actual {r['y']} (no stake)")


def page_tracker(D, mode, treatment):
    bets = D['bets']
    seasons = st.multiselect('Seasons', sorted(bets['season'].unique()), default=sorted(bets['season'].unique()))
    d = bets[bets['season'].isin(seasons)]
    if d.empty:
        st.info('Select at least one season.')
        return
    if D['meta']['mode'] == 'live':
        pending = int(d[f'status_{treatment}'].isna().sum())
        d = d[d[f'status_{treatment}'].notna()]
        st.caption(f'{pending} bets are placed and not yet settled; only settled bets count below.')
        if d.empty:
            st.info('No settled bets yet.')
            return
    s = L.roi_summary(d, mode, treatment)
    other = {m: L.roi_summary(d, m, treatment, n_boot=200)['roi'] for m in L.PRICE_MODES}
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric('Settled bets', f"{s['bets']:,}", help=f"{s['player_games']:,} distinct player-games")
    t2.metric(f'ROI ({L.PRICE_MODES[mode].split(":")[0]})', pct(s['roi']), help=f"95% interval {pct(s['lo'])} to {pct(s['hi'])} (game-clustered)")
    t3.metric('Win rate', f"{100 * s['win_rate']:.1f}%")
    t4.metric('Units', f"{s['units']:+.0f}")
    t5.metric('ROI by price', ' · '.join(f"{k}: {pct(v)}" for k, v in other.items()))
    st.caption(f"95% interval for ROI: {pct(s['lo'])} to {pct(s['hi'])}. " + L.noise_note(s['bets']))
    fig = go.Figure()
    for m in L.PRICE_MODES:
        c = L.cumulative(d, m, treatment)
        fig.add_trace(go.Scatter(x=c['game_date'], y=c['cum_units'], name=L.PRICE_MODES[m], line=dict(width=3 if m == mode else 1.2), opacity=1 if m == mode else 0.6))
        if m == mode:
            fig.add_trace(go.Scatter(x=list(c['game_date']) + list(c['game_date'])[::-1], y=list(c['hi']) + list(c['lo'])[::-1], fill='toself', fillcolor='rgba(31,119,180,0.12)',
                                     line=dict(width=0), name='approx. 95% band', hoverinfo='skip'))
    fig.update_layout(title='Cumulative units (flat 1-unit bets)', height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title='units')
    st.plotly_chart(fig, use_container_width=True)
    b1, b2 = st.columns(2)
    fmt = {'roi': '{:+.1%}', 'units': '{:+.0f}', 'bets': '{:,.0f}'}
    b1.markdown('**By stat**'); b1.dataframe(L.breakdown(d, 'stat', mode, treatment).style.format(fmt), hide_index=True, use_container_width=True)
    b2.markdown('**By side**'); b2.dataframe(L.breakdown(d, 'side', mode, treatment).style.format(fmt), hide_index=True, use_container_width=True)
    b3, b4 = st.columns(2)
    b3.markdown('**By book (best-price quote)**'); b3.dataframe(L.breakdown(d, 'book', mode, treatment).style.format(fmt), hide_index=True, use_container_width=True)
    dd = d.assign(decision=d['minutes_before'].map(T_LABEL))
    b4.markdown('**By decision time**'); b4.dataframe(L.breakdown(dd, 'decision', mode, treatment).style.format(fmt), hide_index=True, use_container_width=True)
    st.subheader('Season summary' if D['meta']['mode'] != 'live' else 'Live record summary (placebo CLV and the information slope need a longer record)')
    sm = D['summary'].copy()
    show = sm[['season', 'kind', 'bets', 'player_games', 'roi_best_stands', 'roi_median_stands', 'roi_flat_stands', 'clv_real_pts', 'clv_placebo_mean_pts', 'clv_excess_pts', 'clv_excess_lower_pts', 'placebo_roi',
               'slope4', 'slope4_lo', 'slope4_hi']]
    st.dataframe(show.style.format({'roi_best_stands': '{:+.1%}', 'roi_median_stands': '{:+.1%}', 'roi_flat_stands': '{:+.1%}', 'placebo_roi': '{:+.1%}', 'clv_real_pts': '{:+.2f}',
                                    'clv_placebo_mean_pts': '{:+.2f}', 'clv_excess_pts': '{:+.2f}', 'clv_excess_lower_pts': '{:+.2f}', 'slope4': '{:.2f}', 'slope4_lo': '{:.2f}', 'slope4_hi': '{:.2f}'}),
                 hide_index=True, use_container_width=True)
    st.caption('CLV is in points of consensus probability toward the bet side by T-5; "excess" subtracts the no-information placebo (same k, rule, prices). Slope = realized-minus-market regressed on model-minus-market over the four traded stats.')


def page_health(D):
    if D['meta']['mode'] == 'live':
        st.warning(D['meta']['disclosure'], icon='⚠️')
        if len(D['checks']):
            st.markdown('**Checks at export time**')
            st.dataframe(D['checks'], hide_index=True, use_container_width=True)
        st.markdown('**Pipeline runs (latest first)**')
        st.dataframe(D['pipeline_health'].head(40), hide_index=True, use_container_width=True)
        st.markdown('**Decision slots (done, missed, no quotes)**')
        st.dataframe(D['slot_log'].head(60), hide_index=True, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
    rel = D['reliability']
    fig = go.Figure()
    for season, g in rel.groupby('season'):
        fig.add_trace(go.Scatter(x=g['mean_p_model'], y=g['hit_rate'], mode='lines+markers', name=f'{season} model'))
    fig.add_trace(go.Scatter(x=[0.3, 0.7], y=[0.3, 0.7], mode='lines', name='perfect', line=dict(dash='dash', color='grey')))
    fig.update_layout(title='Reliability at the book line, traded stats (model P(over) deciles vs hit rate)', height=340, margin=dict(l=10, r=10, t=40, b=10), xaxis_title='model P(over)', yaxis_title='hit rate')
    st.plotly_chart(fig, use_container_width=True)


def page_about(D):
    meta, rule = D['meta'], D['meta']['rule']
    st.markdown(f"""
### What this is
A **paper-trading** view of one frozen rule. No real money, no bet placement. Mode: **{meta['mode']}** ({'the live paper record, one row per rule-qualified bet at first qualification, flat 1 unit' if meta['mode'] == 'live' else 'historical replay'}), built {meta['built_at_utc']}.

### The frozen rule
- Stats: {', '.join(L.STAT_LABEL[s] for s in rule['stats'])}; edge threshold {rule['primary']['e']:.0%} expected value at the best price; earliest decision time (T-4h, T-1h, T-5m) at which a market qualifies;
  at least {rule['primary']['min_books']} books agreeing on the line; one bet per player-game-stat; flat 1 unit.
- Model probability shrunk toward the market: p* = market + k x (model - market), k by stat: {', '.join(f"{L.STAT_LABEL[s]} {v:.2f}" for s, v in rule['k_by_stat'].items())}.
- No rolling recalibration (declared sensitivity only). Rule file hash `{meta['rule_sha256'][:12]}`, dataset hash `{meta['features_db_sha256'][:12]}`, code `{meta['code_commit'][:8]}`.

### Pricing views (choose in the sidebar)
- **Generic (default)**: the median of the bettable books' quotes at the consensus line: what an ordinary bettor gets.
- **Best available**: the best quote among the bettable books at the flag time. This is what the rule was selected on.
- **Flat -110**: the textbook generic bookie.

### What the history says (reports/props/phase5_final_report.md)
The model's signal replicated across three seasons, but the profit depends on price: best-price ROI was about +4.9%, +3.3% and +4.6%; at the median price +0.8%, -1.7% and -0.5%. The approved gate
(median-price ROI non-negative) was **not passed** on the final season. 2023-24 and 2024-25 are in-sample for k and the rule; 2025-26 is the single final holdout, scored once.

### Limits
Quotes can move or disappear before placement; prop limits are small; winning accounts get restricted; early-exit settlement differs by book (see the settlement selector). Unibet no longer quotes in 2025-26.
""")


def main():
    st.title('🏀 NBA Player Props · paper-trading dashboard')
    sources = {name: p for name, p in (('Live paper record', LIVE_DIR), ('Backtest replay', BACKTEST_DIR)) if (p / 'bets.csv').exists()}
    if not sources:
        st.error(f'No data. Build it with: python scripts/export_props_dashboard_data.py --live   (or without --live for the backtest). Looked in {LIVE_DIR} and {BACKTEST_DIR}.')
        return
    st.sidebar.header('Data')
    source = st.sidebar.radio('Source', list(sources), index=0)
    D = load(str(sources[source]))
    if D['meta']['mode'] == 'live':
        st.warning('LIVE PAPER RECORD: frozen rule, no real money. ' + D['meta']['disclosure'], icon='⚠️')
        st.sidebar.caption(f"Record started {D['meta'].get('record_started_utc') or 'not yet'} · model {D['meta']['model_version']} · built {D['meta']['built_at_utc']}")
    else:
        st.warning('BACKTEST MODE: historical replays of the frozen rule, not live bets. 2023-24 and 2024-25 are in-sample for k and the rule; 2025-26 is the final holdout.', icon='⚠️')
    st.sidebar.header('Pricing')
    mode = st.sidebar.radio('Price used for EV and profit', list(L.PRICE_MODES), format_func=lambda m: L.PRICE_MODES[m], index=0)
    treatment = st.sidebar.selectbox('Early-exit settlement', list(L.TREATMENTS), format_func=lambda t: L.TREATMENTS[t], index=0)
    st.sidebar.caption('Team dashboard: `streamlit run dashboard/app.py`')
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
