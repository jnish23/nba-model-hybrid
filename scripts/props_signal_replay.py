"""
Replays the live signal chain (props/live/signal.py) on historical slates from the stored T-grid snapshots and compares the bets with the parity harness's live-feature bets of the
same dates. An integration test of the job's evaluation code, not a result: model v3_final_2025-26 (out of sample for 2025-26), roster rule v2.

    python scripts/props_signal_replay.py --attempt 2 --dates 2025-11-12 2026-01-15 2026-03-20
"""
import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props import schedule as S  # noqa: E402
from props.features import build, history, rows as rows_mod  # noqa: E402
from props.live import at_t as A, signal  # noqa: E402
from props.models import registry  # noqa: E402

SEASON, DB, FDB = '2025-26', 'data/nba_stats.db', 'data/props_features_v3.db'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, default=2)
    ap.add_argument('--dates', nargs='+', required=True)
    ap.add_argument('--n-sims', type=int, default=20000)
    a = ap.parse_args()
    sched = S.load_schedule(SEASON)
    sched = sched[sched['game_id'].str[:3].isin(['002', '004'])]
    cons, bets = pickle.load(open(f'data/props_betting/prices_{SEASON}.pkl', 'rb'))
    for d in (cons, bets):
        d['game_id'] = d['game_id'].astype(str)
    with sqlite3.connect('file:data/props_odds.db?mode=ro', uri=True, timeout=120) as c:
        tips = pd.read_sql('SELECT game_id, tip_ref_utc FROM event_ref WHERE game_id IS NOT NULL', c).drop_duplicates('game_id', keep='last').set_index('game_id')['tip_ref_utc']
        names = pd.read_sql('SELECT game_id, player_id, max(player_name_raw) AS n FROM prop_snapshots WHERE player_id > 0 AND substr(commence_time_utc,1,7) BETWEEN ? AND ? GROUP BY 1,2',
                            c, params=[min(a.dates)[:7], max(a.dates)[:7]])
    nm = {(g, int(p)): n for g, p, n in zip(names['game_id'].astype(str), names['player_id'], names['n'])}
    hist = rows_mod.load_history(DB)
    hist['game_id'] = hist['game_id'].astype(str)
    ctx = signal.Context(hist=hist, pos=history.position_groups(DB), reports=build.load_linked_reports(DB), tips=build.reference_tips_et([SEASON]),
                         tables=A.play_tables(FDB, build.GRID), bundle=registry.load_bundle('v3_final_2025-26', SEASON), rule=json.loads(Path('reports/props/phase5_revised_rule.frozen_rule.json').read_text()),
                         nba_db=DB, n_sims=a.n_sims, model_version='v3_final_2025-26', provenance={'model_sha256': '', 'features_db_sha256': '', 'rule_sha256': '', 'roster_rule': 'v2', 'code_commit': ''})
    got, existing = [], set()
    for date in a.dates:
        g = sched[sched['et_game_date'] == date]
        items = []
        for r in g.itertuples():
            srow = pd.Series(r._asdict()); srow['tip_ref_utc'], srow['season'], srow['game_date'] = tips.get(r.game_id), SEASON, date
            for T in (240, 60, 5):
                cs, bs = cons[(cons.game_id == r.game_id) & (cons.minutes_before == T)], bets[(bets.game_id == r.game_id) & (bets.minutes_before == T)]
                if cs.empty:
                    continue
                items.append(signal.SlotItem(game=srow, slot=T, snapshot_ts_utc=str(cs['snapshot_ts_utc'].iloc[0]), snapshot_minutes=float(T), event_id='', cons=cs, bets=bs,
                                             names={int(p): nm.get((r.game_id, int(p))) for p in cs['player_id'].unique()}))
        items.sort(key=lambda i: -i.slot)
        frame, _ = signal.build_frames(ctx, date, items)
        for it in items:
            sig, new, detail = signal.evaluate_item(ctx, it, frame, existing)
            existing |= set(map(tuple, new[['game_id', 'player_id', 'stat']].itertuples(index=False, name=None))) if len(new) else set()
            got.append(new)
    live = pd.concat([x for x in got if len(x)], ignore_index=True)
    ref = pd.read_csv(f'data/props_live_parity/attempt{a.attempt}/a6_bets_live.csv', dtype={'game_id': str})
    ref = ref[ref['game_id'].isin(set(sched[sched['et_game_date'].isin(a.dates)]['game_id']))]
    key = ['game_id', 'player_id', 'stat']
    both = live.merge(ref[key + ['best_side', 'best_ev']], on=key, suffixes=('', '_ref'))
    print(f'replay bets {len(live)}, harness bets {len(ref)}, shared {len(both)}, Jaccard {len(both) / (len(live) + len(ref) - len(both)):.3f}, side agreement {(both.side == both.best_side).mean():.3f}, '
          f'EV corr {both["ev_best"].corr(both["best_ev"]):.3f}')
    live.to_csv('data/raw/replay_bets.csv', index=False)


if __name__ == '__main__':
    main()
