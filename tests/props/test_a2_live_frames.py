"""
A2 step P1 (reports/props/audit_2026-11/A2_live_plan.md): the production live frames under a pre-game roster rule (props/live/pregame_frames.py, called by props.live.signal.build_frames
and by scripts/props_d1_parity.py), with a row for every quoted player. Synthetic league of tests/props/test_a11_pregame_roster.py, no database.

  (i)   under 'v2' and 'registered' the frames equal those of the code before the change (a verbatim copy of the old build_frames)
  (ii)  a quote for a non-member changes no other row (exact); a quote for a member changes nothing at all
  (iii) a report stamped later than T - 15 minutes changes nothing; the snapshot time, not the nominal slot, decides which reports count; rows are labelled with the slot
  (iv)  a quoted non-member's teammate columns equal the training builder's for the same player as a box-score target who is not a member
  (v)   a quoted player whose team cannot be resolved is skipped with a reason, without raising
  (vi)  the reports of every game of the date place players; the parity script and the signal path give the same frames
"""
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from props.features import injury_at_t as inj, pregame_roster as pg
from props.live import at_t as A, base as live_base, pregame_frames, roster, signal
from props.live.base import build_live_base
from props.live.frame import live_frame
from test_a11_pregame_roster import GRID, POS, _season_dates, base_and_labels, league

ROOT = Path(__file__).resolve().parents[2]
SEASON = '2024-25'
DATE = _season_dates(SEASON)[6]              # the trade date: player 105 is on team 1's previous-game roster but the report of team 3's game places him on team 3


@dataclass
class Env:
    hist: pd.DataFrame
    reports: pd.DataFrame
    tips: pd.Series
    tables: dict
    games: pd.DataFrame           # the games of every date: game_id, game_date, season, home_team_id, away_team_id
    base: pd.DataFrame


def regulars_miss_the_previous_game(hist, reports, date=DATE, per_team=2):
    """The case of the plan's P1 design: regulars who were not listed in their team's previous game and are not on today's reports (so they are not members of today's game)."""
    g = pg.game_table(hist)
    today = set(g.loc[g['game_date'] == date, 'game_id'])
    named = set(reports.loc[reports['gameId'].isin(today), 'personId'])
    drop = []
    for t in sorted(set(hist.loc[hist['game_id'].isin(today), 'team_id'])):
        tg = hist[(hist['team_id'] == t) & (hist['game_date'] < date)].sort_values(['game_date', 'game_id'])
        last = tg[tg['game_id'] == tg['game_id'].iloc[-1]]
        drop += list(last.loc[~last['player_id'].isin(named) & (last['player_id'] != 105), ['game_id', 'player_id']].head(per_team).itertuples(index=False, name=None))
    keep = ~pd.Series(list(zip(hist['game_id'], hist['player_id']))).isin(set(drop)).to_numpy()
    return hist[keep].reset_index(drop=True)


@pytest.fixture(scope='module')
def env():
    hist, reports, tips = league()
    hist = regulars_miss_the_previous_game(hist, reports)
    base, labels = base_and_labels(hist)
    tables = {}
    for x in GRID:
        chosen, games_t = inj.latest_report_rows(reports, tips, x)
        status = inj.status_for_rows(base[['game_id', 'player_id', 'team_id']], chosen, games_t)
        keyed = status.merge(base[['game_id', 'player_id', 'season', 'm_bucket']], on=['game_id', 'player_id']).merge(labels, on=['game_id', 'player_id'])
        tables[x] = inj.play_rate_table(keyed, x, '2019-20')
    return Env(hist, reports, tips, tables, pg.game_table(hist), base)


@pytest.fixture(autouse=True)
def fake_team_context(monkeypatch):
    """build_live_base reads team form and travel from the box-score database; here a flat context table keyed like the real one."""
    def fake(db_path, games):
        recs = [{'game_id': g.game_id, 'team_id': int(t), 'is_playoff': 0, 'season_game_number': 5, 'days_rest': 2.0 + int(t) % 3, 'form': 0.1 * int(t)}
                for g in games.itertuples() for t in (g.home_team_id, g.away_team_id)]
        return pd.DataFrame(recs)
    monkeypatch.setattr(live_base, 'live_team_context', fake)


def date_games(env, date=DATE):
    return env.games[env.games['game_date'] == date].reset_index(drop=True)


def schedule_of(env, date=DATE):
    return date_games(env, date).rename(columns={'game_date': 'et_game_date'})


def make_items(env, date, cons, slots=(60,), snap=None, only=None):
    """cons: {game_id: [player ids]} quoted for every slot of that game. snap: information time (default the slot)."""
    items = []
    for g in date_games(env, date).itertuples():
        if only is not None and g.game_id not in only:
            continue
        srow = pd.Series({'game_id': g.game_id, 'season': g.season, 'game_date': date, 'tip_ref_utc': str(env.tips[g.game_id]), 'home_team': f'T{g.home_team_id}', 'away_team': f'T{g.away_team_id}',
                          'home_team_id': g.home_team_id, 'away_team_id': g.away_team_id, 'et_game_date': date})
        for T in slots:
            ids = np.array(cons.get(g.game_id, []), dtype='int64')
            items.append(signal.SlotItem(game=srow, slot=T, snapshot_ts_utc='', snapshot_minutes=float(T if snap is None else snap), event_id='', cons=pd.DataFrame({'game_id': [g.game_id] * len(ids), 'player_id': ids}),
                                         bets=pd.DataFrame(), names={}))
    return items


def make_ctx(env, rule, schedule=None, reports=None):
    return signal.Context(hist=env.hist, pos=POS, reports=env.reports if reports is None else reports, tips=env.tips, tables=env.tables, bundle={}, rule={}, nba_db='unused', n_sims=1,
                          model_version='t', provenance={}, roster_rule=rule, schedule=schedule)


def frame_of(env, rule, cons, slots=(60,), snap=None, schedule='default', reports=None, only=None, date=DATE):
    sched = schedule_of(env, date) if isinstance(schedule, str) else schedule
    ctx = make_ctx(env, rule, sched, reports)
    frame, unresolved = signal.build_frames(ctx, date, make_items(env, date, cons, slots, snap, only))
    return frame, unresolved


def same(a, b):
    a, b = a.sort_values(['game_id', 'player_id', 'minutes_before']).reset_index(drop=True), b.sort_values(['game_id', 'player_id', 'minutes_before']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def without(frame, game_id, player_ids):
    return frame[~((frame['game_id'] == game_id) & frame['player_id'].isin(set(player_ids)))].reset_index(drop=True)


def non_members(env, frame, game_id, k=2, date=DATE):
    """Players whose latest listed team this season is a team of the game, who have no row in the frame (so they are not members at the slot)."""
    g = date_games(env, date).set_index('game_id').loc[game_id]
    lt = roster.last_team(env.hist[env.hist['game_date'] < date], date, SEASON)
    have = set(frame['player_id'])                                         # no row in any game of the date (a member of another game would be skipped, see test_v)
    out = [p for p, t in sorted(lt.items()) if t in (int(g['home_team_id']), int(g['away_team_id'])) and p not in have]
    assert len(out) >= k, 'the synthetic league has too few non-members on this date'
    return out[:k]


def members_of(frame, game_id):
    return sorted(set(frame.loc[frame['game_id'] == game_id, 'player_id']))


# ---------------------------------------------------------------- (i) v2 and registered are unchanged ----------------------------------------------------------------
def _old_build_frames(ctx, date, items):
    """props.live.signal.build_frames as of commit 48a7f00 (before P1), verbatim."""
    games = pd.DataFrame([i.game for i in items]).drop_duplicates('game_id')
    games = games.assign(game_date=date)
    h = ctx.hist[ctx.hist['game_date'] < date]
    season = str(games['season'].iloc[0])
    per = []
    extras = []
    for it in items:
        rep = A.asof_reports(ctx.reports, ctx.tips, it.snapshot_minutes)
        rp = A.report_players(rep, ctx.tips, it.snapshot_minutes)
        rp = rp[rp['game_id'] == it.game['game_id']]
        qp = it.cons[['game_id', 'player_id']].drop_duplicates()
        per.append((it, rep, rp, qp))
        extras += [rp, qp]
    base_all, unresolved = build_live_base(ctx.hist, ctx.pos, games, pd.concat(extras, ignore_index=True), date, ctx.nba_db,
                                           rosters=roster.previous_game_rosters(h, date, season))
    blocks, frames = [], {}
    for it, rep, rp, qp in per:
        g1 = games[games['game_id'] == it.game['game_id']]
        members = roster.rule_members(ctx.roster_rule, g1, ctx.hist, date, rp, qp)
        base_T = base_all.merge(members, on=['game_id', 'player_id'], how='inner', validate='one_to_one')
        blocks.append(A.live_at_t(base_T, rep, ctx.tips, it.slot, ctx.tables[it.slot], asof_minutes=it.snapshot_minutes))
    frame = live_frame(base_all, blocks)
    return frame, unresolved


@pytest.mark.parametrize('rule', ['v2', 'registered'])
def test_i_v2_and_registered_frames_are_identical_to_the_code_before_the_change(env, rule):
    probe, _ = frame_of(env, 'pregame_all', {})
    gid = date_games(env)['game_id'].iloc[0]
    quoted = members_of(probe, gid)[:4] + non_members(env, probe, gid, 1)
    items = make_items(env, DATE, {gid: quoted}, slots=(240, 60, 5), snap=None)
    for schedule in (schedule_of(env), None):                       # v2 never reads the schedule
        ctx = make_ctx(env, rule, schedule)
        new, u_new = signal.build_frames(ctx, DATE, items)
        old, u_old = _old_build_frames(ctx, DATE, items)
        pd.testing.assert_frame_equal(new, old, check_exact=True)
        assert u_new == u_old
        assert 'skipped_players' not in new.attrs and len(new) > 0


def test_i_the_default_rule_is_still_v2_and_not_a_pregame_rule():
    assert signal.Context.__dataclass_fields__['roster_rule'].default == 'v2' and 'v2' not in roster.PREGAME_RULES
    assert signal.Context.__dataclass_fields__['schedule'].default is None


# ---------------------------------------------------------------- (ii) quotes never change another row ----------------------------------------------------------------
@pytest.mark.parametrize('rule', ['pregame_all', 'v2nq'])
def test_ii_a_quote_for_a_non_member_changes_no_other_row_and_a_quote_for_a_member_changes_nothing(env, rule):
    slots = (240, 60, 5)
    probe, _ = frame_of(env, rule, {}, slots=slots)
    gid = date_games(env)['game_id'].iloc[0]
    counts = probe[probe['game_id'] == gid].groupby('player_id').size()
    mem = sorted(counts[counts == len(slots)].index)                          # members at every slot (a Out player leaves the v2nq set at the slots whose report lists him)
    assert len(mem) >= 6
    q1, q2 = non_members(env, probe, gid, 2)
    a, _ = frame_of(env, rule, {gid: mem[:3]}, slots=slots)
    b, _ = frame_of(env, rule, {gid: mem[:3] + [q1]}, slots=slots)
    c, _ = frame_of(env, rule, {gid: mem[:3] + [q1, q2]}, slots=slots)
    d, _ = frame_of(env, rule, {gid: mem[:3] + [mem[5]]}, slots=slots)
    e, _ = frame_of(env, rule, {gid: [q1]}, slots=slots)
    assert set(b.loc[b['game_id'] == gid, 'player_id']) - set(a.loc[a['game_id'] == gid, 'player_id']) == {q1}
    assert len(b) == len(a) + 3 and len(c) == len(a) + 6                                  # a row per slot for each quoted non-member
    same(without(b, gid, [q1]), a)                                                        # adding q1 changes no other row
    same(without(c, gid, [q2]), b)                                                        # adding q2 on top changes no other row, q1's own row included
    same(without(c, gid, [q1, q2]), a)
    same(without(e, gid, [q1]), a)                                                        # removing every other quote changes nothing either
    same(d, a)                                                                            # a quote for a member changes nothing at all
    for T in slots:                                                                       # his row is labelled with the slot
        assert ((b['player_id'] == q1) & (b['minutes_before'] == T)).sum() == 1


# ---------------------------------------------------------------- (iii) report timing ----------------------------------------------------------------
def _late_report(env, gid, player, team, minutes_before_tip, status='Out'):
    return pd.DataFrame([{'gameId': gid, 'teamId': team, 'personId': player, 'report_ts': env.tips[gid] - pd.Timedelta(minutes=minutes_before_tip), 'status': status,
                          'reason': 'Injury/Illness - Right knee; Sprain', 'not_submitted': False}])


@pytest.mark.parametrize('slot,snap', [(60, 60.0), (60, 57.0), (5, 5.0), (240, 244.0)])
def test_iii_a_report_stamped_later_than_t_minus_15_changes_nothing(env, slot, snap):
    probe, _ = frame_of(env, 'pregame_all', {})
    gid = date_games(env)['game_id'].iloc[0]
    mem = members_of(probe, gid)
    team = int(date_games(env)['home_team_id'].iloc[0])
    quoted = {gid: mem[:3] + non_members(env, probe, gid, 1)}
    ref, _ = frame_of(env, 'pregame_all', quoted, slots=(slot,), snap=snap)
    late = _late_report(env, gid, mem[0], team, snap + 14.0)              # stamped 14 minutes before T: inside the publish lag
    late2 = _late_report(env, gid, mem[1], team, snap + 1.0, 'Questionable')
    new, _ = frame_of(env, 'pregame_all', quoted, slots=(slot,), snap=snap, reports=pd.concat([env.reports, late, late2], ignore_index=True))
    same(new, ref)


def test_iii_the_snapshot_time_decides_which_reports_count_and_the_rows_carry_the_slot(env):
    probe, _ = frame_of(env, 'pregame_all', {})
    gid = date_games(env)['game_id'].iloc[0]
    mem = members_of(probe, gid)
    star = mem[0]
    rep = pd.concat([env.reports, _late_report(env, gid, star, star // 100, 74.0)], ignore_index=True)      # usable at 57 minutes before tip (cutoff 72), not at a nominal 60 (cutoff 75)
    quoted = {gid: mem[:3]}
    at_slot, _ = frame_of(env, 'pregame_all', quoted, slots=(60,), snap=60.0, reports=rep)
    at_snap, _ = frame_of(env, 'pregame_all', quoted, slots=(60,), snap=57.0, reports=rep)
    ref, _ = frame_of(env, 'pregame_all', quoted, slots=(60,), snap=60.0)
    same(at_slot, ref)                                                                    # not yet usable at 60: nothing changes
    assert not at_snap[at_snap['game_id'] == gid].equals(ref[ref['game_id'] == gid])      # usable at 57: the game's members change
    assert (at_snap['minutes_before'] == 60).all() and (at_snap['game_id'].isin(set(date_games(env)['game_id']))).all()
    assert star in members_of(at_snap, gid)                                               # the new report names him: a member whatever his status


# ---------------------------------------------------------------- (iv) equal to the training builder ----------------------------------------------------------------
def _training_blocks(env, rule):
    """{x: (block, members)}: the training builder's at-T block for every box-score row (props.features.build.build_at_t with membership=rule, without the market environment)."""
    rows = env.base
    pre, _ = pg.prepare_members(rows, env.hist, POS, env.reports, env.tips, GRID, rule)
    out = {}
    for x in GRID:
        chosen, games_t = inj.latest_report_rows(env.reports, env.tips, x)
        status = inj.status_for_rows(rows[['game_id', 'player_id', 'team_id']], chosen, games_t)
        members, member_feat = pre[x]
        member_status = inj.status_for_rows(members[['game_id', 'player_id', 'team_id']], chosen, games_t)
        out[x] = (pg.at_t_members_block(rows, status, env.tables[x], members, member_status, member_feat), members)
    return out


@pytest.mark.parametrize('rule', ['pregame_all', 'v2nq'])
def test_iv_a_quoted_non_member_equals_the_training_builder_for_a_box_score_target_who_is_not_a_member(env, rule):
    blocks = _training_blocks(env, rule)
    found, builds, call_ups = 0, 0, 0
    for date in _season_dates(SEASON)[3:]:
        box = env.base[env.base['game_id'].isin(set(date_games(env, date)['game_id']))]
        for gid in sorted(set(box['game_id'])):
            nm = {x: sorted(int(p) for p in box.loc[box['game_id'] == gid, 'player_id'] if (gid, int(p)) not in set(zip(blocks[x][1]['game_id'], blocks[x][1]['player_id']))) for x in GRID}
            union = sorted(set().union(*nm.values()))
            if not union or builds >= 6:
                continue
            builds += 1
            frame, _ = signal.build_frames(make_ctx(env, rule, schedule_of(env, date)), date, make_items(env, date, {gid: union}, slots=GRID))
            for x in GRID:
                skipped = frame.attrs['skipped_players'].get((gid, x), {})
                assert set(skipped.values()) <= {'no_team'}, skipped                      # a call-up with no team this season yet cannot be placed: the only legitimate skip here
                call_ups += len(skipped)
                q_ok = [p for p in nm[x] if p not in skipped]
                live = frame[(frame['game_id'] == gid) & frame['player_id'].isin(q_ok) & (frame['minutes_before'] == x)].sort_values('player_id').reset_index(drop=True)
                assert len(live) == len(q_ok), (date, x, nm[x], len(live))
                train = blocks[x][0]
                train = train[(train['game_id'] == gid) & train['player_id'].isin(q_ok)].sort_values('player_id').reset_index(drop=True)
                av = [c for c in train.columns if c.startswith(('av_', 'opp_av_'))]
                assert len(av) == 15 and list(live['player_id']) == list(train['player_id'])
                for c in av + ['inj_status', 'inj_status_code', 'inj_report_missing']:
                    if train[c].dtype.kind in 'iuO' or c.endswith(('_n_out', '_n_questionable')):
                        assert (live[c].to_numpy() == train[c].to_numpy()).all(), c
                    else:
                        l, t = live[c].to_numpy(float), train[c].to_numpy(float)
                        assert (np.isnan(l) == np.isnan(t)).all() and np.nanmax(np.abs(l - t), initial=0.0) <= 1e-9, c       # float noise of the league priors only (see test_a11_pregame_roster.ULP)
                found += len(q_ok)
    assert found >= 6, 'the synthetic league has too few box-score targets outside the member set'


# ---------------------------------------------------------------- (v) unresolvable quoted players ----------------------------------------------------------------
def test_v_a_quoted_player_whose_team_cannot_be_resolved_is_skipped_with_a_reason_and_nothing_raises(env):
    probe, _ = frame_of(env, 'pregame_all', {})
    g = date_games(env)
    gid, other = g['game_id'].iloc[0], g['game_id'].iloc[1]
    mem = members_of(probe, gid)
    stranger = members_of(probe, other)[-1]                                 # a member of the OTHER game's teams: his latest team is not in this game
    nobody = 987654                                                         # never listed, never reported
    good = non_members(env, probe, gid, 1)[0]
    frame, unresolved = frame_of(env, 'pregame_all', {gid: mem[:2] + [good, stranger, nobody]})
    got = frame.loc[(frame['game_id'] == gid) & (frame['minutes_before'] == 60), 'player_id']
    assert good in set(got) and stranger not in set(got) and nobody not in set(got)
    skipped = frame.attrs['skipped_players'][(gid, 60)]
    assert skipped == {stranger: 'last_team_not_in_game', nobody: 'no_team'}
    # evaluate_item reports them in the slot's detail without breaking the existing field
    only_bad = make_items(env, DATE, {gid: [stranger, nobody]}, only={gid})[0]
    f2, _ = signal.build_frames(make_ctx(env, 'pregame_all', schedule_of(env)), DATE, [only_bad])
    sig, new, detail = signal.evaluate_item(make_ctx(env, 'pregame_all'), only_bad, f2, existing=set())
    assert sig.empty and new.empty
    assert detail['players_without_features'] == sorted([stranger, nobody]) and detail['quoted_players'] == 2
    assert detail['skip_reasons'] == {str(stranger): 'last_team_not_in_game', str(nobody): 'no_team'}
    assert set(detail['skip_reasons'].values()) <= set(pregame_frames.SKIP_REASONS)


def test_v_a_player_the_reports_place_in_another_game_of_the_date_is_skipped_for_this_one(env):
    probe, _ = frame_of(env, 'pregame_all', {})
    g1, g2 = date_games(env)['game_id'].iloc[:2]
    assert 105 in members_of(probe, g2) and 105 not in members_of(probe, g1)
    frame, _ = frame_of(env, 'pregame_all', {g1: members_of(probe, g1)[:2] + [105]})
    assert frame.attrs['skipped_players'][(g1, 60)] == {105: 'member_of_other_game'}
    assert 105 in members_of(frame, g2) and 105 not in members_of(frame, g1)
    same(frame, probe)                                                      # skipping him changes nothing else


def test_v_a_report_naming_a_team_that_is_not_in_the_game_skips_the_player_with_that_reason(env):
    probe, _ = frame_of(env, 'pregame_all', {})
    gid = date_games(env)['game_id'].iloc[0]
    team_out = int(date_games(env)['home_team_id'].iloc[1])               # a team that does not play in game 1
    q = non_members(env, probe, gid, 1)[0]
    rep = pd.concat([env.reports, _late_report(env, gid, q, team_out, 90.0, 'Questionable')], ignore_index=True)       # the latest report usable at T-60
    frame, _ = frame_of(env, 'pregame_all', {gid: [q]}, reports=rep)
    assert frame.attrs['skipped_players'][(gid, 60)] == {q: 'report_team_not_in_game'}
    assert q not in set(frame.loc[frame['game_id'] == gid, 'player_id'])


# ---------------------------------------------------------------- (vi) the date's reports, one code path ----------------------------------------------------------------
def test_vi_a_report_of_another_game_removes_a_player_from_his_old_team_even_when_only_one_game_has_a_due_slot(env):
    g = date_games(env)
    g1 = g['game_id'].iloc[0]                                              # team 1 v team 2; player 105 (team 1 until today) is named by game 2's report for team 3
    full, _ = frame_of(env, 'pregame_all', {})
    only1, _ = frame_of(env, 'pregame_all', {}, only={g1})
    no_sched, _ = frame_of(env, 'pregame_all', {}, only={g1}, schedule=None)
    same(only1[only1['game_id'] == g1], full[full['game_id'] == g1])
    assert 105 not in set(full.loc[full['game_id'] == g1, 'player_id'])
    assert 105 in set(no_sched.loc[no_sched['game_id'] == g1, 'player_id'])          # without the schedule the other game's report is not seen: the documented limitation


def test_vi_the_parity_script_and_the_signal_path_give_the_same_frame(env, monkeypatch):
    sys.path.insert(0, str(ROOT / 'scripts'))
    par = importlib.import_module('props_d1_parity')
    monkeypatch.setattr(par, 'DB', 'unused')
    monkeypatch.setattr(par, 'SEASON', SEASON)
    probe, _ = frame_of(env, 'pregame_all', {})
    gids = list(date_games(env)['game_id'])
    quoted = {gids[0]: members_of(probe, gids[0])[:3] + non_members(env, probe, gids[0], 1), gids[1]: members_of(probe, gids[1])[:2] + non_members(env, probe, gids[1], 1)}
    cons = pd.concat([pd.DataFrame({'game_id': g, 'player_id': p, 'minutes_before': T}) for g, p in quoted.items() for T in GRID], ignore_index=True)
    sched = schedule_of(env)
    skipped = []
    base_all, blocks, unresolved = par.build_live_frames_pregame([DATE], env.hist, POS, env.reports, env.tips, env.tables, sched, 'pregame_all', cons=cons, skipped_out=skipped)
    frame = live_frame(base_all, blocks)
    got, _ = frame_of(env, 'pregame_all', quoted, slots=GRID)
    same(frame, got)
    assert skipped == []
    nq = par.build_live_frames_pregame([DATE], env.hist, POS, env.reports, env.tips, env.tables, sched, 'pregame_all')           # the no-quotes behaviour is still available
    nq_frame = live_frame(nq[0], nq[1])
    assert set(map(tuple, nq_frame[['game_id', 'player_id']].drop_duplicates().to_numpy().tolist())) <= set(map(tuple, frame[['game_id', 'player_id']].drop_duplicates().to_numpy().tolist()))
    assert par.parse_args(['--attempt', '1']).with_quotes is False and par.parse_args(['--attempt', '1', '--with-quotes']).with_quotes is True
