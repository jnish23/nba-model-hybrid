"""Forward-test segments on the props dashboard: the definitions file, the pure assignment and summary, the real page under AppTest (with and without segments in the export), the methodology
sentence, the configuration-driven disclosure, and that the published bundle stays self-sufficient. Synthetic data only; no network."""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import props_board_logic as L
from props.live import export, publish

ROOT = Path(__file__).resolve().parents[2]
SEGMENTS_FILE = ROOT / 'config' / 'props_forward_segments.json'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load('test_wp4_dashboard')                 # make(), write_export(), live_bets(), run_app()
REAL = json.loads(SEGMENTS_FILE.read_text(encoding='utf-8'))
FROZEN_RULE = json.loads((ROOT / 'config' / 'props_frozen_rule.json').read_text(encoding='utf-8'))
C3_RULE = json.loads((ROOT / 'reports' / 'props' / 'audit_2026-11' / 'candidate_rule_a11.json').read_text(encoding='utf-8'))
CAND_CFG = json.loads((ROOT / 'config' / 'props_live_candidate.json').read_text(encoding='utf-8'))
LIVE_CFG = json.loads((ROOT / 'config' / 'props_live.json').read_text(encoding='utf-8'))
V2_CFG = {'model_version': 'v3_live_2026-27r1', 'rule_sha256': 'b597763ab3b13fd9cf295b1de802255ac87fe6e9bbf065103171db49eec2b6fb', 'roster_rule': 'v2'}      # segment 1's configuration (the live config until the switch of 2026-11-14)

# synthetic bets (W.make) run 2026-10-22..10-31 and 2026-11-01..11-10: three test segments cut that span
SEGS = [{'key': 'a', 'label': 'Early', 'to_game_date': '2026-10-31'},
        {'key': 'b', 'label': 'Middle', 'from_game_date': '2026-11-01', 'to_game_date': '2026-11-05', 'model_version': 'v2', 'roster_rule': 'v2'},
        {'key': 'c', 'label': 'Latest', 'from_game_date': '2026-11-06', 'model_version': 'v2', 'roster_rule': 'v2'}]
DIAG_NUMERIC = [c for c in L.DIAG_COLS if c != 'too_small']


# --- the definitions file -----------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_real_definitions_match_the_configurations_they_describe():
    segs = L.clean_segments(REAL)
    assert [s['key'] for s in segs] == ['pre', 'seg1', 'seg2'] and [s['label'] for s in segs] == ['Before the forward test', 'Segment 1', 'Segment 2']
    pre, s1, s2 = segs
    assert pre['to_game_date'] == '2026-11-12' and 'from_game_date' not in pre and not any(k in pre for k in ('model_version', 'rule_sha256', 'roster_rule'))
    assert (s1['from_game_date'], s1['to_game_date']) == ('2026-11-13', '2026-11-13')
    assert (s1['model_version'], s1['rule_sha256'], s1['roster_rule']) == (V2_CFG['model_version'], V2_CFG['rule_sha256'], V2_CFG['roster_rule'])
    assert s2['from_game_date'] == CAND_CFG['effective_from'] == '2026-11-14' and 'to_game_date' not in s2
    assert (s2['model_version'], s2['rule_sha256'], s2['roster_rule']) == (CAND_CFG['model_version'], CAND_CFG['rule_sha256'], CAND_CFG['roster_rule'])
    assert (s2['model_version'], s2['rule_sha256'], s2['roster_rule'], s2['from_game_date']) == (LIVE_CFG['model_version'], LIVE_CFG['rule_sha256'], LIVE_CFG['roster_rule'], LIVE_CFG['effective_from'])      # the open segment is the deployed configuration
    for s in (s1, s2):
        assert (ROOT / s['preregistration']).exists()
    assert L.current_segment(REAL)['key'] == 'seg2'
    assert export.load_segments() == REAL and export.load_segments(ROOT / 'no_such_file.json') is None


def test_segment_assignment_at_the_boundaries():
    d = pd.DataFrame({'game_date': ['2026-11-12', '2026-11-13', '2026-11-14', '2026-10-22', '2027-04-12', None]})
    got = L.assign_segments(d, REAL)
    assert list(got) == ['pre', 'seg1', 'seg2', 'pre', 'seg2', 'unassigned']          # the open-ended last segment takes every later date; an unreadable date is unassigned
    gap = [s for s in L.clean_segments(REAL) if s['key'] != 'seg1']                    # no segment covers 2026-11-13 now
    assert list(L.assign_segments(d, gap)) == ['pre', 'unassigned', 'seg2', 'pre', 'seg2', 'unassigned']
    assert list(L.assign_segments(d, None)) == ['unassigned'] * 6 and list(L.assign_segments(d.iloc[0:0], REAL)) == []
    assert list(L.assign_segments(pd.DataFrame({'x': [1]}), REAL)) == ['unassigned']    # no game_date column
    assert list(L.assign_segments(pd.DataFrame({'game_date': [pd.Timestamp('2026-11-13')]}), REAL)) == ['seg1']
    assert L.current_segment(None) is None and L.current_segment([s for s in REAL if 'to_game_date' in s]) is None


# --- segment_summary ----------------------------------------------------------------------------------------------------------------------------------------------------------

def same(a, b):
    for c in DIAG_NUMERIC:
        assert (np.isnan(a[c]) and np.isnan(b[c])) or a[c] == pytest.approx(b[c]), c
    assert bool(a['too_small']) == bool(b['too_small'])


def test_segment_rows_equal_the_version_summary_of_the_same_subset_and_the_overall_row_the_pooled_one():
    d = W.make(n_dates=20, per_date=30, shock=0.1, seed=11).assign(model_version='v2')
    d.loc[::7, 'status_stands'] = np.nan                                               # some pending bets
    t = L.segment_summary(d, SEGS)
    assert list(t['segment']) == ['a', 'b', 'c', 'all'] and list(t.columns) == L.SEGMENT_COLS
    key = L.assign_segments(d, SEGS)
    for k in 'abc':
        sub = d[key == k]
        ref = L.version_summary(sub).iloc[0]                                          # the 'all' row of the same subset
        row = t[t['segment'] == k].iloc[0]
        same(row, ref)
        assert row['bets'] == ref['bets'] == len(sub) and row['first_game_date'] == ref['first_game_date'] and row['last_game_date'] == ref['last_game_date']
        assert row['game_dates'] == sub['game_date'].nunique()
        flat = L.date_cluster_roi(sub[sub['status_stands'].isin(['win', 'loss'])], 'flat', 'stands', 300, 0)
        assert row['roi_flat'] == pytest.approx(flat['roi']) and row['roi_flat_lo'] == pytest.approx(flat['lo'])
    pooled, over = L.version_summary(d).iloc[0], t[t['segment'] == 'all'].iloc[0]
    same(over, pooled)
    assert over['label'] == L.OVERALL_LABEL and over['bets'] == len(d) == t.loc[t['segment'] != 'all', 'bets'].sum()
    assert over['first_game_date'] == '2026-10-22' and over['mismatched'] == 0


def test_small_segments_are_greyed_empty_segments_stay_and_unassigned_appears_only_when_needed():
    d = W.make(n_dates=4, per_date=10, seed=12)                                         # 40 bets on 2026-10-22..25
    t = L.segment_summary(d, SEGS)
    assert list(t['segment']) == ['a', 'b', 'c', 'all'] and t['too_small'].all()      # fewer than MIN_STAKES stakes everywhere: greyed, kept
    assert t.loc[t['segment'] == 'b', 'bets'].iloc[0] == 0 and np.isnan(t.loc[t['segment'] == 'b', 'roi_median'].iloc[0]) and t.loc[t['segment'] == 'b', 'first_game_date'].iloc[0] == ''
    gap = [SEGS[1], SEGS[2]]                                                           # nothing covers October
    u = L.segment_summary(d, gap)
    assert list(u['segment']) == ['b', 'c', 'unassigned', 'all'] and u.loc[u['segment'] == 'unassigned', 'bets'].iloc[0] == 40 and u.loc[u['segment'] == 'unassigned', 'label'].iloc[0] == L.UNASSIGNED_LABEL
    assert list(L.segment_summary(d, None)['segment']) == ['unassigned', 'all']
    assert list(L.segment_summary(d.iloc[0:0], SEGS)['segment']) == ['a', 'b', 'c', 'all']
    big = W.make(n_dates=20, per_date=30, seed=13)
    assert not L.segment_summary(big, SEGS).loc[lambda x: x['segment'] == 'a', 'too_small'].iloc[0]
    dup = pd.concat([big, big])                                                         # a non-unique index must not break the lookups
    assert L.segment_summary(dup, SEGS)['bets'].tolist() == [2 * n for n in L.segment_summary(big, SEGS)['bets']]


def test_mismatched_bets_are_counted_per_segment_and_never_dropped():
    d = W.make(n_dates=20, per_date=10, seed=14).assign(model_version='v2', roster_rule='v2', rule_sha256='h')
    segs = [dict(s, rule_sha256='h') if s['key'] != 'a' else s for s in SEGS]          # segment 'a' states nothing: nothing to check there
    key = L.assign_segments(d, segs)

    def idx(k):
        return d.index[(key == k).to_numpy()]
    d.loc[idx('a')[:3], 'model_version'] = 'other'                                       # not checked: segment 'a' has no definition
    d.loc[idx('b')[:4], 'model_version'] = 'v1'
    d.loc[idx('b')[4:6], 'rule_sha256'] = 'x'
    d.loc[idx('c')[:1], 'roster_rule'] = 'pregame_all'
    d.loc[idx('c')[1:3], 'roster_rule'] = np.nan                                         # blank where the definition states a value: a difference
    d.loc[idx('b')[0], 'rule_sha256'] = 'x'                                              # the same bet differs twice: still one bet
    t = L.segment_summary(d, segs).set_index('segment')
    assert t['mismatched'].to_dict() == {'a': 0, 'b': 6, 'c': 3, 'all': 9}
    assert t['bets'].to_dict() == {k: int((key == k).sum()) for k in 'abc'} | {'all': len(d)}      # nothing dropped
    assert list(L.segment_mismatch(d, segs)).count(True) == 9
    assert not L.segment_mismatch(d.drop(columns=['model_version', 'roster_rule', 'rule_sha256']), segs).any()      # no such columns in an older export: nothing to compare


# --- the page ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def tracker_frames(app):
    return {i: df for i, df in enumerate(app.dataframe)}


def seg_table(app):
    for df in app.dataframe:
        v = df.value
        if 'Segment' in v.columns and 'Mismatched' in v.columns:
            return v
    return None


def test_the_tracker_shows_the_segment_table_headline_and_selector(tmp_path, monkeypatch):
    W.write_export(tmp_path / 'seg', W.live_bets(), meta_extra={'segments': SEGS})
    app = W.run_app(monkeypatch, tmp_path / 'seg')
    text, labels = W.check_all_pages(app)
    t = seg_table(app)
    assert t is not None and list(t['Segment']) == ['Early', 'Middle', 'Latest', L.OVERALL_LABEL]
    assert list(t.columns[:5]) == ['Segment', 'First date', 'Last date', 'Bets recorded', 'Game dates'] and {'ROI median price [95% CI]', 'ROI best quoted price [95% CI]', 'ROI flat -110 [95% CI]', 'CLV pts (n)', 'Model version', 'Mismatched', 'Readable?'} <= set(t.columns)
    assert t['Bets recorded'].sum() == 2 * len(W.live_bets()) and t['Mismatched'].sum() == 0            # the overall row repeats the pool
    assert list(t['Model version']) == ['mixed', 'v2', 'v2', 'mixed']
    assert any(c.value.startswith('A segment is a stretch of game dates') and 'only the segment rows are forward-test evidence' in c.value and 'mixes configurations' in c.value for c in app.caption)
    assert any(m.value == '**By forward-test segment**' for m in app.markdown)
    order = [m.value for m in app.markdown if m.value in ('**By forward-test segment**', '**By model version**')]
    assert order == ['**By forward-test segment**', '**By model version**']                           # the segment table sits above the version table
    metrics = {m.label: m.value for m in app.metric}
    assert 'Overall ROI, median price (all segments)' in metrics and 'Latest ROI, median price (current segment)' in metrics
    assert metrics['Overall ROI, median price (all segments)'] == metrics['ROI, median price (primary)']      # nothing filtered yet
    head = [c.value for c in app.caption if c.value.startswith('Overall: ')]
    assert head and 'Latest (from 2026-11-06 onward, model v2)' in head[0] and 'Only the segment is forward-test evidence' in head[0]
    sel = [s for s in app.selectbox if s.label == 'Segment'][0]
    assert sel.options == ['All', 'Early', 'Middle', 'Latest'] and sel.value == 'All'
    ver = [s for s in app.selectbox if s.label == 'Model version'][0]
    assert ver.options == ['All', 'v1', 'v2']
    n_all = int([m.value for m in app.metric if m.label == 'Settled bets'][0].replace(',', ''))
    bets = W.live_bets()
    key = L.assign_segments(bets, SEGS)
    expect = {name: int(((key == k) & bets['status_stands'].isin(['win', 'loss'])).sum()) for k, name in (('a', 'Early'), ('b', 'Middle'), ('c', 'Latest'))}
    for name in ('Early', 'Middle', 'Latest'):
        sel.set_value(name).run()
        assert not app.exception, [e.value for e in app.exception]
        n = int([m.value for m in app.metric if m.label == 'Settled bets'][0].replace(',', ''))
        assert n == expect[name] < n_all
        assert seg_table(app) is not None and len(seg_table(app)) == 4                                   # the segment table itself always shows every row
        assert [m.value for m in app.metric if m.label == 'Overall ROI, median price (all segments)'] == [metrics['Overall ROI, median price (all segments)']]     # the headline ignores the filters
        sel = [s for s in app.selectbox if s.label == 'Segment'][0]
    sel.set_value('Early').run()
    ver = [s for s in app.selectbox if s.label == 'Model version'][0]
    ver.set_value('v2').run()
    assert not app.exception and any('No bets for this selection' in i.value for i in app.info)          # Early holds only v1 bets: both filters apply
    app.sidebar.radio[1].set_value('best').run()
    assert not app.exception, [e.value for e in app.exception]


def test_the_page_flags_mismatched_bets(tmp_path, monkeypatch):
    bets = W.live_bets().assign(roster_rule='v2')
    bets.loc[bets['game_date'] >= '2026-11-06', 'roster_rule'] = 'pregame_all'
    W.write_export(tmp_path / 'bad', bets, meta_extra={'segments': SEGS})
    app = W.run_app(monkeypatch, tmp_path / 'bad')
    assert not app.exception, [e.value for e in app.exception]
    assert any("differs from their segment's definition" in w.value for w in app.warning)
    assert seg_table(app).loc[lambda x: x['Segment'] == 'Latest', 'Mismatched'].iloc[0] == int((bets['game_date'] >= '2026-11-06').sum())


def test_an_old_export_without_segments_renders_without_the_table(tmp_path, monkeypatch):
    W.write_export(tmp_path / 'old', W.live_bets())
    app = W.run_app(monkeypatch, tmp_path / 'old')
    text, labels = W.check_all_pages(app)
    assert seg_table(app) is None and not any('forward-test segment' in m.value for m in app.markdown)
    assert not [s for s in app.selectbox if s.label == 'Segment'] and not any('current segment' in l or 'all segments' in l for l in labels)
    assert [s for s in app.selectbox if s.label == 'Model version']                                   # the version selector is unaffected
    assert 'forward-test segments' not in text
    junk = tmp_path / 'junk'
    W.write_export(junk, W.live_bets(), meta_extra={'segments': 'not a list'})
    app = W.run_app(monkeypatch, junk)
    assert not app.exception and seg_table(app) is None


def test_a_segment_selector_without_a_model_version_selector(tmp_path, monkeypatch):
    W.write_export(tmp_path / 'nov', W.live_bets(), new=False, meta_extra={'segments': SEGS})          # no model_version column: only the segment filter is offered
    app = W.run_app(monkeypatch, tmp_path / 'nov')
    assert not app.exception, [e.value for e in app.exception]
    assert [s for s in app.selectbox if s.label == 'Segment'] and not [s for s in app.selectbox if s.label == 'Model version']
    [s for s in app.selectbox if s.label == 'Segment'][0].set_value('Latest').run()
    assert not app.exception, [e.value for e in app.exception]


def test_the_about_page_lists_the_segments_and_the_rule_sentence_follows_the_export(tmp_path, monkeypatch):
    W.write_export(tmp_path / 'c3', W.live_bets(), meta_extra={'segments': REAL, 'rule': C3_RULE})
    app = W.run_app(monkeypatch, tmp_path / 'c3')
    assert not app.exception, [e.value for e in app.exception]
    md = '\n'.join(m.value for m in app.markdown)
    assert '- **Before the forward test**: up to 2026-11-12, mixed model versions.' in md
    assert '- **Segment 1**: on 2026-11-13, model `v3_live_2026-27r1`, roster rule `v2`.' in md
    assert '- **Segment 2**: from 2026-11-14 onward, model `v4_live_2026-27c1`, roster rule `pregame_all`.' in md
    assert 'The first version (`v3_live_2026-27`, October 22-31) was **restored' in md                # the existing sentences stay
    assert 'Shin de-vig' in md and 'candidate rule C3' in md and 'p* = market + k x (model - market)' not in md
    W.write_export(tmp_path / 'fr', W.live_bets())
    app = W.run_app(monkeypatch, tmp_path / 'fr')
    md = '\n'.join(m.value for m in app.markdown)
    assert '- Model probability shrunk toward the market: p* = market + k x (model - market), k by stat: Rebounds 0.26, Assists 0.29, 3-pointers 0.31, Reb + Ast 0.23.' in md
    assert 'forward-test segments, each run' not in md


# --- the methodology sentence --------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_frozen_rule_sentence_is_the_old_one_word_for_word():
    k = FROZEN_RULE['k_by_stat']
    old = f"Model probability shrunk toward the market: p* = market + k x (model - market), k by stat: {', '.join(f'{L.STAT_LABEL[s]} {v:.2f}' for s, v in k.items())}."      # the expression app.py used before
    assert L.pstar_text(FROZEN_RULE) == old
    assert L.pstar_text(FROZEN_RULE) == ('Model probability shrunk toward the market: p* = market + k x (model - market), k by stat: Assists 0.29, Reb + Ast 0.23, Rebounds 0.26, 3-pointers 0.31.')


def test_the_c3_sentence_names_shin_the_intercepts_the_clip_and_all_four_k_values():
    t = L.pstar_text(C3_RULE)
    assert 'candidate rule C3' in t and 'Shin de-vig' in t and 'mean over all books' in t and 'intercept' in t and 'above or below the market' in t
    assert 'clipped to between 0.01 and 0.99' in t
    for stat, v in C3_RULE['k_by_stat'].items():
        if stat in C3_RULE['stats']:
            assert f'{L.STAT_LABEL[stat]} {v:.2f}' in t
        a = C3_RULE['intercept_by_stat'].get(stat)
        if a:
            assert f"{L.STAT_LABEL[stat]} {a['a_pos']:+.3f} / {a['a_neg']:+.3f}" in t
    assert t.count('k by stat:') == 1 and t.endswith('.')
    unknown = L.pstar_text({'candidate': 'C9', 'k_by_stat': {'assists': 0.3, 'ra': 0.2}})
    assert 'C9' in unknown and 'Assists 0.30' in unknown and 'Reb + Ast 0.20' in unknown and 'x (model' not in unknown and 'Shin' not in unknown
    odd = L.pstar_text({**C3_RULE, 'pmarket': {'devig_method': 'whatever', 'consensus_set': 'some_set'}, 'p_star_clip': None})
    assert 'whatever' in odd and 'some_set' in odd and 'the range in the rule file' in odd


# --- the disclosure -------------------------------------------------------------------------------------------------------------------------------------------------------------

OLD_DISCLOSURE = ('Paper record of the frozen rule with the live model and roster rule v2. The live inputs do not reproduce the training inputs (teammate availability cannot be rebuilt '
                  'before the game): in the parity test on 40 held-out 2025-26 dates the live-feature bets earned about 1.7 points less than the training-feature bets (+3.9% vs +5.6% at the '
                  'selected price), A1/A4/A5/A6 were waived by the owner, and October coverage is about 60%. See reports/props/d1_parity_report.md.')


def test_disclosure_for_roster_rule_v2_is_the_old_text_byte_for_byte():
    assert export.DISCLOSURE == OLD_DISCLOSURE
    assert export.disclosure(V2_CFG, FROZEN_RULE) == OLD_DISCLOSURE and export.disclosure(V2_CFG) == OLD_DISCLOSURE
    assert export.disclosure({'roster_rule': 'v2', 'waivers': ['whatever']}) == OLD_DISCLOSURE and export.disclosure({}) == OLD_DISCLOSURE


def test_disclosure_for_a_pregame_roster_rule_follows_the_configuration():
    t = export.disclosure(CAND_CFG, C3_RULE)
    assert 'v4_live_2026-27c1' in t and 'candidate rule C3' in t and 'roster rule pregame_all' in t
    assert 'same code' in t and 'reports/props/audit_2026-11/6a_roster/report.md' in t
    assert 'A4' not in t and 'cannot be rebuilt' not in t and 'frozen rule with' not in t
    assert 'The owner waived: ' + '; '.join(CAND_CFG['waivers']) + '.' in t and CAND_CFG['waivers'] == ['A1 (October coverage ~60%)', 'A5', 'A6']
    assert 'The record before 2026-11-14 was produced by earlier configurations (see the segment table on the Tracker page).' in t
    assert (Path(__file__).resolve().parents[2] / 'reports/props/audit_2026-11/6a_roster/report.md').exists()
    other = export.disclosure({**CAND_CFG, 'waivers': ['A7 (made up)'], 'roster_rule': 'v2nq', 'effective_from': '2027-01-02'}, FROZEN_RULE)
    assert 'The owner waived: A7 (made up).' in other and 'A1' not in other and 'frozen rule' in other and 'roster rule v2nq' in other and 'before 2027-01-02' in other
    none = export.disclosure({'roster_rule': 'pregame_all', 'model_version': 'm'})
    assert 'No waiver is recorded' in none and 'owner waived' not in none and 'before this configuration took effect' in none


def test_the_export_writes_segments_and_the_disclosure_into_meta(tmp_path):
    from props.live import store
    h = W.export_helpers()
    conn = store.connect(tmp_path / 'live.db')
    bets = pd.concat([h._bet(1, game_date='2026-11-13', model_version='v3_live_2026-27r1', rule_sha256=V2_CFG['rule_sha256'], roster_rule='v2'),
                      h._bet(2, game_date='2026-11-14', model_version='v4_live_2026-27c1', rule_sha256='z', roster_rule='pregame_all')])
    store.write_slot(conn, 'g1', 60, 'done', '2026-11-14T00:42:00Z', 'r1', bets=bets)
    export.export_live(tmp_path / 'live.db', tmp_path / 'out', h.CONFIG, n_boot=20)
    meta = json.loads((tmp_path / 'out' / 'meta.json').read_text())
    assert meta['segments'] == REAL and meta['disclosure'] == OLD_DISCLOSURE                           # the test config has roster rule v2
    csv = pd.read_csv(tmp_path / 'out' / 'bets.csv')
    assert csv['rule_sha256'].tolist() == [V2_CFG['rule_sha256'], 'z'] and csv['roster_rule'].tolist() == ['v2', 'pregame_all']
    seg = L.segment_summary(csv, meta['segments']).set_index('segment')
    assert seg['bets'].to_dict() == {'pre': 0, 'seg1': 1, 'seg2': 1, 'all': 2}
    assert seg['mismatched'].to_dict() == {'pre': 0, 'seg1': 0, 'seg2': 1, 'all': 1}                    # bet 2 carries rule hash 'z', not segment 2's
    cand = {**h.CONFIG, **{k: CAND_CFG[k] for k in ('roster_rule', 'waivers', 'effective_from', 'model_version')}, 'rule_file': CAND_CFG['rule_file']}
    export.export_live(tmp_path / 'live.db', tmp_path / 'out2', cand, n_boot=20, segments=[])
    meta2 = json.loads((tmp_path / 'out2' / 'meta.json').read_text())
    assert 'segments' not in meta2 and meta2['disclosure'] == export.disclosure(cand, C3_RULE) and 'candidate rule C3' in meta2['disclosure']
    assert meta2['rule']['candidate'] == 'C3'
    sm = pd.read_csv(tmp_path / 'out' / 'season_summary.csv')
    assert list(sm['model_version']) == ['all', 'v3_live_2026-27r1', 'v4_live_2026-27c1'] and list(sm.columns) == export.SUMMARY_COLS         # season_summary.csv is unchanged: no segment rows


# --- the published bundle -------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_bundle_is_self_sufficient_for_the_segment_features(tmp_path):
    pytest.importorskip('streamlit')
    pytest.importorskip('plotly')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    files = publish.build_bundle(bundle, ROOT, live_dir=tmp_path / 'none', backtest_dir=tmp_path / 'none')
    assert 'core/props_board_logic.py' in files and 'dashboard_props/app.py' in files
    assert not any('props_forward_segments' in f for f in files)                                          # the definitions travel inside meta.json, not as a file
    for f in ('dashboard_props/app.py', 'core/props_board_logic.py'):
        src = (bundle / f).read_text(encoding='utf-8')
        assert not re.search(r'^\s*(from|import)\s+props\b', src, re.M) and 'config/' not in src.replace('config/props_forward_segments.json, copied', '')      # imports nothing the bundle does not ship and reads no config file
    code = ("import sys, json\n"
            f"sys.path.insert(0, {str(bundle)!r})\n"
            "import core.props_board_logic as m\n"
            "segs = json.loads(sys.argv[1])\n"
            "import pandas as pd\n"
            "d = pd.DataFrame({'game_date': ['2026-11-13', '2026-11-14']})\n"
            "print(list(m.assign_segments(d, segs)))\n"
            "print(m.pstar_text(json.loads(sys.argv[2])))\n")
    r = subprocess.run([sys.executable, '-I', '-c', code, json.dumps(REAL), json.dumps(C3_RULE)], cwd=bundle, capture_output=True, text=True)      # -I: only the bundle's own core/ is importable
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.splitlines()[0] == "['seg1', 'seg2']" and 'Shin de-vig' in r.stdout
