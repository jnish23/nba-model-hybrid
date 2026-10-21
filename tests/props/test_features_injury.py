import numpy as np
import pandas as pd
import pytest

from props.features import injury_at_t as inj

TIP = pd.Timestamp('2025-01-15 19:10')            # reference tip, ET naive


def reports():
    rows = [  # gameId, report_ts, personId, status, reason  (each report lists every player who has a status)
        ('0022400001', '2025-01-15 11:00', 1, 'Questionable', 'Injury/Illness - Left knee; Soreness'),
        ('0022400001', '2025-01-15 15:00', 1, 'Out', 'Injury/Illness - Left knee; Soreness'),
        ('0022400001', '2025-01-15 15:00', 2, 'Probable', 'Injury/Illness - N/a; Illness'),
        ('0022400001', '2025-01-15 18:30', 1, 'Out', 'Injury/Illness - Left knee; Soreness'),
        ('0022400001', '2025-01-15 18:30', 2, 'Probable', 'Injury/Illness - N/a; Illness'),
        ('0022400001', '2025-01-15 19:00', 1, 'Out', 'Injury/Illness - Left knee; Soreness'),
        ('0022400001', '2025-01-15 19:00', 2, 'Out', 'Injury/Illness - N/a; Illness'),
        ('0022400001', '2025-01-15 19:30', 2, 'Probable', 'Injury/Illness - N/a; Illness'),        # published after tip - 5: never visible at any T
    ]
    return pd.DataFrame(rows, columns=['gameId', 'report_ts', 'personId', 'status', 'reason'])


def test_reason_class():
    assert inj.reason_class('Injury/Illness - Left knee; Soreness') == 'injury'
    assert inj.reason_class('Injury/Illness - N/a; Illness') == 'illness'
    assert inj.reason_class('Injury/Illness - Left ankle; Injury Management') == 'injury'
    assert inj.reason_class('G League - Two-Way') == 'g_league'
    assert inj.reason_class('Rest') == 'rest' and inj.reason_class('Personal Reasons') == 'personal'
    assert inj.reason_class('-') == 'other'


def test_latest_report_is_the_last_one_at_or_before_T_and_never_a_later_one():
    tip = pd.Series({'0022400001': TIP})
    # cutoffs: T-240 = 15:10, T-60 = 18:10, T-5 = 19:05
    for x, ts in ((240, '2025-01-15 15:00'), (60, '2025-01-15 15:00'), (5, '2025-01-15 19:00')):
        rows, games = inj.latest_report_rows(reports(), tip, x)
        assert games.loc['0022400001', 'report_ts'] == pd.Timestamp(ts)
        assert (rows['report_ts'] <= TIP - pd.Timedelta(minutes=x)).all()
        assert games.loc['0022400001', 'report_age_min'] == pytest.approx((TIP - pd.Timedelta(minutes=x) - pd.Timestamp(ts)).total_seconds() / 60)
    # the 19:30 report (after the reference tip) is invisible even at T-0
    rows, _ = inj.latest_report_rows(reports(), tip, 0)
    assert (rows['report_ts'] <= TIP).all()
    # a game with no report before T has no row at all
    rows, games = inj.latest_report_rows(reports(), pd.Series({'0022400001': pd.Timestamp('2025-01-15 10:00')}), 5)
    assert len(rows) == 0 and len(games) == 0


def test_status_for_rows_listed_unlisted_and_missing_report():
    tip = pd.Series({'0022400001': TIP})
    chosen, games = inj.latest_report_rows(reports(), tip, 60)                       # the 15:00 report: player 1 Out, player 2 Probable
    roster = pd.DataFrame({'game_id': ['0022400001', '0022400001', '0022400001', '0022400002'], 'player_id': [1, 2, 3, 9], 'team_id': [10, 10, 10, 20]})
    chosen = chosen.assign(teamId=10)
    s = inj.status_for_rows(roster, chosen, games).set_index(['game_id', 'player_id'])
    assert s.loc[('0022400001', 1), 'inj_status'] == 'Out' and s.loc[('0022400001', 2), 'inj_status'] == 'Probable'
    assert s.loc[('0022400001', 3), 'inj_status'] == 'None'                                    # on no report: nothing to report
    assert s.loc[('0022400002', 9), 'inj_status'] == 'no_report' and s.loc[('0022400002', 9), 'inj_report_missing'] == 1
    assert s.loc[('0022400001', 1), 'inj_report_age_min'] == pytest.approx(190)


def test_play_rate_table_uses_only_earlier_seasons_and_shrinks_to_the_spec():
    h = pd.DataFrame({'season': ['2020-21'] * 40 + ['2021-22'] * 40, 'inj_status': ['Questionable'] * 80, 'm_bucket': [2] * 80,
                      'lbl_played': [1] * 20 + [0] * 20 + [1] * 40})
    t = inj.play_rate_table(h, 60, '2019-20')
    first, second = t[('2020-21', 'Questionable', 2)], t[('2021-22', 'Questionable', 2)]
    assert first == pytest.approx(0.57)                                                # no earlier season: exactly the spec rate
    assert second == pytest.approx((20 + inj.K_RATE * 0.57) / (40 + inj.K_RATE))       # 2021-22 sees only 2020-21's 20 of 40


def test_attach_p_play_falls_back_to_the_spec_rate_for_an_unseen_cell():
    status = pd.DataFrame({'game_id': ['0022400001'], 'player_id': [1], 'team_id': [10], 'inj_status': ['Doubtful']})
    feat = pd.DataFrame({'game_id': ['0022400001'], 'player_id': [1], 'season': ['2019-20'], 'm_bucket': [3]})
    d = inj.attach_p_play(status, feat, pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], [], []])))
    assert d['inj_p_play'].iloc[0] == inj.SPEC_P_PLAY['Doubtful'] and d['inj_status_code'].iloc[0] == 4


def test_teammate_availability_by_hand():
    # team A: players 1 (the subject), 2 and 3; team B: player 4
    d = pd.DataFrame({'game_id': ['G'] * 4, 'player_id': [1, 2, 3, 4], 'team_id': [10, 10, 10, 20], 'opp_team_id': [20, 20, 20, 10],
                      'inj_p_play': [0.9, 0.0, 0.5, 0.2], 'inj_status_code': [3, 5, 3, 5]})
    feat = pd.DataFrame({'game_id': ['G'] * 4, 'player_id': [1, 2, 3, 4], 'm_min_h15': [30., 20., 10., 25.],
                         'r_usage_pct_h15': [0.2] * 4, 'r_reb_chances_h15': [0.3] * 4, 'r_assists_h15': [0.1] * 4,
                         'r_secondary_ast_h15': [0.0] * 4, 'r_points_h15': [0.5] * 4})
    out = inj.teammate_availability(d, feat).set_index('player_id')
    # for player 1: teammates 2 (out, 20 min) and 3 (50%, 10 min): missing minutes share = (1*20 + 0.5*10) / (20 + 10) = 25/30
    assert out.loc[1, 'av_miss_minutes'] == pytest.approx(25 / 30)
    assert out.loc[1, 'av_n_out'] == 1 and out.loc[1, 'av_n_questionable'] == 1
    # for player 2 (himself out): teammates 1 (p .9, 30 min) and 3 (p .5, 10 min): (0.1*30 + 0.5*10) / 40
    assert out.loc[2, 'av_miss_minutes'] == pytest.approx((0.1 * 30 + 0.5 * 10) / 40)
    # the opponent of player 4 is team A: whole-roster missing share (0.1*30 + 1*20 + 0.5*10) / 60
    assert out.loc[4, 'opp_av_miss_minutes'] == pytest.approx((0.1 * 30 + 20 + 5) / 60)
    assert out.loc[1, 'opp_av_n_out'] == 1                                             # player 4 is Out


# ---- on the real database: the play-probability table must not see the labels it is applied to -----------------------
from conftest import DB_PATH, ROOT  # noqa: E402

PROPS_DB = ROOT / 'data' / 'props_odds.db'


@pytest.mark.db
@pytest.mark.skipif(not PROPS_DB.exists(), reason='props store not available')
def test_play_probabilities_use_only_labels_from_earlier_seasons():
    from props.features import build
    base, labels = build.build_base(str(DB_PATH))
    reports = build.load_linked_reports(str(DB_PATH))
    sub = base[base['season'] >= '2022-23'].reset_index(drop=True)
    lab = labels.merge(sub[['game_id', 'player_id']], on=['game_id', 'player_id'])
    a = build.build_at_t(sub, lab, str(DB_PATH), str(PROPS_DB), reports=reports)
    rng = np.random.default_rng(0)
    shuf = lab.merge(sub[['game_id', 'player_id', 'season']], on=['game_id', 'player_id'])
    sel = (shuf['season'] >= '2024-25').to_numpy()
    shuf.loc[sel, 'lbl_played'] = rng.permutation(shuf.loc[sel, 'lbl_played'].to_numpy())
    b = build.build_at_t(sub, shuf.drop(columns='season'), str(DB_PATH), str(PROPS_DB), reports=reports)
    key = ['game_id', 'player_id', 'minutes_before']
    m = a.merge(b, on=key, suffixes=('_a', '_b')).merge(sub[['game_id', 'player_id', 'season']], on=['game_id', 'player_id'])
    same = m[m['season'] <= '2024-25']
    assert np.allclose(same['inj_p_play_a'], same['inj_p_play_b'])                 # shuffling 2024-25 labels cannot move 2024-25 probabilities
    later = m[m['season'] == '2025-26']
    assert not np.allclose(later['inj_p_play_a'], later['inj_p_play_b'])           # ... but it does move 2025-26's, which learn from 2024-25
