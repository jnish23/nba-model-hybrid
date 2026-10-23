import datetime as dt

from core import odds_fallback as fb
from core.odds_vendor.base import CreditBudget, VendorResponse
from core.odds_vendor.theoddsapi import TheOddsApi
from team.live_odds import split_open_and_current

NOW = dt.datetime(2026, 10, 23, 16, 0, tzinfo=dt.timezone.utc)
EVENT = {
    'id': 'e1', 'commence_time': '2026-10-24T02:10:00Z', 'home_team': 'Los Angeles Clippers', 'away_team': 'Portland Trail Blazers',
    'bookmakers': [
        {'key': 'draftkings', 'markets': [
            {'key': 'h2h', 'outcomes': [{'name': 'Los Angeles Clippers', 'price': -200}, {'name': 'Portland Trail Blazers', 'price': 170}]},
            {'key': 'spreads', 'outcomes': [{'name': 'Los Angeles Clippers', 'price': -110, 'point': -5.5},
                                            {'name': 'Portland Trail Blazers', 'price': -110, 'point': 5.5}]},
            {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -108, 'point': 221.5}, {'name': 'Under', 'price': -112, 'point': 221.5}]}]},
        {'key': 'williamhill_us', 'markets': [
            {'key': 'h2h', 'outcomes': [{'name': 'Los Angeles Clippers', 'price': -190}, {'name': 'Portland Trail Blazers', 'price': 160}]}]},
        {'key': 'bovada', 'markets': [                      # a book the model does not use
            {'key': 'h2h', 'outcomes': [{'name': 'Los Angeles Clippers', 'price': -500}, {'name': 'Portland Trail Blazers', 'price': 400}]}]},
    ]}


def test_event_becomes_the_sbr_game_shape_with_sbr_book_names():
    g = fb.event_to_sbr_game(EVENT)
    assert g['home_ml'] == {'draftkings': -200, 'caesars': -190} and g['away_ml'] == {'draftkings': 170, 'caesars': 160}
    assert g['home_spread'] == {'draftkings': -5.5} and g['away_spread'] == {'draftkings': 5.5}
    assert g['total'] == {'draftkings': 221.5} and g['over_odds'] == {'draftkings': -108} and g['under_odds'] == {'draftkings': -112}
    assert g['home_spread_odds'] == {'draftkings': -110}
    assert 'bovada' not in str(g)


def test_the_converted_game_goes_through_the_normal_odds_path_unchanged():
    g = fb.event_to_sbr_game(EVENT)
    out = split_open_and_current(g, g)
    assert out['odds_home_ml'] == -200 and out['odds_away_ml'] == 170 and out['odds_home_spread'] == -5.5 and out['odds_total'] == 221.5
    assert out['curr_odds_home_spread'] == -5.5 and out['odds_open_missing'] == 0
    assert out['curr_price_over_odds'] == -108


def test_matching_by_nickname_bridges_la_clippers_and_the_api_name():
    games = fb.index_by_matchup([EVENT, {'id': 'e2', 'commence_time': '2026-10-24T00:00:00Z', 'home_team': 'Boston Celtics',
                                         'away_team': 'New York Knicks', 'bookmakers': []}], now=NOW)
    assert list(games) == [('blazers', 'clippers')]             # an event without lines is not indexed
    assert fb.find_game(games, 'Portland Trail Blazers', 'LA Clippers') is games[('blazers', 'clippers')]
    assert fb.find_game(games, 'Portland Trail Blazers', 'Los Angeles Lakers') is None


def test_started_and_far_off_games_are_not_used():
    started = dict(EVENT, commence_time='2026-10-23T15:00:00Z')              # in progress: its lines are live lines
    christmas = dict(EVENT, commence_time='2026-12-25T19:30:00Z')            # a later rematch of the same matchup
    assert fb.index_by_matchup([started, christmas], now=NOW) == {}
    tonight, tomorrow = dict(EVENT), dict(EVENT, commence_time='2026-10-24T20:00:00Z', id='later')
    assert fb.index_by_matchup([tomorrow, tonight], now=NOW)[('blazers', 'clippers')]['commence_time'] == '2026-10-24T02:10:00Z'


def test_has_lines_is_false_for_the_empty_sbr_game():
    assert not fb.has_lines({'home_ml': {}, 'total': {}, 'home_team': 'x'})
    assert not fb.has_lines({})
    assert fb.has_lines({'home_ml': {'draftkings': -110}})


class _Resp:
    def __init__(self, body):
        self._b, self.status_code, self.ok, self.text = body, 200, True, '[]'
        self.headers = {'x-requests-last': '3', 'x-requests-remaining': '4990'}

    def json(self):
        return self._b


class _Session:
    def __init__(self, body):
        self.body, self.calls = body, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        return _Resp(self.body)


def test_bulk_call_costs_three_credits_for_five_bookmakers_and_never_leaks_the_key(tmp_path):
    budget = CreditBudget(tmp_path / 'l.db', per_run=10)
    sess = _Session([EVENT])
    api = TheOddsApi(budget, api_key='SECRETKEY', raw_dir=tmp_path / 'raw', session=sess)
    games = fb.fetch_team_odds(api=api, now=NOW)
    url, params = sess.calls[0]
    assert url.endswith('/sports/basketball_nba/odds') and params['markets'] == 'h2h,spreads,totals'
    assert set(params['bookmakers'].split(',')) == set(fb.API_BOOKMAKERS) and 'regions' not in params
    assert budget.run_used == 3 and list(games) == [('blazers', 'clippers')]
    assert 'SECRETKEY' not in open(next((tmp_path / 'raw').rglob('*.json.gz')), 'rb').read().decode('latin1')


def test_the_backup_never_stops_the_run(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setattr('core.odds_vendor.theoddsapi.load_env', lambda *a, **k: None)
    assert fb.fetch_team_odds(budget_db=tmp_path / 'l.db', raw_dir=tmp_path / 'raw') == {}
    assert 'Odds API backup unavailable' in caplog.text

    class Boom:
        def team_odds(self, **k):
            raise RuntimeError('HTTP 429')
    assert fb.fetch_team_odds(api=Boom()) == {}
