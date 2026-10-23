"""
The Odds API (v4) adapter.

Credit rules (https://the-odds-api.com/liveapi/guides/v4/), used for the pre-flight estimate:
  /sports/{sport}/events                        free
  /sports/{sport}/odds                          markets returned x regions        (every game's team lines in one call)
  /sports/{sport}/events/{id}/odds              markets returned x regions        (props; the estimate is the worst case)
  /historical/sports/{sport}/events             1
  /historical/sports/{sport}/events/{id}/odds   10 x markets returned x regions
When `bookmakers` is given instead of `regions`, every group of 10 bookmakers counts as one region.

The API key comes from the ODDS_API_KEY environment variable (or the constructor) and is never logged or written
to the raw response files. Every response body is saved (gzip JSON) before anything parses it, so a parser fix
never costs credits.
"""
import gzip
import json
import logging
import math
import os
import time
from pathlib import Path

import requests

from core.env import load_env
from core.odds_vendor.base import BudgetExceeded, CreditBudget, VendorResponse, utc_now_iso  # noqa: F401

BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT = 'basketball_nba'
RETRY_STATUS = {429, 500, 502, 503, 504}


class OddsApiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f'HTTP {status}: {message}')
        self.status = status


def _n_regions(regions, bookmakers):
    if bookmakers:
        return math.ceil(len(_as_list(bookmakers)) / 10)
    return len(_as_list(regions))


def _as_list(v):
    return [x for x in (v.split(',') if isinstance(v, str) else list(v or [])) if x]


class TheOddsApi:
    name = 'theoddsapi'

    def __init__(self, budget: CreditBudget, api_key=None, raw_dir='data/raw/props_api', session=None,
                 sport=SPORT, timeout=30, max_retries=2):
        self._key = (api_key or os.environ.get('ODDS_API_KEY') or (load_env() and os.environ.get('ODDS_API_KEY')) or '').strip()
        if not self._key:
            raise RuntimeError('ODDS_API_KEY is not set (put it in the gitignored .env file or the user environment; never commit it)')
        self.budget, self.raw_dir, self.sport = budget, Path(raw_dir), sport
        self.session = session or requests.Session()
        self.timeout, self.max_retries = timeout, max_retries

    # ---- public API -------------------------------------------------------------------------------------
    def events(self, commence_from=None, commence_to=None):
        params = {k: v for k, v in (('commenceTimeFrom', commence_from), ('commenceTimeTo', commence_to)) if v}
        return self._get(f'/sports/{self.sport}/events', params, estimate=0.0, tag='events')

    def team_odds(self, markets=('h2h', 'spreads', 'totals'), regions='us', bookmakers=None):
        """Moneyline / spread / total of EVERY upcoming game in one call (the bulk endpoint): cost = markets x regions."""
        markets = _as_list(markets)
        params = {'markets': ','.join(markets), 'oddsFormat': 'american', 'dateFormat': 'iso'}
        params.update({'bookmakers': ','.join(_as_list(bookmakers))} if bookmakers else {'regions': ','.join(_as_list(regions))})
        est = len(markets) * _n_regions(regions, bookmakers)
        return self._get(f'/sports/{self.sport}/odds', params, estimate=est, tag='team_odds')

    def event_odds(self, event_id, markets, regions='us', bookmakers=None):
        markets = _as_list(markets)
        params = {'markets': ','.join(markets), 'oddsFormat': 'american', 'dateFormat': 'iso'}
        params.update({'bookmakers': ','.join(_as_list(bookmakers))} if bookmakers else {'regions': ','.join(_as_list(regions))})
        est = len(markets) * _n_regions(regions, bookmakers)
        return self._get(f'/sports/{self.sport}/events/{event_id}/odds', params, estimate=est, tag=f'odds_{event_id}')

    def historical_events(self, date_iso):
        return self._get(f'/historical/sports/{self.sport}/events', {'date': date_iso}, estimate=1.0,
                         tag='hist_events', historical=True)

    def historical_event_odds(self, event_id, date_iso, markets, regions='us', bookmakers=None):
        markets = _as_list(markets)
        params = {'date': date_iso, 'markets': ','.join(markets), 'oddsFormat': 'american', 'dateFormat': 'iso'}
        params.update({'bookmakers': ','.join(_as_list(bookmakers))} if bookmakers else {'regions': ','.join(_as_list(regions))})
        est = 10.0 * len(markets) * _n_regions(regions, bookmakers)
        return self._get(f'/historical/sports/{self.sport}/events/{event_id}/odds', params, estimate=est,
                         tag=f'hist_odds_{event_id}', historical=True)

    # ---- internals --------------------------------------------------------------------------------------
    def _get(self, path, params, estimate, tag, historical=False):
        self.budget.check(estimate)                    # raises BudgetExceeded before anything is sent
        url = BASE_URL + path
        resp = None
        for attempt in range(self.max_retries + 1):
            resp = self.session.get(url, params={**params, 'apiKey': self._key}, timeout=self.timeout)
            if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                break
            time.sleep(2 ** attempt)
        fetched = utc_now_iso()
        cost = _float(resp.headers.get('x-requests-last'))
        remaining = _float(resp.headers.get('x-requests-remaining'))
        self.budget.record(self.name, path, params, estimate, cost, remaining, resp.status_code,
                           '' if resp.ok else resp.text[:200].replace(self._key, '***'))
        if not resp.ok:
            raise OddsApiError(resp.status_code, resp.text[:300].replace(self._key, '***'))
        body = resp.json()
        snapshot_ts = body.get('timestamp', fetched) if historical and isinstance(body, dict) else fetched
        payload = body.get('data', body) if historical and isinstance(body, dict) else body
        raw_path = self._save_raw(tag, snapshot_ts, path, params, resp, body)
        return VendorResponse(endpoint=path, payload=payload, snapshot_ts_utc=snapshot_ts, fetched_ts_utc=fetched,
                              cost=cost, remaining=remaining, raw_path=raw_path,
                              meta={k: body.get(k) for k in ('previous_timestamp', 'next_timestamp')}
                              if historical and isinstance(body, dict) else {})

    def _save_raw(self, tag, snapshot_ts, path, params, resp, body):
        try:
            day = snapshot_ts[:10]
            out = self.raw_dir / day / f"{tag}_{snapshot_ts.replace(':', '').replace('-', '')}.json.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            doc = {'request': {'path': path, 'params': params}, 'fetched_ts_utc': utc_now_iso(),
                   'headers': {k: resp.headers.get(k) for k in ('x-requests-last', 'x-requests-remaining', 'x-requests-used')},
                   'response': body}
            with gzip.open(out, 'wt', encoding='utf-8') as f:
                json.dump(doc, f)
            return str(out)
        except OSError as e:
            logging.error(f'could not save the raw response ({e}); continuing')
            return None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
