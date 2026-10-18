"""
sbrscrape with browser-like request headers.

sportsbookreview.com answers sbrscrape's bare `requests.get(url)` (default python-requests
User-Agent) with HTTP 403, and sbrscrape swallows the error, so every odds fetch silently came
back empty. Import Scoreboard from here instead of from sbrscrape:

    from core.sbr import Scoreboard
"""
import requests
import sbrscrape

HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/126.0 Safari/537.36'),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
}
TIMEOUT = 30


class _RequestsWithHeaders:
    """Stand-in for the `requests` module inside sbrscrape: same API, default headers + timeout."""

    def __getattr__(self, name):
        return getattr(requests, name)

    @staticmethod
    def get(url, **kwargs):
        kwargs['headers'] = {**HEADERS, **(kwargs.get('headers') or {})}
        kwargs.setdefault('timeout', TIMEOUT)
        return requests.get(url, **kwargs)


sbrscrape.requests = _RequestsWithHeaders()
Scoreboard = sbrscrape.Scoreboard
