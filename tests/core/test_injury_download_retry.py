"""core.scrape_injury_reports.live_with_retries: the injury-report download is tried 3 times with a short backoff (injectable sleep,
stubbed download, no network). The existing live() and fetch() are not changed."""
import logging

import pytest

import core.scrape_injury_reports as sir


class Sleeper:
    def __init__(self):
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)


def _live(results):
    """A stand-in for live(): returns / raises the items in order; records the as_of it was asked for."""
    asked = []

    def live(as_of=None):
        asked.append(as_of)
        item = results[min(len(asked) - 1, len(results) - 1)]
        if isinstance(item, Exception):
            raise item
        return item
    live.asked = asked
    return live


def test_a_first_try_success_returns_at_once_without_sleeping():
    sl, live = Sleeper(), _live(['p.pdf'])
    assert sir.live_with_retries(as_of='cutoff', sleep_fn=sl, live_fn=live) == 'p.pdf'
    assert sl.slept == [] and live.asked == ['cutoff']


def test_none_and_exceptions_are_retried_with_a_backoff_between_tries():
    sl, live = Sleeper(), _live([None, TimeoutError('slow'), 'p.pdf'])
    assert sir.live_with_retries(as_of='cutoff', sleep_fn=sl, live_fn=live) == 'p.pdf'
    assert len(live.asked) == 3 and sl.slept == [10, 30]


def test_three_failures_return_none_log_an_error_and_do_not_sleep_after_the_last(caplog):
    sl, live = Sleeper(), _live([None, RuntimeError('x'), None])
    with caplog.at_level(logging.INFO):
        assert sir.live_with_retries(as_of='cutoff', sleep_fn=sl, live_fn=live) is None
    assert len(live.asked) == 3 and sl.slept == [10, 30]
    assert any(r.levelno == logging.ERROR and 'download FAILED' in r.getMessage() for r in caplog.records)


def test_attempts_and_backoff_are_configurable_and_it_never_raises():
    sl, live = Sleeper(), _live([ValueError('x')])
    assert sir.live_with_retries(attempts=2, backoff_seconds=(1,), sleep_fn=sl, live_fn=live) is None
    assert len(live.asked) == 2 and sl.slept == [1]


def test_defaults_are_three_attempts_and_a_short_backoff():
    assert sir.LIVE_ATTEMPTS == 3 and sir.LIVE_BACKOFF_SECONDS == (10, 30)


def test_it_calls_the_module_level_live_so_it_can_be_replaced(monkeypatch):
    live = _live(['q.pdf'])
    monkeypatch.setattr(sir, 'live', live)
    assert sir.live_with_retries(as_of='t', sleep_fn=Sleeper()) == 'q.pdf' and live.asked == ['t']
