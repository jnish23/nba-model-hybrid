"""Vendor-agnostic pieces: the response container and the credit budget every adapter must go through."""
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """A request was refused because it could push credit use past a configured cap. Nothing was sent."""


@dataclass
class VendorResponse:
    """One API response, vendor-neutral. payload is the decoded JSON body."""
    endpoint: str
    payload: object
    snapshot_ts_utc: str            # when the odds are as of (historical: the vendor's snapshot time; live: fetch time)
    fetched_ts_utc: str
    cost: float | None = None       # credits the vendor says this call cost (None if it did not say)
    remaining: float | None = None
    raw_path: str | None = None
    meta: dict = field(default_factory=dict)


def utc_now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class CreditBudget:
    """
    Hard caps on credit use, persisted in SQLite so they survive restarts and are shared by every process.

    A request is checked before it is sent against its worst-case cost (estimate): the call is refused if it
    would exceed the per-run, per-UTC-day or total cap. After the call the vendor-reported cost is recorded.
    total counts everything ever recorded in the ledger, so set it to the plan's allowance minus a reserve.
    (The total is not reset when the vendor's monthly quota resets: raise it or clear the ledger then.)
    """

    def __init__(self, db_path, per_run=None, per_day=None, total=None):
        self.db_path = str(db_path)
        self.per_run, self.per_day, self.total = per_run, per_day, total
        self.run_used = 0.0
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS api_ledger (
                ts_utc TEXT, vendor TEXT, endpoint TEXT, params TEXT, estimate REAL, cost REAL, remaining REAL,
                status INTEGER, note TEXT)""")

    def _used(self, since=None):
        with sqlite3.connect(self.db_path) as conn:
            q, args = "SELECT COALESCE(SUM(COALESCE(cost, estimate)), 0) FROM api_ledger", ()
            if since:
                q, args = q + " WHERE ts_utc >= ?", (since,)
            return float(conn.execute(q, args).fetchone()[0])

    def used_today(self):
        return self._used(datetime.now(timezone.utc).strftime('%Y-%m-%dT00:00:00Z'))

    def check(self, estimate):
        if estimate <= 0:
            return                      # free calls (events listings) are never blocked by a credit cap
        with self._lock:
            if self.per_run is not None and self.run_used + estimate > self.per_run:
                raise BudgetExceeded(f'per-run cap {self.per_run}: used {self.run_used:g} + estimate {estimate:g}')
            if self.per_day is not None and self.used_today() + estimate > self.per_day:
                raise BudgetExceeded(f'per-day cap {self.per_day}: used {self.used_today():g} + estimate {estimate:g}')
            if self.total is not None and self._used() + estimate > self.total:
                raise BudgetExceeded(f'total cap {self.total}: used {self._used():g} + estimate {estimate:g}')

    def record(self, vendor, endpoint, params, estimate, cost, remaining, status, note=''):
        with self._lock:
            self.run_used += cost if cost is not None else estimate
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO api_ledger VALUES (?,?,?,?,?,?,?,?,?)",
                             (utc_now_iso(), vendor, endpoint, json.dumps(params, sort_keys=True), estimate, cost,
                              remaining, status, note))
        logging.info(f'{vendor} {endpoint}: cost {cost}, remaining {remaining} (run total {self.run_used:g})')
