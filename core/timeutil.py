"""
Time handling for odds work. Rule: every stored timestamp is UTC ISO-8601 with a trailing Z ('2026-10-21T23:30:00Z').
Eastern time exists only for the NBA's notion of a game date (the ET calendar date of tip-off) and for display.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
UTC = timezone.utc


def parse_utc(ts):
    """ISO string (Z or offset; naive strings are taken as UTC) -> aware UTC datetime."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def to_utc_iso(ts):
    return parse_utc(ts).strftime('%Y-%m-%dT%H:%M:%SZ')


def et_game_date(commence_time_utc):
    """The NBA game date: the Eastern calendar date of tip-off ('2026-10-21T01:00:00Z' -> '2026-10-20')."""
    return parse_utc(commence_time_utc).astimezone(ET).strftime('%Y-%m-%d')


def minutes_to_tip(snapshot_ts_utc, commence_time_utc):
    return (parse_utc(commence_time_utc) - parse_utc(snapshot_ts_utc)).total_seconds() / 60.0


def now_utc_iso():
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
