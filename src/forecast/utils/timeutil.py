"""UTC <-> exchange time helpers. No naive datetimes leave this module."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

EXCHANGE_TZ = "America/New_York"

TIMEFRAME_MINUTES = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "1Day": 390}


def utc(dt: datetime | date | str) -> datetime:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def now_utc() -> datetime:
    return datetime.now(UTC)


def parse_duration(s: str) -> timedelta:
    """'3M' -> ~3 months, '2W', '10D'. Months are 30.44 days (block sizing only)."""
    n, unit = int(s[:-1]), s[-1].upper()
    days = {"D": 1, "W": 7, "M": 30.44, "Y": 365.25}[unit]
    return timedelta(days=n * days)
