"""Early closes of the New York exchanges (NYSE / Nasdaq cash equities), for minute bars.

On an early-close day the regular session is 09:30-13:00 New York, 210 minutes instead of 390. What
trades between 13:00 and 16:00 on such a day is the late session, not the regular one, so those bars
are dropped on load (``alpaca.regular_session``) and the time of day of a row is measured against the
day's own session length (``features.build.time_of_day``). The dates are published years ahead, so
knowing a day's close at any moment of that day is not look-ahead.

A table, not a rule: the exchange decides each year (there was no early close on 2020-07-02 or
2026-07-02, for instance, when 4 July fell on a Saturday). 2026-2028 are from the NYSE Group calendar
announced 2025-12-23; 2016-2025 were written from memory of the published calendars and not re-checked
against a source (no minute track loads those years). Full holidays are not listed: a closed day has
no bars. Unscheduled closures and shortened days are not known here either.
Days after ``KNOWN_THROUGH`` are treated as full sessions; ``tests/test_calendar.py`` keeps the data
pins of the committed minute configs inside the table.
"""
from __future__ import annotations

from datetime import date

import numpy as np

FULL_SECONDS = 23400       # 09:30-16:00
EARLY_SECONDS = 12600      # 09:30-13:00
KNOWN_FROM, KNOWN_THROUGH = date(2016, 1, 1), date(2028, 12, 31)

EARLY_CLOSES: frozenset[date] = frozenset(date.fromisoformat(d) for d in (
    "2016-11-25",
    "2017-07-03", "2017-11-24",
    "2018-07-03", "2018-11-23", "2018-12-24",
    "2019-07-03", "2019-11-29", "2019-12-24",
    "2020-11-27", "2020-12-24",
    "2021-11-26",
    "2022-11-25",
    "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
    "2027-11-26",
    "2028-07-03", "2028-11-24",
))


def session_seconds(day: date) -> int:
    """Length of the regular session that opens 09:30 New York on ``day``."""
    return EARLY_SECONDS if day in EARLY_CLOSES else FULL_SECONDS


def early_closes_between(first: date, last: date) -> list[date]:
    """Early-close days in ``first..last`` (inclusive), sorted."""
    return sorted(d for d in EARLY_CLOSES if first <= d <= last)


def session_seconds_of(days: np.ndarray) -> np.ndarray:
    """``session_seconds`` for an array of New York dates given as days since 1970-01-01."""
    early = np.array(sorted((d - date(1970, 1, 1)).days for d in EARLY_CLOSES), dtype=np.int64)
    return np.where(np.isin(np.asarray(days, dtype=np.int64), early), EARLY_SECONDS, FULL_SECONDS)
