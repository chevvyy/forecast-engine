"""Exchange calendar: early closes, the 210-minute session, and the time of day on such a day."""
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
import yaml

from forecast.data import alpaca, calendar
from forecast.features.build import time_of_day
from forecast.utils.common import REPO

NY = ZoneInfo("America/New_York")


def _minutes(day: date) -> pl.DataFrame:
    """One bar per minute from 08:00 to 17:59 New York on ``day`` (pre-market and late session included)."""
    t0 = datetime(day.year, day.month, day.day, 8, tzinfo=NY).astimezone(UTC)
    t = [t0 + timedelta(minutes=i) for i in range(600)]
    return pl.DataFrame({"t": t}, schema={"t": pl.Datetime("us", "UTC")})


def test_table_matches_the_published_2026_to_2028_calendar():
    assert calendar.early_closes_between(date(2026, 1, 1), date(2028, 12, 31)) == [
        date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26), date(2028, 7, 3), date(2028, 11, 24)]
    assert date(2026, 7, 2) not in calendar.EARLY_CLOSES          # 4 July on a Saturday: no early close
    assert all(d.weekday() < 5 and calendar.KNOWN_FROM <= d <= calendar.KNOWN_THROUGH
               for d in calendar.EARLY_CLOSES)
    assert calendar.session_seconds(date(2026, 11, 27)) == 210 * 60
    assert calendar.session_seconds(date(2026, 11, 25)) == 390 * 60


def test_half_day_session_has_210_minutes_not_390():
    half = alpaca.regular_session(_minutes(date(2026, 11, 27)))["t"].dt.convert_time_zone("America/New_York")
    full = alpaca.regular_session(_minutes(date(2026, 11, 25)))["t"].dt.convert_time_zone("America/New_York")
    assert (half.len(), half.min().strftime("%H:%M"), half.max().strftime("%H:%M")) == (210, "09:30", "12:59")
    assert (full.len(), full.min().strftime("%H:%M"), full.max().strftime("%H:%M")) == (390, "09:30", "15:59")
    u = lambda *a: datetime(*a, tzinfo=UTC)  # noqa: E731
    assert alpaca.has_session_minute(u(2026, 11, 27, 17, 59), u(2026, 11, 27, 23, 0))      # 12:59 EST
    assert not alpaca.has_session_minute(u(2026, 11, 27, 18, 0), u(2026, 11, 27, 23, 0))   # 13:00: closed
    assert alpaca.has_session_minute(u(2026, 11, 25, 18, 0), u(2026, 11, 25, 23, 0))       # a full day


@pytest.mark.leakage
def test_time_of_day_on_a_half_day_is_the_share_of_the_short_session():
    def at(*a):
        return int(datetime(*a, tzinfo=NY).timestamp())

    tod = time_of_day(np.array([at(2026, 11, 27, 9, 30), at(2026, 11, 27, 12, 45), at(2026, 11, 27, 13, 0),
                                at(2026, 11, 27, 15, 0), at(2026, 11, 25, 12, 45), at(2028, 7, 3, 11, 15)]))
    np.testing.assert_allclose(tod, [0.0, 195 / 210, 1.0, 1.0, 195 / 390, 0.5])
    # the clock and the published calendar only: the same instants give the same answer whatever
    # order or company they come in (nothing is estimated from the rows themselves)
    one = time_of_day(np.array([at(2026, 11, 27, 12, 45)]))
    assert one[0] == tod[1]


def test_minute_data_pins_sit_inside_the_table_and_minute_2_has_no_half_day():
    for path in sorted((REPO / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = cfg.get("data") or {}
        if not str(data.get("interval", "1d")).endswith("m") or "end" not in data:
            continue
        first, last = date.fromisoformat(str(data["start"])), date.fromisoformat(str(data["end"]))
        assert first >= calendar.KNOWN_FROM and last <= calendar.KNOWN_THROUGH, path.name
        if cfg["eval"]["version"] == "minute-2":
            # The calendar landed inside minute-2 without a version bump because it changes no bar and
            # no time of day in this window. A window that contains a half-day needs a new version.
            assert calendar.early_closes_between(first, last) == [], path.name
