from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple

from .calendars import BusinessCalendar


def next_working_moment(calendar: BusinessCalendar, start: datetime) -> datetime:
    return calendar.next_working_moment(start)


def add_business_duration(
    calendar: BusinessCalendar, start: datetime, duration: timedelta
) -> datetime:
    return calendar.add_business_timedelta(start, duration)


def add_mixed_duration(
    calendar: BusinessCalendar,
    start: datetime,
    business: timedelta,
    calendar_time: timedelta,
) -> Tuple[datetime, datetime]:
    """Add a mix of business time and absolute calendar time.

    Returns a tuple of (after_business, after_calendar).
    """
    after_business = calendar.add_business_timedelta(start, business)
    after_calendar = after_business + calendar_time
    return after_business, after_calendar


__all__ = [
    "next_working_moment",
    "add_business_duration",
    "add_mixed_duration",
]

