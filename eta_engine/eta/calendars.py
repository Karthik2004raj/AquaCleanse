from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from dateutil import tz
import holidays as holidays_lib


TimeInterval = Tuple[time, time]


@dataclass
class WorkingHours:
    """Represents weekly working hours by weekday with one or more intervals per day.

    - Weekday numbering follows datetime.weekday(): Monday=0, Sunday=6
    - Each interval is a tuple of (start_time, end_time) with start < end
    """

    weekday_to_intervals: Dict[int, List[TimeInterval]]

    def intervals_for(self, day: date) -> List[TimeInterval]:
        return self.weekday_to_intervals.get(day.weekday(), [])

    def is_within(self, dt: datetime) -> bool:
        intervals = self.weekday_to_intervals.get(dt.weekday(), [])
        for start_time, end_time in intervals:
            if start_time <= dt.timetz().replace(tzinfo=None) < end_time:
                return True
        return False

    @staticmethod
    def simple_weekdays(start: time, end: time, weekend: Iterable[int] = (5, 6)) -> "WorkingHours":
        mapping: Dict[int, List[TimeInterval]] = {}
        for weekday in range(7):
            if weekday in weekend:
                continue
            mapping[weekday] = [(start, end)]
        return WorkingHours(weekday_to_intervals=mapping)


class HolidayProvider:
    """Abstract holiday provider interface."""

    def is_holiday(self, day: date) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class HolidaysLibraryProvider(HolidayProvider):
    """Holiday provider based on the `holidays` library.

    Args:
        country: ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB')
        subdiv: Optional subdivision such as state/province code
        observed: Count observed days when holidays fall on weekends
        expand: Expand dynamically to new years when queried
    """

    def __init__(
        self,
        country: str,
        subdiv: Optional[str] = None,
        observed: bool = True,
        expand: bool = True,
    ) -> None:
        self._holidays = holidays_lib.country_holidays(
            country=country, subdiv=subdiv, observed=observed, expand=expand
        )

    def is_holiday(self, day: date) -> bool:
        return day in self._holidays


class BusinessCalendar:
    """Business calendar with working hours, holidays, and timezone.

    Provides utilities to query working moments and add business time.
    """

    def __init__(
        self,
        working_hours: WorkingHours,
        holiday_provider: Optional[HolidayProvider] = None,
        timezone: Optional[str] = "UTC",
    ) -> None:
        self.working_hours = working_hours
        self.holiday_provider = holiday_provider
        self.tzinfo = tz.gettz(timezone) if isinstance(timezone, str) else timezone

    # ---------- Public API ----------
    def is_working_datetime(self, dt: datetime) -> bool:
        localized = self._ensure_tz(dt)
        if self._is_holiday(localized.date()):
            return False
        return self.working_hours.is_within(localized)

    def next_working_moment(self, dt: datetime) -> datetime:
        """Return dt if it is within working time; otherwise the next opening moment."""
        current = self._ensure_tz(dt)
        if self.is_working_datetime(current):
            return current
        # If not working, find the next interval start that is not a holiday
        probe_date = current.date()
        for _ in range(500):  # safety bound ~ 1.5 years
            if not self._is_holiday(probe_date):
                intervals = self.working_hours.intervals_for(probe_date)
                for start_time, end_time in intervals:
                    start_dt = self._combine(probe_date, start_time)
                    if current <= start_dt:
                        return start_dt
            probe_date = probe_date + timedelta(days=1)
        raise RuntimeError("Failed to find next working moment within search bound")

    def add_business_timedelta(self, start: datetime, delta: timedelta) -> datetime:
        """Add business time to a start moment, skipping non-working periods.

        Only supports non-negative durations.
        """
        if delta.total_seconds() < 0:
            raise ValueError("Negative business durations are not supported")
        if delta.total_seconds() == 0:
            return self.next_working_moment(start)

        remaining_seconds = int(delta.total_seconds())
        current = self.next_working_moment(start)

        while remaining_seconds > 0:
            interval_end = self._current_interval_end(current)
            if interval_end is None:
                # Not within working hours; jump to next opening
                current = self.next_working_moment(current)
                continue

            seconds_available = int((interval_end - current).total_seconds())
            if remaining_seconds <= seconds_available:
                return current + timedelta(seconds=remaining_seconds)
            # Consume the whole interval and move to next
            remaining_seconds -= seconds_available
            current = self._next_interval_start_after(interval_end)

        return current

    # ---------- Internals ----------
    def _current_interval_end(self, dtm: datetime) -> Optional[datetime]:
        """End of the current working interval containing dtm, or None if outside."""
        if self._is_holiday(dtm.date()):
            return None
        for start_time, end_time in self.working_hours.intervals_for(dtm.date()):
            start_dt = self._combine(dtm.date(), start_time)
            end_dt = self._combine(dtm.date(), end_time)
            if start_dt <= dtm < end_dt:
                return end_dt
        return None

    def _next_interval_start_after(self, dtm: datetime) -> datetime:
        probe_date = dtm.date()
        # Try remaining intervals today after dtm
        if not self._is_holiday(probe_date):
            for start_time, _ in self.working_hours.intervals_for(probe_date):
                candidate = self._combine(probe_date, start_time)
                if candidate > dtm:
                    return candidate
        # Move to next day with an opening interval
        probe_date = probe_date + timedelta(days=1)
        for _ in range(500):
            if not self._is_holiday(probe_date):
                intervals = self.working_hours.intervals_for(probe_date)
                if intervals:
                    start_time = intervals[0][0]
                    return self._combine(probe_date, start_time)
            probe_date = probe_date + timedelta(days=1)
        raise RuntimeError("Failed to locate next interval start within search bound")

    def _is_holiday(self, d: date) -> bool:
        if self.holiday_provider is None:
            return False
        return self.holiday_provider.is_holiday(d)

    def _combine(self, d: date, t: time) -> datetime:
        return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second, tzinfo=self.tzinfo)

    def _ensure_tz(self, dtm: datetime) -> datetime:
        if dtm.tzinfo is None:
            return dtm.replace(tzinfo=self.tzinfo)
        return dtm.astimezone(self.tzinfo)


__all__ = [
    "WorkingHours",
    "HolidayProvider",
    "HolidaysLibraryProvider",
    "BusinessCalendar",
]

