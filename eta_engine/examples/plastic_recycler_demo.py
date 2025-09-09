from __future__ import annotations

from datetime import datetime, time

from dateutil import tz

from eta.calendars import BusinessCalendar, HolidaysLibraryProvider, WorkingHours
from eta.eta_engine import ETAEngine
from eta.models import DurationSpec, DurationType, Pipeline, PipelineStep


def build_calendar():
    # Example: 9-17 Monday-Friday, US holidays, timezone UTC
    working = WorkingHours.simple_weekdays(start=time(9, 0), end=time(17, 0))
    holidays = HolidaysLibraryProvider(country="US")
    return BusinessCalendar(working_hours=working, holiday_provider=holidays, timezone="UTC")


def build_pipeline() -> Pipeline:
    # An example recycler->supplier pipeline with mixed business and calendar durations
    return Pipeline(
        steps=[
            PipelineStep(
                name="Recycler intake & sorting",
                duration=DurationSpec(kind=DurationType.BUSINESS, hours=4),
                description="Initial sorting and preparation at recycler",
            ),
            PipelineStep(
                name="Pickup scheduling",
                duration=DurationSpec(kind=DurationType.BUSINESS, hours=8),
                description="Book and confirm carrier pickup window",
            ),
            PipelineStep(
                name="Linehaul transit",
                duration=DurationSpec(kind=DurationType.CALENDAR, hours=36),
                description="Over-the-road transit not constrained by facility hours",
            ),
            PipelineStep(
                name="Supplier inbound & QA",
                duration=DurationSpec(kind=DurationType.BUSINESS, hours=6),
                description="Receiving, QA inspection, and putaway",
            ),
        ]
    )


def main():
    calendar = build_calendar()
    engine = ETAEngine(calendar)

    # Start now in UTC
    start = datetime.now(tz=tz.UTC)
    pipeline = build_pipeline()

    breakdown = engine.compute(start, pipeline)
    eta = breakdown[-1].end

    print("Start:", start.isoformat())
    for r in breakdown:
        print(f"{r.step.name}: {r.start.isoformat()} -> {r.end.isoformat()}")
    print("ETA:", eta.isoformat())


if __name__ == "__main__":
    main()

