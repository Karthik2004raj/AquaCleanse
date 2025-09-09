from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from .calendars import BusinessCalendar
from .models import DurationType, Pipeline, PipelineStep


@dataclass
class StepResult:
    step: PipelineStep
    start: datetime
    end: datetime


class ETAEngine:
    def __init__(self, calendar: BusinessCalendar) -> None:
        self.calendar = calendar

    def compute(self, start: datetime, pipeline: Pipeline) -> List[StepResult]:
        """Compute start/end for each step and return the detailed breakdown."""
        results: List[StepResult] = []
        current = self.calendar.next_working_moment(start)

        for step in pipeline.steps:
            step_start = current
            if step.duration.kind == DurationType.BUSINESS:
                step_end = self.calendar.add_business_timedelta(step_start, step.duration.to_timedelta())
            else:
                # Calendar time addition from the business-aligned step start
                step_end = step_start + step.duration.to_timedelta()
            results.append(StepResult(step=step, start=step_start, end=step_end))
            current = step_end

        return results

    def compute_eta(self, start: datetime, pipeline: Pipeline) -> datetime:
        breakdown = self.compute(start, pipeline)
        return breakdown[-1].end if breakdown else start


__all__ = [
    "ETAEngine",
    "StepResult",
]

