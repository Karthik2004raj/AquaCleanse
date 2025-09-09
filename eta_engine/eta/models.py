from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DurationType(str, Enum):
    BUSINESS = "business_hours"
    CALENDAR = "calendar_hours"


class DurationSpec(BaseModel):
    kind: DurationType = Field(description="Whether to count business or calendar hours")
    hours: float = Field(ge=0, description="Number of hours for this step")

    def to_timedelta(self) -> timedelta:
        # Convert hours to seconds via timedelta
        return timedelta(seconds=int(self.hours * 3600))


class PipelineStep(BaseModel):
    name: str
    duration: DurationSpec
    description: Optional[str] = None


class Pipeline(BaseModel):
    steps: List[PipelineStep]

