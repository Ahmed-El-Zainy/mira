"""api/models.py – request / response Pydantic models."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    query: str = Field(..., examples=["Analyse the near-term prospects of Tesla, Inc. (TSLA)."])


class MonitoringRequest(BaseModel):
    ticker: str = Field(..., examples=["TSLA"])
    cadence_hours: int = Field(24, ge=1, le=168)


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    result: dict | None = None
    error: str | None = None
    tool_calls_used: int = 0
    budget_exceeded: bool | None = False
    tag: str | None = ""
    created_at: str | None = None
