"""storage/models.py – shared data models for persistence."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class JobLog(BaseModel):
    timestamp: str
    job_id: str
    event_type: str
    tool_name: str | None = None
    inputs: dict | None = None
    outputs: Any = None
    error: Any = None
    latency_ms: float | None = None
    success: bool | None = None
